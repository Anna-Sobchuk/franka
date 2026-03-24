#!/usr/bin/env python3
"""
session_monitor.py
──────────────────
Passive telemetry and automatic watchdog for the Franka FR3 pick-and-place
system.  Run alongside the existing nodes — no changes to any other script
are required or made.

    python3 ~/franka/scripts/session_monitor.py

──────────────────────────────────────────────────────────────────────────────
Design contract
──────────────────────────────────────────────────────────────────────────────
READ-ONLY subscriptions (never published to, never modified):
  /detected_object/point                        — detection events
  /franka_picker/done                           — cycle completion events
  /franka_gripper_1/franka_gripper/joint_states — live jaw width
  /camera/color/image_raw                       — camera heartbeat (FPS only)

WRITE: new topics, not consumed by any existing node:
  /session_monitor/cycle_count        Int32    total completed cycles this session
  /session_monitor/last_cycle_s       Float64  duration of most recent cycle (s)
  /session_monitor/mean_cycle_s       Float64  mean cycle duration over session (s)
  /session_monitor/min_grasp_width_cm Float64  min jaw width seen during last grasp
  /session_monitor/mean_grasp_width_cm Float64 mean grasp width over session
  /session_monitor/gripper_width_cm   Float64  current live jaw separation (cm)
  /session_monitor/camera_fps         Float64  rolling camera frame rate estimate
  /session_monitor/armed              Bool     detector armed state (inferred)
  /session_monitor/status             String   human-readable state string

WATCHDOG write (same semantics as picker's own signal — Bool(True)):
  /franka_picker/done  — published ONLY when picker is stuck > WATCHDOG_TIMEOUT s.
                         Equivalent to the manual fix documented in README:
                         rostopic pub /franka_picker/done std_msgs/Bool "data: true" -1
──────────────────────────────────────────────────────────────────────────────
What the terminal summary shows (every SUMMARY_INTERVAL seconds):
  • Session elapsed time
  • Number of completed pick-and-place cycles
  • Last / mean / min / max cycle duration
  • Current gripper jaw width
  • Min grasp width per cycle (quality indicator — lower = tighter contact)
  • Mean grasp width across all completed grasps
  • Camera frame rate
  • Current system state (ARMED / BUSY / IDLE)
  • Watchdog configuration and intervention count
──────────────────────────────────────────────────────────────────────────────
"""

import threading
import time
from collections import deque
from datetime import timedelta

import rospy
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Float64, Int32, String

# ── Tunable parameters ────────────────────────────────────────────────────────
GRIPPER_NS       = '/franka_gripper_1/franka_gripper'
WATCHDOG_TIMEOUT = 90.0   # seconds — auto re-arm if picker busy longer than this
SUMMARY_INTERVAL = 15.0   # seconds between printed session summaries
CYCLE_HISTORY    = 100    # rolling window size for cycle-time statistics
CAM_FPS_WINDOW   = 30     # frames used for rolling FPS estimate
# ─────────────────────────────────────────────────────────────────────────────


class SessionMonitor:
    """
    Passively observes the Franka pick-and-place pipeline and:
      1. Measures per-cycle timing and grasp quality metrics.
      2. Publishes those metrics to /session_monitor/* for external tools.
      3. Prints a formatted session summary to the ROS log periodically.
      4. Automatically re-arms the detector if the picker becomes stuck.

    All state mutations happen inside self._lock so the class is safe for
    concurrent callbacks from multiple ROS spinner threads.
    """

    def __init__(self) -> None:
        rospy.init_node('session_monitor', anonymous=False)
        self._lock = threading.Lock()

        # ── Timing ───────────────────────────────────────────────────
        self._session_start    : float              = time.time()
        self._cycle_start_time : float | None       = None
        self._cycle_durations  : deque[float]       = deque(maxlen=CYCLE_HISTORY)
        self._cycle_count      : int                = 0

        # ── System state (inferred from topic activity) ───────────────
        self._picker_busy : bool = False
        self._armed       : bool = True   # assumed armed at startup

        # ── Gripper tracking ─────────────────────────────────────────
        # Current live jaw separation, updated at ~40 Hz from joint_states
        self._gripper_width_cm : float | None = None
        # Minimum jaw width observed during each active cycle.
        # This captures the actual grasp contact width (not the post-release
        # open width), which is a quality indicator for each pick attempt.
        self._cycle_min_gripper_cm : float | None = None
        # Stored min-grasp-width for every completed cycle
        self._grasp_widths_cm : list[float] = []

        # ── Camera health ────────────────────────────────────────────
        self._cam_timestamps : deque[float] = deque(maxlen=CAM_FPS_WINDOW)

        # ── Watchdog ─────────────────────────────────────────────────
        # Guard flag prevents multiple watchdog triggers for the same stuck cycle
        self._watchdog_armed_for_cycle : bool = False
        self._watchdog_count           : int  = 0

        # ── Subscribers (all pre-existing topics, read-only) ──────────
        rospy.Subscriber(
            '/detected_object/point', PointStamped,
            self._on_detection, queue_size=1)

        rospy.Subscriber(
            '/franka_picker/done', Bool,
            self._on_done, queue_size=1)

        rospy.Subscriber(
            f'{GRIPPER_NS}/joint_states', JointState,
            self._on_gripper_state, queue_size=1)

        # Lightweight subscription — we only record the arrival timestamp,
        # never decode the image payload, so callback overhead is negligible.
        rospy.Subscriber(
            '/camera/color/image_raw', Image,
            self._on_camera_frame, queue_size=1)

        # ── Publishers (new topics — not consumed by any existing node) ─
        self._pub_cycle_count      = rospy.Publisher(
            '/session_monitor/cycle_count',        Int32,   queue_size=1, latch=True)
        self._pub_last_cycle       = rospy.Publisher(
            '/session_monitor/last_cycle_s',       Float64, queue_size=1, latch=True)
        self._pub_mean_cycle       = rospy.Publisher(
            '/session_monitor/mean_cycle_s',       Float64, queue_size=1, latch=True)
        self._pub_min_grasp        = rospy.Publisher(
            '/session_monitor/min_grasp_width_cm', Float64, queue_size=1, latch=True)
        self._pub_mean_grasp       = rospy.Publisher(
            '/session_monitor/mean_grasp_width_cm',Float64, queue_size=1, latch=True)
        self._pub_gripper_live     = rospy.Publisher(
            '/session_monitor/gripper_width_cm',   Float64, queue_size=1, latch=True)
        self._pub_camera_fps       = rospy.Publisher(
            '/session_monitor/camera_fps',         Float64, queue_size=1, latch=True)
        self._pub_armed            = rospy.Publisher(
            '/session_monitor/armed',              Bool,    queue_size=1, latch=True)
        self._pub_status           = rospy.Publisher(
            '/session_monitor/status',             String,  queue_size=1, latch=True)

        # Watchdog publisher — publishes Bool(True) to /franka_picker/done.
        # This is identical to the picker's own done signal, so object_detector.py
        # will call its done_cb → re-arm, exactly as if the picker finished normally.
        self._pub_rearm = rospy.Publisher(
            '/franka_picker/done', Bool, queue_size=1)

        # ── Timers ────────────────────────────────────────────────────
        rospy.Timer(rospy.Duration(1.0),              self._watchdog_tick)
        rospy.Timer(rospy.Duration(0.5),              self._publish_metrics)
        rospy.Timer(rospy.Duration(SUMMARY_INTERVAL), self._summary_tick)

        self._print_startup_banner()

    # =========================================================================
    # Callbacks — all existing-topic subscribers
    # =========================================================================

    def _on_detection(self, msg: PointStamped) -> None:
        """
        Fires when object_detector.py publishes a stable can position.
        Marks the start of a new pick cycle and arms the watchdog for it.
        """
        with self._lock:
            self._cycle_start_time        = time.time()
            self._picker_busy             = True
            self._armed                   = False
            self._watchdog_armed_for_cycle = True
            self._cycle_min_gripper_cm    = self._gripper_width_cm  # current open width

        rospy.logdebug(
            f"[Monitor] Cycle started — "
            f"can at cam ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})")

    def _on_done(self, msg: Bool) -> None:
        """
        Fires when franka_picker.py (or the watchdog) publishes done=True.
        Records cycle duration and the minimum gripper width seen during the cycle.
        """
        if not msg.data:
            return

        with self._lock:
            if self._cycle_start_time is not None and self._picker_busy:
                # Record timing
                duration = time.time() - self._cycle_start_time
                self._cycle_durations.append(duration)
                self._cycle_count += 1

                # Record grasp quality: minimum jaw width seen during this cycle.
                # The gripper closes in step 5 (grasp) and re-opens in step 9
                # (release).  The minimum during the cycle is the contact width.
                if self._cycle_min_gripper_cm is not None:
                    self._grasp_widths_cm.append(self._cycle_min_gripper_cm)

            # Reset cycle state
            self._picker_busy             = False
            self._armed                   = True
            self._cycle_start_time        = None
            self._cycle_min_gripper_cm    = None
            self._watchdog_armed_for_cycle = False

    def _on_gripper_state(self, msg: JointState) -> None:
        """
        Updates live gripper width and tracks the per-cycle minimum.
        The Franka Hand reports two finger positions; their sum is the jaw gap.
        """
        if len(msg.position) < 2:
            return

        width_cm = sum(msg.position) * 100.0   # metres → centimetres

        with self._lock:
            self._gripper_width_cm = width_cm

            # Track minimum during an active cycle — this captures the grasp
            # contact width when the gripper closes around the can (step 5).
            if self._picker_busy and self._cycle_min_gripper_cm is not None:
                if width_cm < self._cycle_min_gripper_cm:
                    self._cycle_min_gripper_cm = width_cm

    def _on_camera_frame(self, msg: Image) -> None:
        """
        Records the frame arrival timestamp for rolling FPS estimation.
        The image payload is never decoded — overhead is a single list append.
        """
        with self._lock:
            self._cam_timestamps.append(time.time())

    # =========================================================================
    # Timer callbacks
    # =========================================================================

    def _watchdog_tick(self, _event) -> None:
        """
        Runs every second.  If the picker has been busy longer than
        WATCHDOG_TIMEOUT and we have not already intervened for this cycle,
        force-publish True to /franka_picker/done to un-stick the detector.

        This is the automated equivalent of the manual recovery command from
        the project README:
            rostopic pub /franka_picker/done std_msgs/Bool "data: true" -1
        """
        fire = False
        count = 0

        with self._lock:
            stuck = (
                self._picker_busy
                and self._watchdog_armed_for_cycle
                and self._cycle_start_time is not None
                and (time.time() - self._cycle_start_time) > WATCHDOG_TIMEOUT
            )
            if stuck:
                fire                           = True
                self._watchdog_count          += 1
                count                          = self._watchdog_count
                # Disarm so we do not fire again for the same stuck event
                self._watchdog_armed_for_cycle = False
                # Reset cycle state — the done_cb will see busy=False and skip recording
                self._picker_busy             = False
                self._armed                   = True
                self._cycle_start_time        = None
                self._cycle_min_gripper_cm    = None

        if fire:
            rospy.logwarn(
                f"[Monitor] WATCHDOG: picker busy >{WATCHDOG_TIMEOUT:.0f}s — "
                f"publishing re-arm signal  (total interventions: {count})")
            self._pub_rearm.publish(Bool(data=True))

    def _publish_metrics(self, _event) -> None:
        """Publishes all metrics to /session_monitor/* at 2 Hz."""
        with self._lock:
            cycles       = self._cycle_count
            durations    = list(self._cycle_durations)
            grasp_ws     = list(self._grasp_widths_cm)
            gripper_live = self._gripper_width_cm or 0.0
            armed        = self._armed
            busy         = self._picker_busy
            t_stamps     = list(self._cam_timestamps)

        last_c  = durations[-1]                  if durations  else 0.0
        mean_c  = sum(durations) / len(durations) if durations  else 0.0
        min_g   = grasp_ws[-1]                   if grasp_ws   else 0.0
        mean_g  = sum(grasp_ws) / len(grasp_ws)  if grasp_ws   else 0.0

        fps = 0.0
        if len(t_stamps) >= 2:
            span = t_stamps[-1] - t_stamps[0]
            if span > 0:
                fps = (len(t_stamps) - 1) / span

        if armed:
            status = "ARMED"
        elif busy:
            status = "BUSY"
        else:
            status = "IDLE"

        self._pub_cycle_count.publish(Int32(data=cycles))
        self._pub_last_cycle.publish(Float64(data=last_c))
        self._pub_mean_cycle.publish(Float64(data=mean_c))
        self._pub_min_grasp.publish(Float64(data=min_g))
        self._pub_mean_grasp.publish(Float64(data=mean_g))
        self._pub_gripper_live.publish(Float64(data=gripper_live))
        self._pub_camera_fps.publish(Float64(data=fps))
        self._pub_armed.publish(Bool(data=armed))
        self._pub_status.publish(String(data=status))

    def _summary_tick(self, _event) -> None:
        self._print_summary()

    # =========================================================================
    # Terminal display
    # =========================================================================

    def _print_startup_banner(self) -> None:
        SEP = "=" * 62
        rospy.loginfo(SEP)
        rospy.loginfo("  Franka FR3 Session Monitor  —  passive observer")
        rospy.loginfo(f"  Watchdog timeout  :  {WATCHDOG_TIMEOUT:.0f} s")
        rospy.loginfo(f"  Summary interval  :  {SUMMARY_INTERVAL:.0f} s")
        rospy.loginfo(f"  Published metrics :  /session_monitor/*")
        rospy.loginfo("  Subscribes to  (read-only):")
        rospy.loginfo("    /detected_object/point")
        rospy.loginfo("    /franka_picker/done")
        rospy.loginfo(f"    {GRIPPER_NS}/joint_states")
        rospy.loginfo("    /camera/color/image_raw")
        rospy.loginfo(SEP)

    def _print_summary(self) -> None:
        """Prints a formatted session report to the ROS log."""

        # --- Snapshot of state (brief lock hold) -------------------------
        with self._lock:
            cycles       = self._cycle_count
            durations    = list(self._cycle_durations)
            grasp_ws     = list(self._grasp_widths_cm)
            busy         = self._picker_busy
            armed        = self._armed
            wcount       = self._watchdog_count
            gripper_live = self._gripper_width_cm
            t_stamps     = list(self._cam_timestamps)
            t_session    = self._session_start
            cycle_start  = self._cycle_start_time

        # --- Derived values (computed outside lock) ----------------------
        elapsed     = time.time() - t_session
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        last_c  = durations[-1]                    if durations else None
        mean_c  = sum(durations) / len(durations)  if durations else None
        min_c   = min(durations)                   if durations else None
        max_c   = max(durations)                   if durations else None

        min_g   = grasp_ws[-1]                     if grasp_ws  else None
        mean_g  = sum(grasp_ws)  / len(grasp_ws)   if grasp_ws  else None

        fps = 0.0
        if len(t_stamps) >= 2:
            span = t_stamps[-1] - t_stamps[0]
            if span > 0:
                fps = (len(t_stamps) - 1) / span

        if armed:
            state_str = "ARMED  —  waiting for can"
        elif busy and cycle_start is not None:
            elapsed_in_cycle = time.time() - cycle_start
            state_str = f"BUSY   —  cycle running  ({elapsed_in_cycle:.0f}s elapsed)"
        else:
            state_str = "IDLE"

        wd_str = (
            f"timeout {WATCHDOG_TIMEOUT:.0f}s  |  "
            + ("no interventions" if wcount == 0
               else f"{wcount} intervention{'s' if wcount > 1 else ''}")
        )

        # --- Format lines -------------------------------------------------
        SEP  = "-" * 62
        SEP2 = "=" * 62

        def row(label: str, value: str) -> str:
            return f"  {label:<24}:  {value}"

        lines = [
            SEP2,
            f"  FRANKA FR3  |  session monitor  |  {time.strftime('%H:%M:%S')}",
            SEP,
            row("Session elapsed",   elapsed_str),
            row("Completed cycles",  str(cycles)),
            SEP,
        ]

        if durations:
            lines += [
                row("Last cycle",        f"{last_c:.1f} s"),
                row("Mean cycle",        f"{mean_c:.1f} s"),
                row("Min  /  Max",       f"{min_c:.1f} s  /  {max_c:.1f} s"),
            ]
        else:
            lines.append(row("Cycle times", "no completed cycles yet"))

        lines.append(SEP)

        if gripper_live is not None:
            lines.append(row("Gripper width (live)", f"{gripper_live:.1f} cm"))

        if min_g is not None:
            lines.append(row("Last grasp width",     f"{min_g:.1f} cm  (min during cycle)"))
        if mean_g is not None:
            lines.append(
                row("Mean grasp width",
                    f"{mean_g:.1f} cm  (over {len(grasp_ws)} grasps)"))

        lines += [
            row("Camera FPS",         f"{fps:.1f}"),
            row("System state",       state_str),
            SEP,
            row("Watchdog",           wd_str),
            SEP2,
        ]

        for line in lines:
            rospy.loginfo(line)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    try:
        SessionMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
