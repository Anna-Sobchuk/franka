#!/usr/bin/env python3
"""
Franka Can Picker — FrankaPy version.
Subscribes to /detected_object/point from YOLO detector,
picks up the can and places it behind the robot.

Run inside Docker (with franka-interface running):
  python3 ~/franka/scripts/franka_picker.py

Run detector separately on HOST:
  python3 ~/franka/scripts/object_detector.py
"""
import rospy
import actionlib
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
from franka_gripper.msg import (MoveAction, MoveGoal,
                                GraspAction, GraspGoal,
                                GraspEpsilon)
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState
import time

# ── Motion durations ──────────────────────────────────────────────────────────
CART_DURATION  = 6    # seconds for Cartesian moves
JOINT_DURATION = 8    # seconds for joint moves

# ── Can defaults (overridden by detector topics) ──────────────────────────────
CAN_HEIGHT   = 0.11
CAN_DIAMETER = 0.075

# ── Pick geometry ─────────────────────────────────────────────────────────────
GRASP_Z_OFFSET = 0.01   # how high above can surface to grasp (from detector Z)
HOVER_ABOVE    = 0.15   # metres above grasp point for hover
LIFT_ABOVE     = 0.30   # metres above grasp point for safe rotation

# ── Two-step rotation (tested and working) ────────────────────────────────────
MID_JOINTS = [
    np.radians(90),
    np.radians(-45),
    np.radians(0),
    np.radians(-135),
    np.radians(0),
    np.radians(90),
    np.radians(45),
]
BEHIND_JOINTS = [
    np.radians(165),
    np.radians(-45),
    np.radians(0),
    np.radians(-135),
    np.radians(0),
    np.radians(90),
    np.radians(45),
]

# ── Place height behind robot ─────────────────────────────────────────────────
PLACE_Z = CAN_HEIGHT - 0.02   # just above table

# ── Gripper namespace ─────────────────────────────────────────────────────────
GRIPPER_NS = '/franka_gripper_1/franka_gripper'

# ── Direct camera→robot mapping (no TF publisher needed) ─────────────────────
# Learned from 4 calibration points using linear regression
# robot_x = cx*ax + cy*bx + cz*cx_ + dx
# robot_y = cx*ay + cy*by + cz*cy_ + dy
CAM2ROBOT_X = [0.1396, 1.6647, 0.3942, 0.0982]   # [cx, cy, cz, 1]
CAM2ROBOT_Y = [1.0213, -0.0325, -0.1186, 0.0191]     # [cx, cy, cz, 1]
TABLE_Z = 0.0983  # mean table height in robot frame

ROTATION_DOWN = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1]
], dtype=float)


def make_pose(x, y, z, tilt_deg=0):
    """Create a pose pointing down. For far reaches use tilt_deg>0 to tilt
    the gripper slightly forward — avoids wrist singularity at extended X."""
    if tilt_deg == 0:
        rot = ROTATION_DOWN
    else:
        # Tilt forward around Y axis by tilt_deg degrees
        angle = np.radians(tilt_deg)
        c, s = np.cos(angle), np.sin(angle)
        tilt = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        rot = ROTATION_DOWN @ tilt
    return RigidTransform(
        rotation=rot,
        translation=np.array([x, y, z]),
        from_frame='franka_tool',
        to_frame='world'
    )


def get_gripper_width():
    js = rospy.wait_for_message(
        f'{GRIPPER_NS}/joint_states', JointState, timeout=5.0)
    return sum(js.position)


def goto(fa, pose):
    fa.goto_pose(pose, use_impedance=False, ignore_virtual_walls=True,
                 duration=CART_DURATION)


def goto_joints(fa, joints):
    fa.goto_joints(joints, use_impedance=False, ignore_virtual_walls=True,
                   duration=JOINT_DURATION)


class GripperController:
    def __init__(self):
        self.move_client  = actionlib.SimpleActionClient(
            f'{GRIPPER_NS}/move', MoveAction)
        self.grasp_client = actionlib.SimpleActionClient(
            f'{GRIPPER_NS}/grasp', GraspAction)
        rospy.loginfo("Waiting for gripper servers...")
        self.move_client.wait_for_server(timeout=rospy.Duration(10))
        self.grasp_client.wait_for_server(timeout=rospy.Duration(10))
        rospy.loginfo("Gripper ready!")

    def open(self, width=0.08, speed=0.05):
        goal = MoveGoal(width=width, speed=speed)
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result(rospy.Duration(10))
        rospy.loginfo(f"Gripper opened — {get_gripper_width()*100:.1f}cm")

    def close(self):
        goal = GraspGoal()
        goal.width   = 0.07
        goal.speed   = 0.01
        goal.force   = 5.0
        goal.epsilon = GraspEpsilon(inner=0.05, outer=0.05)
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result(rospy.Duration(10))
        result = self.grasp_client.get_result()
        if result is None:
            rospy.logwarn("Gripper result is None — timeout or no can present")
            return False
        rospy.loginfo(f"Gripper closed — success:{result.success} "
                      f"width:{get_gripper_width()*100:.1f}cm")
        return result.success


class FrankaPicker:
    def __init__(self):
        # FrankaArm() calls rospy.init_node() internally
        rospy.loginfo("Connecting to FrankaArm...")
        self.fa = FrankaArm()
        rospy.loginfo("FrankaArm connected!")

        self.gripper = GripperController()

        # Can dimensions updated by detector
        self.can_height   = CAN_HEIGHT
        self.can_diameter = CAN_DIAMETER

        # Busy flag — ignore detections while moving
        self.busy = False

        # Subscribers
        rospy.Subscriber('/detected_can/diameter', Float32, self.diameter_cb)
        rospy.Subscriber('/detected_can/height',   Float32, self.height_cb)
        rospy.Subscriber('/detected_object/point', PointStamped, self.point_cb)

        # Publisher — tells detector we're done so it can re-arm
        self.done_pub = rospy.Publisher('/franka_picker/done', Bool, queue_size=1)

        rospy.loginfo("Franka Picker ready — waiting for detections...")

    def diameter_cb(self, msg):
        self.can_diameter = msg.data

    def height_cb(self, msg):
        self.can_height = msg.data

    def point_cb(self, msg):
        """Called when detector publishes a stable can position."""
        if self.busy:
            return
        self.busy = True
        rospy.loginfo("=" * 50)
        rospy.loginfo("Can detected — starting pick & place")
        rospy.loginfo("=" * 50)

        try:
            # ── Direct camera→robot mapping (no TF publisher needed) ──
            cam_x, cam_y, cam_z = msg.point.x, msg.point.y, msg.point.z
            cx = CAM2ROBOT_X[0]*cam_x + CAM2ROBOT_X[1]*cam_y + CAM2ROBOT_X[2]*cam_z + CAM2ROBOT_X[3]
            cy = CAM2ROBOT_Y[0]*cam_x + CAM2ROBOT_Y[1]*cam_y + CAM2ROBOT_Y[2]*cam_z + CAM2ROBOT_Y[3]
            cz = TABLE_Z

            rospy.loginfo(f"Camera raw: X={cam_x:.3f} Y={cam_y:.3f} Z={cam_z:.3f}")
            rospy.loginfo(f"Can in robot frame: X={cx:.3f} Y={cy:.3f} Z={cz:.3f}")
            rospy.loginfo(f"Diameter={self.can_diameter*100:.1f}cm  "
                          f"Height={self.can_height*100:.1f}cm")

            # ── Sanity check — reject unreachable positions ─────────────
            if not (0.25 < cx < 0.80) or not (-0.40 < cy < 0.40):
                rospy.logwarn(f"Position out of workspace (X={cx:.3f} Y={cy:.3f}) — skipping")
                self.busy = False
                self.done_pub.publish(Bool(data=True))
                return

            # ── Grasp height — same as manual_test.py ──────────────────
            grasp_z = self.can_height - 0.01
            hover_z = grasp_z + HOVER_ABOVE
            lift_z  = grasp_z + LIFT_ABOVE

            # ── Step 1: Home ────────────────────────────────────────────
            rospy.loginfo("Step 1: Going home...")
            self.fa.reset_joints()
            time.sleep(0.5)

            # ── Step 2: Open gripper ────────────────────────────────────
            rospy.loginfo("Step 2: Opening gripper...")
            self.gripper.open(width=0.08)

            # ── Step 3: Hover above can ─────────────────────────────────
            # Use slight forward tilt for far reaches to avoid wrist singularity
            tilt = 10 if cx > 0.60 else 0
            rospy.loginfo(f"Step 3: Hovering above can ({cx:.3f}, {cy:.3f}, {hover_z:.3f}) tilt={tilt}°...")
            goto(self.fa, make_pose(cx, cy, hover_z, tilt_deg=tilt))
            time.sleep(0.5)

            # ── Step 4: Lower to grasp height ───────────────────────────
            # First move to intermediate Z to avoid FrankaInterface silently
            # rejecting large downward moves from hover
            mid_z = (hover_z + grasp_z) / 2
            rospy.loginfo(f"Step 4a: Lowering to mid Z={mid_z:.3f}m...")
            goto(self.fa, make_pose(cx, cy, mid_z, tilt_deg=tilt))
            time.sleep(0.3)
            rospy.loginfo(f"Step 4b: Lowering to grasp Z={grasp_z:.3f}m...")
            goto(self.fa, make_pose(cx, cy, grasp_z, tilt_deg=tilt))
            time.sleep(0.5)

            # Verify robot actually reached grasp height
            actual_z = self.fa.get_pose().translation[2]
            rospy.loginfo(f"Actual Z after lowering: {actual_z:.3f}m (target {grasp_z:.3f}m)")
            if actual_z > grasp_z + 0.05:
                rospy.logwarn(f"Robot did not lower to target (still at Z={actual_z:.3f}) — going home and retrying")
                self.gripper.open(width=0.08)
                self.fa.reset_joints()   # go home so camera sees fresh scene
                rospy.sleep(2.0)         # wait for arm to clear view
                self.busy = False
                self.done_pub.publish(Bool(data=True))
                return

            # ── Step 5: Grasp ───────────────────────────────────────────
            rospy.loginfo("Step 5: Grasping can...")
            self.gripper.close()
            time.sleep(0.5)

            # Check we actually grasped something — width > 2cm means can is there
            gripper_w = get_gripper_width()
            rospy.loginfo(f"Gripper width after grasp: {gripper_w*100:.1f}cm")
            if gripper_w < 0.020:
                rospy.logwarn("Gripper closed to 0 — missed can! Going home and retrying.")
                self.gripper.open(width=0.08)
                self.fa.reset_joints()   # go home so camera gets fresh view
                rospy.sleep(2.0)
                self.busy = False
                self.done_pub.publish(Bool(data=True))
                return

            # ── Step 6: Lift ────────────────────────────────────────────
            rospy.loginfo(f"Step 6: Lifting to Z={lift_z:.3f}m...")
            goto(self.fa, make_pose(cx, cy, lift_z))
            time.sleep(0.5)

            # ── Step 7a: Move to safe centered config (joint 0 = 0°) ───
            # Same as home but lifted — always reachable from any pick position
            SAFE_LIFTED_JOINTS = [
                np.radians(0),
                np.radians(-45),
                np.radians(0),
                np.radians(-135),
                np.radians(0),
                np.radians(90),
                np.radians(45),
            ]
            rospy.loginfo("Step 7a: Moving to safe centered config (0°)...")
            goto_joints(self.fa, SAFE_LIFTED_JOINTS)
            time.sleep(0.5)

            # Check object still in gripper after first rotation
            w_after_7a = get_gripper_width()
            rospy.loginfo(f"Gripper width after 7a: {w_after_7a*100:.1f}cm")
            if w_after_7a < 0.020:
                rospy.logwarn("Object dropped during rotation — aborting")
                self.fa.reset_joints()
                self.busy = False
                rospy.sleep(2.0)
                self.done_pub.publish(Bool(data=True))
                return

            # ── Step 7b: Rotate directly to behind robot (165°) ─────────
            rospy.loginfo("Step 7b: Rotating directly to behind robot (165°)...")
            goto_joints(self.fa, BEHIND_JOINTS)
            time.sleep(0.5)

            pose_after = self.fa.get_pose()
            rospy.loginfo(f"Pose after rotation: "
                          f"X={pose_after.translation[0]:.3f} "
                          f"Y={pose_after.translation[1]:.3f} "
                          f"Z={pose_after.translation[2]:.3f}")

            # ── Step 8: Lower can to table ──────────────────────────────
            rospy.loginfo(f"Step 8: Lowering can to table Z={PLACE_Z:.3f}m...")
            current = self.fa.get_pose()
            place_pose = RigidTransform(
                rotation=current.rotation,
                translation=np.array([
                    current.translation[0],
                    current.translation[1],
                    PLACE_Z
                ]),
                from_frame='franka_tool',
                to_frame='world'
            )
            goto(self.fa, place_pose)
            time.sleep(0.5)

            # ── Step 9: Release ─────────────────────────────────────────
            rospy.loginfo("Step 9: Releasing can...")
            self.gripper.open(width=0.08, speed=0.02)
            time.sleep(0.5)

            # ── Step 10: Lift away ──────────────────────────────────────
            rospy.loginfo("Step 10: Lifting away...")
            current = self.fa.get_pose()
            retreat = RigidTransform(
                rotation=current.rotation,
                translation=np.array([
                    current.translation[0],
                    current.translation[1],
                    lift_z
                ]),
                from_frame='franka_tool',
                to_frame='world'
            )
            goto(self.fa, retreat)
            time.sleep(0.5)

            # ── Step 11: Home ───────────────────────────────────────────
            rospy.loginfo("Step 11: Returning home...")
            self.fa.reset_joints()

            rospy.loginfo("=" * 50)
            rospy.loginfo("COMPLETE! Can placed behind robot.")
            rospy.loginfo("=" * 50)

        except Exception as e:
            rospy.logerr(f"Pick & place failed: {e}")
            import traceback; traceback.print_exc()
            try:
                self.fa.reset_joints()
            except:
                pass
        finally:
            self.busy = False
            rospy.sleep(2.0)  # wait for arm to clear camera view before re-arming
            self.done_pub.publish(Bool(data=True))  # re-arm detector


if __name__ == '__main__':
    try:
        FrankaPicker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
