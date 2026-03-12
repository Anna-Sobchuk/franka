#!/home/raicam/franka_project_Anna/venv/bin python3
"""
YOLO Can Detector — pipeline version
=====================================
Detects a can, waits for N stable frames, publishes ONCE to /detected_object/point,
then disarms until Franka signals it's done via /franka_picker/done.

Topics published:
  /detected_object/point      → PointStamped  (camera frame, triggers picker)
  /detected_can/diameter      → Float32
  /detected_can/height        → Float32
  /object_detector/yolo_debug → Image (annotated)

Topics subscribed:
  /camera/color/camera_info
  /camera/color/image_raw
  /camera/aligned_depth_to_color/image_raw
  /franka_picker/done         → Bool  (re-arms detector after each pick)

Run on HOST (outside docker):
  python3 object_detector.py
"""
import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

# ── Stability settings ────────────────────────────────────────────────────────
STABILITY_FRAMES   = 10      # consecutive frames with a detection before triggering
STABILITY_MAX_DRIFT = 0.02   # metres — how much the 3D point can move between frames
                             # and still count as "same can"

# ── Can physical dimensions ───────────────────────────────────────────────────
CAN_DIAMETER = 0.075   # 7.5 cm
CAN_HEIGHT   = 0.110   # 11 cm

# ── YOLO class IDs to try for cans ───────────────────────────────────────────
# 39=bottle, 41=cup, 44=vase — adjust or add your custom class if you have one
YOLO_CLASSES = [39, 41, 44]
YOLO_CONF    = 0.3


class YOLOCanDetector:
    def __init__(self):
        rospy.init_node('can_detector_yolo')
        self.bridge = CvBridge()

        self.model = YOLO('yolov8n.pt')

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_image = None

        # ── Pipeline state ────────────────────────────────────────────
        self.armed        = True    # False while robot is busy
        self.stable_count = 0       # consecutive frames with stable detection
        self.last_stable  = None    # last 3D point (x, y, z) in camera frame

        # ── Subscribers ───────────────────────────────────────────────
        self.info_sub  = rospy.Subscriber('/camera/color/camera_info',
                                          CameraInfo, self.info_cb)
        self.color_sub = rospy.Subscriber('/camera/color/image_raw',
                                          Image, self.color_cb)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',
                                          Image, self.depth_cb)
        rospy.Subscriber('/franka_picker/done', Bool, self.done_cb)

        # ── Publishers ────────────────────────────────────────────────
        self.point_pub    = rospy.Publisher('/detected_object/point',
                                            PointStamped, queue_size=1)
        self.diameter_pub = rospy.Publisher('/detected_can/diameter',
                                            Float32, queue_size=1)
        self.height_pub   = rospy.Publisher('/detected_can/height',
                                            Float32, queue_size=1)
        self.debug_pub    = rospy.Publisher('/object_detector/yolo_debug',
                                            Image, queue_size=1)

        rospy.loginfo("=" * 55)
        rospy.loginfo("YOLO Can Detector ready — armed, waiting for can...")
        rospy.loginfo(f"  Stability required: {STABILITY_FRAMES} frames")
        rospy.loginfo(f"  Max drift:          {STABILITY_MAX_DRIFT*100:.0f} cm")
        rospy.loginfo("=" * 55)

    # ── Re-arm when robot finishes ────────────────────────────────────────────
    def done_cb(self, msg):
        if msg.data:
            self.armed        = True
            self.stable_count = 0
            self.last_stable  = None
            rospy.loginfo("Detector re-armed — ready for next can.")

    def info_cb(self, msg):
        self.fx = msg.K[0];  self.fy = msg.K[4]
        self.cx = msg.K[2];  self.cy = msg.K[5]
        rospy.loginfo(f"Camera calibrated: fx={self.fx:.1f} fy={self.fy:.1f}")
        self.info_sub.unregister()

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    # ── Main detection loop ───────────────────────────────────────────────────
    def color_cb(self, msg):
        if self.depth_image is None or self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # ── Always annotate debug image regardless of armed state ─────
        # (so you can see what the camera sees even while robot moves)
        results = self.model(frame, verbose=False, conf=YOLO_CONF,
                             classes=YOLO_CLASSES)

        best = None   # closest valid detection this frame

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                px = int((x1 + x2) / 2)
                py = int((y1 + y2) / 2)

                # ── Depth sample (5×5 median for stability) ───────────
                roi = self.depth_image[
                    max(0, py-2):min(self.depth_image.shape[0], py+3),
                    max(0, px-2):min(self.depth_image.shape[1], px+3)
                ]
                valid = roi[roi > 0]
                if valid.size < 5:
                    continue

                Z = np.median(valid) / 1000.0   # mm → m (surface of can)
                if not (0.15 < Z < 1.5):
                    continue

                # ── Size filter ───────────────────────────────────────
                est_diam = ((x2 - x1) * Z) / self.fx
                if not (0.05 < est_diam < 0.12):
                    continue

                # ── 3D position in camera frame ───────────────────────
                X = (px - self.cx) * Z / self.fx
                Y = (py - self.cy) * Z / self.fy

                # Keep closest
                if best is None or Z < best['Z']:
                    best = {
                        'X': X, 'Y': Y, 'Z': Z,
                        'diam': est_diam,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'px': px, 'py': py
                    }

                # Draw box
                color = (0, 255, 0) if self.armed else (128, 128, 128)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame,
                            f"D={est_diam*100:.1f}cm Z={Z:.2f}m",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── Stability logic (only when armed) ────────────────────────────
        if self.armed and best is not None:
            pt = (best['X'], best['Y'], best['Z'])

            if self.last_stable is None:
                # First detection this cycle
                self.stable_count = 1
                self.last_stable  = pt
            else:
                drift = np.linalg.norm(np.array(pt) - np.array(self.last_stable))
                if drift < STABILITY_MAX_DRIFT:
                    self.stable_count += 1
                    self.last_stable   = pt
                else:
                    # Can moved — restart count
                    rospy.logdebug(f"Can drifted {drift*100:.1f}cm — resetting stability count")
                    self.stable_count = 1
                    self.last_stable  = pt

            # ── Trigger if stable enough ──────────────────────────────
            if self.stable_count >= STABILITY_FRAMES:
                self.armed        = False   # disarm until robot is done
                self.stable_count = 0
                self.last_stable  = None

                rospy.loginfo("=" * 50)
                rospy.loginfo(f"STABLE detection confirmed — triggering picker")
                rospy.loginfo(f"  Camera frame: X={best['X']:.3f} "
                              f"Y={best['Y']:.3f}  Z={best['Z']:.3f}")
                rospy.loginfo(f"  Diameter: {best['diam']*100:.1f} cm")
                rospy.loginfo("=" * 50)

                # Publish can position
                pt_msg = PointStamped()
                pt_msg.header.stamp    = rospy.Time.now()
                pt_msg.header.frame_id = 'camera_color_optical_frame'
                pt_msg.point.x = best['X']
                pt_msg.point.y = best['Y']
                pt_msg.point.z = best['Z']
                self.point_pub.publish(pt_msg)

                # Publish dimensions
                self.diameter_pub.publish(Float32(data=best['diam']))
                self.height_pub.publish(Float32(data=CAN_HEIGHT))

        elif not self.armed:
            # Show disarmed overlay on debug image
            cv2.putText(frame, "ROBOT BUSY — disarmed",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2)
        elif best is None:
            # Reset stability if no can seen
            self.stable_count = 0
            self.last_stable  = None

        # ── Stability progress bar on debug image ─────────────────────
        if self.armed and self.stable_count > 0:
            bar_w  = int((self.stable_count / STABILITY_FRAMES) * 200)
            cv2.rectangle(frame, (10, frame.shape[0]-25),
                          (210, frame.shape[0]-10), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, frame.shape[0]-25),
                          (10 + bar_w, frame.shape[0]-10), (0, 255, 0), -1)
            cv2.putText(frame,
                        f"Stability {self.stable_count}/{STABILITY_FRAMES}",
                        (10, frame.shape[0]-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))


if __name__ == '__main__':
    try:
        YOLOCanDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
