#!/usr/bin/env python3
"""
YOLO Can Detector
Finds a can, waits for stable detection, publishes XYZ once to /detected_object/point.
Re-arms when franka_picker finishes via /franka_picker/done.

Topics published:
  /detected_object/point      → PointStamped  (camera frame, triggers picker)
  /object_detector/yolo_debug → Image         (annotated debug feed)

Topics subscribed:
  /camera/color/camera_info
  /camera/color/image_raw
  /camera/aligned_depth_to_color/image_raw
  /franka_picker/done
"""
import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge

STABILITY_FRAMES    = 10     # consecutive frames before triggering
STABILITY_MAX_DRIFT = 0.02   # metres — max movement between frames to count as stable


class YOLOCanDetector:
    def __init__(self):
        rospy.init_node('can_detector_yolo')
        self.bridge = CvBridge()
        self.model  = YOLO('yolov8n.pt')

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_image = None

        # State
        self.armed        = True
        self.stable_count = 0
        self.last_pos     = None   # (X, Y, Z) in camera frame

        # Subscribers
        self.info_sub = rospy.Subscriber('/camera/color/camera_info',
                                         CameraInfo, self.info_cb)
        rospy.Subscriber('/camera/color/image_raw',
                         Image, self.color_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',
                         Image, self.depth_cb)
        rospy.Subscriber('/franka_picker/done',
                         Bool, self.done_cb)

        # Publishers
        self.point_pub = rospy.Publisher('/detected_object/point',
                                         PointStamped, queue_size=1)
        self.debug_pub = rospy.Publisher('/object_detector/yolo_debug',
                                         Image, queue_size=1)

        rospy.loginfo("=" * 55)
        rospy.loginfo("YOLO Can Detector ready — armed.")
        rospy.loginfo(f"  Stability required: {STABILITY_FRAMES} frames")
        rospy.loginfo(f"  Max drift:          {STABILITY_MAX_DRIFT*100:.0f} cm")
        rospy.loginfo("=" * 55)

    def done_cb(self, msg):
        if msg.data:
            self.armed        = True
            self.stable_count = 0
            self.last_pos     = None
            rospy.loginfo("Detector re-armed — ready for next can.")

    def info_cb(self, msg):
        self.fx, self.fy = msg.K[0], msg.K[4]
        self.cx, self.cy = msg.K[2], msg.K[5]
        rospy.loginfo(f"Camera calibrated: fx={self.fx:.1f} fy={self.fy:.1f}")
        self.info_sub.unregister()

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def color_cb(self, msg):
        if self.depth_image is None or self.fx is None:
            return

        frame   = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(frame, verbose=False, conf=0.15)

        best    = None
        best_px = None   # pixel coords of best detection for drawing

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                px = int((x1 + x2) / 2)
                py = int((y1 + y2) / 2)

                # Depth — 5x5 median for stability
                roi = self.depth_image[
                    max(0, py-2):min(self.depth_image.shape[0], py+3),
                    max(0, px-2):min(self.depth_image.shape[1], px+3)
                ]
                valid = roi[roi > 0]
                if valid.size < 5:
                    continue

                Z = np.median(valid) / 1000.0   # mm → m
                if not (0.15 < Z < 1.5):
                    continue

                # Size filter — roughly can-sized (5–12 cm diameter)
                est_diam = ((x2 - x1) * Z) / self.fx
                if not (0.05 < est_diam < 0.12):
                    continue

                X = (px - self.cx) * Z / self.fx
                Y = (py - self.cy) * Z / self.fy

                # Draw bounding box and info
                color = (0, 255, 0) if self.armed else (128, 128, 128)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                cv2.putText(frame,
                            f"D={est_diam*100:.1f}cm Z={Z:.2f}m",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Keep closest detection
                if best is None or Z < best[2]:
                    best    = (X, Y, Z)
                    best_px = (px, py)

        # ── Armed/disarmed overlay ────────────────────────────────────
        if not self.armed:
            cv2.putText(frame, "ROBOT BUSY — disarmed",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "ARMED — scanning",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # ── Stability progress bar ────────────────────────────────────
        if self.armed and self.stable_count > 0:
            bar_w = int((self.stable_count / STABILITY_FRAMES) * 200)
            cv2.rectangle(frame,
                          (10, frame.shape[0] - 25),
                          (210, frame.shape[0] - 10),
                          (50, 50, 50), -1)
            cv2.rectangle(frame,
                          (10, frame.shape[0] - 25),
                          (10 + bar_w, frame.shape[0] - 10),
                          (0, 255, 0), -1)
            cv2.putText(frame,
                        f"Stability {self.stable_count}/{STABILITY_FRAMES}",
                        (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # ── Stability check — only trigger when armed ─────────────────
        if self.armed and best is not None:
            if self.last_pos is None:
                self.stable_count = 1
                self.last_pos     = best
            else:
                drift = np.linalg.norm(np.array(best) - np.array(self.last_pos))
                if drift < STABILITY_MAX_DRIFT:
                    self.stable_count += 1
                    self.last_pos      = best
                else:
                    rospy.logdebug(f"Can drifted {drift*100:.1f}cm — resetting stability")
                    self.stable_count = 1
                    self.last_pos     = best

            if self.stable_count >= STABILITY_FRAMES:
                self.armed        = False
                self.stable_count = 0
                self.last_pos     = None

                X, Y, Z = best
                rospy.loginfo("=" * 50)
                rospy.loginfo(f"Can confirmed — sending to picker")
                rospy.loginfo(f"  Camera frame: X={X:.3f}  Y={Y:.3f}  Z={Z:.3f}")
                rospy.loginfo("=" * 50)

                pt = PointStamped()
                pt.header.stamp    = rospy.Time.now()
                pt.header.frame_id = 'camera_color_optical_frame'
                pt.point.x, pt.point.y, pt.point.z = X, Y, Z
                self.point_pub.publish(pt)

        elif best is None:
            self.stable_count = 0
            self.last_pos     = None

        # ── Always publish debug image ────────────────────────────────
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))


if __name__ == '__main__':
    try:
        YOLOCanDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
