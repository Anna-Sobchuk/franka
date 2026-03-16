#!/usr/bin/env python3
"""
YOLO Can Detector with HoughCircles fallback.
Finds a metal can, waits for stable detection, publishes XYZ once to /detected_object/point.
Re-arms when franka_picker finishes via /franka_picker/done.
"""
import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge

# ── Detection parameters ──────────────────────────────────────────────────────
STABILITY_FRAMES    = 8
STABILITY_MAX_DRIFT = 0.025  # metres

YOLO_CONF    = 0.10
YOLO_CLASSES = [39, 40, 41, 44]  # bottle, wine glass, cup, bottle variant

MIN_DIAM = 0.05
MAX_DIAM = 0.13
MIN_DEPTH = 0.15
MAX_DEPTH = 1.50

HOUGH_MIN_RADIUS = 18
HOUGH_MAX_RADIUS = 80
HOUGH_PARAM1     = 60
HOUGH_PARAM2     = 28


def get_depth(depth_image, px, py, window=11):
    """Robust depth estimate using median over a window, ignoring zeros."""
    h, w = depth_image.shape
    half = window // 2
    roi = depth_image[
        max(0, py - half):min(h, py + half + 1),
        max(0, px - half):min(w, px + half + 1)
    ]
    valid = roi[roi > 0]
    if valid.size < 5:
        return None
    return np.median(valid) / 1000.0


class YOLOCanDetector:
    def __init__(self):
        rospy.init_node('can_detector_yolo')
        self.bridge = CvBridge()
        self.model  = YOLO('yolov8n.pt')

        self.fx = self.fy = self.cx = self.cy = None
        self.depth_image = None

        self.armed        = True
        self.stable_count = 0
        self.last_pos     = None

        self.info_sub = rospy.Subscriber('/camera/color/camera_info',
                                         CameraInfo, self.info_cb)
        rospy.Subscriber('/camera/color/image_raw',   Image, self.color_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',
                         Image, self.depth_cb)
        rospy.Subscriber('/franka_picker/done', Bool, self.done_cb)

        self.point_pub = rospy.Publisher('/detected_object/point',
                                         PointStamped, queue_size=1)
        self.debug_pub = rospy.Publisher('/object_detector/yolo_debug',
                                         Image, queue_size=1)

        rospy.loginfo("=" * 55)
        rospy.loginfo("YOLO Can Detector ready — armed.")
        rospy.loginfo(f"  Stability: {STABILITY_FRAMES} frames")
        rospy.loginfo(f"  YOLO conf={YOLO_CONF}  HoughCircles fallback: ON")
        rospy.loginfo("=" * 55)

    def done_cb(self, msg):
        if msg.data:
            self.armed        = True
            self.stable_count = 0
            self.last_pos     = None
            rospy.loginfo("Detector re-armed.")

    def info_cb(self, msg):
        self.fx, self.fy = msg.K[0], msg.K[4]
        self.cx, self.cy = msg.K[2], msg.K[5]
        rospy.loginfo(f"Camera calibrated: fx={self.fx:.1f} fy={self.fy:.1f}")
        self.info_sub.unregister()

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def detect_yolo(self, frame):
        results = self.model(frame, verbose=False, conf=YOLO_CONF, classes=YOLO_CLASSES)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                px = int((x1 + x2) / 2)
                py = int((y1 + y2) / 2)
                Z = get_depth(self.depth_image, px, py)
                if Z is None or not (MIN_DEPTH < Z < MAX_DEPTH):
                    continue
                est_diam = ((x2 - x1) * Z) / self.fx
                if not (MIN_DIAM < est_diam < MAX_DIAM):
                    continue
                X = (px - self.cx) * Z / self.fx
                Y = (py - self.cy) * Z / self.fy
                cls_name = self.model.names[int(box.cls[0])]
                detections.append((X, Y, Z, px, py, est_diam, f"YOLO:{cls_name}"))
        return detections

    def detect_hough(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray  = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=60,
            param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
            minRadius=HOUGH_MIN_RADIUS, maxRadius=HOUGH_MAX_RADIUS
        )
        detections = []
        if circles is None:
            return detections
        for c in circles[0]:
            px, py, radius = int(c[0]), int(c[1]), int(c[2])
            Z = get_depth(self.depth_image, px, py)
            if Z is None or not (MIN_DEPTH < Z < MAX_DEPTH):
                continue
            est_diam = (radius * 2 * Z) / self.fx
            if not (MIN_DIAM < est_diam < MAX_DIAM):
                continue
            X = (px - self.cx) * Z / self.fx
            Y = (py - self.cy) * Z / self.fy
            detections.append((X, Y, Z, px, py, est_diam, "Hough"))
        return detections

    def color_cb(self, msg):
        if self.depth_image is None or self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        yolo_dets  = self.detect_yolo(frame)
        hough_dets = self.detect_hough(frame)
        all_dets   = yolo_dets if yolo_dets else hough_dets
        best       = min(all_dets, key=lambda d: d[2]) if all_dets else None

        # Draw YOLO detections (green)
        for X, Y, Z, px, py, diam, label in yolo_dets:
            col = (0, 255, 0) if self.armed else (128, 128, 128)
            r = int((diam / Z) * self.fx / 2)
            cv2.rectangle(frame, (px-r, py-r), (px+r, py+r), col, 2)
            cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} D={diam*100:.1f}cm Z={Z:.2f}m",
                        (px-r, py-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Draw Hough detections (orange)
        for X, Y, Z, px, py, diam, label in hough_dets:
            col = (0, 165, 255) if self.armed else (128, 128, 128)
            r = int((diam / Z) * self.fx / 2)
            cv2.circle(frame, (px, py), r, col, 2)
            cv2.circle(frame, (px, py), 4, col, -1)
            cv2.putText(frame, f"Hough D={diam*100:.1f}cm Z={Z:.2f}m",
                        (px-r, py-r-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # Status overlay
        if not self.armed:
            cv2.putText(frame, "ROBOT BUSY", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            src = "YOLO" if yolo_dets else ("HOUGH" if hough_dets else "searching...")
            cv2.putText(frame, f"ARMED [{src}]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Stability bar
        if self.armed and self.stable_count > 0:
            bar_w = int((self.stable_count / STABILITY_FRAMES) * 200)
            cv2.rectangle(frame, (10, frame.shape[0]-25), (210, frame.shape[0]-10), (50,50,50), -1)
            cv2.rectangle(frame, (10, frame.shape[0]-25), (10+bar_w, frame.shape[0]-10), (0,255,0), -1)
            cv2.putText(frame, f"Stability {self.stable_count}/{STABILITY_FRAMES}",
                        (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Stability logic
        if self.armed and best is not None:
            X, Y, Z, px, py, diam, label = best
            pos = (X, Y, Z)
            if self.last_pos is None:
                self.stable_count = 1
                self.last_pos     = pos
            else:
                drift = np.linalg.norm(np.array(pos) - np.array(self.last_pos))
                if drift < STABILITY_MAX_DRIFT:
                    self.stable_count += 1
                    self.last_pos      = pos
                else:
                    self.stable_count = 1
                    self.last_pos      = pos

            if self.stable_count >= STABILITY_FRAMES:
                self.armed        = False
                self.stable_count = 0
                self.last_pos     = None
                rospy.loginfo("=" * 50)
                rospy.loginfo(f"Can confirmed via {label} — sending to picker")
                rospy.loginfo(f"  Camera: X={X:.3f}  Y={Y:.3f}  Z={Z:.3f}  D={diam*100:.1f}cm")
                rospy.loginfo("=" * 50)
                pt = PointStamped()
                pt.header.stamp    = rospy.Time.now()
                pt.header.frame_id = 'camera_color_optical_frame'
                pt.point.x, pt.point.y, pt.point.z = X, Y, Z
                self.point_pub.publish(pt)
        elif best is None:
            self.stable_count = 0
            self.last_pos     = None

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))


if __name__ == '__main__':
    try:
        YOLOCanDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
