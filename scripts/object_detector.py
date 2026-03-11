#!/home/raicam/franka_project_Anna/venv/bin
import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from cv_bridge import CvBridge

class YOLOCanDetector:
    def __init__(self):
        rospy.init_node('can_detector_yolo')
        self.bridge = CvBridge()
        
        # Load YOLOv8 nano (fastest)
        self.model = YOLO('yolov8n.pt')
        
        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_image = None
        
        # Can physical dimensions
        self.CAN_DIAMETER = 0.075  # 7.5cm
        self.CAN_HEIGHT = 0.11    # 
        
        # Subscribers
        self.info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.info_cb)
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.color_cb)
        self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_cb)
        
        # Publishers
        self.point_pub = rospy.Publisher('/detected_object/point', PointStamped, queue_size=1)
        self.diameter_pub = rospy.Publisher('/detected_can/diameter', Float32, queue_size=1)
        self.height_pub = rospy.Publisher('/detected_can/height', Float32, queue_size=1)
        self.debug_pub = rospy.Publisher('/object_detector/yolo_debug', Image, queue_size=1)
        
        rospy.loginfo("YOLO Can Detector ready - waiting for camera...")

    def info_cb(self, msg):
        """Get camera calibration once"""
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]
        rospy.loginfo(f"✓ Camera calibrated: fx={self.fx:.1f}, fy={self.fy:.1f}")
        self.info_sub.unregister()

    def depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

    def color_cb(self, msg):
        if self.depth_image is None or self.fx is None:
            return
        
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Run YOLO detection (class 39 = bottle, class 41 = cup - adjust for your objects)
        # For cans, you might want class 39 (bottle) or train custom model
        results = self.model(frame, verbose=False, conf=0.3, classes=[39, 41, 44])  # bottle, cup, vase
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                px = int((x1 + x2) / 2)
                py = int((y1 + y2) / 2)
                
                # Sample depth in center region (5x5 for stability)
                y_start = max(0, py - 2)
                y_end = min(self.depth_image.shape[0], py + 3)
                x_start = max(0, px - 2)
                x_end = min(self.depth_image.shape[1], px + 3)
                
                roi = self.depth_image[y_start:y_end, x_start:x_end]
                valid_depths = roi[roi > 0]
                
                if valid_depths.size < 5:
                    continue
                
                # Distance to CAN SURFACE (not center yet)
                Z_surface = np.median(valid_depths) / 1000.0
                
                if Z_surface < 0.15 or Z_surface > 1.0:
                    continue
                
                # Estimate can width from bounding box
                bbox_width_pixels = x2 - x1
                estimated_diameter = (bbox_width_pixels * Z_surface) / self.fx
                
                # Filter: should be roughly can-sized (5-12cm diameter)
                if estimated_diameter < 0.05 or estimated_diameter > 0.12:
                    continue
                
                # Convert pixel center to 3D coordinates IN CAMERA FRAME
                X = (px - self.cx) * Z_surface / self.fx
                Y = (py - self.cy) * Z_surface / self.fy
                
                # Z is the SURFACE of the can (gripper will approach from above)
                Z = Z_surface
                
                detections.append({
                    'position': (X, Y, Z),
                    'diameter': estimated_diameter,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center_px': (px, py)
                })
                
                # Draw on frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)
                
                info_text = f"Can: D={estimated_diameter*100:.1f}cm Z={Z:.2f}m"
                cv2.putText(frame, info_text, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Publish closest detection
        if detections:
            closest = min(detections, key=lambda d: d['position'][2])
            
            pt = PointStamped()
            pt.header.stamp = rospy.Time.now()
            pt.header.frame_id = "camera_color_optical_frame"
            pt.point.x = closest['position'][0]
            pt.point.y = closest['position'][1]
            pt.point.z = closest['position'][2]  # Surface of can
            self.point_pub.publish(pt)
            
            self.diameter_pub.publish(Float32(data=closest['diameter']))
            self.height_pub.publish(Float32(data=self.CAN_HEIGHT))
            
            rospy.loginfo(f"✓ Can detected at ({pt.point.x:.2f}, {pt.point.y:.2f}, {pt.point.z:.2f})")
        
        # Publish debug image
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))

if __name__ == '__main__':
    try:
        YOLOCanDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
