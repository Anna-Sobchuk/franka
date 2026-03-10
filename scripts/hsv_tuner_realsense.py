#!/usr/bin/env python3
"""
HSV Color Tuner for RealSense Camera
Run this to find the right color values for your object
"""
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class HSVTuner:
    def __init__(self):
        self.bridge = CvBridge()
        self.frame = None
        
        rospy.init_node('hsv_tuner', anonymous=True)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_cb)
        
        cv2.namedWindow("HSV Tuner")
        cv2.createTrackbar("H Low",  "HSV Tuner", 0,   179, self.nothing)
        cv2.createTrackbar("S Low",  "HSV Tuner", 100, 255, self.nothing)
        cv2.createTrackbar("V Low",  "HSV Tuner", 70,  255, self.nothing)
        cv2.createTrackbar("H High", "HSV Tuner", 10,  179, self.nothing)
        cv2.createTrackbar("S High", "HSV Tuner", 255, 255, self.nothing)
        cv2.createTrackbar("V High", "HSV Tuner", 255, 255, self.nothing)
        
        print("=" * 60)
        print("HSV Color Tuner for RealSense")
        print("=" * 60)
        print("1. Adjust sliders until only your object is WHITE in mask")
        print("2. Note the 6 numbers (H/S/V Low and High)")
        print("3. Put them in object_detector.py:")
        print("   self.lower_color = np.array([H_LOW, S_LOW, V_LOW])")
        print("   self.upper_color = np.array([H_HIGH, S_HIGH, V_HIGH])")
        print("4. Press 'q' to quit")
        print("=" * 60)
        
    def nothing(self, x):
        pass
    
    def image_cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.frame is not None:
                hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                
                h_low = cv2.getTrackbarPos("H Low", "HSV Tuner")
                s_low = cv2.getTrackbarPos("S Low", "HSV Tuner")
                v_low = cv2.getTrackbarPos("V Low", "HSV Tuner")
                h_high = cv2.getTrackbarPos("H High", "HSV Tuner")
                s_high = cv2.getTrackbarPos("S High", "HSV Tuner")
                v_high = cv2.getTrackbarPos("V High", "HSV Tuner")
                
                lower = np.array([h_low, s_low, v_low])
                upper = np.array([h_high, s_high, v_high])
                
                mask = cv2.inRange(hsv, lower, upper)
                result = cv2.bitwise_and(self.frame, self.frame, mask=mask)
                
                # Display values on frame
                text = f"HSV Low: [{h_low}, {s_low}, {v_low}]  High: [{h_high}, {s_high}, {v_high}]"
                cv2.putText(self.frame, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imshow("Original", self.frame)
                cv2.imshow("Mask (tune until object is WHITE)", mask)
                cv2.imshow("Result", result)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            rate.sleep()
        
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        tuner = HSVTuner()
        tuner.run()
    except rospy.ROSInterruptException:
        pass
