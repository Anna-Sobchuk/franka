#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32

def talker():
    point_pub = rospy.Publisher('/detected_object/point', PointStamped, queue_size=10)
    diameter_pub = rospy.Publisher('/detected_can/diameter', Float32, queue_size=10)
    height_pub = rospy.Publisher('/detected_can/height', Float32, queue_size=10)
    
    rospy.init_node('fake_camera', anonymous=True)
    
    rospy.sleep(3)  # Wait for subscribers to connect
    
    # Publish ONCE, then stop
    msg = PointStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "camera_color_optical_frame"
    
    msg.point.x = 0.0
    msg.point.y = 0.0
    msg.point.z = 0.30  # 30cm from camera
    
    diameter_pub.publish(Float32(data=0.085))
    height_pub.publish(Float32(data=0.085))
    
    rospy.loginfo("🎥 Fake Camera: Can detected at Z=0.30m, D=8.5cm")
    rospy.loginfo("   (Published once, then stopping)")
    point_pub.publish(msg)
    
    rospy.sleep(1)  # Give time for message to be received
    rospy.loginfo("Fake camera done. Robot should now pick and place.")
    
    # Keep node alive but don't publish more
    rospy.spin()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
