#!/usr/bin/env python3
import rospy
import tf2_ros
import tf.transformations as tft
import math
import geometry_msgs.msg

# Publish static transform: camera → robot_base
# YOU MUST MEASURE THESE VALUES on robot day!

if __name__ == '__main__':
    rospy.init_node('camera_tf_publisher')
    
    broadcaster = tf2_ros.StaticTransformBroadcaster()
    
    static_transform = geometry_msgs.msg.TransformStamped()
    static_transform.header.stamp = rospy.Time.now()
    static_transform.header.frame_id = "panda_link0"
    static_transform.child_frame_id = "camera_color_optical_frame"
    
    # CHANGE THESE to your actual camera position!
    # Example: camera is 50cm forward, 0cm left/right, 30cm above robot base
    static_transform.transform.translation.x = 0.75 # meters forward
    static_transform.transform.translation.y = 0.00  # meters left(+)/right(-)
    static_transform.transform.translation.z = 0.70  # meters up
    
    rot_down = tft.quaternion_from_euler(math.pi, 0, 0)
    
    # Then rotate 30° around Y axis (tilt forward)
    rot_tilt = tft.quaternion_from_euler(0, math.radians(-30), 0)
    
    # Combine rotations
    combined_quat = tft.quaternion_multiply(rot_down, rot_tilt)
    
    static_transform.transform.rotation.x = combined_quat[0]
    static_transform.transform.rotation.y = combined_quat[1]
    static_transform.transform.rotation.z = combined_quat[2]
    static_transform.transform.rotation.w = combined_quat[3]
    
    broadcaster.sendTransform(static_transform)
    
    rospy.loginfo("=" * 60)
    rospy.loginfo("Camera TF published:")
    rospy.loginfo(f"  Position: X=0.75m (forward), Y=0.00m (centered), Z=0.70m (up)")
    rospy.loginfo(f"  Rotation: 30° tilt down + forward")
    rospy.loginfo("=" * 60)
    
    rospy.spin()
