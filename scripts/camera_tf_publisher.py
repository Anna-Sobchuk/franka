#!/usr/bin/env python3
"""
Static TF publisher: panda_link0 → camera_color_optical_frame
Calibrated on 2026-03-12 using can touch point method.
"""
import rospy
import tf2_ros
import tf.transformations as tft
import math
import geometry_msgs.msg

if __name__ == '__main__':
    rospy.init_node('camera_tf_publisher')

    broadcaster = tf2_ros.StaticTransformBroadcaster()

    static_transform = geometry_msgs.msg.TransformStamped()
    static_transform.header.stamp = rospy.Time.now()
    static_transform.header.frame_id = "panda_link0"
    static_transform.child_frame_id = "camera_color_optical_frame"

    # Calibrated values — measured 2026-03-12
    # Method: gripper touched can top (robot frame) vs detector point (camera frame)
    static_transform.transform.translation.x = 0.7556
    static_transform.transform.translation.y = 0.0796
    static_transform.transform.translation.z = 0.7649

    # Rotation: 180deg flip around X + 30deg tilt forward around Y
    rot_down = tft.quaternion_from_euler(math.pi, 0, 0)
    rot_tilt = tft.quaternion_from_euler(0, math.radians(-30), 0)
    combined_quat = tft.quaternion_multiply(rot_down, rot_tilt)

    static_transform.transform.rotation.x = combined_quat[0]
    static_transform.transform.rotation.y = combined_quat[1]
    static_transform.transform.rotation.z = combined_quat[2]
    static_transform.transform.rotation.w = combined_quat[3]

    broadcaster.sendTransform(static_transform)

    rospy.loginfo("=" * 60)
    rospy.loginfo("Camera TF published (calibrated):")
    rospy.loginfo("  X=0.7556  Y=0.0796  Z=0.7649")
    rospy.loginfo("  Rotation: 180deg X flip + 30deg Y tilt")
    rospy.loginfo("=" * 60)

    rospy.spin()
