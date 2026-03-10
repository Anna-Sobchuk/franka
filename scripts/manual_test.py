#!/usr/bin/env python3
"""
Manual test: Move to position 50cm in front of robot and grasp
"""
import sys
import rospy
import moveit_commander
import geometry_msgs.msg
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal

def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('manual_test')
    
    arm = moveit_commander.MoveGroupCommander("panda_arm")
    arm.set_max_velocity_scaling_factor(0.2)  # Slow and safe
    
    # Gripper action clients
    gripper_move = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
    gripper_grasp = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
    
    rospy.loginfo("Waiting for gripper...")
    gripper_move.wait_for_server(timeout=rospy.Duration(5))
    gripper_grasp.wait_for_server(timeout=rospy.Duration(5))
    
    # Step 1: Go to home position
    rospy.loginfo("Going to home position...")
    home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
    arm.set_joint_value_target(home_joints)
    arm.go(wait=True)
    arm.stop()
    rospy.sleep(1)
    
    # Step 2: Open gripper
    rospy.loginfo("Opening gripper...")
    goal = MoveGoal()
    goal.width = 0.08
    goal.speed = 0.05
    gripper_move.send_goal(goal)
    gripper_move.wait_for_result()
    rospy.sleep(1)
    
    # Step 3: Move above can (50cm forward, 20cm up from table)
    rospy.loginfo("Moving above can position (50cm forward)...")
    target = geometry_msgs.msg.Pose()
    target.position.x = 0.50  # 50cm forward
    target.position.y = 0.00  # centered
    target.position.z = 0.20  # 20cm above table
    
    # Gripper pointing down
    target.orientation.x = 1.0
    target.orientation.y = 0.0
    target.orientation.z = 0.0
    target.orientation.w = 0.0
    
    arm.set_pose_target(target)
    success = arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    
    if not success:
        rospy.logerr("Could not reach position!")
        return
    
    rospy.sleep(1)
    
    # Step 4: Lower to grasp height (adjust based on your can)
    rospy.loginfo("Lowering to can...")
    target.position.z = 0.08  # 8cm above table (middle of 11cm can)
    
    arm.set_pose_target(target)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.sleep(1)
    
    # Step 5: Close gripper
    rospy.loginfo("Grasping can...")
    grasp_goal = GraspGoal()
    grasp_goal.width = 0.075  # 7.5cm can diameter
    grasp_goal.speed = 0.05
    grasp_goal.force = 40
    grasp_goal.epsilon.inner = 0.005
    grasp_goal.epsilon.outer = 0.005
    gripper_grasp.send_goal(grasp_goal)
    gripper_grasp.wait_for_result()
    
    result = gripper_grasp.get_result()
    rospy.loginfo(f"Grasp result: {result.success}")
    rospy.sleep(2)
    
    # Step 6: Lift
    rospy.loginfo("Lifting...")
    target.position.z = 0.25
    arm.set_pose_target(target)
    arm.go(wait=True)
    arm.stop()
    arm.clear_pose_targets()
    rospy.sleep(1)
    
    # Step 7: Return home
    rospy.loginfo("Returning home...")
    arm.set_joint_value_target(home_joints)
    arm.go(wait=True)
    arm.stop()
    
    rospy.loginfo("✓ Test complete!")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
