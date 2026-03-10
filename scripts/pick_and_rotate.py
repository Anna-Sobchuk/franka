#!/usr/bin/env python3
import rospy
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped, Pose
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
import math
import sys
import tf.transformations as tft

class PandaPickAndRotate:
    def __init__(self):
        rospy.init_node('panda_pick_and_rotate', anonymous=True)

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.arm = moveit_commander.MoveGroupCommander("panda_arm")
        self.arm.set_planning_time(10.0)
        self.arm.set_num_planning_attempts(10)
        self.arm.set_max_velocity_scaling_factor(0.2)
        self.arm.set_max_acceleration_scaling_factor(0.1)

        self.gripper_grasp = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.gripper_move = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)

        rospy.loginfo("Waiting for gripper action servers...")
        self.gripper_grasp.wait_for_server(timeout=rospy.Duration(10))
        self.gripper_move.wait_for_server(timeout=rospy.Duration(10))
        rospy.loginfo("Gripper ready.")

        self._add_table_collision_object()

        self.task_done = False
        self.obj_sub = rospy.Subscriber('/detected_object/point', PointStamped, self.object_cb)
        rospy.loginfo("Ready. Waiting for object detection...")

    def _add_table_collision_object(self):
        rospy.sleep(1)
        table_pose = geometry_msgs.msg.PoseStamped()
        table_pose.header.frame_id = "panda_link0"
        table_pose.pose.position.x = 0.5
        table_pose.pose.position.y = 0.0
        table_pose.pose.position.z = -0.05
        table_pose.pose.orientation.w = 1.0
        self.scene.add_box("table", table_pose, size=(1.2, 1.2, 0.1))

    def open_gripper(self, width=0.08):
        goal = MoveGoal()
        goal.width = width
        goal.speed = 0.05
        self.gripper_move.send_goal(goal)
        self.gripper_move.wait_for_result(timeout=rospy.Duration(10))

    def close_gripper(self, width=0.03, force=30):
        goal = GraspGoal()
        goal.width = width
        goal.speed = 0.05
        goal.force = force
        goal.epsilon.inner = 0.01
        goal.epsilon.outer = 0.01
        self.gripper_grasp.send_goal(goal)
        self.gripper_grasp.wait_for_result(timeout=rospy.Duration(10))
        return self.gripper_grasp.get_result().success

    def move_to_pose(self, x, y, z, roll=math.pi, pitch=0, yaw=0):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        q = tft.quaternion_from_euler(roll, pitch, yaw)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        self.arm.set_pose_target(pose)
        success, plan, _, _ = self.arm.plan()
        if not success:
            rospy.logwarn("Planning failed!")
            return False
        result = self.arm.execute(plan, wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        return result

    def move_to_joint_config(self, joint_angles):
        self.arm.set_joint_value_target(joint_angles)
        success, plan, _, _ = self.arm.plan()
        if not success:
            rospy.logwarn("Joint planning failed!")
            return False
        result = self.arm.execute(plan, wait=True)
        self.arm.stop()
        return result

    def go_to_home(self):
        self.arm.set_named_target("ready")
        self.arm.go(wait=True)
        self.arm.stop()

    def object_cb(self, msg):
        if self.task_done:
            return
        self.task_done = True
        self.run_pick_and_rotate(msg)

    def run_pick_and_rotate(self, obj_point):
        x = obj_point.point.x
        y = obj_point.point.y
        z = obj_point.point.z

        rospy.loginfo(f"Object at x={x:.3f} y={y:.3f} z={z:.3f}")


        rospy.loginfo("Step 1: Opening gripper...")
        self.open_gripper(width=0.08)

        rospy.loginfo("Step 2: Moving above object...")
        if not self.move_to_pose(x, y, z + 0.15):
            rospy.logerr("Could not reach above object!")
            self.task_done = False
            return
        rospy.sleep(0.5)

        rospy.loginfo("Step 3: Lowering to object...")
        if not self.move_to_pose(x, y, z + 0.01):
            rospy.logerr("Could not reach object!")
            self.task_done = False
            return
        rospy.sleep(0.5)

        rospy.loginfo("Step 4: Grasping...")
        self.close_gripper(width=0.04, force=40)
        rospy.sleep(0.5)

        rospy.loginfo("Step 5: Lifting...")
        self.move_to_pose(x, y, z + 0.20)
        rospy.sleep(0.5)

        rospy.loginfo("Step 6: Rotating base 90 degrees...")
        joints = list(self.arm.get_current_joint_values())
        joints[0] += math.pi / 2
        joints[0] = max(-2.8973, min(2.8973, joints[0]))
        self.move_to_joint_config(joints)
        rospy.sleep(0.5)

        rospy.loginfo("Step 7: Placing down...")
        cur = self.arm.get_current_pose().pose.position
        self.move_to_pose(cur.x, cur.y, z + 0.05)
        rospy.sleep(0.5)

        rospy.loginfo("Step 8: Releasing...")
        self.open_gripper(width=0.08)
        rospy.sleep(0.5)

        rospy.loginfo("Step 9: Going home...")
        self.go_to_home()

        rospy.loginfo("Task complete!")

if __name__ == '__main__':
    try:
        node = PandaPickAndRotate()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
