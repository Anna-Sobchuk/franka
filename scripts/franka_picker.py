#!/usr/bin/env python3
import sys
import rospy
import tf2_ros
import tf2_geometry_msgs
import moveit_commander
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
import actionlib
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal
import math

class FrankaPicker:
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('franka_can_picker', anonymous=True)
        
        # MoveIt setup
        self.arm = moveit_commander.MoveGroupCommander("panda_arm")
        self.arm.set_max_velocity_scaling_factor(0.3)  # 30% speed
        self.arm.set_max_acceleration_scaling_factor(0.2)
        self.arm.set_planning_time(10.0)
        
        # Gripper action clients 
        self.gripper_grasp = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        self.gripper_move = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        
        rospy.loginfo("Waiting for gripper action servers...")
        grasp_available = self.gripper_grasp.wait_for_server(timeout=rospy.Duration(10))
        move_available = self.gripper_move.wait_for_server(timeout=rospy.Duration(10))
        
        if grasp_available and move_available:
        	rospy.loginfo("✓ Gripper ready")
        	self.simulation_mode = False
        else:
        	rospy.logwarn("Simulation mode")
        	self.simulation_mode = True
        
        # TF2 for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Can dimensions (updated from detector)
        self.can_diameter = 0.085  # Default
        self.can_height = 0.085    # Default
        
        # State
        self.is_moving = False
        
        # Subscribers
        self.point_sub = rospy.Subscriber('/detected_object/point', PointStamped, self.target_cb)
        self.diameter_sub = rospy.Subscriber('/detected_can/diameter', Float32, self.diameter_cb)
        self.height_sub = rospy.Subscriber('/detected_can/height', Float32, self.height_cb)
        
        rospy.loginfo("Franka Picker ready - waiting for cans...")
    
    def diameter_cb(self, msg):
        self.can_diameter = msg.data
    
    def height_cb(self, msg):
        self.can_height = msg.data
    
    def open_gripper(self, width=0.08):
        """Open gripper to specified width"""
        if self.simulation_mode:
            rospy.loginfo(f"[SIM] Gripper opened to {width*100:.1f}cm")
            return
        
        goal = MoveGoal()
        goal.width = width
        goal.speed = 0.05
        self.gripper_move.send_goal(goal)
        self.gripper_move.wait_for_result(timeout=rospy.Duration(5))
        rospy.loginfo(f"Gripper opened to {width*100:.1f}cm")
    
    def close_gripper(self, width, force=40):
        """Close gripper to grasp object"""
        if self.simulation_mode:
            rospy.loginfo(f"[SIM] Gripper closed to {width*100:.1f}cm with {force}N force")
            return True
        
        goal = GraspGoal()
        goal.width = width
        goal.speed = 0.05
        goal.force = force
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.005
        self.gripper_grasp.send_goal(goal)
        self.gripper_grasp.wait_for_result(timeout=rospy.Duration(5))
        result = self.gripper_grasp.get_result()
        rospy.loginfo(f"Grasp result: {result.success}")
        return result.success
        
    def target_cb(self, msg):
        """Callback when new can is detected"""
        if self.is_moving:
            rospy.logwarn("Already moving, ignoring new detection")
            return
        
        self.is_moving = True
        rospy.loginfo("=" * 50)
        rospy.loginfo("NEW CAN DETECTED - Starting pick and place")
        rospy.loginfo("=" * 50)
        
        try:
            # Transform from camera frame to robot base frame
            rospy.loginfo("Waiting for TF transform...")
            transform = self.tf_buffer.lookup_transform(
                "panda_link0",              # Target frame (robot base)
                msg.header.frame_id,        # Source frame (camera)
                rospy.Time(0),              # Latest available
                rospy.Duration(5.0)
            )
            
            target_in_base = tf2_geometry_msgs.do_transform_point(msg, transform)
            
            rospy.loginfo(f"Can position in robot base frame:")
            
            rospy.loginfo(f"  X: {target_in_base.point.x:.3f}m")
            rospy.loginfo(f"  Y: {target_in_base.point.y:.3f}m")
            rospy.loginfo(f"  Z: {target_in_base.point.z:.3f}m (table surface)")
            rospy.loginfo(f"  Diameter: {self.can_diameter*100:.1f}cm")
            rospy.loginfo(f"  Height: {self.can_height*100:.1f}cm")
            
            # Execute pick
            success = self.execute_pick(target_in_base.point)
            
            if success:
                # Execute place
                self.execute_place()
            else:
                rospy.logwarn("Pick failed, aborting")
                self.go_home()
            
        except Exception as e:
            rospy.logerr(f"Error in pick-place: {e}")
            import traceback
            traceback.print_exc()
            self.go_home()
        
        finally:
            self.is_moving = False
    
    def execute_pick(self, point):
        """Pick up can at given position"""
        # Point.z is the table surface where the can sits
        # We grasp at middle of can for stability
        grasp_height = point.z + (self.can_height / 2)
        
        # Step 1: Open gripper wide
        rospy.loginfo("Step 1: Opening gripper...")
        self.open_gripper(width=0.08)
        rospy.sleep(0.5)
        
        # Step 2: Move to pre-grasp position (above can)
        rospy.loginfo("Step 2: Moving above can...")
        hover_pose = geometry_msgs.msg.Pose()
        hover_pose.position.x = point.x
        hover_pose.position.y = point.y
        hover_pose.position.z = grasp_height + 0.15  # 15cm above grasp point
        
        # Gripper pointing DOWN (correct quaternion)
        hover_pose.orientation.x = 1.0
        hover_pose.orientation.y = 0.0
        hover_pose.orientation.z = 0.0
        hover_pose.orientation.w = 0.0
        
        self.arm.set_pose_target(hover_pose)
        success = self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        
        if not success:
            rospy.logerr("Failed to reach hover position!")
            return False
        
        rospy.sleep(0.5)
        
        # Step 3: Approach can (Cartesian path for straight-down motion)
        rospy.loginfo("Step 3: Lowering to grasp height...")
        approach_pose = geometry_msgs.msg.Pose()
        approach_pose.position.x = point.x
        approach_pose.position.y = point.y
        approach_pose.position.z = grasp_height + 0.01  # 1cm above middle
        approach_pose.orientation = hover_pose.orientation
        
        waypoints = [approach_pose]
        (plan, fraction) = self.arm.compute_cartesian_path(waypoints, 0.01, True)
        
        if fraction < 0.9:
            rospy.logwarn(f"Cartesian path only {fraction*100:.0f}% complete")
        
        self.arm.execute(plan, wait=True)
        self.arm.stop()
        rospy.sleep(0.5)
        
        # Step 4: Close gripper around can
        rospy.loginfo("Step 4: Grasping can...")
        grasp_width = self.can_diameter + 0.01
        success = self.close_gripper(width=grasp_width, force=35)
        rospy.sleep(1.0)
        
        if not success:
            rospy.logwarn("Grasp may have failed, continuing anyway...")
        
        # Step 5: Lift can
        rospy.loginfo("Step 5: Lifting can...")
        lift_pose = approach_pose
        lift_pose.position.z = grasp_height + 0.20  # Lift 20cm
        
        waypoints = [lift_pose]
        (plan, fraction) = self.arm.compute_cartesian_path(waypoints, 0.01, True)
        self.arm.execute(plan, wait=True)
        self.arm.stop()
        rospy.sleep(0.5)
        
        rospy.loginfo("✓ Pick complete!")
        return True
    
    def execute_place(self):
        """Place can in trash bin"""
        rospy.loginfo("Step 6: Moving to trash bin...")
        
        # Trash bin location (adjust to YOUR setup)
        drop_pose = geometry_msgs.msg.Pose()
        drop_pose.position.x = 0.1   # 30cm forward from robot
        
        drop_pose.position.y = 0.1  # 40cm to the right
        drop_pose.position.z = 0.2   # 40cm height
        
        # Keep gripper pointing down
        drop_pose.orientation.x = 1.0
        drop_pose.orientation.y = 0.0
        drop_pose.orientation.z = 0.0
        drop_pose.orientation.w = 0.0
        
        self.arm.set_pose_target(drop_pose)
        self.arm.go(wait=True)
        self.arm.stop()
        self.arm.clear_pose_targets()
        rospy.sleep(0.5)
        
        # Release can
        rospy.loginfo("Step 7: Releasing can...")
        self.open_gripper(width=0.08)
        rospy.sleep(1.0)
        
        # Return home
        self.go_home()
        
        rospy.loginfo("=" * 50)
        rospy.loginfo("✓ CAN RECYCLED! Ready for next one.")
        rospy.loginfo("=" * 50)
    
    def go_home(self):
        """Return to home position"""
        rospy.loginfo("Step 8: Returning home...")
        
        # Franka's "ready" pose
        home_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
        self.arm.set_joint_value_target(home_joints)
        self.arm.go(wait=True)
        self.arm.stop()
        
        rospy.loginfo("✓ Home position reached")

if __name__ == '__main__':
    try:
        FrankaPicker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
