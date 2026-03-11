#!/usr/bin/env python3
"""
Manual test for Franka Panda - ROS2 Humble + MoveIt2
Move to position in front of robot and attempt grasp.
Run this FIRST to verify robot moves before the full pipeline.

Usage:
  cd ~/franka_project_Anna/catkin_ws
  source install/setup.bash
  python3 src/panda_pick_rotate/scripts/manual_test.py
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (MotionPlanRequest, Constraints, JointConstraint)
from control_msgs.action import GripperCommand
import time


class ManualTest(Node):
    def __init__(self):
        super().__init__('manual_test')

        self.move_client    = ActionClient(self, MoveGroup, '/move_action')
        self.gripper_client = ActionClient(
            self, GripperCommand, '/panda_hand_controller/gripper_cmd'
        )

        self.get_logger().info("Waiting for MoveGroup...")
        if not self.move_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(
                "MoveGroup not available!\n"
                "Start it with:\n"
                "  ros2 launch franka_moveit_config moveit.launch.py robot_ip:=<IP>"
            )
            return

        self.get_logger().info("MoveGroup ready — starting test")
        self.run_test()

    # ── Helpers ───────────────────────────────────────────────────────────
    def move_joints(self, values, label=""):
        """Send joint goal to MoveGroup and wait for result."""
        names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                 'panda_joint5', 'panda_joint6', 'panda_joint7']

        goal = MoveGroup.Goal()
        req  = MotionPlanRequest()
        req.group_name                    = "panda_arm"
        req.num_planning_attempts         = 10
        req.allowed_planning_time         = 10.0
        req.max_velocity_scaling_factor   = 0.2
        req.max_acceleration_scaling_factor = 0.1

        c = Constraints()
        for name, val in zip(names, values):
            jc = JointConstraint()
            jc.joint_name     = name
            jc.position       = val
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight          = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints.append(c)

        goal.request = req
        goal.planning_options.plan_only      = False
        goal.planning_options.replan         = True
        goal.planning_options.replan_attempts = 5

        self.get_logger().info(f"Moving to: {label}")
        fut = self.move_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)

        if not fut.result():
            self.get_logger().error(f"Goal rejected: {label}")
            return False

        res_fut = fut.result().get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=30.0)

        ok = res_fut.result().result.error_code.val == 1
        if ok:
            self.get_logger().info(f"Done: {label}")
        else:
            code = res_fut.result().result.error_code.val
            self.get_logger().error(f"Failed (code {code}): {label}")
        return ok

    def gripper(self, pos, effort=40.0):
        """pos: 0.04=open, 0.0=closed"""
        if not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn("Gripper not available — skipping")
            return
        goal = GripperCommand.Goal()
        goal.command.position   = pos
        goal.command.max_effort = effort
        fut = self.gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        state = "open" if pos > 0.01 else "closed"
        self.get_logger().info(f"Gripper {state}")

    # ── Test sequence ──────────────────────────────────────────────────────
    def run_test(self):
        self.get_logger().info("=" * 50)

        # Joint configs — adjust these based on your actual robot position!
        HOME      = [0.0, -0.785,  0.0, -2.356, 0.0, 1.571, 0.785]
        ABOVE_CAN = [0.0,  0.200,  0.0, -1.800, 0.0, 2.000, 0.785]
        AT_CAN    = [0.0,  0.400,  0.0, -1.500, 0.0, 1.900, 0.785]

        steps = [
            ("Step 1: Go home",          lambda: self.move_joints(HOME,      "home")),
            ("Step 2: Open gripper",      lambda: self.gripper(0.04, 20.0)),
            ("Step 3: Move above can",    lambda: self.move_joints(ABOVE_CAN, "above can")),
            ("Step 4: Lower to can",      lambda: self.move_joints(AT_CAN,    "at can")),
            ("Step 5: Close gripper",     lambda: self.gripper(0.0, 40.0)),
            ("Step 6: Lift",              lambda: self.move_joints(ABOVE_CAN, "lift")),
            ("Step 7: Return home",       lambda: self.move_joints(HOME,      "return home")),
            ("Step 8: Release",           lambda: self.gripper(0.04, 20.0)),
        ]

        for label, action in steps:
            self.get_logger().info(f"\n{label}")
            result = action()
            if result is False:   # motion planning failed
                self.get_logger().error(f"FAILED at: {label} — stopping")
                break
            time.sleep(0.5)

        self.get_logger().info("=" * 50)
        self.get_logger().info("TEST COMPLETE")
        self.get_logger().info("=" * 50)


def main():
    rclpy.init()
    node = ManualTest()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
