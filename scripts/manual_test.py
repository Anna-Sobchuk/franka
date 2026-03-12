#!/usr/bin/env python3
"""
Manual test for Franka Panda using FrankaPy + direct actionlib gripper.
Picks can from in front of robot, rotates via 90deg midpoint to behind,
lowers and places can gently on table.

Run inside Docker:
  python3 ~/franka/scripts/manual_test.py
"""
import rospy
import actionlib
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
from franka_gripper.msg import (MoveAction, MoveGoal,
                                GraspAction, GraspGoal,
                                GraspEpsilon)
from sensor_msgs.msg import JointState
import time

# ── Adjust these to match your actual setup ───────────────────────────────────
CAN_X        = 0.50   # metres forward from robot base
CAN_Y        = 0.00   # metres left(+) / right(-)
CAN_Z_TABLE  = 0.02   # table surface height above robot base
CAN_HEIGHT   = 0.12   # can height
CAN_DIAMETER = 0.075  # 7.5cm diameter

GRASP_Z = CAN_Z_TABLE + CAN_HEIGHT / 2   # middle of can
HOVER_Z = GRASP_Z + 0.15                 # above can before lowering
LIFT_Z  = GRASP_Z + 0.30                 # high lift for safe rotation
PLACE_Z = CAN_Z_TABLE + CAN_HEIGHT + 0.01  # 1cm above table when placing

# Two-step rotation — go via 90deg sideways to avoid getting stuck
MID_JOINTS = [
    np.radians(90),    # face sideways
    np.radians(-45),
    np.radians(0),
    np.radians(-135),
    np.radians(0),
    np.radians(90),
    np.radians(45),
]

BEHIND_JOINTS = [
    np.radians(165),   # face behind
    np.radians(-45),
    np.radians(0),
    np.radians(-135),
    np.radians(0),
    np.radians(90),
    np.radians(45),
]

# Gripper namespace
GRIPPER_NS = '/franka_gripper_1/franka_gripper'

ROTATION_DOWN = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1]
], dtype=float)


def make_pose(x, y, z):
    return RigidTransform(
        rotation=ROTATION_DOWN,
        translation=np.array([x, y, z]),
        from_frame='franka_tool',
        to_frame='world'
    )


def get_gripper_width():
    js = rospy.wait_for_message(
        f'{GRIPPER_NS}/joint_states', JointState, timeout=5.0)
    return sum(js.position)


class GripperController:
    def __init__(self):
        self.move_client  = actionlib.SimpleActionClient(
            f'{GRIPPER_NS}/move', MoveAction)
        self.grasp_client = actionlib.SimpleActionClient(
            f'{GRIPPER_NS}/grasp', GraspAction)
        print("  Waiting for gripper action servers...")
        self.move_client.wait_for_server(timeout=rospy.Duration(5))
        self.grasp_client.wait_for_server(timeout=rospy.Duration(5))
        print("  Gripper ready!")

    def open(self, width=0.08, speed=0.05):
        goal = MoveGoal()
        goal.width = width
        goal.speed = speed
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result(rospy.Duration(10))
        actual = get_gripper_width()
        print(f"  Gripper opened — actual: {actual*100:.1f}cm")

    def close(self):
        """Close slowly with low force — stops at first contact with can."""
        goal = GraspGoal()
        goal.width   = 0.07
        goal.speed   = 0.01
        goal.force   = 5.0
        goal.epsilon = GraspEpsilon(inner=0.05, outer=0.05)
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result(rospy.Duration(10))
        result = self.grasp_client.get_result()
        actual = get_gripper_width()
        print(f"  Gripper closed — success: {result.success}  "
              f"actual: {actual*100:.1f}cm")
        return result.success


def main():
    print("=" * 55)
    print("  FrankaPy Manual Test — Pick & Place Behind Robot")
    print("=" * 55)

    print("\nConnecting to FrankaArm...")
    fa = FrankaArm()
    print("Connected!")

    print("\nConnecting to gripper...")
    gripper = GripperController()

    pose = fa.get_pose()
    print(f"\nCurrent EE: X={pose.translation[0]:.3f} "
          f"Y={pose.translation[1]:.3f} Z={pose.translation[2]:.3f}")

    # ── Step 1: Home ──────────────────────────────────────────────────
    print("\nStep 1: Going home...")
    fa.reset_joints()
    time.sleep(0.5)

    # ── Step 2: Open gripper ──────────────────────────────────────────
    print("\nStep 2: Opening gripper...")
    gripper.open(width=0.08)

    # ── Step 3: Hover above can ───────────────────────────────────────
    print(f"\nStep 3: Hovering above can ({CAN_X}, {CAN_Y}, {HOVER_Z:.3f})...")
    fa.goto_pose(make_pose(CAN_X, CAN_Y, HOVER_Z), use_impedance=False)
    time.sleep(0.5)

    # ── Step 4: Lower to grasp height ────────────────────────────────
    print(f"\nStep 4: Lowering to grasp Z={GRASP_Z:.3f}m...")
    fa.goto_pose(make_pose(CAN_X, CAN_Y, GRASP_Z), use_impedance=False)
    time.sleep(0.5)

    # ── Step 5: Grasp ─────────────────────────────────────────────────
    print("\nStep 5: Grasping can...")
    gripper.close()
    time.sleep(0.5)

    # ── Step 6: Lift high for safe rotation ──────────────────────────
    print(f"\nStep 6: Lifting to Z={LIFT_Z:.3f}m...")
    fa.goto_pose(make_pose(CAN_X, CAN_Y, LIFT_Z), use_impedance=False)
    time.sleep(0.5)

    # ── Step 7a: Rotate to sideways midpoint (90deg) ──────────────────
    print("\nStep 7a: Rotating to sideways midpoint (90°)...")
    fa.goto_joints(MID_JOINTS, use_impedance=False, ignore_virtual_walls=True)
    time.sleep(0.5)

    # ── Step 7b: Rotate to behind robot (165deg) ──────────────────────
    print("\nStep 7b: Rotating to behind robot (165°)...")
    fa.goto_joints(BEHIND_JOINTS, use_impedance=False, ignore_virtual_walls=True)
    time.sleep(0.5)

    pose_after = fa.get_pose()
    print(f"  Pose after rotation: X={pose_after.translation[0]:.3f} "
          f"Y={pose_after.translation[1]:.3f} Z={pose_after.translation[2]:.3f}")

    # ── Step 8: Lower can gently to table ────────────────────────────
    print(f"\nStep 8: Lowering can to table Z={PLACE_Z:.3f}m...")
    current = fa.get_pose()
    place_pose = RigidTransform(
        rotation=current.rotation,
        translation=np.array([
            current.translation[0],
            current.translation[1],
            PLACE_Z
        ]),
        from_frame='franka_tool',
        to_frame='world'
    )
    fa.goto_pose(place_pose, use_impedance=False, ignore_virtual_walls=True)
    time.sleep(0.5)

    # ── Step 9: Open gripper slowly to place can ─────────────────────
    print("\nStep 9: Placing can — opening gripper slowly...")
    gripper.open(width=0.08, speed=0.02)
    time.sleep(0.5)

    # ── Step 10: Lift straight up away from can ───────────────────────
    print("\nStep 10: Lifting away from placed can...")
    current = fa.get_pose()
    retreat_pose = RigidTransform(
        rotation=current.rotation,
        translation=np.array([
            current.translation[0],
            current.translation[1],
            LIFT_Z
        ]),
        from_frame='franka_tool',
        to_frame='world'
    )
    fa.goto_pose(retreat_pose, use_impedance=False, ignore_virtual_walls=True)
    time.sleep(0.5)

    # ── Step 11: Return home ──────────────────────────────────────────
    print("\nStep 11: Returning home...")
    fa.reset_joints()

    print("\n" + "=" * 55)
    print("  COMPLETE! Can placed behind robot.")
    print("=" * 55)


if __name__ == '__main__':
    main()
