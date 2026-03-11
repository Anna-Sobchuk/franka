#!/usr/bin/env python3
"""
Manual test for Franka Panda using FrankaPy.
Tests basic movement and gripper before running full pick pipeline.

Run from inside the Docker:
  cd ~/frankapy
  python3 /root/frankapy_ws/src/panda_pick_rotate/scripts/manual_test.py

OR just:
  python3 manual_test.py
"""
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import time

# ── Can position in robot base frame (metres) ────────────────────────────────
# ADJUST THESE to where your can actually is on the table!
CAN_X        = 0.50   # forward from robot base
CAN_Y        = 0.00   # left/right (0 = centred)
CAN_Z_TABLE  = 0.02   # height of table surface above robot base
CAN_HEIGHT   = 0.11   # 11cm tall can
CAN_DIAMETER = 0.075  # 7.5cm diameter

# Grasp at middle of can height
GRASP_Z  = CAN_Z_TABLE + CAN_HEIGHT / 2   # ~0.075m
HOVER_Z  = GRASP_Z + 0.15                 # 15cm above grasp
LIFT_Z   = GRASP_Z + 0.25                 # 25cm above grasp (after pick)

# Drop position (adjust to your trash bin / target location)
DROP_X = 0.10
DROP_Y = 0.40
DROP_Z = 0.20

# Gripper pointing straight down
ROTATION_DOWN = np.array([
    [ 1,  0,  0],
    [ 0, -1,  0],
    [ 0,  0, -1]
], dtype=float)


def make_pose(x, y, z, rotation=ROTATION_DOWN):
    """Helper: create RigidTransform for a given XYZ position."""
    return RigidTransform(
        rotation=rotation,
        translation=np.array([x, y, z]),
        from_frame='franka_tool',
        to_frame='world'
    )


def main():
    print("=" * 55)
    print("  FrankaPy Manual Test — Can Pick")
    print("=" * 55)

    print("\nConnecting to FrankaArm...")
    fa = FrankaArm()
    print("Connected!")

    # ── Step 1: Go to home ───────────────────────────────────────────
    print("\nStep 1: Going to home position...")
    fa.reset_joints()
    time.sleep(0.5)

    # ── Step 2: Open gripper ─────────────────────────────────────────
    print("\nStep 2: Opening gripper...")
    fa.open_gripper()
    time.sleep(0.5)

    # ── Step 3: Print current pose (sanity check) ────────────────────
    current_pose = fa.get_pose()
    print(f"\nCurrent EE pose:")
    print(f"  X={current_pose.translation[0]:.3f}  "
          f"Y={current_pose.translation[1]:.3f}  "
          f"Z={current_pose.translation[2]:.3f}")
    print(f"  Gripper width: {fa.get_gripper_width()*100:.1f}cm")

    # ── Step 4: Move to hover above can ─────────────────────────────
    print(f"\nStep 3: Moving to hover above can "
          f"({CAN_X:.2f}, {CAN_Y:.2f}, {HOVER_Z:.2f})...")
    hover_pose = make_pose(CAN_X, CAN_Y, HOVER_Z)
    fa.goto_pose(hover_pose, use_impedance=False)
    time.sleep(0.5)

    # ── Step 5: Lower to grasp height ───────────────────────────────
    print(f"\nStep 4: Lowering to grasp height Z={GRASP_Z:.3f}m...")
    grasp_pose = make_pose(CAN_X, CAN_Y, GRASP_Z)
    fa.goto_pose(grasp_pose, use_impedance=False)
    time.sleep(0.5)

    # ── Step 6: Close gripper ────────────────────────────────────────
    print("\nStep 5: Closing gripper...")
    fa.close_gripper()
    time.sleep(1.0)

    print(f"  Gripper width after grasp: {fa.get_gripper_width()*100:.1f}cm")
    print(f"  Is grasped: {fa.get_gripper_is_grasped()}")

    # ── Step 7: Lift ─────────────────────────────────────────────────
    print(f"\nStep 6: Lifting to Z={LIFT_Z:.3f}m...")
    lift_pose = make_pose(CAN_X, CAN_Y, LIFT_Z)
    fa.goto_pose(lift_pose, use_impedance=False)
    time.sleep(0.5)

    # ── Step 8: Move to drop position ───────────────────────────────
    print(f"\nStep 7: Moving to drop position "
          f"({DROP_X:.2f}, {DROP_Y:.2f}, {DROP_Z:.2f})...")
    drop_pose = make_pose(DROP_X, DROP_Y, DROP_Z)
    fa.goto_pose(drop_pose, use_impedance=False)
    time.sleep(0.5)

    # ── Step 9: Release ──────────────────────────────────────────────
    print("\nStep 8: Releasing...")
    fa.open_gripper()
    time.sleep(0.5)

    # ── Step 10: Go home ─────────────────────────────────────────────
    print("\nStep 9: Returning home...")
    fa.reset_joints()

    print("\n" + "=" * 55)
    print("  TEST COMPLETE!")
    print("  Adjust CAN_X, CAN_Y, CAN_Z_TABLE at top of")
    print("  script to match your actual can position.")
    print("=" * 55)


if __name__ == '__main__':
    main()
