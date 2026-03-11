#!/usr/bin/env python3
"""
Manual test for Franka Panda using FrankaPy.

Run inside Docker:
  python3 ~/franka/scripts/manual_test.py
  python3 ~/franka/scripts/manual_test.py --skip-gripper   # arm only, no gripper
"""
import argparse
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
import time

# ── Adjust these to match your setup ─────────────────────────────────────────
CAN_X        = 0.50   # metres forward from robot base
CAN_Y        = 0.00   # metres left(+) / right(-)
CAN_Z_TABLE  = 0.02   # table surface height above robot base
CAN_HEIGHT   = 0.11   # can is 11cm tall
CAN_DIAMETER = 0.075  # can is 7.5cm wide

GRASP_Z = CAN_Z_TABLE + CAN_HEIGHT / 2   # middle of can
HOVER_Z = GRASP_Z + 0.15                 # 15cm above grasp point
LIFT_Z  = GRASP_Z + 0.25                 # after pick

DROP_X, DROP_Y, DROP_Z = 0.10, 0.40, 0.20

# Gripper pointing straight down
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


def safe_open_gripper(fa, skip=False):
    if skip:
        print("  [skipped - arm only mode]")
        return
    try:
        fa.open_gripper(block=False)
        time.sleep(2.0)  # wait without blocking
        print(f"  Gripper width: {fa.get_gripper_width()*100:.1f}cm")
    except Exception as e:
        print(f"  Gripper open failed: {e} — continuing")


def safe_close_gripper(fa, skip=False):
    if skip:
        print("  [skipped - arm only mode]")
        return
    try:
        fa.close_gripper(grasp=True, block=False)
        time.sleep(2.0)
        print(f"  Gripper width: {fa.get_gripper_width()*100:.1f}cm")
        print(f"  Is grasped: {fa.get_gripper_is_grasped()}")
    except Exception as e:
        print(f"  Gripper close failed: {e} — continuing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-gripper', action='store_true',
                        help='Skip all gripper commands (arm movement only)')
    args = parser.parse_args()

    print("=" * 55)
    print("  FrankaPy Manual Test")
    if args.skip_gripper:
        print("  MODE: Arm only (gripper skipped)")
    print("=" * 55)

    print("\nConnecting to FrankaArm...")
    fa = FrankaArm()
    print("Connected!")

    # Current state
    pose = fa.get_pose()
    print(f"\nCurrent EE position:")
    print(f"  X={pose.translation[0]:.3f}  "
          f"Y={pose.translation[1]:.3f}  "
          f"Z={pose.translation[2]:.3f}")

    # Step 1: Home
    print("\nStep 1: Going home...")
    fa.reset_joints()
    time.sleep(0.5)
    print("  Done")

    # Step 2: Open gripper
    print("\nStep 2: Opening gripper...")
    safe_open_gripper(fa, skip=args.skip_gripper)

    # Step 3: Hover above can
    print(f"\nStep 3: Moving above can ({CAN_X}, {CAN_Y}, {HOVER_Z:.3f})...")
    fa.goto_pose(make_pose(CAN_X, CAN_Y, HOVER_Z), use_impedance=False)
    time.sleep(0.5)
    print("  Done")

    # Step 4: Lower to grasp
    print(f"\nStep 4: Lowering to grasp height Z={GRASP_Z:.3f}m...")
    fa.goto_pose(make_pose(CAN_X, CAN_Y, GRASP_Z), use_impedance=False)
    time.sleep(0.5)
    print("  Done")

    # Step 5: Close gripper
    print("\nStep 5: Closing gripper...")
    safe_close_gripper(fa, skip=args.skip_gripper)

    # Step 6: Lift
    print(f"\nStep 6: Lifting to Z={LIFT_Z:.3f}m...")
    fa.goto_pose(make_pose(CAN_X, CAN_Y, LIFT_Z), use_impedance=False)
    time.sleep(0.5)
    print("  Done")

    # Step 7: Move to drop
    print(f"\nStep 7: Moving to drop ({DROP_X}, {DROP_Y}, {DROP_Z})...")
    fa.goto_pose(make_pose(DROP_X, DROP_Y, DROP_Z), use_impedance=False)
    time.sleep(0.5)
    print("  Done")

    # Step 8: Release
    print("\nStep 8: Releasing...")
    safe_open_gripper(fa, skip=args.skip_gripper)

    # Step 9: Home
    print("\nStep 9: Returning home...")
    fa.reset_joints()

    print("\n" + "=" * 55)
    print("  TEST COMPLETE!")
    print("  Adjust CAN_X, CAN_Y, CAN_Z_TABLE at top of")
    print("  script to match your actual can position.")
    print("=" * 55)


if __name__ == '__main__':
    main()
