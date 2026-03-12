#!/usr/bin/env python3
"""
Franka Can Picker — FrankaPy version.
Subscribes to /detected_object/point from YOLO detector,
picks up the can and places it behind the robot.

Run inside Docker (with franka-interface running):
  python3 ~/franka/scripts/franka_picker.py

Run detector separately on HOST:
  python3 ~/franka/scripts/object_detector.py
"""
import rospy
import actionlib
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from autolab_core import RigidTransform
from frankapy import FrankaArm
from franka_gripper.msg import (MoveAction, MoveGoal,
                                GraspAction, GraspGoal,
                                GraspEpsilon)
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Bool
from sensor_msgs.msg import JointState
import time

# ── Motion durations ──────────────────────────────────────────────────────────
CART_DURATION  = 6    # seconds for Cartesian moves
JOINT_DURATION = 8    # seconds for joint moves

# ── Can defaults (overridden by detector topics) ──────────────────────────────
CAN_HEIGHT   = 0.11
CAN_DIAMETER = 0.075

# ── Pick geometry ─────────────────────────────────────────────────────────────
GRASP_Z_OFFSET = 0.01   # how high above can surface to grasp (from detector Z)
HOVER_ABOVE    = 0.15   # metres above grasp point for hover
LIFT_ABOVE     = 0.30   # metres above grasp point for safe rotation

# ── Two-step rotation (tested and working) ────────────────────────────────────
MID_JOINTS = [
    np.radians(90),
    np.radians(-45),
    np.radians(0),
    np.radians(-135),
    np.radians(0),
    np.radians(90),
    np.radians(45),
]
BEHIND_JOINTS = [
    np.radians(165),
    np.radians(-45),
    np.radians(0),
    np.radians(-135),
    np.radians(0),
    np.radians(90),
    np.radians(45),
]

# ── Place height behind robot ─────────────────────────────────────────────────
PLACE_Z = CAN_HEIGHT - 0.02   # just above table

# ── Gripper namespace ─────────────────────────────────────────────────────────
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


def goto(fa, pose):
    fa.goto_pose(pose, use_impedance=False, ignore_virtual_walls=True,
                 duration=CART_DURATION)


def goto_joints(fa, joints):
    fa.goto_joints(joints, use_impedance=False, ignore_virtual_walls=True,
                   duration=JOINT_DURATION)


class GripperController:
    def __init__(self):
        self.move_client  = actionlib.SimpleActionClient(
            f'{GRIPPER_NS}/move', MoveAction)
        self.grasp_client = actionlib.SimpleActionClient(
            f'{GRIPPER_NS}/grasp', GraspAction)
        rospy.loginfo("Waiting for gripper servers...")
        self.move_client.wait_for_server(timeout=rospy.Duration(10))
        self.grasp_client.wait_for_server(timeout=rospy.Duration(10))
        rospy.loginfo("Gripper ready!")

    def open(self, width=0.08, speed=0.05):
        goal = MoveGoal(width=width, speed=speed)
        self.move_client.send_goal(goal)
        self.move_client.wait_for_result(rospy.Duration(10))
        rospy.loginfo(f"Gripper opened — {get_gripper_width()*100:.1f}cm")

    def close(self):
        goal = GraspGoal()
        goal.width   = 0.07
        goal.speed   = 0.01
        goal.force   = 5.0
        goal.epsilon = GraspEpsilon(inner=0.05, outer=0.05)
        self.grasp_client.send_goal(goal)
        self.grasp_client.wait_for_result(rospy.Duration(10))
        result = self.grasp_client.get_result()
        rospy.loginfo(f"Gripper closed — success:{result.success} "
                      f"width:{get_gripper_width()*100:.1f}cm")
        return result.success


class FrankaPicker:
    def __init__(self):
        rospy.init_node('franka_can_picker')

        rospy.loginfo("Connecting to FrankaArm...")
        self.fa = FrankaArm()
        rospy.loginfo("FrankaArm connected!")

        self.gripper = GripperController()

        # TF for camera → robot transform
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Can dimensions updated by detector
        self.can_height   = CAN_HEIGHT
        self.can_diameter = CAN_DIAMETER

        # Busy flag — ignore detections while moving
        self.busy = False

        # Subscribers
        rospy.Subscriber('/detected_can/diameter', Float32, self.diameter_cb)
        rospy.Subscriber('/detected_can/height',   Float32, self.height_cb)
        rospy.Subscriber('/detected_object/point', PointStamped, self.point_cb)

        # Publisher — tells detector we're done so it can re-arm
        self.done_pub = rospy.Publisher('/franka_picker/done', Bool, queue_size=1)

        rospy.loginfo("Franka Picker ready — waiting for detections...")

    def diameter_cb(self, msg):
        self.can_diameter = msg.data

    def height_cb(self, msg):
        self.can_height = msg.data

    def point_cb(self, msg):
        """Called when detector publishes a stable can position."""
        if self.busy:
            return
        self.busy = True
        rospy.loginfo("=" * 50)
        rospy.loginfo("Can detected — starting pick & place")
        rospy.loginfo("=" * 50)

        try:
            # ── Transform camera coords → robot base frame ─────────────
            rospy.loginfo("Looking up TF transform...")
            transform = self.tf_buffer.lookup_transform(
                'panda_link0',
                msg.header.frame_id,
                rospy.Time(0),
                rospy.Duration(5.0)
            )
            target = tf2_geometry_msgs.do_transform_point(msg, transform)
            cx = target.point.x
            cy = target.point.y
            cz = target.point.z   # surface of can in robot frame

            rospy.loginfo(f"Can in robot frame: X={cx:.3f} Y={cy:.3f} Z={cz:.3f}")
            rospy.loginfo(f"Diameter={self.can_diameter*100:.1f}cm  "
                          f"Height={self.can_height*100:.1f}cm")

            # ── Grasp at middle of can ──────────────────────────────────
            grasp_z = cz + self.can_height / 2
            hover_z = grasp_z + HOVER_ABOVE
            lift_z  = grasp_z + LIFT_ABOVE

            # ── Step 1: Home ────────────────────────────────────────────
            rospy.loginfo("Step 1: Going home...")
            self.fa.reset_joints()
            time.sleep(0.5)

            # ── Step 2: Open gripper ────────────────────────────────────
            rospy.loginfo("Step 2: Opening gripper...")
            self.gripper.open(width=0.08)

            # ── Step 3: Hover above can ─────────────────────────────────
            rospy.loginfo(f"Step 3: Hovering above can ({cx:.3f}, {cy:.3f}, {hover_z:.3f})...")
            goto(self.fa, make_pose(cx, cy, hover_z))
            time.sleep(0.5)

            # ── Step 4: Lower to grasp height ───────────────────────────
            rospy.loginfo(f"Step 4: Lowering to grasp Z={grasp_z:.3f}m...")
            goto(self.fa, make_pose(cx, cy, grasp_z))
            time.sleep(0.5)

            # ── Step 5: Grasp ───────────────────────────────────────────
            rospy.loginfo("Step 5: Grasping can...")
            self.gripper.close()
            time.sleep(0.5)

            # ── Step 6: Lift ────────────────────────────────────────────
            rospy.loginfo(f"Step 6: Lifting to Z={lift_z:.3f}m...")
            goto(self.fa, make_pose(cx, cy, lift_z))
            time.sleep(0.5)

            # ── Step 7a: Rotate to sideways midpoint ────────────────────
            rospy.loginfo("Step 7a: Rotating to sideways midpoint (90°)...")
            goto_joints(self.fa, MID_JOINTS)
            time.sleep(0.5)

            # ── Step 7b: Rotate to behind robot ─────────────────────────
            rospy.loginfo("Step 7b: Rotating to behind robot (165°)...")
            goto_joints(self.fa, BEHIND_JOINTS)
            time.sleep(0.5)

            pose_after = self.fa.get_pose()
            rospy.loginfo(f"Pose after rotation: "
                          f"X={pose_after.translation[0]:.3f} "
                          f"Y={pose_after.translation[1]:.3f} "
                          f"Z={pose_after.translation[2]:.3f}")

            # ── Step 8: Lower can to table ──────────────────────────────
            rospy.loginfo(f"Step 8: Lowering can to table Z={PLACE_Z:.3f}m...")
            current = self.fa.get_pose()
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
            goto(self.fa, place_pose)
            time.sleep(0.5)

            # ── Step 9: Release ─────────────────────────────────────────
            rospy.loginfo("Step 9: Releasing can...")
            self.gripper.open(width=0.08, speed=0.02)
            time.sleep(0.5)

            # ── Step 10: Lift away ──────────────────────────────────────
            rospy.loginfo("Step 10: Lifting away...")
            current = self.fa.get_pose()
            retreat = RigidTransform(
                rotation=current.rotation,
                translation=np.array([
                    current.translation[0],
                    current.translation[1],
                    lift_z
                ]),
                from_frame='franka_tool',
                to_frame='world'
            )
            goto(self.fa, retreat)
            time.sleep(0.5)

            # ── Step 11: Home ───────────────────────────────────────────
            rospy.loginfo("Step 11: Returning home...")
            self.fa.reset_joints()

            rospy.loginfo("=" * 50)
            rospy.loginfo("COMPLETE! Can placed behind robot.")
            rospy.loginfo("=" * 50)

        except Exception as e:
            rospy.logerr(f"Pick & place failed: {e}")
            import traceback; traceback.print_exc()
            try:
                self.fa.reset_joints()
            except:
                pass
        finally:
            self.busy = False
            self.done_pub.publish(Bool(data=True))  # re-arm detector


if __name__ == '__main__':
    try:
        FrankaPicker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
