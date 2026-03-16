# Franka FR3 Can Pick & Place

Autonomous pick-and-place system using a Franka FR3 robot arm, Intel RealSense D435 depth camera, and YOLOv8 object detection. The robot detects a metal can on the table, picks it up, and places it behind itself.

---

## System Overview

```
RealSense D435
     │  /camera/color/image_raw
     │  /camera/aligned_depth_to_color/image_raw
     ▼
object_detector.py  ──►  /detected_object/point  ──►  franka_picker.py
  (YOLO + Hough)                                        (FrankaArm)
                     ◄──  /franka_picker/done    ◄──
```

The detector publishes a 3D point when it has seen a stable can for several frames. The picker receives the point, transforms it to robot coordinates, executes the pick-and-place sequence, then tells the detector to re-arm for the next can.

---

## Hardware

- **Robot:** Franka FR3, IP `147.250.35.11`
- **Camera:** Intel RealSense D435, mounted on a fixed tripod above the table
- **Host machine:** Ubuntu 22.04, ROS2 Humble (host), ROS1 Noetic (inside Docker)

---

## Software Stack

| Component | Where it runs |
|---|---|
| `franka-interface` C++ backend | Docker |
| `franka_ros_interface` ROS nodes | Docker |
| `franka_gripper` ROS nodes | Docker |
| `realsense2_camera` | Docker |
| `object_detector.py` | Docker |
| `franka_picker.py` | Docker |
| `rqt_image_view` (optional) | Docker |

Everything runs inside the same Docker container. The host machine only provides the display for GUI tools.

---

## First-Time Setup

### 1. Allow display access (run once per host session)
```bash
xhost +local:root
xhost +local:docker
```

### 2. Start the Docker container
```bash
docker run -it --rm \
  --gpus all \
  --name franka_frida-18-11-v2 \
  --net=host \
  --ipc=host \
  --privileged \
  -v /dev/bus/usb:/dev/bus/usb \
  --ulimit rtprio=99 \
  --ulimit memlock=-1 \
  --cap-add=sys_nice \
  --security-opt seccomp=unconfined \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XDG_RUNTIME_DIR=/tmp \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ~/franka_ws:/ws \
  -w /ws \
  u2is-franka-panda-frida-21-11 \
  bash
```

### 3. Clone the repo and install dependencies (first time only, inside Docker)
```bash
git clone https://github.com/Anna-Sobchuk/franka.git ~/franka
pip install ultralytics --ignore-installed psutil
sudo apt-get install -y ros-noetic-realsense2-camera ros-noetic-realsense2-description
```

---

## Running the System

Open **7 terminals**, each connected to the running container:
```bash
# From any new terminal on host:
docker exec -it franka_frida-18-11-v2 bash
```

Run these in order and wait for each to finish starting before the next:

### Terminal 1 — Franka Interface (C++ backend)
```bash
cd ~/franka-interface/build
./franka_interface --robot_ip 147.250.35.11
```

### Terminal 2 — Franka ROS Interface
```bash
cd ~/franka-interface/catkin_ws
source devel/setup.bash
roslaunch franka_ros_interface franka_ros_interface.launch robot_ip:=147.250.35.11
```

### Terminal 3 — Gripper
```bash
source ~/franka-interface/catkin_ws/devel/setup.bash
roslaunch franka_gripper franka_gripper.launch robot_ip:=147.250.35.11 __ns:=/franka_gripper_1
```

### Terminal 4 — RealSense Camera
```bash
source /opt/ros/noetic/setup.bash
roslaunch realsense2_camera rs_aligned_depth.launch \
  enable_infra1:=false enable_infra2:=false \
  color_fps:=15 depth_fps:=15
```

### Terminal 5 — Object Detector
```bash
python3 ~/franka/scripts/object_detector.py
```

### Terminal 6 — Franka Picker
```bash
python3 ~/franka/scripts/franka_picker.py
```

### Terminal 7 — Visual Debug (optional)
```bash
rqt_image_view
# Select /object_detector/yolo_debug from the dropdown
```

Once all terminals are running, place a can on the table in front of the robot and the system will pick it up automatically.

---

## How It Works

### Object Detection (`object_detector.py`)

The detector subscribes to the RealSense color and depth streams. For each frame it runs two detection methods in parallel:

**YOLOv8** — a neural network trained on COCO that detects objects at low confidence (0.10). All detected boxes are passed through a size filter that keeps only objects with an estimated diameter between 5 and 12 cm, matching a standard food can.

**HoughCircles** — a classical computer vision method that finds circular shapes by edge detection. It uses CLAHE contrast enhancement first to improve detection of metallic surfaces. This acts as a fallback when YOLO misses the can.

Both methods feed a shared candidate list. Duplicate detections are removed with non-maximum suppression. Depth is read using a large 12×12 pixel window with the 20th percentile value — this handles the reflective metal surface of the can, which causes many invalid depth pixels.

A **stability buffer** collects detections over time. The can must be visible and stationary for 8 consecutive frames before the position is published. This prevents false triggers from moving hands or transient detections.

Once stable, a `PointStamped` message is published to `/detected_object/point` in the camera optical frame, and the detector locks until it receives `True` on `/franka_picker/done`.

### Coordinate Mapping (`franka_picker.py`)

Instead of a TF transform tree, the picker uses a direct linear regression mapping learned from 7 measured calibration points. The camera coordinates `(cx, cy, cz)` are mapped to robot base frame coordinates `(rx, ry)` by:

```
rx = 0.1396·cx + 1.6647·cy + 0.3942·cz + 0.0982
ry = 1.0213·cx − 0.0325·cy − 0.1186·cz + 0.0141
```

The Z coordinate (grasp height) is fixed based on the known can height (11 cm) rather than derived from the camera, since the table surface is always at the same height.

Positions outside the reachable workspace (`X < 0.25` or `X > 0.80`, `|Y| > 0.40`) are rejected before any motion begins.

### Pick and Place Sequence (`franka_picker.py`)

1. **Home** — reset to home joint configuration
2. **Open gripper** — open to 8 cm
3. **Hover** — move above the can at Z = 0.250 m. For positions with X > 0.60, the gripper tilts 10° forward to avoid wrist singularity
4. **Lower (two steps)** — descend to Z = 0.175 m, then to grasp height Z = 0.100 m. After reaching the target, the actual Z is verified — if the robot did not lower correctly it aborts and goes home
5. **Grasp** — close gripper with 5 N force. Gripper width is checked — if below 2 cm the can was missed and the robot aborts
6. **Lift** — raise to Z = 0.400 m
7. **Rotate to 0°** — move all joints to a safe centered configuration (joint 0 = 0°). Gripper width is checked again to detect any drop during rotation
8. **Rotate to 165°** — move directly to the behind-robot configuration in one joint-space move
9. **Lower to table** — descend to placement height behind the robot
10. **Release** — open gripper slowly
11. **Lift away** — raise clear of the placed can
12. **Home** — return to home position and re-arm the detector

### Gripper Control

FrankaPy's built-in gripper methods are not used because they do not work with the non-standard gripper namespace (`/franka_gripper_1/franka_gripper/`). Instead, the gripper is controlled directly via ROS `actionlib` with `MoveGoal` (open) and `GraspGoal` (close). The grasp parameters are: width = 7 cm, speed = 1 cm/s, force = 5 N, epsilon = ±5 cm. The large epsilon means the gripper accepts any final width, stopping at first contact with the can.

---

## Recalibration

If the camera or tripod is moved, the coordinate mapping needs to be recalibrated. Place the can at several known positions, record the camera output (`rostopic echo /detected_object/point`) and the robot EE position when touching the can (`fa.run_guide_mode(duration=10)`), then fit new regression coefficients and update `CAM2ROBOT_X` and `CAM2ROBOT_Y` in `franka_picker.py`.

---

## ROS Topics

| Topic | Type | Description |
|---|---|---|
| `/detected_object/point` | `PointStamped` | Can position in camera frame |
| `/detected_can/diameter` | `Float32` | Estimated can diameter (metres) |
| `/detected_can/height` | `Float32` | Can height (metres, fixed 0.11) |
| `/franka_picker/done` | `Bool` | Picker signals detector to re-arm |
| `/object_detector/yolo_debug` | `Image` | Annotated camera feed with detections |
| `/camera/color/image_raw` | `Image` | Raw color stream |
| `/camera/aligned_depth_to_color/image_raw` | `Image` | Depth aligned to color |

---

## Troubleshooting

**Robot not reachable (`ping 147.250.35.11` fails)** — check the control box is powered on and the ethernet cable is connected. Open `http://147.250.35.11` in a browser to check the Franka Desk interface.

**`roslaunch` command not found** — you are on the host machine, not inside Docker. Run `docker exec -it franka_frida-18-11-v2 bash` first.

**Picker hangs at `wait_for_service`** — franka_ros_interface (Terminal 2) is not running or still starting. Wait for it to finish or restart it.

**Detector not publishing** — the `robot_busy` flag may be stuck. Re-arm with: `rostopic pub /franka_picker/done std_msgs/Bool "data: true" -1`

**Camera fails to start** — unplug and replug the USB cable, then relaunch. Make sure the camera is on a USB 3.0 (blue) port.

**After computer restart** — the `--rm` flag in the docker run command deletes the container on shutdown. You need to rerun the full `docker run` command and reinstall ultralytics inside Docker.
