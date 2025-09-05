# Data Collection Pipeline

## Script Overview: `isaac_data_recorder.py`
Captures a synchronized dataset (RGB, depth, semantic segmentation, camera trajectory) from a scripted camera moving along interpolated keyframes inside an Isaac Sim warehouse scene.

### What it does
- Loads predefined warehouse USD environment.
- Spawns a camera (`/World/Camera`) and applies a keyframe path (position + Euler orientation).
- Interpolates translation + orientation per simulation step.
- Renders frames after multiple internal render substeps for quality.
- Queries Replicator annotators for RGB (`LdrColor`), depth (`distance_to_image_plane`), and semantic segmentation.
- Saves:
  - RGB images (JPEG)
  - 16-bit depth (PNG, normalized 0.01–10 m)
  - Colored semantic mask + JSON label/color mapping
  - Camera extrinsic matrix per frame (ROS frame) in `traj.txt`
- Computes and prints camera intrinsics (fx, fy, cx, cy) and depth scale each run.

### How it works (flow)
1. Initialize SimulationApp (headless optional) + enable ROS2 bridge.
2. Load environment via USD reference; add ground plane.
3. Initialize camera, set resolution (1280x720), apertures, pose.
4. Compute intrinsics from USD camera attributes.
5. Warm-up frames for stable rendering.
6. For each time index `i` until keyframes exhausted:
   - Interpolate pose -> set camera.
   - Advance world multiple render substeps.
   - Fetch annotator buffers.
   - Post-process + write outputs.
7. Append homogeneous transform (flattened 4x4) to trajectory file.
8. Close simulation cleanly.

### Key Customization Points
- Edit keyframes list for new paths.
- Adjust MIN_DEPTH / MAX_DEPTH for different sensor ranges.
- Swap annotator (`distance_to_camera`) if preferred.
- Add more modalities (normals, instance segmentation) via Replicator.

---

### Task
- collects rgb, depth, instance segmentation, 3D bbox from a custom scene using isaac sim
- publish them to `ros2` topics
- read and save them to disk 


### Important
- [ ] Make Sure objects in custom scene are annotated, otherwise bbox3d will not work

### How to run
1. Start the docker container
```docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   nvcr.io/nvidia/isaac-lab:2.1.0
   ```
2. Run **isaac sim**
```
 ./isaaclab.sh -p ./isaac_sim/data_writer_omni.py
```
3. Run data collector
```
 ./isaaclab.sh -p ./isaac_sim/data_saver.py
```

4. To run just to get bbq images/frames
```
 ./isaaclab.sh -p ./isaac_sim/isaac_data_recorder.py
```

### Camera Configuration
- Prim path: `/World/Camera`
- Resolution: 1280 (W) x 720 (H)
- Focal length: (USD camera attribute `focalLength` as loaded from stage)
- Horizontal aperture: 80.0 (mm units in USD)
- Vertical aperture: 80.0 * 720 / 1280 = 45.0
- Intrinsics (computed each run):
  - fx = focal_length / horizontalAperture * image_width
  - fy = focal_length / verticalAperture * image_height
  - cx = image_width / 2
  - cy = image_height / 2
- Printed each run: `fx, fy, cx, cy, png_depth_scale`
- png_depth_scale = (MAX_DEPTH - MIN_DEPTH) / 65535 with MIN_DEPTH=0.01 m, MAX_DEPTH=10.0 m
- Pose set per keyframe path (interpolated translation + Euler → quaternion)

### Coordinate Frames
- Camera pose saved twice internally: local/world and ROS-style (`camera_axes="ros"`).
- Trajectory file uses ROS camera pose transformed into a 4x4 matrix (row-major flatten).

### Generated Data (per frame index N -> zero padded 6 digits)
- RGB: `frameNNNNNN.jpg` (BGR saved via OpenCV, 8-bit)
- Depth: `depthNNNNNN.png` (16-bit unsigned, metric depth normalized into [0, 65535])
  - To convert pixel value p back to meters: depth_m = p / 65535 * (MAX_DEPTH - MIN_DEPTH) + MIN_DEPTH
- Semantic segmentation (colored): `semanticNNNNNN.png`
  - Color map is deterministic, background id 0 = black.
- Semantic label info: `semanticNNNNNN_info.json`
  - Structure: `{ seg_id: { "label": <string>, "color_bgr": [B,G,R] } }`
- Camera trajectory: `traj.txt`
  - Each line = 16 values (row-major 4x4 transform matrix of camera in ROS frame)

### Keyframe Motion
- Keyframes define `time`, `translation`, `euler_angles` (degrees)
- Linear interpolation of translation and Euler angles (then converted to quaternion)
- Stops when interpolation runs out (after last keyframe interval)

### Depth Details
- Raw depth source: `distance_to_image_plane` annotator
- Clamped to [0.01, 10.0] meters before scaling
- Stored as 16-bit PNG for compactness (avoid float32 size)

### Segmentation Details
- Annotator: `semantic_segmentation`
- Uses `idToLabels` mapping from replicator
- Custom color map sized by max(seg_id, number_of_labels)

### Trajectory File Usage Example
```
with open('traj.txt') as f:
    for line in f:
        T = np.fromstring(line.strip(), sep=' ').reshape(4,4)
```

### 3D Bounding Boxes
- 3D bbox collection requires properly annotated prims (semantic instance + bbox metadata in the USD)
- If multiple boxes appear for same object: check duplications (instancing) or LOD variants

### About Camera inside isaac sim
- Provide position + orientation each keyframe.
- Each new keyframe pose overwrites previous orientation (no cumulative yaw accumulation).

### Future to do
- [ ] Use the custom scene; not from isaac sim assets
- [ ] how to chceck visually 3d bbox are correct? How to visualize them?
- [ ] why some ther are multiple bboxes for the same object?
- [ ] warnings about depricated functions