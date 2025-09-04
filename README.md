# Data Collection Pipeline

### Task
- collects rgb, depth, instance segmentation, 3D bbox from a custom scene using isaac sim
- publish them to `ros2` topics
- read and save them to disk 


### Important
- [ ] Make sure objects in custom scene are annotated, otherwise bbox3d will not work

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

### About Camera inside isaac sim
- To camera you should give position, orientation.
- Every new position comes with original orientation

### Future to do
- [ ] Use the custom scene; not from isaac sim assets
- [ ] how to chceck visually 3d bbox are correct? How to visualize them?
- [ ] why some ther are multiple bboxes for the same object?
- [ ] warnings about depricated functions