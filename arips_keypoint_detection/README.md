# arips_keypoint_detection

Generic keypoint detection for RGB-D images: dataset extraction from rosbags, an annotation tool,
small heatmap models (< 1M parameters) for RGB, RGB-D or depth-only input, and a ROS 2 detector node.

## Workflow

```bash
# 1. extract every 10th RGB frame (+ closest depth) from all bags below a folder -> <bags>/dataset
ros2 run arips_keypoint_detection extract_rgbd_dataset --bags BAGS_DIR --skip 10
#    bags without the depth topic are extracted rgb-only; --no-depth extracts only rgb for all bags

# 2. annotate (left click = keypoint 0, right click = keypoint 1, arrows / ctrl+arrows to navigate)
ros2 run arips_keypoint_detection keypoint_annotator --dataset DATASET_DIR
#    a new annotation can start from a preset (config/presets/*.yaml: tag, description, keypoints)

# 3. create a random train/test split of the annotated images (or --all for one split with everything)
ros2 run arips_keypoint_detection create_split DATASET_DIR/config_door_handle.json foobar --test_ratio 0.1
#    -> split_door_handle_foobar_train.json, split_door_handle_foobar_test.json  ({"images": [...]})
#    add --by-bag to put whole recordings into train or test (test size as close to the ratio as possible)

# 4. train on the splits (--test-split is optional; several split files can be combined)
ros2 run arips_keypoint_detection train_keypoints --tag door_handle \
    --train-split DATASET_DIR/split_door_handle_foobar_train.json \
    --test-split DATASET_DIR/split_door_handle_foobar_test.json --model mobilenet --inputs rgb depth
#    -> keypoint_train/door_handle_foobar_mobilenet_rgbd/ with best.pt and model.onnx (exported automatically,
#       --no-export to skip); the splits used are stored as "split" in best.json
#    alternative without split files: --data DATASET_DIR (validation = last bag)
#    important parameters: see "Training parameters" below

# 5. compare models / input combinations on the same test images
ros2 run arips_keypoint_detection evaluate_keypoints keypoint_train/*/best.pt \
    --split-file DATASET_DIR/split_door_handle_foobar_test.json --preview /tmp/kp

# 6. run the node (training already exported model.onnx; re-export manually e.g. with another threshold:
#    ros2 run arips_keypoint_detection export_keypoints_onnx keypoint_train/.../best.pt --threshold 0.4)
ros2 run arips_keypoint_detection keypoint_detector_node --ros-args \
    -p model:=$PWD/keypoint_train/door_handle_mobilenet_rgbd/model.onnx
```

## Dataset format

Flat folder:
- `<bag>_<sec>_<nsec>_rgb.jpg`: RGB image.
- `<bag>_<sec>_<nsec>_depth.png`: depth, uint16 mm, 0 = invalid. Each file is named with its own stamp, and
  RGB and depth are associated by bag + nearest stamp.
- `<rgb file>_<tag>.json`: `{"0": [x, y], "1": [x, y]}` in RGB pixel coordinates. Keypoints that aren't
  visible are left out, and `{}` means none is visible.
- `config_<tag>.json`: `{"tag": ..., "keypoints": {"0": name, ...}}`.
- `split_<tag>[_<name>]_<train|test|all>.json`: `{"images": [<rgb file name>, ...]}`, created by `create_split`.

## Models (`--model`)

- `mobilenet` (default): ImageNet-pretrained MobileNetV3-Small encoder cut at stride 16 + light U-Net
  decoder, ~0.2M params. The pretrained encoder gets 0.3x learning rate.
- `tiny_unet`: inverted-residual U-Net from scratch, ~0.18M params.

Every enabled input (`rgb`: 3 ch normalised; `depth`: 2 ch = depth scaled from 0.2–5 m + valid mask) has
its own stride-2 stem. The stems are summed into a shared trunk, so all input combinations use the same
network.

Output: one heatmap per keypoint at 1/4 input resolution (default input 320x240).
- Training targets are Gaussians with σ = 2 heatmap px (≈ 16 px in a 640x480 image), which tolerates
  annotation jitter.
- Loss: quality focal loss.
- Decoding: argmax + log-parabola refinement. Confidence = peak value, visible = confidence ≥ threshold.

Training: strong augmentation (affine pan/rotation/zoom/shear; brightness/contrast/gamma/hue/saturation,
noise, blur, JPEG; depth noise, holes and dropout; modality dropout for RGB-D), AdamW + cosine schedule,
EMA weights, best checkpoint by a clipped validation pixel error.

## Training parameters (`train_keypoints`)

| Parameter | Default | Meaning |
|---|---|---|
| `--epochs` | 100 | Number of epochs. An epoch is a fixed number of augmented samples, not one pass over the data, so the training length doesn't depend on the dataset size. |
| `--samples-per-epoch` | 1600 | Augmented samples per epoch (at least the number of training images). Total steps = epochs × samples / batch. |
| `--batch` | 32 | Batch size. |
| `--lr` | 2e-3 | Peak learning rate (AdamW, linear warmup then cosine decay to 1 %). |
| `--warmup-epochs` | 3 | Linear warmup length. |
| `--encoder-lr-scale` | 0.3 | Learning rate factor for the pretrained encoder (`mobilenet`). |
| `--weight-decay` | 1e-4 / 5e-4 | Default for `mobilenet` / `tiny_unet`. |
| `--ema` | 0.998 | EMA decay of the weights. The EMA weights are validated and saved. |
| `--model` | `mobilenet` | `mobilenet` or `tiny_unet`, see above. |
| `--inputs` | `rgb` | Any of `rgb`, `depth`. |
| `--input-size W H` | 320 240 | Network input size. The heatmaps are 1/4 of it. |
| `--sigma` | 2.0 | Target Gaussian σ in heatmap pixels. |
| `--loss` | `qfl` | Heatmap loss. |
| `--threshold` | 0.3 | Visibility threshold on the heatmap peak (stored in the metadata, the node uses it). |
| `--hflip` | off | Horizontal flip augmentation. Only use it if the keypoints are symmetric or their order doesn't matter. |
| `--no-augment` | off | Disable augmentation. |
| `--no-pretrained` | off | Don't use ImageNet weights. |
| `--val-bags` / `--val-fraction` | 1 / – | Validation split when training with `--data` instead of split files. |
| `--overfit N` | – | Sanity check: train and validate on the first N images, without augmentation. |
| `--out` | auto | Output folder, default `keypoint_train/<split name or tag>_<model>_<inputs>`. |
| `--no-export` | off | Skip the automatic ONNX export of `best.pt` to `model.onnx` after training. |
| `--workers` / `--seed` | 8 / 0 | Data loader workers, random seed. |

Training needs torch, torchvision and onnx (workspace venv). The node only needs onnxruntime and falls back
to `cv2.dnn`.

## Node topics

- Subscribes to `rgb_topic` / `depth_topic` (raw `Image`, or `CompressedImage` when the topic ends with
  `/compressed` / `/compressedDepth`). Depth is also subscribed for RGB-only models unless `use_depth:=false`.
- Subscribes to `camera_info_topic` (default: `<image namespace>/camera_info`, e.g. `/kinect/rgb/camera_info`).
- Publishes:
  - `~/keypoints` (`arips_keypoint_msgs/KeypointArray`): per keypoint the depth (median of the valid pixels in
    a (2·`depth_radius`+1)² window, default 7x7) and the 3D position in the camera optical frame, computed
    from the depth and the camera_info projection matrix. The depth image must be registered to the RGB image.
  - `~/markers` (`visualization_msgs/MarkerArray`): a sphere per keypoint (`marker_size`, default 3 cm),
    colored per keypoint, with color saturation = confidence.
  - `~/image_annotated`.
