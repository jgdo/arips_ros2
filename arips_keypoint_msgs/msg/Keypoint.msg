# One detected keypoint in image coordinates of the original (RGB, or depth for depth-only models) image.
# Pixel coordinates use the corner convention: (0, 0) is the top-left corner of the first pixel.
int32 index
string name
float32 x
float32 y
# detector confidence (heatmap peak value, 0..1)
float32 confidence
# confidence >= threshold
bool visible
# depth (Z along the optical axis) at (x, y) in metres: median of the valid depth pixels in a
# (2 * depth_radius + 1)^2 window (has_depth = false if no valid depth there)
bool has_depth
float32 depth
# 3D point in the camera optical frame of KeypointArray.header (x right, y down, z forward), in metres,
# from depth and the camera_info projection matrix (has_position = false without depth or camera_info)
bool has_position
geometry_msgs/Point position
