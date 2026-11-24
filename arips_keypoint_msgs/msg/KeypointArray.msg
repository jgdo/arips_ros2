# All keypoints of one model for one image; always contains all keypoints of the model, see visible.
# header: stamp and frame of the image; the frame is the camera optical frame of Keypoint.position
std_msgs/Header header
string tag
Keypoint[] keypoints
