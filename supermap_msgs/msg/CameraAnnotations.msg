# Per-image instance masks for labelling a dense cloud.
# header.stamp is the image stamp; header.frame_id is the camera optical frame.
std_msgs/Header header
# "yoloe" or "manual".
string source
# Masks must be on a rectified image (D == 0, R == I) matching camera_info.
bool rectified
sensor_msgs/CameraInfo camera_info
# Optional explicit pose. When has_world_from_camera is false, consumers look it up in TF.
bool has_world_from_camera
string world_frame
geometry_msgs/Transform world_from_camera
string model
MaskDetection[] detections
