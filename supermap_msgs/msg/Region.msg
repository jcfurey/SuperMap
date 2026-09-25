# A geometric surface region of the dense map. Class distributions may be mixed.
uint32 id
uint32 voxel_count
# Axis-aligned bounds in the map frame (identity orientation).
vision_msgs/BoundingBox3D bbox
# Parallel arrays: voxels per semantic label inside this region.
string[] labels
uint32[] label_voxel_counts
