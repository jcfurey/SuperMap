# One open-vocabulary instance mask on a rectified pinhole image grid.
string label
float32 score
# Foreground spans in row-major order, flattened as [start0, length0, start1, length1, ...].
# These are plain spans over the flattened width*height grid, not COCO RLE.
uint32[] mask_runs
