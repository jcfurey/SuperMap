"""Range of camera masks from a spinning LiDAR mounted apart from the camera.

The rig drives toward a bulk bag standing on the ground in front of a wall.
Its LiDAR sits 0.4 m above and 0.1 m behind the camera, so ground returns
form rings at fixed ranges from the sensor, and the LiDAR sees parts of the
wall the camera cannot. A segmentation mask bled a few pixels onto the
ground holds a ring in front of the bag; taken as the bag's depth, that ring
put the box (and the dense labels) on the ground in front of the object and
slid them along with the vehicle.
"""
import numpy as np
import pytest

from semantic_mapping.dense_cloud import CameraLabels, DenseCloudConfig, DenseCloudPipeline, foreground_layer
from semantic_mapping.geometry_utils import (
    invert_se3, occlusion_grid_for, occlusion_visible, rasterize_depth, splat_depth_buffer, transform_points,
)
from semantic_mapping.pipeline import PipelineConfig, SemanticMappingPipeline
from semantic_mapping.types import CameraIntrinsics, Detection2D, Observation, StampedPose

W, H, F = 320, 240, 225.0
K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]])
INTRINSICS = CameraIntrinsics(fx=F, fy=F, cx=W / 2, cy=H / 2, width=W, height=H)
R_WORLD_FROM_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # looks along +x, z up
BAG = ([20.0, -0.5, 0.0], [21.0, 0.5, 1.0])
WALL_X = 45.0


def camera_pose(x=0.0):
    T = np.eye(4)
    T[:3, :3] = R_WORLD_FROM_CAM
    T[:3, 3] = [x, 0.0, 1.2]
    return T


def lidar_pose(x=0.0):
    T = np.eye(4)
    T[:3, 3] = [x - 0.1, 0.0, 1.6]
    return T


def _raycast(origin, dirs, boxes):
    """Nearest hit per ray: (t, id) with id -1 ground, -2 wall, i box i."""
    t_best = np.full(len(dirs), np.inf)
    ident = np.full(len(dirs), -3)
    with np.errstate(divide="ignore", invalid="ignore"):
        for t, i in ((np.where(dirs[:, 2] < 0, -origin[2] / dirs[:, 2], np.inf), -1),
                     (np.where(dirs[:, 0] > 0, (WALL_X - origin[0]) / dirs[:, 0], np.inf), -2)):
            better = (t > 0) & (t < t_best)
            t_best[better], ident[better] = t[better], i
        for i, (lo, hi) in enumerate(boxes):
            t1, t2 = (np.asarray(lo) - origin) / dirs, (np.asarray(hi) - origin) / dirs
            near, far = np.nanmax(np.minimum(t1, t2), axis=1), np.nanmin(np.maximum(t1, t2), axis=1)
            better = (near <= far) & (near > 0) & (near < t_best)
            t_best[better], ident[better] = near[better], i
    return t_best, ident


def lidar_scan(x=0.0, boxes=(BAG,), beams=64, columns=1024):
    """World-frame returns of a 64-beam, 45 degree spinning LiDAR, and what each hit."""
    elevation = np.deg2rad(np.linspace(-22.5, 22.5, beams))
    azimuth = np.deg2rad(np.linspace(-40.0, 40.0, columns * 80 // 360))  # the part facing the camera's view
    el, az = np.meshgrid(elevation, azimuth, indexing="ij")
    dirs = np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1).reshape(-1, 3)
    origin = lidar_pose(x)[:3, 3]
    t, ident = _raycast(origin, dirs, boxes)
    keep = np.isfinite(t)
    return origin + t[keep, None] * dirs[keep], ident[keep]


def silhouette(x=0.0, boxes=(BAG,)):
    """Per-pixel object id seen by the camera."""
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack(((us - K[0, 2]) / F, (vs - K[1, 2]) / F, np.ones_like(us, float)), -1).reshape(-1, 3)
    return _raycast(camera_pose(x)[:3, 3], rays @ R_WORLD_FROM_CAM.T, boxes)[1].reshape(H, W)


def bag_detection(x=0.0, bleed_down=3):
    """The bag's mask, bled ``bleed_down`` pixels onto the ground like a loose segmentation."""
    mask = silhouette(x) == 0
    grown = mask.copy()
    for k in range(1, bleed_down + 1):
        grown[k:] |= mask[:-k]
    ys, xs = np.nonzero(grown)
    return Detection2D(bbox=np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float),
                       label="bulk bag", score=0.9, mask=grown)


def depth_image(x=0.0, splat=True):
    world, _ = lidar_scan(x)
    points_cam = transform_points(invert_se3(camera_pose(x)), world)
    if not splat:
        return rasterize_depth(points_cam, K, W, H)
    return rasterize_depth(points_cam, K, W, H, splat_radius_m=0.05, occlusion_grid_px=occlusion_grid_for(W, H))


def lift(config, x=0.0, splat=True, bleed_down=3):
    pipeline = SemanticMappingPipeline(config)
    return pipeline._detection_points_world(bag_detection(x, bleed_down), depth_image(x, splat), K, camera_pose(x))


# --------------------------------------------------------------- occlusion
def test_one_pixel_zbuffer_shows_the_wall_through_a_barrier_and_splatting_hides_it():
    barrier = ([10.0, -1.0, 0.0], [10.6, 1.0, 1.0])
    world, ident = lidar_scan(boxes=(barrier,))
    points_cam = transform_points(invert_se3(camera_pose()), world)
    inside = silhouette(boxes=(barrier,)) == 0
    one_pixel = rasterize_depth(points_cam, K, W, H)
    splatted = rasterize_depth(points_cam, K, W, H, splat_radius_m=0.05, occlusion_grid_px=occlusion_grid_for(W, H))
    # The LiDAR, above the camera, reaches wall the barrier hides from the camera.
    assert np.count_nonzero(one_pixel[inside] > 12.0) > 20
    assert np.count_nonzero(splatted[inside] > 12.0) == 0
    barrier_readings = (one_pixel[inside] > 0) & (one_pixel[inside] < 11.0)
    assert np.count_nonzero((splatted[inside] > 0) & (splatted[inside] < 11.0)) == np.count_nonzero(barrier_readings)
    assert (ident == -2).any()


def test_occlusion_grid_only_hides_more_of_the_grazing_ground():
    world, ident = lidar_scan()
    cam = transform_points(invert_se3(camera_pose()), world)
    uv = cam @ K.T
    us = np.round(uv[:, 0] / uv[:, 2]).astype(np.int64)
    vs = np.round(uv[:, 1] / uv[:, 2]).astype(np.int64)
    ok = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    us, vs, z, ident = us[ok], vs[ok], cam[ok, 2], ident[ok]
    exact = occlusion_visible(us, vs, z, F, W, H, radius_m=0.05)
    coarse = occlusion_visible(us, vs, z, F, W, H, radius_m=0.05, grid_px=2)
    np.testing.assert_array_equal(exact[ident == 0], coarse[ident == 0])  # the bag
    assert np.mean(exact[ident == -2] == coarse[ident == -2]) > 0.99  # the wall
    assert np.count_nonzero(coarse & ~exact) < 0.01 * len(z)
    # The exact test is the dense path's footprint z-buffer.
    buffer = splat_depth_buffer(us, vs, z, F, W, H, 0.05, 8)
    np.testing.assert_array_equal(exact, z <= buffer[vs * W + us] + 0.3)
    assert occlusion_visible(us, vs, z, F, W, H, radius_m=0.0).all()
    assert [occlusion_grid_for(*size) for size in ((640, 480), (1280, 960), (2448, 2048))] == [1, 2, 4]


def _cloud_on_pixels(us, vs, z):
    z = np.broadcast_to(np.asarray(z, float), np.shape(us))
    return np.stack(((us - K[0, 2]) * z / F, (vs - K[1, 2]) * z / F, z), -1).reshape(-1, 3)


def test_dense_clouds_keep_the_background_beside_a_silhouette():
    # A depth camera's cloud: one point per pixel, a board at 2 m in front of a wall at 6 m.
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    board = (abs(us - W / 2) < 40) & (abs(vs - H / 2) < 30)
    points = _cloud_on_pixels(us, vs, np.where(board, 2.0, 6.0))
    one_pixel = rasterize_depth(points, K, W, H)
    np.testing.assert_array_equal(rasterize_depth(points, K, W, H, splat_radius_m=0.05), one_pixel)


def test_sparse_samples_hide_the_background_between_them():
    # Board samples on even pixels, wall samples on the odd pixels between them.
    us, vs = np.meshgrid(np.arange(0, W, 2), np.arange(0, H, 2))
    board = (abs(us - W / 2) < 40) & (abs(vs - H / 2) < 30)
    points = np.vstack((_cloud_on_pixels(us[board], vs[board], 2.0), _cloud_on_pixels(us + 1, vs + 1, 6.0)))
    inside = np.zeros((H, W), bool)
    inside[H // 2 - 28:H // 2 + 28, W // 2 - 38:W // 2 + 38] = True
    assert (rasterize_depth(points, K, W, H)[inside] > 3.0).any()
    splatted = rasterize_depth(points, K, W, H, splat_radius_m=0.05)
    assert not (splatted[inside] > 3.0).any()
    assert (splatted[vs[board], us[board]] == 2.0).all()  # every board sample stays
    assert (splatted[~inside] > 3.0).any()  # the wall around the board stays


def test_rasterize_depth_is_unchanged_without_a_splat_radius():
    cam = transform_points(invert_se3(camera_pose()), lidar_scan()[0])
    np.testing.assert_array_equal(rasterize_depth(cam, K, W, H), rasterize_depth(cam, K, W, H, splat_radius_m=0.0))


# ------------------------------------------------------- tracker mask range
def _nearest_layer(**kwargs):
    return PipelineConfig(foreground_depth_gap_m=0.75, foreground_depth_labels=["bulk bag"], **kwargs)


@pytest.mark.parametrize("x", [0.0, 0.5, 1.0])
def test_bled_mask_used_to_place_the_bag_on_a_ground_ring_in_front(x):
    # Documents the failure: nearest layer = the ring, whose range tracks the sensor.
    points = lift(_nearest_layer(ground_exclusion=False), x=x, splat=False)
    assert len(points) and points[:, 0].max() < 19.5 and np.ptp(points[:, 2]) < 0.1


@pytest.mark.parametrize("config", [PipelineConfig(), _nearest_layer()], ids=["all returns", "nearest layer"])
@pytest.mark.parametrize("x", [0.0, 0.5, 1.0])
def test_ground_exclusion_keeps_the_mask_on_the_object(config, x):
    points = lift(config, x=x)
    assert len(points)
    assert points[:, 0].min() > 19.9 and points[:, 0].max() < 21.1
    assert points[:, 2].min() > 0.1 and points[:, 2].max() > 0.6


def test_tracked_box_stays_put_while_driving():
    boxes = {}
    for name, config in (("before", _nearest_layer(ground_exclusion=False)), ("after", _nearest_layer())):
        pipeline = SemanticMappingPipeline(config)
        boxes[name] = []
        for i, x in enumerate(np.arange(0.0, 1.61, 0.4)):
            observation = Observation(stamp=i * 0.1, pose=StampedPose(i * 0.1, camera_pose(x)),
                                      intrinsics=INTRINSICS, depth=depth_image(x, splat=name == "after"),
                                      detections=[bag_detection(x)])
            result = pipeline.process_frame(observation)
            assert result.detection_instance_ids == [1]
            boxes[name].append(pipeline.object_map.objects[1].bbox3d.copy())
    before, after = np.array(boxes["before"]), np.array(boxes["after"])
    assert before[:, 3].max() < 19.6 and np.ptp(before[:, 3]) > 0.5  # on the rings, following the vehicle
    assert after[:, 0].min() > 19.8 and after[:, 3].max() < 21.2  # on the bag, every frame


def test_flat_and_ground_classes_keep_their_ground_returns():
    depth = depth_image()
    mask = np.zeros((H, W), bool)
    mask[int(H / 2 + F * 1.2 / 14.0):int(H / 2 + F * 1.2 / 10.0), 140:180] = True  # ground 10-14 m ahead
    ys, xs = np.nonzero(mask)
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
    pipeline = SemanticMappingPipeline(PipelineConfig())
    reference = SemanticMappingPipeline(PipelineConfig(ground_exclusion=False))
    for label in ("rug", "manhole cover"):  # a ground class, and a class lying flat with nothing above ground
        detection = Detection2D(bbox=box, label=label, score=0.9, mask=mask)
        points = pipeline._detection_points_world(detection, depth, K, camera_pose())
        assert len(points) > 5
        np.testing.assert_array_equal(points, reference._detection_points_world(detection, depth, K, camera_pose()))


def test_a_world_frame_that_is_not_z_up_disables_ground_exclusion():
    # The camera's optical frame as "world": z is depth, and the lowest-z
    # returns would otherwise be taken for the ground and removed.
    detection, depth = bag_detection(), depth_image()
    on = SemanticMappingPipeline(PipelineConfig())._detection_points_world(detection, depth, K, np.eye(4))
    off = SemanticMappingPipeline(PipelineConfig(ground_exclusion=False))._detection_points_world(
        detection, depth, K, np.eye(4))
    np.testing.assert_array_equal(on, off)


def test_ground_exclusion_settings_are_validated():
    for bad in ({"ground_exclusion_min_returns": 0}, {"ground_surface_labels": "floor"}):
        with pytest.raises(ValueError):
            PipelineConfig(**bad)
    for bad in ({"ground_context_px": -1}, {"ground_surface_labels": "floor"}, {"ground_clearance": 0.0},
                {"ground_exclusion": 1}):
        with pytest.raises(ValueError):
            DenseCloudConfig(**bad)


# --------------------------------------------------------- dense labeller
def _dense_labels(ground_exclusion=True, frames=(0.0, 0.5, 1.0), bleed_down=3, label="bulk bag"):
    pipeline = DenseCloudPipeline(DenseCloudConfig(input_mode="scan", ground_exclusion=ground_exclusion))
    for i, x in enumerate(frames):
        world, _ = lidar_scan(x)
        pipeline.update(world, float(i))
        detection = bag_detection(x, bleed_down)
        detection.label = label
        pipeline.annotate(CameraLabels(float(i), INTRINSICS, camera_pose(x), [detection]))
    return pipeline.points[pipeline.semantic > 0]


def test_dense_labels_no_longer_smear_along_the_ground_in_front_of_the_object():
    before = _dense_labels(ground_exclusion=False)
    ground = before[before[:, 0] < 19.9]
    assert len(ground) and np.ptp(ground[:, 0]) > 0.4  # a ring per frame, following the vehicle
    after = _dense_labels()
    assert len(after) > 20
    assert after[:, 0].min() > 19.9 and after[:, 2].min() > 0.1


def test_dense_ground_class_keeps_its_ground_labels():
    labelled = _dense_labels(label="rug", frames=(0.0,))
    assert (labelled[:, 2] < 0.1).any()


def test_dense_labels_use_pixel_centres_at_integer_coordinates():
    pipeline = DenseCloudPipeline(DenseCloudConfig(min_region_voxels=1))
    # u = 10 * 0.33 / 2 + 5 = 6.65: pixel 7 under the pinhole convention (floor gave 6).
    pipeline.update(np.array([[0.33, 0.0, 2.0]]), 1.0)
    intr = CameraIntrinsics(10.0, 10.0, 5.0, 5.0, 10, 10)
    for column, expected in ((6, 0), (7, 1)):
        mask = np.zeros((10, 10), bool)
        mask[5, column] = True
        detection = Detection2D(np.array([0.0, 0.0, 10.0, 10.0]), "chair", 1.0, mask=mask)
        fresh = DenseCloudPipeline(DenseCloudConfig(min_region_voxels=1))
        fresh.update(np.array([[0.33, 0.0, 2.0]]), 1.0)
        assert fresh.annotate(CameraLabels(1.0, intr, np.eye(4), [detection])).semantic_ids.tolist() == [expected]


def test_dense_foreground_layer_is_unchanged():
    depths = np.array([0.5, 3.0, 3.1, 3.2, 3.05, 18.0, 18.2])
    assert foreground_layer(depths, DenseCloudConfig()).tolist() == [False, True, True, True, True, False, False]
