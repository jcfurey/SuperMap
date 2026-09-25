"""Platform (robot self) mask from a URDF: geometry, rendering and detection filtering."""
import struct

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from semantic_mapping.platform_mask import (
    PlatformModel, exclude_platform, load_mesh, resolve_mesh_path, simplify_mesh, urdf_origin,
)
from semantic_mapping.types import CameraIntrinsics, Detection2D

INTR = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0, width=100, height=80)


def urdf(*links: str) -> str:
    return "<robot name='r'>" + "".join(links) + "</robot>"


def link(name: str, body: str = "") -> str:
    return f"<link name='{name}'>{body}</link>"


def visual(geometry: str, xyz="0 0 0", rpy="0 0 0", kind="visual") -> str:
    return f"<{kind}><origin xyz='{xyz}' rpy='{rpy}'/><geometry>{geometry}</geometry></{kind}>"


def at(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def test_urdf_origin_is_fixed_axis_roll_pitch_yaw():
    element = type("E", (), {"get": lambda self, key, default: {"rpy": "0.3 -0.2 1.1", "xyz": "1 2 3"}[key]})()
    T = urdf_origin(element)
    np.testing.assert_allclose(T[:3, :3], Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])
    np.testing.assert_array_equal(urdf_origin(None), np.eye(4))


def test_box_in_front_of_camera_projects_to_its_rectangle():
    model = PlatformModel.from_urdf(urdf(link("body", visual("<box size='1 1 0.2'/>"))))
    mask = model.render({"body": at(z=5.0)}, INTR)
    # Near face at z = 4.9: x, y in [-0.5, 0.5] -> u in 50 +- 10.2, v in 40 +- 10.2.
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    assert (cols[0], cols[-1], rows[0], rows[-1]) == (40, 60, 30, 50)
    assert mask[30:51, 40:61].all() and mask.sum() == 21 * 21
    assert not mask.flags.writeable


def test_padding_dilates_and_links_are_placed_by_their_poses():
    model = PlatformModel.from_urdf(urdf(link("a", visual("<sphere radius='0.2'/>")),
                                         link("b", visual("<cylinder radius='0.1' length='0.4'/>"))), padding_px=3)
    mask = model.render({"a": at(x=-1.0, z=5.0), "b": at(x=1.0, z=5.0)}, INTR)
    assert mask[40, 30] and mask[40, 70] and not mask[40, 50]
    assert mask[40, 30 - 4 - 3] and not mask[40, 30 - 4 - 3 - 3]


def test_primitive_enclosing_the_camera_is_left_out():
    model = PlatformModel.from_urdf(urdf(link("chassis", visual("<box size='4 4 4'/>")),
                                         link("blade", visual("<box size='1 1 1'/>"))))
    mask = model.render({"chassis": at(z=0.5), "blade": at(z=5.0)}, INTR, camera_key="cam")
    assert mask.any() and mask.mean() < 0.2
    assert model.inside_warnings == {("cam", "chassis")}


def test_geometry_crossing_the_near_plane_is_clipped_not_dropped():
    # A long slab under the camera, from behind it to 10 m ahead: only its far part is in view.
    model = PlatformModel.from_urdf(urdf(link("deck", visual("<box size='2 0.1 12'/>"))), near_clip_m=0.15)
    mask = model.render({"deck": at(y=1.0, z=4.0)}, INTR)
    assert mask[79, 50] and not mask[:40].any()
    # A camera's own housing, all nearer than the clip distance, disappears entirely.
    housing = PlatformModel.from_urdf(urdf(link("lens", visual("<cylinder radius='0.02' length='0.06'/>"))))
    assert not housing.render({"lens": at(z=0.06)}, INTR).any()


def test_render_is_cached_until_a_link_moves():
    model = PlatformModel.from_urdf(urdf(link("hood", visual("<box size='1 1 1'/>")),
                                         link("blade", visual("<box size='1 1 1'/>"))))
    poses = {"hood": at(x=-1.5, z=5.0), "blade": at(x=1.5, z=5.0)}
    first = model.render(poses, INTR)
    assert model.render({k: v.copy() for k, v in poses.items()}, INTR) is first
    moved = model.render({**poses, "blade": at(x=1.5, y=1.0, z=5.0)}, INTR)
    assert moved is not first and not np.array_equal(moved, first)
    np.testing.assert_array_equal(moved[:, :35], first[:, :35])
    other = model.render(poses, CameraIntrinsics(50.0, 50.0, 25.0, 20.0, 50, 40), camera_key="small")
    assert other.shape == (40, 50)


def _binary_stl(triangles) -> bytes:
    data = bytearray(80) + struct.pack("<I", len(triangles))
    for triangle in triangles:
        data += struct.pack("<3f", 0, 0, 0) + struct.pack("<9f", *np.ravel(triangle)) + b"\0\0"
    return bytes(data)


SQUARE = [[[-1, -1, 0], [1, -1, 0], [1, 1, 0]], [[-1, -1, 0], [1, 1, 0], [-1, 1, 0]]]


def test_meshes_load_from_binary_and_ascii_stl_and_obj(tmp_path):
    (tmp_path / "b.stl").write_bytes(_binary_stl(SQUARE))
    (tmp_path / "a.stl").write_text("solid s\n" + "".join(
        "facet normal 0 0 1\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t)
        + "endloop\nendfacet\n" for t in SQUARE) + "endsolid s\n")
    (tmp_path / "q.obj").write_text("v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nf 1/1 2/2 3/3 4/4\n")
    for name in ("b.stl", "a.stl", "q.obj"):
        vertices, faces = load_mesh(tmp_path / name)
        triangles = vertices[faces]
        assert triangles.shape == (2, 3, 3)
        np.testing.assert_allclose(np.abs(triangles).max(axis=(0, 1)), [1, 1, 0])


def test_package_meshes_resolve_through_search_paths_with_scale(tmp_path):
    share = tmp_path / "install" / "share" / "robot_description" / "meshes"
    share.mkdir(parents=True)
    (share / "blade.stl").write_bytes(_binary_stl(SQUARE))
    uri = "package://robot_description/meshes/blade.stl"
    assert resolve_mesh_path(uri, [tmp_path / "install"]) == share / "blade.stl"
    with pytest.raises(FileNotFoundError):
        resolve_mesh_path("package://no_such_pkg_for_platform_test/m.stl", [tmp_path])
    geometry = f"<mesh filename='{uri}' scale='0.5 0.5 0.5'/>"
    model = PlatformModel.from_urdf(urdf(link("blade", visual(geometry))), mesh_paths=[tmp_path / "install"])
    assert model.parts[0].kind == "mesh"
    np.testing.assert_allclose(np.abs(model.parts[0].vertices).max(axis=0), [0.5, 0.5, 0])
    # A mesh is a surface: the square covers exactly its projection and no camera-containment test applies.
    mask = model.render({"blade": at(z=5.0)}, INTR)
    assert mask[40, 50] and mask.sum() == 21 * 21


def test_unloadable_visual_falls_back_to_collision_and_is_reported():
    missing = visual("<mesh filename='package://no_such_pkg_for_platform_test/body.stl'/>")
    body = link("body", missing + visual("<box size='1 1 1'/>", kind="collision"))
    flat = link("track", visual("<cylinder radius='0' length='0'/>") + visual("<box size='1 1 1'/>", kind="collision"))
    model = PlatformModel.from_urdf(urdf(body, flat, link("frame_only")))
    assert [(p.link, p.kind) for p in model.parts] == [("body", "box"), ("track", "box")]
    assert model.links == ["body", "track"]
    assert len(model.skipped) == 2 and "no_such_pkg" in model.skipped[0] and "degenerate" in model.skipped[1]
    only_collision = PlatformModel.from_urdf(urdf(body), geometry="collision")
    assert only_collision.parts[0].kind == "box" and not only_collision.skipped


def test_exclude_links_and_invalid_descriptions():
    two = urdf(link("a", visual("<box size='1 1 1'/>")), link("cam_housing", visual("<box size='1 1 1'/>")))
    assert PlatformModel.from_urdf(two, exclude_links=["cam_housing"]).links == ["a"]
    with pytest.raises(ValueError, match="no usable link geometry"):
        PlatformModel.from_urdf(urdf(link("base")))
    with pytest.raises(ValueError, match="URDF"):
        PlatformModel.from_urdf("<sdf/>")
    with pytest.raises(ValueError, match="geometry"):
        PlatformModel.from_urdf(two, geometry="inertial")


def test_vertex_clustering_simplifies_a_dense_mesh_within_one_cell():
    n = 60
    grid = np.stack(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij"), axis=-1).reshape(-1, 2)
    vertices = np.column_stack([grid, np.zeros(len(grid))])
    index = np.arange(n * n).reshape(n, n)
    a, b, c, d = index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    merged, simplified = simplify_mesh(vertices, faces, 0.1)
    assert len(simplified) < len(faces) / 20 and len(merged) <= 121
    np.testing.assert_allclose(merged.min(axis=0)[:2], 0, atol=0.05)
    np.testing.assert_allclose(merged.max(axis=0)[:2], 1, atol=0.05)
    assert simplify_mesh(vertices, faces, 0.0)[1] is faces


def test_detections_on_the_platform_are_dropped_or_trimmed():
    platform = np.zeros((80, 100), bool)
    platform[50:] = True  # the hood fills the bottom of the image
    on_hood = np.zeros_like(platform)
    on_hood[45:80, 20:60] = True
    person = np.zeros_like(platform)
    person[10:60, 70:80] = True
    detections = [Detection2D(np.array([20., 45., 60., 80.]), "bulldozer", 0.9, mask=on_hood),
                  Detection2D(np.array([70., 10., 80., 60.]), "person", 0.8, mask=person),
                  Detection2D(np.array([0., 55., 100., 80.]), "truck", 0.7),
                  Detection2D(np.array([0., 0., 30., 40.]), "sign", 0.7)]
    kept, dropped = exclude_platform(detections, platform, max_overlap=0.5)
    assert dropped == 2 and [d.label for d in kept] == ["person", "sign"]
    assert kept[0].mask.sum() == 40 * 10 and not (kept[0].mask & platform).any()
    np.testing.assert_array_equal(kept[0].bbox, [70, 10, 80, 50])
    assert detections[1].mask.sum() == 50 * 10  # the input is untouched
    assert kept[1] is detections[3]
    assert exclude_platform(detections, platform, max_overlap=1.0)[1] == 0


def test_a_mask_entirely_on_the_platform_is_dropped_even_at_full_overlap():
    platform = np.zeros((20, 20), bool)
    platform[:10] = True
    mask = np.zeros_like(platform)
    mask[2:5, 2:5] = True  # used to leave an empty mask whose box raised IndexError
    empty = np.zeros_like(platform)
    detections = [Detection2D(np.array([2., 2., 5., 5.]), "hood", 0.9, mask=mask),
                  Detection2D(np.array([0., 0., 1., 1.]), "nothing", 0.9, mask=empty)]
    assert exclude_platform(detections, platform, max_overlap=1.0) == ([], 2)


def test_platform_exclusion_on_mask_crops_matches_whole_image_masks():
    rng = np.random.default_rng(11)
    platform = rng.random((60, 90)) > .6
    for overlap in (0.2, 0.5, 0.9):
        for _ in range(20):
            mask = np.zeros_like(platform)
            y, x = rng.integers(0, 60), rng.integers(0, 90)
            mask[y:y + rng.integers(1, 30), x:x + rng.integers(1, 40)] = rng.random() > .1
            area, on = mask.sum(), (mask & platform).sum()
            kept, dropped = exclude_platform([Detection2D(np.array([0., 0., 1., 1.]), "x", .9, mask=mask)], platform, overlap)
            if not area or on > overlap * area:
                assert (kept, dropped) == ([], 1)
            else:
                np.testing.assert_array_equal(kept[0].mask, mask & ~platform)
                if on:
                    ys, xs = np.nonzero(mask & ~platform)
                    np.testing.assert_array_equal(kept[0].bbox, [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
