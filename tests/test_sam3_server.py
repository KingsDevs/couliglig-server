"""SAM3 server (sam3_server/): cloud processing, and the ZMQ protocol end to end
with a fake SAM3 model and a fake GraspGen server. No torch needed.

Needs requirements-sam3.txt's light packages (pyzmq, msgpack, msgpack-numpy,
opencv-python-headless, scikit-learn); skipped without them.
"""

import base64
import threading

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
pytest.importorskip("msgpack_numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("sklearn")

from sam3_server import processing as P  # noqa: E402
from sam3_server.client import SAM3Client  # noqa: E402
from sam3_server.rpc import ZmqClient, pack, unpack  # noqa: E402
from sam3_server.server import SAM3Server  # noqa: E402

H, W = 48, 64


# ---------------------------------------------------------------------------
# Scene: a table plane at z = 1.0 m with a 10 x 10 cm box (top at z = 0.9) in
# the middle of the image. The mask covers the box plus a rim of table.
# ---------------------------------------------------------------------------

def scene():
    v, u = np.mgrid[0:H, 0:W]
    pc = np.zeros((H, W, 3), dtype=np.float32)
    pc[..., 0] = (u - W / 2) * 0.005
    pc[..., 1] = (v - H / 2) * 0.005
    pc[..., 2] = 1.0
    box = (abs(u - W / 2) < 10) & (abs(v - H / 2) < 10)
    pc[..., 2][box] = 0.9
    mask = ((abs(u - W / 2) < 16) & (abs(v - H / 2) < 16)).astype(np.uint8)
    rgb = np.full((H, W, 3), 80, dtype=np.uint8)
    return rgb, pc, mask, box


class FakeSegmenter:
    info = {"model_id": "fake/sam3", "fp16": False}

    def __init__(self, mask):
        self.mask = mask
        self.calls = []

    def __call__(self, rgb, text):
        self.calls.append((rgb.shape, text))
        if text == "nothing":
            return None
        return self.mask, 0.9, 2


class FakeGraspGen:
    """REP server answering GraspGen's infer with n grasps at the cloud centroid."""

    def __init__(self, n=5, port=None):
        self.n = n
        self.requests = []
        self.sock = zmq.Context.instance().socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        if port is None:
            port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        else:
            self.sock.bind(f"tcp://127.0.0.1:{port}")
        self.port = port
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            if not self.sock.poll(50):
                continue
            req = unpack(self.sock.recv())
            self.requests.append(req)
            pts = np.asarray(req["point_cloud"])
            grasps = np.tile(np.eye(4, dtype=np.float32), (self.n, 1, 1))
            grasps[:, :3, 3] = pts.mean(axis=0)
            grasps[:, 3, 3] = 0          # the server must fix the homogeneous row
            conf = np.linspace(0.1, 0.9, self.n).astype(np.float32)
            self.sock.send(pack({"grasps": grasps, "confidences": conf}))

    def close(self):
        self.stop.set()
        self.thread.join(2)
        self.sock.close()


def free_port():
    s = zmq.Context.instance().socket(zmq.REP)
    port = s.bind_to_random_port("tcp://127.0.0.1")
    s.close()
    return port


@pytest.fixture
def serve():
    """serve(segmenter, graspgen, **kw) -> SAM3Client to a running server."""
    stops, threads, clients = [], [], []

    def start(segmenter, graspgen, **kw):
        server = SAM3Server(segmenter, graspgen, host="127.0.0.1", port=0, **kw)
        endpoint = server.bind()
        stop = threading.Event()
        t = threading.Thread(target=server.serve_forever, args=(stop,), daemon=True)
        t.start()
        stops.append(stop)
        threads.append(t)
        client = SAM3Client("127.0.0.1", int(endpoint.rsplit(":", 1)[1]), timeout_ms=5000)
        clients.append(client)
        return client

    yield start
    for c in clients:
        c.close()
    for s in stops:
        s.set()
    for t in threads:
        t.join(2)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def test_decode_image_payload_variants():
    rgb = np.random.default_rng(0).integers(0, 255, (8, 6, 3), dtype=np.uint8)
    assert np.array_equal(P.decode_image_payload(rgb), rgb)
    flat = {"data": rgb.ravel().tolist(), "shape": [8, 6, 3], "dtype": "uint8"}
    assert np.array_equal(P.decode_image_payload(flat), rgb)
    ok, png = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(png.tobytes()).decode()
    assert np.array_equal(P.decode_image_payload({"base64": b64}), rgb)   # back to RGB
    assert np.array_equal(P.decode_image_payload(b64), rgb)
    with pytest.raises(ValueError):
        P.decode_image_payload({"nope": 1})
    with pytest.raises(ValueError):
        P.decode_image_payload(base64.b64encode(b"not an image").decode())


def test_masked_points_resizes_mask_and_finds_seed():
    rgb, pc, mask, box = scene()
    big_mask = cv2.resize(mask, (W * 2, H * 2), interpolation=cv2.INTER_NEAREST)
    pts, seed, mask_resized = P.masked_points(pc, big_mask, (H * 2, W * 2, 3))
    assert mask_resized.shape == (H, W)
    assert len(pts) == int(mask.sum())
    assert seed is not None and seed[2] == pytest.approx(0.9)       # on the box


def test_masked_points_drops_nan_and_dilates_when_all_nan():
    rgb, pc, mask, box = scene()
    pc[mask == 1] = np.nan
    pts, seed, _ = P.masked_points(pc, mask, rgb.shape)
    assert seed is None
    assert len(pts) > 0 and np.isfinite(pts).all()    # boundary points via dilation


def test_masked_points_flat_cloud_passes_through():
    pts = np.ones((20, 3), dtype=np.float32)
    out, seed, mres = P.masked_points(pts, np.ones((4, 4), np.uint8), (4, 4, 3))
    assert out is pts and seed is None and mres is None


def test_plane_removal_and_cluster_keep_the_box():
    rgb, pc, mask, box = scene()
    pts, seed, _ = P.masked_points(pc, mask, rgb.shape)
    kw = dict(enabled=True, dist_thresh=0.005, min_ratio=0.25, ransac_iters=100,
              dbscan_min_samples=10, rng=np.random.default_rng(1))
    no_plane = P.remove_support_plane(pts, seed, **kw)
    assert np.allclose(no_plane[:, 2], 0.9)
    assert len(no_plane) == int((box & (mask == 1)).sum())
    # Seed on the dominant plane: that is a flat object, keep everything.
    table_seed = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    assert len(P.remove_support_plane(pts, table_seed, **kw)) == len(pts)
    assert P.remove_support_plane(pts, seed, **{**kw, "enabled": False}) is pts


def test_cluster_prefers_the_one_nearest_the_seed():
    rng = np.random.default_rng(2)
    small = rng.normal([0, 0, 0.5], 0.002, (30, 3))
    large = rng.normal([0.3, 0, 0.5], 0.002, (300, 3))
    pts = np.vstack([small, large]).astype(np.float32)
    kw = dict(enabled=True, eps=0.01, min_samples=10)
    assert len(P.largest_object_cluster(pts, np.array([0, 0, 0.5]), **kw)) == 30
    assert len(P.largest_object_cluster(pts, None, **kw)) == 300


def test_heuristic_pose_is_a_rotation_along_the_long_axis():
    rng = np.random.default_rng(3)
    pts = np.column_stack([rng.uniform(-0.01, 0.01, 200),
                           rng.uniform(-0.10, 0.10, 200),      # long along camera y
                           rng.uniform(0.79, 0.81, 200)])
    pose = P.heuristic_pose(pts)
    r = pose[:3, :3].astype(np.float64)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-5)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(pose[:3, 0], P.LOOK_DOWN_CAM, atol=1e-5)
    assert abs(pose[1, 2]) > 0.99                                 # local Z ~ camera y
    assert np.allclose(pose[:3, 3], pts.mean(axis=0), atol=1e-5)
    assert P.heuristic_pose(pts[:2]) is None


def test_format_grasps_ranks_and_truncates():
    grasps = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    conf = np.array([0.2, 0.9, 0.5, 0.1], dtype=np.float32)
    r = P.format_grasps(grasps, conf, top_n=2)
    assert [g["candidate_index"] for g in r["grasps"]] == [1, 2]
    assert [g["rank"] for g in r["grasps"]] == [0, 1]
    assert r["best_grasp"] is r["grasps"][0]
    assert r["num_candidates"] == 4 and r["num_grasps"] == 2
    assert r["frame"] == "zed_camera" and r["mode"] == "auto"
    assert P.format_grasps(grasps, conf, top_n=0)["num_grasps"] == 4


def test_server_requires_graspgen_in_auto_mode():
    with pytest.raises(ValueError):
        SAM3Server(FakeSegmenter(None), None, pose_mode="auto")
    with pytest.raises(ValueError):
        SAM3Server(FakeSegmenter(None), None, pose_mode="bogus")


# ---------------------------------------------------------------------------
# Protocol, end to end over ZMQ
# ---------------------------------------------------------------------------

def test_health_metadata_and_unknown_action(serve):
    gg = FakeGraspGen()
    try:
        c = serve(FakeSegmenter(None), ZmqClient("127.0.0.1", gg.port))
        assert c.health()
        meta = c.metadata()
        assert meta["model_id"] == "fake/sam3"
        assert meta["graspgen_target"] == f"tcp://127.0.0.1:{gg.port}"
        assert meta["pose_mode"] == "auto"
        with pytest.raises(RuntimeError, match="Unknown action"):
            c._rpc.request({"action": "dance"})
        assert c.health()                       # still serving after an error reply
    finally:
        gg.close()


def test_infer_forwards_the_cleaned_box_to_graspgen(serve):
    rgb, pc, mask, box = scene()
    gg = FakeGraspGen(n=5)
    try:
        seg = FakeSegmenter(mask)
        c = serve(seg, ZmqClient("127.0.0.1", gg.port), top_n=3)
        r = c.infer(rgb, pc, "the box")
        assert seg.calls == [((H, W, 3), "the box")]
        sent = np.asarray(gg.requests[0]["point_cloud"])
        assert np.allclose(sent[:, 2], 0.9)                 # table removed
        assert gg.requests[0]["num_grasps"] == 100
        assert r["mode"] == "auto" and r["num_grasps"] == 3 and r["num_candidates"] == 5
        assert [g["candidate_index"] for g in r["grasps"]] == [4, 3, 2]
        assert r["best_grasp"]["confidence"] == pytest.approx(0.9)
        assert np.asarray(r["best_grasp"]["pose"])[3, 3] == 1
        assert np.allclose(r["best_grasp"]["position"], sent.mean(axis=0), atol=1e-5)
        assert set(r["timing"]) >= {"sam_ms", "graspgen_ms", "total_handler_ms"}
    finally:
        gg.close()


def test_infer_without_a_match_or_inputs(serve):
    rgb, pc, mask, _ = scene()
    gg = FakeGraspGen()
    try:
        c = serve(FakeSegmenter(mask), ZmqClient("127.0.0.1", gg.port))
        r = c.infer(rgb, pc, "nothing")
        assert r["grasps"] == [] and r["best_grasp"] is None
        assert "no instance" in r["message"]
        r = c._rpc.request({"action": "infer", "text": "x"})
        assert r["grasps"] == [] and "requires" in r["message"]
        assert gg.requests == []
    finally:
        gg.close()


def test_manual_mode_skips_graspgen(serve):
    rgb, pc, mask, _ = scene()
    c = serve(FakeSegmenter(mask), None, pose_mode="manual")
    r = c.infer(rgb, pc, "the box")
    assert r["mode"] == "manual" and r["num_grasps"] == 1
    assert np.allclose(np.asarray(r["best_grasp"]["pose"])[:3, 0], P.LOOK_DOWN_CAM, atol=1e-5)


def test_segment_returns_the_mask_only(serve):
    rgb, pc, mask, _ = scene()
    c = serve(FakeSegmenter(mask), None, pose_mode="manual")
    r = c.segment(rgb, "the box")
    assert r["message"] == "ok" and r["n_instances"] == 2
    assert np.array_equal(np.asarray(r["mask"]), mask)
    assert c.segment(rgb, "nothing")["mask"] is None
    assert "Empty text" in c.segment(rgb, "  ")["message"]
    assert "Bad image" in c._rpc.request({"action": "segment", "text": "x"})["message"]


def test_graspgen_down_then_back(serve):
    """A timed-out GraspGen request must not wedge later ones (REQ socket rebuilt)."""
    rgb, pc, mask, _ = scene()
    port = free_port()
    c = serve(FakeSegmenter(mask), ZmqClient("127.0.0.1", port, timeout_ms=300))
    r = c.infer(rgb, pc, "the box")
    assert r["grasps"] == [] and "GraspGen forward failed" in r["message"]

    gg = FakeGraspGen(n=2, port=port)                # GraspGen comes up on that port
    try:
        r = c.infer(rgb, pc, "the box")
        assert r["num_grasps"] == 2, r.get("message")
    finally:
        gg.close()
