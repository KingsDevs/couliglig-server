"""Image/cloud decoding, 3D cleanup and grasp formatting for the SAM3 server.

No torch here, so the server's tests can run without a GPU stack.
"""

import base64

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

FRAME = "zed_camera"


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_image_payload(image_payload) -> np.ndarray:
    """Decode an image payload and return an RGB (H, W, 3) uint8 array.

    Accepts a raw array, {"data", "shape", "dtype"}, {"base64"} or a base64 string
    (an encoded PNG/JPEG). Raw arrays are assumed RGB already.
    """
    if isinstance(image_payload, dict):
        if "data" in image_payload:
            image = np.asarray(
                image_payload["data"],
                dtype=image_payload.get("dtype", np.uint8),
            )
            if "shape" in image_payload:
                image = image.reshape(image_payload["shape"])
            return image
        if "base64" not in image_payload:
            raise ValueError("image dict must include 'data' or 'base64'")
        encoded_image = image_payload["base64"]
    elif isinstance(image_payload, str):
        encoded_image = image_payload
    else:
        return np.asarray(image_payload, dtype=np.uint8)

    image_buffer = np.frombuffer(base64.b64decode(encoded_image), dtype=np.uint8)
    bgr = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image payload")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_point_cloud(pc_payload) -> np.ndarray:
    """Point cloud payload -> (H, W, 3) organized or (N, 3) float32 array."""
    return np.asarray(pc_payload, dtype=np.float32)


# ---------------------------------------------------------------------------
# Mask -> 3D points
# ---------------------------------------------------------------------------

def masked_points(pc: np.ndarray, mask: np.ndarray, image_shape):
    """Project a 2D mask (image resolution) into an organized cloud.

    Returns (pts (N, 3) finite only, seed_xyz or None, mask_resized or None).
    seed_xyz is the cloud point under the mask centroid, used to pick the object
    cluster. A flat (N, 3) cloud is returned as is (no mask can apply to it).
    Textureless surfaces leave depth NaN inside the mask; then a dilated mask
    picks up the finite boundary points.
    """
    if pc.ndim != 3:
        return pc, None, None

    h_c, w_c = pc.shape[:2]
    mask_resized = cv2.resize(mask, (w_c, h_c), interpolation=cv2.INTER_NEAREST)
    pts = pc[mask_resized == 1]
    pts = pts[np.isfinite(pts).all(axis=1)]

    seed_xyz = None
    ys, xs = np.where(mask == 1)
    if len(xs) > 0:
        u, v = int(round(xs.mean())), int(round(ys.mean()))
        u_c = max(0, min(int(round(u * w_c / image_shape[1])), w_c - 1))
        v_c = max(0, min(int(round(v * h_c / image_shape[0])), h_c - 1))
        cand = pc[v_c, u_c]
        if np.all(np.isfinite(cand)):
            seed_xyz = cand

    if len(pts) == 0 and mask_resized.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        pts_d = pc[cv2.dilate(mask_resized, kernel) == 1]
        pts = pts_d[np.isfinite(pts_d).all(axis=1)]

    return pts, seed_xyz, mask_resized


# ---------------------------------------------------------------------------
# 3D cloud cleanup (the 2D mask drags in table/background + flying pixels)
# ---------------------------------------------------------------------------

def remove_support_plane(
    pts, seed_xyz, *, enabled, dist_thresh, min_ratio, ransac_iters,
    dbscan_min_samples, rng=None,
):
    """RANSAC-fit the dominant plane and drop its inliers (the floor/table).

    DBSCAN can't separate an object from the surface it touches because they're
    one connected component. Removing the support plane cuts that link so the
    object survives clustering on its own.

    Skips removal (returns pts unchanged) when no plane explains at least
    `min_ratio` of the points, when the seed point lies on the best plane (a flat
    object, not the floor), or when removal would leave too few points to grasp.
    """
    n = len(pts)
    if not enabled or n < 3 * dbscan_min_samples:
        return pts

    rng = rng if rng is not None else np.random.default_rng()
    best_inliers, best_count, best_plane = None, 0, None
    for _ in range(ransac_iters):
        p0, p1, p2 = pts[rng.choice(n, 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(normal)
        if nn < 1e-9:
            continue
        normal = normal / nn
        inliers = np.abs((pts - p0) @ normal) < dist_thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers, best_plane = count, inliers, (normal, p0)

    if best_inliers is None or best_count < min_ratio * n:
        return pts

    if seed_xyz is not None and np.all(np.isfinite(seed_xyz)):
        normal, p0 = best_plane
        if abs(float((seed_xyz - p0) @ normal)) < dist_thresh:
            return pts

    keep = ~best_inliers
    if int(keep.sum()) < dbscan_min_samples:
        return pts
    return pts[keep]


def largest_object_cluster(pts, seed_xyz, *, enabled, eps, min_samples):
    """Keep the dominant 3D cluster of the masked points.

    When `seed_xyz` is given, the cluster whose nearest member is closest to the
    seed is kept (so a small object beside a large background blob still wins);
    otherwise the largest cluster wins. Falls back to the input unchanged when
    clustering is disabled, the cloud is too small, or every point is noise.
    """
    if not enabled or len(pts) < min_samples:
        return pts

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts).labels_
    valid = labels >= 0
    if not valid.any():
        return pts

    if seed_xyz is not None and np.all(np.isfinite(seed_xyz)):
        best_label, best_d = None, np.inf
        for lab in np.unique(labels[valid]):
            d = np.linalg.norm(pts[labels == lab] - seed_xyz, axis=1).min()
            if d < best_d:
                best_d, best_label = d, lab
        keep = labels == best_label
    else:
        keep = labels == int(np.bincount(labels[valid]).argmax())

    return pts[keep]


# ---------------------------------------------------------------------------
# Grasp poses / results
# ---------------------------------------------------------------------------

# Fixed "looking down" approach direction in the point cloud frame (ZED
# `zed_left_camera_frame`): world-down (arm `base_link` -Z) rotated through the
# static RoArm-M3 TF chain (base_link -> zed_camera [rpy 0 pi/2 0] ->
# zed_camera_link [rpy pi 0.384 0] -> zed_left_camera_frame), i.e. it includes
# the camera's ~22 deg tilt and the 180 deg mounting flip. Recompute it if that
# calibration changes. Used by the manual pose mode.
LOOK_DOWN_CAM = np.array([0.927172, 0.0, -0.374636])


def heuristic_pose(pts: np.ndarray):
    """Top-down grasp pose from the segmented cloud (no GraspGen).

    Position is the centroid; local X (approach) looks straight down
    (LOOK_DOWN_CAM); local Z follows the object's long axis; local Y (finger
    closing) is across it. Returns a (4, 4) float32 pose, or None when the
    geometry is degenerate.
    """
    if len(pts) < 3:
        return None

    x = LOOK_DOWN_CAM
    center = pts.mean(axis=0)
    cov = np.cov((pts - center).T)
    if not np.all(np.isfinite(cov)):
        return None
    _, evecs = np.linalg.eigh(cov)      # ascending eigenvalues
    long_axis = evecs[:, -1]

    z = long_axis - (long_axis @ x) * x
    if np.linalg.norm(z) < 1e-6:
        # Long axis ~parallel to the approach: any in-plane reference will do.
        ref = np.array([1.0, 0.0, 0.0]) if abs(x[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        z = ref - (ref @ x) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = center
    return pose


def _grasp_entry(pose: np.ndarray, confidence: float, index: int, rank: int) -> dict:
    pose = pose.astype(np.float32, copy=False)
    return {
        "pose": pose,
        "position": pose[:3, 3].astype(np.float32, copy=False),
        "rotation_matrix": pose[:3, :3].astype(np.float32, copy=False),
        "confidence": float(confidence),
        "candidate_index": int(index),
        "rank": rank,
    }


def format_grasps(grasps: np.ndarray, confidences: np.ndarray, top_n: int) -> dict:
    """The top-N candidate grasps, best first (top_n <= 0: all of them).

    The robot side runs IK/reachability over them in order and takes the first
    reachable one. Poses are in the zed_camera frame, metres.
    """
    order = np.argsort(confidences)[::-1]
    if top_n > 0:
        order = order[:top_n]
    ranked = [_grasp_entry(grasps[idx], confidences[idx], idx, rank)
              for rank, idx in enumerate(order)]
    return {
        "grasps": ranked,
        "best_grasp": ranked[0] if ranked else None,
        "num_grasps": len(ranked),
        "num_candidates": int(len(grasps)),
        "best_confidence": ranked[0]["confidence"] if ranked else 0.0,
        "frame": FRAME,
        "units": "meters",
        "mode": "auto",
    }


def heuristic_grasp_result(pose: np.ndarray) -> dict:
    """format_grasps' shape for the single manual-mode pose."""
    grasp = _grasp_entry(pose, 1.0, 0, 0)
    return {
        "grasps": [grasp],
        "best_grasp": grasp,
        "num_grasps": 1,
        "num_candidates": 1,
        "best_confidence": 1.0,
        "frame": FRAME,
        "units": "meters",
        "mode": "manual",
    }


def empty_grasp_result(reason: str) -> dict:
    return {
        "grasps": [],
        "best_grasp": None,
        "num_grasps": 0,
        "num_candidates": 0,
        "message": reason,
    }


# ---------------------------------------------------------------------------
# Debug overlay
# ---------------------------------------------------------------------------

def draw_overlay(rgb, mask, score: float, n_pts: int, text: str) -> np.ndarray:
    """BGR image with the mask tinted green, its centroid and a label."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    on = mask == 1
    bgr[on] = (bgr[on] * 0.6 + np.array([0, 200, 0]) * 0.4).astype(np.uint8)
    ys, xs = np.where(on)
    if len(xs) > 0:
        u, v = int(round(xs.mean())), int(round(ys.mean()))
        cv2.circle(bgr, (u, v), 6, (0, 0, 255), -1)
        cv2.circle(bgr, (u, v), 8, (255, 255, 255), 2)
    put_label(bgr, f"'{text}'  score={score:.3f}  pts={n_pts}", (0, 0, 0))
    return bgr


def put_label(bgr, label: str, color) -> None:
    for c, thickness in (((255, 255, 255), 2), (color, 1)):
        cv2.putText(bgr, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    c, thickness, cv2.LINE_AA)
