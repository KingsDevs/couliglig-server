"""SAM3 text-prompt segmentation + grasp server (ZMQ REP, default port 5557).

The text names the object to pick (e.g. "the red mug"). SAM3 returns every
instance of that concept; the best-scoring one's mask is projected into the
organized point cloud, cleaned in 3D, and forwarded to the GraspGen ZMQ server
(GraspGen-couliglig, its own machine and environment) for 6-DOF grasps.

Protocol (msgpack over ZMQ REQ/REP, numpy via msgpack-numpy):
    {"action": "infer",   "image": ..., "point_cloud": ndarray, "text": str}
        -> ranked grasps in the zed_camera frame, metres (see processing.format_grasps)
    {"action": "segment", "image": ..., "text": str}
        -> {"mask": uint8 (H, W) | None, "score", "n_instances", "message"}
           (segmentation only: no cloud, no GraspGen)
    {"action": "metadata"} / {"action": "health"}
    Errors: {"error": str}.

pose_mode "manual" skips GraspGen and returns one heuristic top-down pose.
"""

import logging
import os
import time
import zlib

import cv2
import numpy as np
import zmq

from sam3_server import processing as P
from sam3_server.rpc import ZmqClient, pack, unpack

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff", ".tif"}


class SAM3Server:
    """segmenter: callable (rgb, text) -> (mask, score, n_instances) | None, with an
    `info` dict for metadata (Sam3Segmenter). graspgen: ZmqClient to the GraspGen
    server, or None (manual pose mode only)."""

    def __init__(
        self,
        segmenter,
        graspgen: ZmqClient | None,
        *,
        host: str = "0.0.0.0",
        port: int = 5557,
        pose_mode: str = "auto",
        top_n: int = 20,
        visualize: bool = False,
        vis_path: str = "/tmp/sam3_seg.png",
        dump_cloud: str = "",
        # GraspGen request parameters
        num_grasps: int = 100,
        topk_num_grasps: int = 20,
        min_grasps: int = 10,
        max_tries: int = 3,
        remove_outliers: bool = True,
        min_points: int = 32,
        # 3D cleanup
        dbscan: bool = True,
        dbscan_eps: float = 0.01,
        dbscan_min_samples: int = 10,
        plane_removal: bool = True,
        plane_dist_thresh: float = 0.005,
        plane_min_ratio: float = 0.25,
        plane_ransac_iters: int = 100,
    ) -> None:
        if pose_mode not in ("auto", "manual"):
            raise ValueError(f"pose_mode must be 'auto' or 'manual', got {pose_mode!r}")
        if pose_mode == "auto" and graspgen is None:
            raise ValueError("pose_mode 'auto' needs a GraspGen client")
        self._segmenter = segmenter
        self._graspgen = graspgen
        self._host = host
        self._port = port
        self._pose_mode = pose_mode
        self._top_n = top_n
        self._visualize = visualize
        # cv2.imwrite picks the format from the extension.
        if vis_path and os.path.splitext(vis_path)[1].lower() not in _IMAGE_EXTS:
            vis_path += ".png"
        self._vis_path = vis_path
        self._dump_cloud = dump_cloud
        self._graspgen_params = {
            "grasp_threshold": -1.0,
            "num_grasps": num_grasps,
            "topk_num_grasps": topk_num_grasps,
            "min_grasps": min_grasps,
            "max_tries": max_tries,
            "remove_outliers": remove_outliers,
        }
        self._min_points = min_points
        self._dbscan = dbscan
        self._dbscan_eps = dbscan_eps
        self._dbscan_min_samples = dbscan_min_samples
        self._plane_removal = plane_removal
        self._plane_dist_thresh = plane_dist_thresh
        self._plane_min_ratio = plane_min_ratio
        self._plane_ransac_iters = plane_ransac_iters

        self._metadata = {
            **getattr(segmenter, "info", {}),
            "port": port,
            "graspgen_target": graspgen.address if graspgen is not None else None,
            "top_n": top_n,
            "pose_mode": pose_mode,
            "frame": P.FRAME,
            "units": "meters",
        }
        self._socket = None

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def handle(self, request: dict) -> dict:
        action = request.get("action") if isinstance(request, dict) else None
        if action == "health":
            return {"status": "ok"}
        if action == "metadata":
            return self._metadata
        if action == "infer":
            return self._handle_infer(request)
        if action == "segment":
            return self._handle_segment(request)
        return {"error": f"Unknown action: {action}"}

    def _handle_segment(self, request: dict) -> dict:
        """Text -> 2D mask at the input image's resolution. Projecting it into a
        cloud is the caller's business (the ROS sam3_segment_node does that)."""
        def miss(message):
            return {"mask": None, "score": 0.0, "n_instances": 0, "message": message}

        t_start = time.perf_counter()
        text = (request.get("text") or "").strip()
        if not text:
            return miss("Empty text; SAM3 needs a concept to segment.")
        try:
            rgb = P.decode_image_payload(request["image"])
        except (KeyError, ValueError) as exc:
            return miss(f"Bad image payload: {exc}")

        t_decode = time.perf_counter()
        seg = self._segmenter(rgb, text)
        t_sam = time.perf_counter()
        if seg is None:
            logger.info("No SAM3 instance for %r", text)
            return miss(f"SAM3 found no instance matching {text!r}")

        mask, score, n_instances = seg
        logger.info("Segment: %d instance(s) for %r, best score %.3f, %.1f%% of pixels",
                    n_instances, text, score, mask.mean() * 100)
        return {
            "mask": np.ascontiguousarray(mask, dtype=np.uint8),
            "score": float(score),
            "n_instances": int(n_instances),
            "message": "ok",
            "timing": {
                "decode_ms": (t_decode - t_start) * 1000,
                "sam_ms": (t_sam - t_decode) * 1000,
                "total_handler_ms": (t_sam - t_start) * 1000,
            },
        }

    def _handle_infer(self, request: dict) -> dict:
        image = request.get("image")
        point_cloud = request.get("point_cloud")
        text = request.get("text")
        if image is None or point_cloud is None or text is None:
            return P.empty_grasp_result("infer requires 'image', 'point_cloud', and 'text'")

        t_start = time.perf_counter()
        rgb = P.decode_image_payload(image)
        if isinstance(image, dict):
            logger.info("image client_frame=%s client_crc=%s server_crc=%s text=%r",
                        image.get("frame"), image.get("checksum"),
                        zlib.crc32(np.ascontiguousarray(rgb).tobytes()) & 0xFFFFFFFF, text)
        pc = P.decode_point_cloud(point_cloud)
        t_decode = time.perf_counter()

        seg = self._segmenter(rgb, text)
        t_sam = time.perf_counter()
        if seg is None:
            logger.info("SAM3 found no instance for %r", text)
            if self._visualize:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                P.put_label(bgr, f"no match: '{text}'", (0, 0, 255))
                self._write_overlay(bgr)
            return P.empty_grasp_result(f"SAM3 found no instance for '{text}'")
        mask, score, n_instances = seg
        logger.info("SAM3 found %d instance(s) for %r; best score=%.3f, coverage=%.1f%%",
                    n_instances, text, score, mask.mean() * 100)

        pts, seed_xyz, _ = P.masked_points(pc, mask, rgb.shape)

        # Drop table/background left by the 2D mask: first the support plane the
        # object rests on, then everything but the object's cluster.
        if len(pts) >= self._dbscan_min_samples:
            n_before = len(pts)
            pts = P.remove_support_plane(
                pts, seed_xyz,
                enabled=self._plane_removal,
                dist_thresh=self._plane_dist_thresh,
                min_ratio=self._plane_min_ratio,
                ransac_iters=self._plane_ransac_iters,
                dbscan_min_samples=self._dbscan_min_samples,
            )
            n_plane = len(pts)
            pts = P.largest_object_cluster(
                pts, seed_xyz,
                enabled=self._dbscan,
                eps=self._dbscan_eps,
                min_samples=self._dbscan_min_samples,
            )
            logger.info("3D cleanup: %d -> %d (plane) -> %d (cluster) points (seed=%s)",
                        n_before, n_plane, len(pts), "yes" if seed_xyz is not None else "no")
        t_mask = time.perf_counter()

        if self._dump_cloud and len(pts):
            try:
                np.save(self._dump_cloud, np.asarray(pts, dtype=np.float32))
            except Exception as exc:
                logger.warning("Could not dump cloud to %s: %s", self._dump_cloud, exc)

        result = self._grasps(pts)
        t_grasp = time.perf_counter()

        if self._visualize:
            self._write_overlay(P.draw_overlay(rgb, mask, score, len(pts), text))
        t_vis = time.perf_counter()

        result["timing"] = {
            "decode_ms": (t_decode - t_start) * 1000,
            "sam_ms": (t_sam - t_decode) * 1000,
            "mask_ms": (t_mask - t_sam) * 1000,
            "graspgen_ms": (t_grasp - t_mask) * 1000,
            "visualize_ms": (t_vis - t_grasp) * 1000,
            "total_handler_ms": (t_vis - t_start) * 1000,
        }
        logger.info("Infer %r: %d points, %d grasp(s) | %s", text, len(pts),
                    result["num_grasps"],
                    ", ".join(f"{k}={v:.1f}" for k, v in result["timing"].items()))
        return result

    def _grasps(self, pts: np.ndarray) -> dict:
        if pts.ndim != 2 or pts.shape[1] != 3:
            return P.empty_grasp_result(f"Segmented point cloud must be (N, 3), got {pts.shape}")
        if len(pts) < self._min_points:
            return P.empty_grasp_result(
                f"Only {len(pts)} segmented points; need at least {self._min_points}")

        if self._pose_mode == "manual":
            pose = P.heuristic_pose(pts)
            if pose is None:
                return P.empty_grasp_result("Heuristic pose is degenerate")
            return P.heuristic_grasp_result(pose)

        try:
            response = self._graspgen.request({
                "action": "infer",
                "point_cloud": np.ascontiguousarray(pts, dtype=np.float32),
                **self._graspgen_params,
            })
        except (zmq.error.ZMQError, RuntimeError) as exc:
            return P.empty_grasp_result(f"GraspGen forward failed: {exc!r}")

        # np.array (not asarray): a writable copy, the decoded one is read-only.
        grasps = np.array(response.get("grasps", []), dtype=np.float32)
        confidences = np.asarray(response.get("confidences", []), dtype=np.float32)
        if len(grasps) == 0:
            return P.empty_grasp_result("GraspGen returned no grasps")
        grasps[:, 3, 3] = 1
        return P.format_grasps(grasps, confidences, self._top_n)

    def _write_overlay(self, bgr) -> None:
        """Overlay PNG, overwritten each request (the container is headless)."""
        if not self._vis_path:
            return
        try:
            cv2.imwrite(self._vis_path, bgr)
        except Exception as exc:
            logger.warning("Could not write %s: %s", self._vis_path, exc)

    # ------------------------------------------------------------------
    # Serving
    # ------------------------------------------------------------------

    def bind(self) -> str:
        """Bind the REP socket; returns the endpoint (port 0 picks a free one)."""
        self._socket = zmq.Context.instance().socket(zmq.REP)
        self._socket.setsockopt(zmq.LINGER, 0)
        port = "*" if self._port == 0 else self._port
        self._socket.bind(f"tcp://{self._host}:{port}")
        endpoint = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        logger.info("SAM3 server listening on %s", endpoint)
        return endpoint

    def serve_forever(self, stop=None) -> None:
        """Answer requests until KeyboardInterrupt or stop (a threading.Event) is set."""
        if self._socket is None:
            self.bind()
        try:
            while stop is None or not stop.is_set():
                if not self._socket.poll(200):
                    continue
                raw = self._socket.recv()
                try:
                    response = self.handle(unpack(raw))
                except Exception as exc:
                    logger.exception("Error handling request")
                    response = {"error": str(exc)}
                self._socket.send(pack(response))
        except KeyboardInterrupt:
            logger.info("Shutting down")
        finally:
            self._socket.close()
            self._socket = None
