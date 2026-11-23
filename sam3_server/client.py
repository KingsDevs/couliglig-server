"""Client for the SAM3 server (needs only pyzmq, msgpack, msgpack-numpy, numpy).

    from sam3_server.client import SAM3Client
    c = SAM3Client("192.168.20.108", 5557)
    c.health()                                   # True / False
    r = c.infer(rgb, organized_cloud, "the red mug")
    r["grasps"]                                  # ranked, zed_camera frame, metres

The ROS nodes in couliglig-main keep their own copy of this protocol.
"""

import numpy as np
import zmq

from sam3_server.rpc import ZmqClient


class SAM3Client:
    def __init__(self, host: str = "localhost", port: int = 5557, timeout_ms: int = 60_000):
        self._rpc = ZmqClient(host, port, timeout_ms)

    def health(self) -> bool:
        try:
            return self._rpc.request({"action": "health"}).get("status") == "ok"
        except (zmq.error.ZMQError, RuntimeError):
            return False

    def metadata(self) -> dict:
        return self._rpc.request({"action": "metadata"})

    def segment(self, image, text: str) -> dict:
        """image: RGB (H, W, 3) uint8, or a payload dict the server accepts."""
        return self._rpc.request({"action": "segment", "image": image, "text": text})

    def infer(self, image, point_cloud, text: str) -> dict:
        return self._rpc.request({
            "action": "infer",
            "image": image,
            "point_cloud": np.asarray(point_cloud, dtype=np.float32),
            "text": text,
        })

    def close(self) -> None:
        self._rpc.close()
