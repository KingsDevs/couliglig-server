"""msgpack-over-ZMQ request/reply, as spoken by the SAM3 and GraspGen servers.

Requests are msgpack dicts with an "action"; numpy arrays travel via
msgpack-numpy. A reply carrying "error" raises RuntimeError.
"""

import msgpack
import msgpack_numpy
import zmq

msgpack_numpy.patch()


def pack(obj) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)


def unpack(raw: bytes):
    return msgpack.unpackb(raw, raw=False)


class ZmqClient:
    """One REQ socket to tcp://host:port.

    A REQ socket that timed out stays stuck waiting for its reply and refuses
    every later send, so after any transport error the socket is dropped and the
    next request opens a new one.
    """

    def __init__(self, host: str, port: int, timeout_ms: int = 60_000) -> None:
        self.address = f"tcp://{host}:{port}"
        self._timeout_ms = timeout_ms
        self._ctx = zmq.Context.instance()
        self._socket = None

    def request(self, payload: dict) -> dict:
        """Send payload, return the reply. Raises zmq.error.ZMQError (zmq.Again on
        timeout) when the server can't be reached, RuntimeError on a server error."""
        if self._socket is None:
            sock = self._ctx.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(self.address)
            self._socket = sock
        try:
            self._socket.send(pack(payload))
            raw = self._socket.recv()
        except zmq.error.ZMQError:
            self.close()
            raise
        response = unpack(raw)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
