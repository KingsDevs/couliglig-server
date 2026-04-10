import argparse
import asyncio
import json
import logging
import socket
from typing import Optional

try:
    import websockets
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: websockets\n"
        "Install with: pip install websockets"
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def detect_local_ip() -> str:
    """Best-effort local IP detection for registration payload."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]


async def robot_ws_loop(
    server_base_url: str,
    hostname: str,
    ip: Optional[str],
    namespace: str,
    ros_domain_id: int,
    heartbeat_interval: int,
    reconnect_delay: int,
) -> None:
    ws_url = f"{server_base_url.rstrip('/').replace('http://', 'ws://').replace('https://', 'wss://')}/robots/ws"
    robot_ip = ip or detect_local_ip()

    register_payload = {
        "hostname": hostname,
        "ip": robot_ip,
        "namespace": namespace,
        "ros_domain_id": ros_domain_id,
    }

    while True:
        try:
            logging.info("Connecting to %s", ws_url)
            async with websockets.connect(ws_url, ping_interval=None) as ws:
                await ws.send(json.dumps(register_payload))
                logging.info("Registered as %s (%s)", hostname, robot_ip)

                while True:
                    await ws.send(json.dumps({"type": "heartbeat"}))
                    logging.debug("heartbeat sent")
                    await asyncio.sleep(heartbeat_interval)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logging.warning("WebSocket disconnected (%s). Reconnecting in %ss...", exc, reconnect_delay)
            await asyncio.sleep(reconnect_delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot WebSocket heartbeat client")
    parser.add_argument("--server", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--hostname", default=socket.gethostname(), help="Robot hostname")
    parser.add_argument("--ip", default=None, help="Robot IP (auto-detect if omitted)")
    parser.add_argument("--namespace", default="couliglig", help="Robot namespace")
    parser.add_argument("--ros-domain-id", type=int, default=0, help="ROS domain ID")
    parser.add_argument("--heartbeat-interval", type=int, default=5, help="Heartbeat interval in seconds")
    parser.add_argument("--reconnect-delay", type=int, default=3, help="Reconnect delay in seconds")
    args = parser.parse_args()

    try:
        asyncio.run(
            robot_ws_loop(
                server_base_url=args.server,
                hostname=args.hostname,
                ip=args.ip,
                namespace=args.namespace,
                ros_domain_id=args.ros_domain_id,
                heartbeat_interval=args.heartbeat_interval,
                reconnect_delay=args.reconnect_delay,
            )
        )
    except KeyboardInterrupt:
        logging.info("Shutting down")
