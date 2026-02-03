import time
import json
import logging
import os
import requests
import schedule
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_KEY = "robot_ips"
HEARTBEAT_PATH = "/status/health"
TIMEOUT = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [robot-watcher] %(message)s"
)

redis_client = redis.StrictRedis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True
)

def check_robots():
    raw = redis_client.get(REDIS_KEY)

    if not raw:
        logging.info("No robots registered")
        return

    robot_ips = json.loads(raw)
    updated = robot_ips.copy()

    for hostname, ip in robot_ips.items():
        url = f"http://{ip}:8000{HEARTBEAT_PATH}"
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            
            response = r.json()
            if response.get("hostname") != hostname:
                raise RuntimeError("Hostname mismatch")
            
            updated[hostname] = {
                "hostname": hostname,
                "ip": ip,
                "namespace": response.get("namespace", "couliglig"),
                "ros_domain_id": response.get("ros_domain_id", 0),
                "timestamp": response.get("timestamp", "")
            }

    
            logging.info(f"{hostname} ({ip}) OK")

        except Exception as e:
            logging.warning(f"{hostname} ({ip}) FAILED → removing ({e})")
            updated.pop(hostname, None)

    # Update Redis only if something changed
    if updated != robot_ips:
        redis_client.set(REDIS_KEY, json.dumps(updated))
        logging.info("Redis robot_ips updated")

def main():
    logging.info("Starting robot watcher scheduler")

    # run every 5s
    schedule.every(5).seconds.do(check_robots)

    # run once on startup
    check_robots()

    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
