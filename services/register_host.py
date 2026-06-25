import os
import logging

def get_hosts_file() -> str | None:
    hosts_file = os.getenv("HOSTS_FILE")
    return hosts_file or None


def register_host(hostname: str, ip: str) -> bool:
    hosts_file = get_hosts_file()
    if not hosts_file:
        logging.info("HOSTS_FILE is unset; skipping hosts file registration for %s", hostname)
        return False

    entry = f"{ip} {hostname}.lan {hostname}\n"

    # Read existing lines
    try:
        with open(hosts_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    # Remove old entries
    lines = [l for l in lines if hostname not in l]
    lines.append(entry)

    # Write in-place (no rename)
    with open(hosts_file, "w") as f:
        f.writelines(lines)
        f.flush()
        os.fsync(f.fileno())

    return True
