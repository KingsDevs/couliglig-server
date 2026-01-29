import os

HOSTS_FILE = os.getenv("HOSTS_FILE", "/data/robots.hosts")
# HOSTS_FILE = "/home/karlshane/couliglig-server/temp.hosts"

def register_host(hostname: str, ip: str):
    entry = f"{ip} {hostname}.lan {hostname}\n"

    # Read existing lines
    try:
        with open(HOSTS_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    # Remove old entries
    lines = [l for l in lines if hostname not in l]
    lines.append(entry)

    # Write in-place (no rename)
    with open(HOSTS_FILE, "w") as f:
        f.writelines(lines)
        f.flush()
        os.fsync(f.fileno())