import os
import tempfile

HOSTS_FILE = "/data/robots.hosts"
# HOSTS_FILE = "/home/karlshane/couliglig-server/temp.hosts"

def register_host(hostname: str, ip: str):
    entry = f"{ip} {hostname}.lan {hostname}\n"

    try:
        with open(HOSTS_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    lines = [l for l in lines if hostname not in l]
    lines.append(entry)

    dir_name = os.path.dirname(HOSTS_FILE)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tmp:
        tmp.writelines(lines)
        temp_name = tmp.name

    os.replace(temp_name, HOSTS_FILE)