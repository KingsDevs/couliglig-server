HOSTS_FILE = "/data/robots.hosts"

def register_host(hostname: str, ip: str):
    entry = f"{ip} {hostname}.lan {hostname}\n"

    try:
        with open(HOSTS_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    lines = [l for l in lines if hostname not in l]
    lines.append(entry)

    with open(HOSTS_FILE, "w") as f:
        f.writelines(lines)