from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import register_router, docks_router, maps_router, actions_router, rl_router
from fastapi.responses import HTMLResponse
from services import get_db_session
from fastapi.staticfiles import StaticFiles
import html
import os

# Establish a database session at startup to ensure the database is initialized
session = get_db_session()
session.close()

HOSTS_FILE = os.getenv("HOSTS_FILE", "/data/robots.hosts")
# HOSTS_FILE = "/home/karlshane/couliglig-server/temp.hosts"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(register_router)
app.include_router(docks_router)
app.include_router(maps_router)
app.include_router(actions_router)
app.include_router(rl_router)
app.mount("/static", StaticFiles(directory="static"), name="static")

def read_hosts_lines() -> list[str]:
    try:
        with open(HOSTS_FILE, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    # clean up: ignore blanks + comments
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


@app.get("/", response_class=HTMLResponse)
def index():
    lines = read_hosts_lines()

    rows = []
    for line in lines:
        parts = line.split()
        ip = parts[0] if len(parts) > 0 else ""
        names = " ".join(parts[1:]) if len(parts) > 1 else ""
        rows.append((ip, names))

    rows_html = "\n".join(
        f"<tr><td>{html.escape(ip)}</td><td>{html.escape(names)}</td></tr>"
        for ip, names in rows
    )

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Robot Registry</title>
  <meta http-equiv="refresh" content="3" />
  <style>
    body {{ font-family: Arial, sans-serif; padding: 16px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 900px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f6f6f6; }}
    code {{ background: #f3f3f3; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>Robot Registry</h2>
  <p>File: <code>{html.escape(HOSTS_FILE)}</code></p>
  <p>Auto-refresh: 3s</p>

  <table>
    <thead>
      <tr><th>IP</th><th>Hostnames</th></tr>
    </thead>
    <tbody>
      {rows_html if rows_html else "<tr><td colspan='2'><em>No entries</em></td></tr>"}
    </tbody>
  </table>
</body>
</html>
"""

# add comment for the command for debugging
# uvicorn main:app --reload