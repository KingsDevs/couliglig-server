# robot_watcher.py — retired
#
# Robot liveness is now tracked via WebSocket connections.
# Robots connect to WS /robots/ws and send periodic heartbeats.
# On disconnect the server removes the robot from Redis immediately.
#
# See routes/robots.py — robot_websocket()
