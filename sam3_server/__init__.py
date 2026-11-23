"""SAM3 text-prompt segmentation + grasp server (moved from GraspGen-couliglig).

Runs as its own container on the Jetson (`sam3` compose profile), apart from the
FastAPI app: its dependencies (torch, transformers>=5, opencv, scikit-learn) are in
requirements-sam3.txt / Dockerfile.sam3, never in requirements.txt.

    python -m sam3_server --port 5557 --graspgen-host 192.168.20.113
"""
