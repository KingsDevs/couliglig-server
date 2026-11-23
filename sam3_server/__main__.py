"""python -m sam3_server: load SAM3 and serve on ZMQ (see sam3_server/server.py)."""

import argparse
import logging
import os

from sam3_server.rpc import ZmqClient
from sam3_server.server import SAM3Server


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SAM3 text-prompt segmentation + grasp server")
    p.add_argument("--model-id", default=os.getenv("SAM3_MODEL_ID", "facebook/sam3"),
                   help="Hugging Face model id")
    p.add_argument("--host", default="0.0.0.0", help="ZMQ REP bind address")
    p.add_argument("--port", type=int, default=int(os.getenv("SAM3_PORT", "5557")))
    p.add_argument("--graspgen-host", default=os.getenv("GRASPGEN_HOST", "localhost"),
                   help="GraspGen ZMQ server (GraspGen-couliglig)")
    p.add_argument("--graspgen-port", type=int, default=int(os.getenv("GRASPGEN_PORT", "5556")))
    p.add_argument("--graspgen-timeout-ms", type=int,
                   default=int(os.getenv("GRASPGEN_TIMEOUT_MS", "60000")))
    p.add_argument("--device", default=os.getenv("SAM3_DEVICE", "cuda"), help="cuda / cpu")
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   help="Load SAM3 in fp32 (about twice the memory)")
    p.set_defaults(fp16=_env_flag("SAM3_FP16", "1"))
    p.add_argument("--threshold", type=float, default=float(os.getenv("SAM3_THRESHOLD", "0.5")),
                   help="Instance score threshold")
    p.add_argument("--mask-threshold", type=float,
                   default=float(os.getenv("SAM3_MASK_THRESHOLD", "0.5")),
                   help="Per-mask binarization threshold")
    p.add_argument("--top-n", type=int, default=int(os.getenv("SAM3_TOP_N", "20")),
                   help="Ranked candidate grasps to return (the robot picks the first reachable)")
    p.add_argument("--pose-mode", choices=("auto", "manual"),
                   default=os.getenv("SAM3_POSE_MODE", "auto"),
                   help="'auto' asks GraspGen; 'manual' returns one top-down heuristic pose")
    p.add_argument("--visualize", action="store_true", default=_env_flag("SAM3_VISUALIZE", "0"),
                   help="Write the segmentation overlay to --vis-path each request")
    p.add_argument("--vis-path", default=os.getenv("SAM3_VIS_PATH", "/tmp/sam3_seg.png"))
    p.add_argument("--dump-cloud", default=os.getenv("SAM3_DUMP_CLOUD", ""),
                   help="Save the cleaned cloud sent to GraspGen to this .npy (debug)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args(argv)

    from sam3_server.segmenter import Sam3Segmenter   # torch: only when actually serving

    segmenter = Sam3Segmenter(
        model_id=args.model_id,
        device=args.device,
        fp16=args.fp16,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
    )
    graspgen = None
    if args.pose_mode == "auto":
        graspgen = ZmqClient(args.graspgen_host, args.graspgen_port, args.graspgen_timeout_ms)
        logging.getLogger(__name__).info("GraspGen -> %s", graspgen.address)
    SAM3Server(
        segmenter,
        graspgen,
        host=args.host,
        port=args.port,
        pose_mode=args.pose_mode,
        top_n=args.top_n,
        visualize=args.visualize,
        vis_path=args.vis_path,
        dump_cloud=args.dump_cloud,
    ).serve_forever()


if __name__ == "__main__":
    main()
