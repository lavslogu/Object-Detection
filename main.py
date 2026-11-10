"""Application entry point. Missing requirements are installed on first run."""

import argparse
import importlib.metadata
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_requirements() -> None:
    """Install declared packages when absent, then continue with this interpreter."""
    requirements = ROOT / "requirements.txt"
    missing = []
    for package, module in (("ultralytics", "ultralytics"),
                            ("opencv-python", "cv2")):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
        else:
            # Importability can differ from distribution metadata in broken installs.
            try:
                __import__(module)
            except ImportError:
                missing.append(package)
    if missing:
        print(f"Installing missing project requirements: {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live YOLO webcam object detection")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", "yolo26n.pt"),
                        help="Ultralytics model/checkpoint (default: yolo26n.pt)")
    parser.add_argument("--confidence", type=float, default=0.35,
                        help="Minimum detection confidence from 0 to 1")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between saved frames (default: 1)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.confidence <= 1:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.interval <= 0:
        raise SystemExit("--interval must be greater than zero")
    try:
        ensure_requirements()
    except subprocess.CalledProcessError as exc:
        print(f"Dependency installation failed (exit code {exc.returncode}).", file=sys.stderr)
        return exc.returncode or 1

    from src.camera.detector import CameraDetector
    from src.logging.setup import configure_logging

    logger = configure_logging(ROOT / "src" / "logging" / "logs")
    logger.info("Application starting; python=%s platform=%s", sys.version, sys.platform)
    logger.info("Configuration: camera=%d model=%s confidence=%.2f interval=%.2fs",
                args.camera, args.model, args.confidence, args.interval)
    try:
        CameraDetector(args.model, args.camera, args.confidence, args.interval,
                       ROOT / "src" / "storage", logger).run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Application failed")
        return 1
    logger.info("Application stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
