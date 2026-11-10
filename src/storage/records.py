"""CSV and image persistence for sampled camera frames."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2


class DetectionStore:
    """Store one annotated image and one CSV row per sampled frame."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.frames_dir = root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = root / "detections.csv"
        self._csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=("timestamp", "frame_path", "objects"),
        )
        if self.csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._csv_file.flush()

    def save(self, frame, timestamp: datetime, objects: Iterable[str]) -> Path:
        stamp = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        path = self.frames_dir / f"frame_{stamp}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not write frame to {path}")
        self._writer.writerow({
            "timestamp": timestamp.isoformat(timespec="milliseconds"),
            "frame_path": str(path.relative_to(self.root)),
            "objects": "; ".join(objects),
        })
        self._csv_file.flush()
        return path

    def close(self) -> None:
        self._csv_file.close()
