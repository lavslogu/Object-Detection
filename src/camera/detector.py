"""Run webcam inference with Ultralytics YOLO."""

import logging
import colorsys
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.storage.records import DetectionStore


class CameraDetector:
    def __init__(self, model_name: str, camera_index: int, confidence: float,
                 sample_interval: float, output_dir: Path, logger: logging.Logger):
        self.model_name = model_name
        self.camera_index = camera_index
        self.confidence = confidence
        self.sample_interval = sample_interval
        self.output_dir = output_dir
        self.logger = logger
    @staticmethod
    def _color(index: int) -> tuple[int, int, int]:
        # Golden-angle hue spacing keeps boxes distinct even for many objects.
        hue = (index * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
        return int(blue * 255), int(green * 255), int(red * 255)

    def run(self) -> None:
        self.logger.info("Loading model %s", self.model_name)
        model = YOLO(self.model_name)
        self.logger.info("Model loaded; classes=%d", len(model.names))
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open camera index {self.camera_index}. "
                "Check camera permissions or choose another --camera index."
            )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.logger.info("Camera opened (index=%d)", self.camera_index)
        store = DetectionStore(self.output_dir)
        next_sample = time.monotonic()
        frame_count = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                frame_count += 1
                result = model.predict(frame, conf=self.confidence, verbose=False)[0]
                object_labels: list[str] = []
                if result.boxes is not None:
                    for detection_index, box in enumerate(result.boxes):
                        class_id = int(box.cls[0])
                        name = str(model.names[class_id])
                        score = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        color = self._color(detection_index)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{name} {score:.2f}"
                        (width, height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
                        )
                        top = max(y1, height + baseline + 6)
                        label_x = max(0, x2 - width - 8)
                        cv2.rectangle(frame, (label_x, top - height - baseline - 6),
                                      (label_x + width + 8, top), color, -1)
                        cv2.putText(frame, label, (label_x + 4, top - baseline - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                        object_labels.append(f"{name} ({score:.2f})")

                now = time.monotonic()
                if now >= next_sample:
                    timestamp = datetime.now().astimezone()
                    path = store.save(frame, timestamp, object_labels)
                    self.logger.info("Saved sample %s; detections=%d", path,
                                     len(object_labels))
                    next_sample = now + self.sample_interval

                cv2.imshow("YOLO Live Object Detection  |  Q or Esc to quit", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            store.close()
            capture.release()
            cv2.destroyAllWindows()
            self.logger.info("Camera released after %d frames", frame_count)
