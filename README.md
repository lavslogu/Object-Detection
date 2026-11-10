# Live YOLO Object Detection

A Python webcam application that runs Ultralytics YOLO26 object detection, draws a uniquely colored box and confidence label for every detected object, and saves one annotated frame per second with a timestamp and detections in a CSV file.

## Requirements

- Python 3.10 or newer
- A working webcam and permission to access it
- Internet access on the first run to install missing Python packages and download the model weights

The application uses the current Ultralytics YOLO26 nano checkpoint (`yolo26n.pt`) by default. The first inference downloads the checkpoint automatically. `main.py` checks for the required packages and installs them from `requirements.txt` when missing. A virtual environment is recommended so these dependencies stay isolated.

## Run in VS Code

1. Open this project folder in VS Code.
2. Select a Python 3.10+ interpreter (or create a virtual environment in the integrated terminal).
3. Run:

   ```powershell
   python main.py
   ```

4. Press **Q** or **Esc** in the camera window to stop.

Alternatively, install dependencies yourself with `python -m pip install -r requirements.txt`, then run `python main.py`.

## Options

```text
python main.py --camera 0 --model yolo26n.pt --confidence 0.35 --interval 1
```

- `--camera`: camera device index (default `0`)
- `--model`: Ultralytics model name or local checkpoint path (default `yolo26n.pt`)
- `--confidence`: detection confidence threshold from `0` to `1`
- `--interval`: seconds between saved frames (default `1`)

The model can also be selected with the `YOLO_MODEL` environment variable. If the webcam is unavailable, check OS camera privacy permissions and try another camera index.

## Output

- `src/storage/detections.csv`: timestamp, relative frame path, and semicolon-separated detected class names with confidence scores. Rows append across runs.
- `src/storage/frames/`: annotated JPEG images corresponding to CSV rows.
- `src/logging/logs/application.log`: technical startup configuration, model and camera status, saved samples, shutdown, and error tracebacks. Logs rotate automatically.

Generated frames, logs, model weights, Python caches, and virtual environments are excluded from Git by `.gitignore`.

## Notes

- Video frames are processed live; a sample is saved immediately and then at most once per configured interval. Processing delays can make actual samples less frequent than one per second.
- Bounding boxes use distinct stable colors per detected class for the duration of a run.
- A compatible Ultralytics/PyTorch installation is needed for your hardware. CPU inference works, but can be slower than a supported GPU.
