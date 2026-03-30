# Custom Hand Gesture Recognition

Real-time custom hand gesture recognition in Python using Google MediaPipe hand landmarks and OpenCV.

## Demo

- Live demo: [Try it in your browser](https://tirthpatel-au.github.io/custom-hand-gesture-recognition/)
- Repository: [custom-hand-gesture-recognition](https://github.com/tirthpatel-au/custom-hand-gesture-recognition)
- Local Python demo: `python gesture_recognizer.py --mirror`

The live demo is powered by a browser-based MediaPipe version in [`docs/`](docs). After GitHub Pages is enabled for the repository, pushes to `main` redeploy it automatically through [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml).
## Features

- Real-time hand landmark detection from webcam or video
- Gesture labeling with a lightweight rule-based classifier
- Bounding box, handedness label, confidence, and landmark overlay
- Example gestures:
  - Open Palm
  - Fist
  - Thumbs Up
  - Peace
  - Rock
  - Okay
  - Pointing Up
- Automatic download of the official MediaPipe hand landmarker model on first run
- Works with Python 3.13 using the newer MediaPipe Tasks API

## Tech Stack

- Python
- OpenCV
- MediaPipe Tasks Vision API
- NumPy

## Project Structure

```text
.
|-- .github/
|   `-- workflows/
|       `-- deploy-pages.yml
|-- docs/
|   |-- app.js
|   |-- index.html
|   `-- styles.css
|-- gesture_recognizer.py
|-- requirements.txt
|-- LICENSE
|-- .gitignore
```

## Requirements

- Windows, macOS, or Linux
- Python 3.11+ installed and available on PATH
- Webcam for live testing

## Installation

From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Usage

Run with your webcam:

```powershell
python gesture_recognizer.py --mirror
```

Try the browser version:

- Open [the live demo](https://tirthpatel-au.github.io/custom-hand-gesture-recognition/)
- Click `Start Camera`
- Allow webcam access
- Test gestures directly in the browser

Run on a video file:

```powershell
python gesture_recognizer.py --source path\to\video.mp4
```

Track more than one hand:

```powershell
python gesture_recognizer.py --mirror --max-hands 2
```

## Command Line Options

- `--source`: camera index or video file path, default is `0`
- `--max-hands`: maximum number of hands to detect, default is `1`
- `--mirror`: flips the preview horizontally for a selfie-camera view

## How It Works

1. OpenCV reads frames from a webcam or video file.
2. MediaPipe Hand Landmarker detects 21 landmarks for each hand.
3. Landmark geometry is converted into raised-finger states.
4. A custom rule-based classifier maps those states to gesture names.
5. The app draws the result directly on the video frame in real time.

## First Run Behavior

On first run, the script downloads Google's official hand landmark model file:

- `hand_landmarker.task`

The file is stored locally in the project folder and is ignored by git.

The browser demo does not download this local file into the repo. It loads the official MediaPipe web model from Google's hosted model storage at runtime.

## Troubleshooting

If PowerShell blocks virtual environment activation:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
```

If `python` is not recognized:

- Reinstall Python and enable `Add Python to PATH`

If your hand is detected but the label is occasionally wrong:

- Keep the full hand inside the frame
- Use better lighting
- Face your palm more directly toward the camera
- Move more slowly for thumb-heavy gestures

## Future Improvements

- Add temporal smoothing to reduce flicker
- Add a training mode for custom gestures
- Save landmark datasets for ML classification
- Trigger desktop shortcuts or app actions from gestures
- Add a recorded preview GIF alongside the live demo link

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
