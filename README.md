# stylecam

Restyle your webcam in real time using [fal.ai](https://fal.ai) and output to OBS Virtual Camera. Use it as a camera source in Zoom, Google Meet, etc.

## Prerequisites

- macOS
- Python 3.10+
- [OBS](https://obsproject.com/) (for virtual camera output)
- A [fal.ai](https://fal.ai) API key

## Setup

Install OBS if you don't have it:

```
brew install --cask obs
```

Open OBS once, go to **Tools > Start Virtual Camera**, then close it. This registers the virtual camera device.

Then:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Add your fal.ai key to `.env`:

```
FAL_KEY=your-key-here
```

Then run:

```
python stylecam.py
```

Select **OBS Virtual Camera** as the camera in your video app.

## Live prompt

Edit `prompt.txt` while running to change the style on the fly. The file is polled automatically.

## Config

Tunable constants at the top of `stylecam.py`:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | 1280x720 | Webcam capture resolution |
| `FPS` | 30 | Virtual camera output framerate |
| `JPEG_QUALITY` | 50 | JPEG compression for upload (lower = faster) |
| `NUM_STEPS` | 2 | Inference steps (1-8, lower = faster) |
| `SEED` | 42 | Generation seed |
