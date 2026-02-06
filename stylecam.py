#!/usr/bin/env python3
"""
stylecam - restyle your webcam in real time via fal.ai, output to OBS Virtual Camera
"""

import asyncio
import base64
import os
import time
from io import BytesIO
from typing import Optional

import cv2
import msgpack
import numpy as np
import pyvirtualcam
import websockets
from PIL import Image

# fal.ai auth via FAL_KEY env var
FAL_KEY = os.environ.get("FAL_KEY")
if not FAL_KEY:
    raise RuntimeError("FAL_KEY environment variable is required")

FAL_APP = "fal-ai/flux-2/klein/realtime"
FAL_REST_URL = "https://rest.alpha.fal.ai"
TOKEN_EXPIRATION_SECONDS = 10

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS = 30

# Processing settings
DEFAULT_PROMPT = "anime style"
PROMPT_FILE = "prompt.txt"  # Update this file to change the prompt live
JPEG_QUALITY = 50  # Lower = faster upload
NUM_STEPS = 2
SEED = 42


class KleinVirtualCamera:
    def __init__(self):
        self.webcam: Optional[cv2.VideoCapture] = None
        self.virtual_cam: Optional[pyvirtualcam.Camera] = None
        self.websocket = None
        self.token: Optional[str] = None
        self.running = False

        self.frame_id = 0
        self.pending_frames: dict[str, float] = {}  # request_id -> send_time
        self.latest_processed_frame: Optional[np.ndarray] = None

        # Prompt management
        self.prompt = DEFAULT_PROMPT
        self.prompt_file_mtime = 0.0
        self._init_prompt_file()

        # Stats
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0.0
        self.latency_ms = 0.0

    def _init_prompt_file(self):
        """Create prompt file if it doesn't exist"""
        import os
        prompt_path = os.path.join(os.path.dirname(__file__) or ".", PROMPT_FILE)
        self.prompt_path = prompt_path
        if not os.path.exists(prompt_path):
            with open(prompt_path, "w") as f:
                f.write(self.prompt)
            print(f"Created {PROMPT_FILE} - edit this file to change the prompt live!")
        else:
            self._load_prompt()

    def _load_prompt(self):
        """Load prompt from file"""
        import os
        try:
            mtime = os.path.getmtime(self.prompt_path)
            if mtime > self.prompt_file_mtime:
                with open(self.prompt_path, "r") as f:
                    new_prompt = f.read().strip()
                if new_prompt and new_prompt != self.prompt:
                    self.prompt = new_prompt
                    print(f"\n[PROMPT] Updated to: {self.prompt}")
                self.prompt_file_mtime = mtime
        except Exception as e:
            pass  # Silently ignore file read errors

    async def fetch_token(self) -> str:
        """Fetch a short-lived JWT token from the fal.ai REST API"""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{FAL_REST_URL}/tokens/realtime",
                json={
                    "app": FAL_APP,
                    "token_expiration": TOKEN_EXPIRATION_SECONDS,
                },
                headers={
                    "Authorization": f"Key {FAL_KEY}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if resp.status != 201:
                    body = await resp.text()
                    raise RuntimeError(f"Failed to fetch token: {resp.status} {body}")
                return await resp.text()

    async def _token_refresh_loop(self):
        """Refresh the JWT token before it expires and reconnect the WebSocket"""
        refresh_interval = TOKEN_EXPIRATION_SECONDS * 0.9
        while self.running:
            await asyncio.sleep(refresh_interval)
            if not self.running:
                break
            try:
                self.token = await self.fetch_token()
                # Reconnect with new token
                if self.websocket:
                    await self.websocket.close()
                ws_url = self._build_ws_url()
                self.websocket = await websockets.connect(
                    ws_url, max_size=10 * 768 *768 
                )
                print(f"\n[TOKEN] Refreshed token and reconnected")
            except Exception as e:
                print(f"\n[TOKEN] Failed to refresh: {e}")

    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with the current JWT token"""
        return f"wss://fal.run/{FAL_APP}?fal_jwt_token={self.token}"

    async def connect_websocket(self):
        """Connect to fal.ai Klein realtime WebSocket"""
        print(f"Connecting to fal.ai Klein realtime...")
        print(f"App: {FAL_APP}")

        self.token = await self.fetch_token()
        ws_url = self._build_ws_url()
        self.websocket = await websockets.connect(
            ws_url,
            max_size=10 * 1024 * 1024  # 10MB max message size
        )
        print("Connected!")

    def setup_webcam(self):
        """Initialize webcam capture"""
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.webcam.set(cv2.CAP_PROP_FPS, FPS)

        if not self.webcam.isOpened():
            raise RuntimeError("Could not open webcam")

        print(f"Webcam initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {FPS}fps")

    def setup_virtual_camera(self):
        """Initialize OBS virtual camera output"""
        print(f"Initializing virtual camera: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {FPS}fps")
        try:
            self.virtual_cam = pyvirtualcam.Camera(
                width=CAMERA_WIDTH,
                height=CAMERA_HEIGHT,
                fps=FPS,
                backend='obs',
                print_fps=True  # Debug: print actual output FPS
            )
            print(f"Virtual camera started: {self.virtual_cam.device}")
            print(f"Virtual camera format: {self.virtual_cam.fmt}")
        except Exception as e:
            print(f"Failed to create virtual camera: {e}")
            raise

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from webcam"""
        ret, frame = self.webcam.read()
        if not ret:
            return None
        return frame

    def frame_to_jpeg_bytes(self, frame: np.ndarray) -> bytes:
        """Convert OpenCV frame to JPEG bytes"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Encode as JPEG
        img = Image.fromarray(rgb_frame)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=JPEG_QUALITY)

        return buffer.getvalue()

    def bytes_to_frame(self, img_bytes: bytes) -> Optional[np.ndarray]:
        """Convert JPEG bytes to OpenCV frame (RGB)"""
        try:
            img = Image.open(BytesIO(img_bytes))
            frame = np.array(img)

            # Resize if needed
            if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))

            return frame
        except Exception as e:
            print(f"Error decoding frame: {e}")
            return None

    async def send_frame(self, frame: np.ndarray):
        """Send frame to fal.ai Klein"""
        if self.websocket is None:
            return

        # Check for prompt updates
        self._load_prompt()

        self.frame_id += 1
        request_id = f"req_{self.frame_id}"

        # Convert to JPEG bytes, then base64 data URI (runner expects image_url)
        image_bytes = self.frame_to_jpeg_bytes(frame)
        image_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()

        request = {
            "prompt": self.prompt,
            "image_url": image_url,
            "image_size": "square",
            "num_inference_steps": NUM_STEPS,
            "request_id": request_id,
            "seed": SEED,
            "output_feedback_strength": 1.0,
            "schedule_mu": 2.3,
        }

        # Track send time
        self.pending_frames[request_id] = time.time()

        try:
            # Send as msgpack (like the working HTML)
            packed = msgpack.packb(request)
            await self.websocket.send(packed)
        except Exception as e:
            print(f"Send error: {e}")
            self.pending_frames.pop(request_id, None)

    async def receive_frames(self):
        """Receive processed frames from fal.ai Klein"""
        frame_count = 0
        debug_responses = 0  # Count responses for detailed debug
        while self.running:
            try:
                data = await self.websocket.recv()

                # Decode msgpack response
                if isinstance(data, bytes):
                    response = msgpack.unpackb(data, raw=False)
                else:
                    import json
                    response = json.loads(data)

                # Debug: print all keys and detailed structure for first few responses
                debug_responses += 1
                print(f"\n[DEBUG #{debug_responses}] Response keys: {list(response.keys())}")

                if debug_responses <= 3:
                    # Show detailed structure for first 3 responses
                    for key, value in response.items():
                        if isinstance(value, bytes):
                            print(f"  {key}: <bytes len={len(value)}>")
                        elif isinstance(value, dict):
                            print(f"  {key}: dict with keys {list(value.keys())}")
                        elif isinstance(value, list):
                            print(f"  {key}: list with {len(value)} items")
                            if len(value) > 0:
                                first = value[0]
                                if isinstance(first, bytes):
                                    print(f"    [0]: <bytes len={len(first)}>")
                                elif isinstance(first, dict):
                                    print(f"    [0]: dict with keys {list(first.keys())}")
                        else:
                            print(f"  {key}: {type(value).__name__} = {repr(value)[:100]}")

                # Check for errors
                if "error" in response:
                    print(f"Server error: {response['error']}")
                    continue

                # Get request ID for latency tracking
                request_id = response.get("request_id")
                if request_id and request_id in self.pending_frames:
                    send_time = self.pending_frames.pop(request_id)
                    self.latency_ms = (time.time() - send_time) * 1000

                # Try to get image data from various possible keys
                # fal.ai realtime typically returns image_bytes directly
                img_bytes = None

                # Check for direct image_bytes (most likely for realtime API)
                if "image_bytes" in response:
                    img_bytes = response["image_bytes"]
                    if isinstance(img_bytes, bytes):
                        print(f"[DEBUG] Found image_bytes: {len(img_bytes)} bytes")

                # Check for images array (common format)
                if img_bytes is None and "images" in response:
                    images = response["images"]
                    if isinstance(images, list) and len(images) > 0:
                        first_img = images[0]
                        if isinstance(first_img, bytes):
                            img_bytes = first_img
                        elif isinstance(first_img, dict):
                            # Could have content, url, or data key
                            content = first_img.get("content") or first_img.get("data")
                            if isinstance(content, bytes):
                                img_bytes = content
                            elif isinstance(content, str):
                                import base64
                                if content.startswith("data:"):
                                    content = content.split(",", 1)[1]
                                img_bytes = base64.b64decode(content)
                            # Check for URL that we'd need to fetch
                            elif "url" in first_img:
                                print(f"[DEBUG] Image is URL: {first_img['url'][:80]}...")

                # Check for single image key
                if img_bytes is None and "image" in response:
                    image_data = response["image"]
                    if isinstance(image_data, bytes):
                        img_bytes = image_data
                    elif isinstance(image_data, dict):
                        content = image_data.get("content") or image_data.get("data")
                        if isinstance(content, bytes):
                            img_bytes = content

                # Check for output key
                if img_bytes is None and "output" in response:
                    output = response["output"]
                    if isinstance(output, bytes):
                        img_bytes = output

                if img_bytes:
                    print(f"[DEBUG] Processing {len(img_bytes)} bytes of image data")
                    # Decode to frame
                    frame = self.bytes_to_frame(img_bytes)
                    if frame is not None:
                        frame_count += 1
                        self.latest_processed_frame = frame
                        print(f"[DEBUG] SUCCESS: Decoded frame #{frame_count}: {frame.shape}")
                    else:
                        print("[DEBUG] FAILED to decode frame from bytes")
                else:
                    print("[DEBUG] No image bytes found in response")

                # Log timings if available
                timings = response.get("timings")
                if timings:
                    total = timings.get("total", 0)
                    gpu_id = timings.get("gpu_id", "?")
                    print(f"[timing] GPU{gpu_id} total={total*1000:.0f}ms pending={len(self.pending_frames)}")

            except websockets.ConnectionClosed as e:
                print(f"\nWebSocket connection closed: {e}")
                break
            except Exception as e:
                print(f"\nError receiving frame: {e}")
                import traceback
                traceback.print_exc()

    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        elapsed = time.time() - self.fps_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_time = time.time()

    async def run(self):
        """Main loop"""
        self.running = True

        # Setup
        self.setup_webcam()
        self.setup_virtual_camera()
        await self.connect_websocket()

        # Start receive and token refresh tasks
        receive_task = asyncio.create_task(self.receive_frames())
        token_refresh_task = asyncio.create_task(self._token_refresh_loop())

        print("\n" + "="*50)
        print("Klein Virtual Camera running!")
        print(f"Prompt: {self.prompt}")
        print(f"Edit '{PROMPT_FILE}' to change the prompt live!")
        print("Select 'OBS Virtual Camera' in your video app")
        print("Press Ctrl+C to stop")
        print("="*50 + "\n")

        # Send at ~10fps to avoid overwhelming the API
        frame_interval = 1.0 / 10
        last_frame_time = time.time()
        max_pending = 4  # Don't queue too many requests

        try:
            while self.running:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                # Send to Klein (throttled)
                now = time.time()
                if now - last_frame_time >= frame_interval:
                    # Only send if not too many pending
                    if len(self.pending_frames) < max_pending:
                        await self.send_frame(frame)
                        last_frame_time = now

                # Output to virtual camera
                output_frame = self.latest_processed_frame
                if output_frame is None:
                    # No processed frame yet, show original (converted to RGB)
                    output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Debug: log frame details periodically
                if self.fps_counter == 0:
                    print(f"\n[VCAM] Sending frame: shape={output_frame.shape}, dtype={output_frame.dtype}, range=[{output_frame.min()}-{output_frame.max()}]")

                self.virtual_cam.send(output_frame)
                self.virtual_cam.sleep_until_next_frame()

                # Update stats
                self.update_fps()

                # Print stats occasionally
                if self.fps_counter == 1:
                    print(f"\rFPS: {self.current_fps:.1f} | Latency: {self.latency_ms:.0f}ms | Pending: {len(self.pending_frames)}   ", end="", flush=True)

                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.running = False
            receive_task.cancel()
            token_refresh_task.cancel()

            if self.websocket:
                await self.websocket.close()
            if self.webcam:
                self.webcam.release()

            print("Stopped.")


def main():
    print("Klein Virtual Camera")
    print("====================\n")

    camera = KleinVirtualCamera()
    asyncio.run(camera.run())


if __name__ == "__main__":
    main()
