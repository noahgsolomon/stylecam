#!/usr/bin/env python3
"""
stylecam - restyle your webcam in real time via fal.ai, output to OBS Virtual Camera
"""

import asyncio
import base64
import json
import os
import time
from io import BytesIO
from typing import Optional

import aiohttp
import cv2
import msgpack
import numpy as np
import pyvirtualcam
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# fal.ai auth — FAL_KEY is "key_id:key_secret"
FAL_KEY = os.environ.get("FAL_KEY")
if not FAL_KEY or ":" not in FAL_KEY:
    raise RuntimeError("FAL_KEY is required (format: key_id:key_secret). Set it in .env or export it.")
FAL_KEY_ID, FAL_KEY_SECRET = FAL_KEY.split(":", 1)

FAL_APP = "fal-ai/flux-2/klein/realtime"

# Camera settings
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FPS = 30

# Processing settings
DEFAULT_PROMPT = "anime style"
PROMPT_FILE = "prompt.txt"
JPEG_QUALITY = 50
NUM_STEPS = 2
SEED = 42


class StyleCam:
    def __init__(self):
        self.webcam: Optional[cv2.VideoCapture] = None
        self.virtual_cam: Optional[pyvirtualcam.Camera] = None
        self.websocket = None
        self.running = False
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_lock = asyncio.Lock()
        self._ws_generation = 0

        self.frame_id = 0
        self.pending_frames: dict[str, float] = {}
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
        prompt_path = os.path.join(os.path.dirname(__file__) or ".", PROMPT_FILE)
        self.prompt_path = prompt_path
        if not os.path.exists(prompt_path):
            with open(prompt_path, "w") as f:
                f.write(self.prompt)
            print(f"Created {PROMPT_FILE} - edit this file to change the prompt live!")
        else:
            self._load_prompt()

    def _load_prompt(self):
        try:
            mtime = os.path.getmtime(self.prompt_path)
            if mtime > self.prompt_file_mtime:
                with open(self.prompt_path, "r") as f:
                    new_prompt = f.read().strip()
                if new_prompt and new_prompt != self.prompt:
                    self.prompt = new_prompt
                    print(f"\n[PROMPT] Updated to: {self.prompt}")
                self.prompt_file_mtime = mtime
        except Exception:
            pass

    def _build_ws_url(self) -> str:
        return (
            f"wss://fal.run/{FAL_APP}"
            f"?fal_key_id={FAL_KEY_ID}&fal_key_secret={FAL_KEY_SECRET}"
        )

    async def _connect(self):
        """Establish or re-establish the WebSocket connection."""
        async with self._ws_lock:
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()

            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()

            ws_url = self._build_ws_url()
            self.websocket = await self._session.ws_connect(
                ws_url, compress=False, max_msg_size=10 * 1024 * 1024,
            )
            self._ws_generation += 1
            self.pending_frames.clear()
            print(f"[WS] Connected (gen={self._ws_generation})")

    async def _reconnect(self, reason: str = ""):
        """Reconnect with backoff."""
        delay = 1.0
        max_delay = 10.0
        while self.running:
            print(f"[WS] Reconnecting ({reason})...")
            try:
                await self._connect()
                return
            except Exception as e:
                print(f"[WS] Reconnect failed: {e}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                delay = min(delay * 1.5, max_delay)

    def setup_webcam(self):
        self.webcam = cv2.VideoCapture(0)
        self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.webcam.set(cv2.CAP_PROP_FPS, FPS)
        if not self.webcam.isOpened():
            raise RuntimeError("Could not open webcam")
        print(f"Webcam: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {FPS}fps")

    def setup_virtual_camera(self):
        self.virtual_cam = pyvirtualcam.Camera(
            width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=FPS, backend='obs',
        )
        print(f"Virtual camera: {self.virtual_cam.device}")

    def capture_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.webcam.read()
        return frame if ret else None

    def frame_to_jpeg_bytes(self, frame: np.ndarray) -> bytes:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=JPEG_QUALITY)
        return buffer.getvalue()

    def bytes_to_frame(self, img_bytes: bytes) -> Optional[np.ndarray]:
        try:
            img = Image.open(BytesIO(img_bytes))
            frame = np.array(img)
            if frame.shape[1] != CAMERA_WIDTH or frame.shape[0] != CAMERA_HEIGHT:
                frame = cv2.resize(frame, (CAMERA_WIDTH, CAMERA_HEIGHT))
            return frame
        except Exception as e:
            print(f"Error decoding frame: {e}")
            return None

    async def send_frame(self, frame: np.ndarray):
        ws = self.websocket
        if ws is None or ws.closed:
            return

        self._load_prompt()
        self.frame_id += 1
        request_id = f"req_{self.frame_id}"

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

        self.pending_frames[request_id] = time.time()
        try:
            packed = msgpack.packb(request)
            await ws.send_bytes(packed)
        except Exception as e:
            print(f"Send error: {e}")
            self.pending_frames.pop(request_id, None)

    def _extract_image_bytes(self, response: dict) -> Optional[bytes]:
        """Extract image bytes from response."""
        # image dict with content (like demo expects)
        if "image" in response:
            v = response["image"]
            if isinstance(v, bytes):
                return v
            if isinstance(v, dict):
                content = v.get("content") or v.get("data")
                if isinstance(content, bytes):
                    return content

        # images array (RawImage list)
        if "images" in response:
            images = response["images"]
            if isinstance(images, list) and len(images) > 0:
                first = images[0]
                if isinstance(first, bytes):
                    return first
                if isinstance(first, dict):
                    content = first.get("content") or first.get("data")
                    if isinstance(content, bytes):
                        return content

        # Direct image_bytes
        if "image_bytes" in response:
            v = response["image_bytes"]
            if isinstance(v, bytes):
                return v

        # output key
        if "output" in response:
            v = response["output"]
            if isinstance(v, bytes):
                return v

        return None

    async def receive_loop(self):
        """Receive frames from fal.ai. Reconnects on disconnect."""
        frame_count = 0
        debug_responses = 0

        while self.running:
            ws = self.websocket
            if ws is None or ws.closed:
                await asyncio.sleep(0.1)
                continue

            gen = self._ws_generation

            try:
                msg = await ws.receive(timeout=30)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"\n[WS] Receive error: {e}")
                if gen == self._ws_generation:
                    await self._reconnect(f"receive error: {e}")
                continue

            if gen != self._ws_generation:
                continue

            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.ERROR):
                print(f"\n[WS] Connection lost: type={msg.type} data={msg.data}")
                if gen == self._ws_generation:
                    await self._reconnect(f"close type={msg.type}")
                continue

            # Decode response
            data = msg.data
            try:
                if isinstance(data, bytes):
                    response = msgpack.unpackb(data, raw=False)
                else:
                    response = json.loads(data)
            except Exception as e:
                print(f"[WS] Decode error: {e}")
                continue

            debug_responses += 1
            if debug_responses <= 5:
                print(f"\n[RECV #{debug_responses}] keys={list(response.keys())}")
                for key, value in response.items():
                    if isinstance(value, bytes):
                        print(f"  {key}: <bytes len={len(value)}>")
                    elif isinstance(value, (dict, list)):
                        print(f"  {key}: {type(value).__name__} len={len(value)}")
                    else:
                        print(f"  {key}: {repr(value)[:80]}")

            if "error" in response:
                print(f"[WS] Server error: {response['error']}")
                continue

            # Latency tracking
            request_id = response.get("request_id")
            if request_id and request_id in self.pending_frames:
                send_time = self.pending_frames.pop(request_id)
                self.latency_ms = (time.time() - send_time) * 1000

            # Extract and decode image
            img_bytes = self._extract_image_bytes(response)
            if img_bytes:
                frame = self.bytes_to_frame(img_bytes)
                if frame is not None:
                    frame_count += 1
                    self.latest_processed_frame = frame
                    if frame_count <= 3 or frame_count % 30 == 0:
                        print(f"[OK] Frame #{frame_count} latency={self.latency_ms:.0f}ms")

    def update_fps(self):
        self.fps_counter += 1
        elapsed = time.time() - self.fps_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_counter / elapsed
            self.fps_counter = 0
            self.fps_time = time.time()

    async def run(self):
        self.running = True

        self.setup_webcam()
        self.setup_virtual_camera()
        await self._connect()

        receive_task = asyncio.create_task(self.receive_loop())

        print("\n" + "=" * 50)
        print("stylecam running!")
        print(f"Prompt: {self.prompt}")
        print(f"Edit '{PROMPT_FILE}' to change the prompt live!")
        print("Select 'OBS Virtual Camera' in your video app")
        print("Press Ctrl+C to stop")
        print("=" * 50 + "\n")

        send_interval = 1.0 / 15  # 15fps, backend handles buffering
        last_send_time = time.time()

        try:
            while self.running:
                frame = self.capture_frame()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                now = time.time()
                if now - last_send_time >= send_interval:
                    await self.send_frame(frame)
                    last_send_time = now

                # Output to virtual camera
                output_frame = self.latest_processed_frame
                if output_frame is None:
                    output_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.virtual_cam.send(output_frame)
                self.virtual_cam.sleep_until_next_frame()

                self.update_fps()
                if self.fps_counter == 1:
                    print(f"\rFPS: {self.current_fps:.1f} | Latency: {self.latency_ms:.0f}ms | Pending: {len(self.pending_frames)}   ", end="", flush=True)

                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.running = False
            receive_task.cancel()
            if self.websocket and not self.websocket.closed:
                await self.websocket.close()
            if self._session and not self._session.closed:
                await self._session.close()
            if self.webcam:
                self.webcam.release()
            print("Stopped.")


def main():
    print("stylecam")
    print("========\n")
    cam = StyleCam()
    asyncio.run(cam.run())


if __name__ == "__main__":
    main()
