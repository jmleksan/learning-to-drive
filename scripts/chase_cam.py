"""Film the ego car from behind while another script drives it.

Run a drive script in one terminal and this in another. It waits for the
vehicle named "hero" (see utils/vehicles.spawn_vehicle), attaches a camera
behind it, and saves every Nth frame as a JPEG. It never ticks the world, so
the drive script keeps control of time. If the car is respawned, as in RL
episodes, the camera follows the new one.

  python -m scripts.chase_cam record data/clips/step6 --seconds 120
  python -m scripts.chase_cam gif data/clips/step6 docs/images/step6_drive.gif --start 20 --seconds 15
"""
from __future__ import annotations

import argparse
import csv
import queue
import time
from pathlib import Path

import carla
from PIL import Image, ImageDraw

from utils.connection import connect_client

WIDTH, HEIGHT, FOV = 640, 360, 100.0
CHASE = carla.Transform(carla.Location(x=-7.5, z=3.8), carla.Rotation(pitch=-6.0))
SIM_FPS = 20


def _find_hero(world: carla.World) -> carla.Actor | None:
    heroes = [a for a in world.get_actors().filter("vehicle.*") if a.attributes.get("role_name") == "hero"]
    return max(heroes, key=lambda a: a.id) if heroes else None


def record(out: Path, seconds: float, every: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    client = connect_client()
    world = None
    frames: queue.Queue[carla.Image] = queue.Queue()
    camera = None
    hero_id = None
    first_frame = None
    saved = 0
    print("[chase] waiting for the hero vehicle")
    try:
        while True:
            if hero_id is None:
                # The driver may load a new map first, which replaces the world.
                try:
                    world = client.get_world()
                    hero = _find_hero(world)
                except RuntimeError:
                    time.sleep(2.0)
                    continue
                if hero is None:
                    time.sleep(0.5)
                    continue
            else:
                hero = _find_hero(world)
            if hero is not None and hero.id != hero_id:
                if camera is not None and camera.is_alive:
                    camera.stop()
                    camera.destroy()
                bp = world.get_blueprint_library().find("sensor.camera.rgb")
                bp.set_attribute("image_size_x", str(WIDTH))
                bp.set_attribute("image_size_y", str(HEIGHT))
                bp.set_attribute("fov", str(FOV))
                camera = world.spawn_actor(bp, CHASE, attach_to=hero, attachment_type=carla.AttachmentType.SpringArmGhost)
                camera.listen(frames.put)
                hero_id = hero.id
                print(f"[chase] following vehicle {hero_id}")
            try:
                image = frames.get(timeout=1.0)
            except queue.Empty:
                if hero_id is not None and _find_hero(world) is None and first_frame is not None:
                    print("[chase] hero gone, stopping")
                    break
                continue
            if first_frame is None:
                first_frame = image.frame
            elapsed = (image.frame - first_frame) / SIM_FPS
            if elapsed > seconds:
                break
            if (image.frame - first_frame) % every:
                continue
            rgb = Image.frombuffer("RGBA", (image.width, image.height), bytes(image.raw_data)).convert("RGB")
            b, g, r = rgb.split()
            Image.merge("RGB", (r, g, b)).save(out / f"{image.frame:08d}.jpg", quality=88)
            saved += 1
    except KeyboardInterrupt:
        pass
    finally:
        if camera is not None and camera.is_alive:
            camera.stop()
            camera.destroy()
        time.sleep(0.2)
    print(f"[chase] saved {saved} frames to {out}")


def _load_decisions(log: Path) -> dict[int, str]:
    """pipeline.drive's drive_log.csv -> caption per drive frame."""
    with log.open(newline="") as fh:
        return {
            int(r["frame"]): f"{r['behavior']}  {r['reason']}  {float(r['speed_kmh']):.0f} km/h"
            for r in csv.DictReader(fh)
        }


def _caption(image: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(image)
    box = draw.textbbox((6, 6), text)
    draw.rectangle((box[0] - 4, box[1] - 3, box[2] + 4, box[3] + 3), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))


def make_gif(
    frames_dir: Path, out: Path, start: float, seconds: float, fps: int, width: int, speed: float,
    log: Path | None = None, log_offset: float = 0.0,
) -> None:
    files = sorted(frames_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"no frames in {frames_dir}")
    stamps = [int(f.stem) for f in files]
    t0 = stamps[0]
    picked, next_t = [], start
    for f, s in zip(files, stamps):
        t = (s - t0) / SIM_FPS
        if t < next_t:
            continue
        if t > start + seconds:
            break
        picked.append(f)
        next_t += speed / fps
    images = [Image.open(f).convert("RGB") for f in picked]
    height = round(images[0].height * width / images[0].width)
    images = [im.resize((width, height), Image.LANCZOS) for im in images]
    if log is not None:
        decisions = _load_decisions(log)
        for f, im in zip(picked, images):
            t = (int(f.stem) - t0) / SIM_FPS + log_offset
            text = decisions.get(round(t * SIM_FPS))
            if text:
                _caption(im, text)
    palette = images[len(images) // 2].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0, optimize=True)
    print(f"[gif] {len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="save chase-camera frames while another script drives")
    rec.add_argument("out", type=Path)
    rec.add_argument("--seconds", type=float, default=120.0, help="sim seconds to record")
    rec.add_argument("--every", type=int, default=2, help="keep every Nth sim frame")
    gif = sub.add_parser("gif", help="turn saved frames into a GIF")
    gif.add_argument("frames", type=Path)
    gif.add_argument("out", type=Path)
    gif.add_argument("--start", type=float, default=0.0, help="sim seconds into the recording")
    gif.add_argument("--seconds", type=float, default=15.0, help="sim seconds of driving to include")
    gif.add_argument("--fps", type=int, default=8)
    gif.add_argument("--width", type=int, default=360)
    gif.add_argument("--speed", type=float, default=1.0, help="playback speed, 2 = twice real time")
    gif.add_argument("--log", type=Path, help="pipeline.drive drive_log.csv: caption each frame with its decision")
    gif.add_argument("--log-offset", type=float, default=0.0, help="drive-log seconds minus recording seconds")
    args = parser.parse_args()
    if args.cmd == "record":
        record(args.out, args.seconds, args.every)
    else:
        make_gif(args.frames, args.out, args.start, args.seconds, args.fps, args.width, args.speed, args.log, args.log_offset)


if __name__ == "__main__":
    main()
