import sys
import json
import subprocess
from pathlib import Path

import cv2
import torch


def format_duration(seconds: float) -> str:
    """Convert seconds into HH:MM:SS."""
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"


def get_video_info(video_path: Path) -> dict:
    """
    Read video/audio metadata using FFprobe.
    """

    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)

    video_stream = None
    audio_stream = None

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream

        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise RuntimeError("No video stream was found.")

    duration = float(
        data.get("format", {}).get(
            "duration",
            video_stream.get("duration", 0)
        )
    )

    fps_text = video_stream.get("avg_frame_rate", "0/1")

    try:
        numerator, denominator = fps_text.split("/")
        fps = (
            float(numerator) / float(denominator)
            if float(denominator) != 0
            else 0
        )
    except (ValueError, ZeroDivisionError):
        fps = 0

    return {
        "file_name": video_path.name,
        "file_path": str(video_path.resolve()),
        "duration_seconds": duration,
        "duration": format_duration(duration),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": round(fps, 3),
        "video_codec": video_stream.get("codec_name", "unknown"),
        "has_audio": audio_stream is not None,
        "audio_codec": (
            audio_stream.get("codec_name", "unknown")
            if audio_stream
            else None
        ),
    }


def test_opencv(video_path: Path) -> bool:
    """Verify that OpenCV can decode the selected recording."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        return False

    success, frame = capture.read()
    capture.release()

    return success and frame is not None


def print_system_info():
    print("=" * 60)
    print("STREAM HIGHLIGHT FINDER")
    print("=" * 60)

    print("\nAI SYSTEM")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"VRAM: {memory:.2f} GB")

    else:
        print("GPU: CPU MODE")


def main():

    print_system_info()

    if len(sys.argv) < 2:
        print("\nNo recording provided.")
        print("\nUsage:")
        print('python main.py "path\\to\\recording.mp4"')
        return

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        print(f"\nERROR: File not found:")
        print(video_path)
        return

    print("\n" + "=" * 60)
    print("RECORDING")
    print("=" * 60)

    try:
        info = get_video_info(video_path)

    except subprocess.CalledProcessError:
        print("\nERROR: FFprobe could not analyze this recording.")
        return

    except Exception as error:
        print(f"\nERROR: {error}")
        return

    print(f"\nFile:       {info['file_name']}")
    print(f"Duration:   {info['duration']}")
    print(
        f"Resolution: {info['width']}x{info['height']}"
    )
    print(f"FPS:        {info['fps']}")
    print(f"Video:      {info['video_codec']}")

    if info["has_audio"]:
        print(f"Audio:      {info['audio_codec']}")
    else:
        print("Audio:      NONE")

    print("\nTesting video decoding...")

    if test_opencv(video_path):
        print("OpenCV:     OK")
    else:
        print("OpenCV:     FAILED")

    print("\n" + "=" * 60)
    print("READY FOR ANALYSIS")
    print("=" * 60)


if __name__ == "__main__":
    main()