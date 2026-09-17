import sys
import json
import subprocess
from pathlib import Path

import cv2
import torch

from analyzers.audio_analyzer import AudioAnalyzer, format_time
from analyzers.video_analyzer import VideoAnalyzer
from analyzers.speech_analyzer import SpeechAnalyzer


def format_duration(seconds: float) -> str:
    """Convert seconds into HH:MM:SS."""
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"


def get_video_info(video_path: Path) -> dict:
    """Read video/audio metadata using FFprobe."""

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
            video_stream.get("duration", 0),
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
    """Verify that OpenCV can decode the recording."""

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

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

        memory = (
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3
        )

        print(f"VRAM: {memory:.2f} GB")

    else:
        print("GPU: CPU MODE")


def run_audio_analysis(video_path: Path):
    """Run audio activity detection."""

    print("\n" + "=" * 60)
    print("AUDIO ACTIVITY ANALYSIS")
    print("=" * 60)

    try:

        audio_analyzer = AudioAnalyzer()

        audio_result = audio_analyzer.analyze(
            video_path
        )

        regions = audio_result["regions"]

        print(
            f"\nMedian energy: "
            f"{audio_result['median_energy']:.5f}"
        )

        print(
            f"Activity threshold: "
            f"{audio_result['threshold']:.5f}"
        )

        print(
            f"\nAudio candidate regions found: "
            f"{len(regions)}"
        )

        if regions:

            print()

            for index, region in enumerate(
                regions,
                start=1,
            ):

                start = format_time(
                    region["start"]
                )

                end = format_time(
                    region["end"]
                )

                print(
                    f"#{index:02}  "
                    f"{start} -> {end}  "
                    f"Peak: "
                    f"{region['peak_energy']:.5f}"
                )

        else:

            print(
                "\nNo unusually active "
                "audio regions detected."
            )

    except Exception as error:

        print(
            f"\nAudio analysis failed: {error}"
        )

def run_video_analysis(video_path: Path):
    """Run visual activity detection."""

    print("\n" + "=" * 60)
    print("VISUAL ACTIVITY ANALYSIS")
    print("=" * 60)

    try:
        analyzer = VideoAnalyzer()

        result = analyzer.analyze(
            video_path
        )

        regions = result["regions"]

        print(
            f"\nMedian motion: "
            f"{result['median_motion']:.3f}"
        )

        print(
            f"Activity threshold: "
            f"{result['threshold']:.3f}"
        )

        print(
            f"\nVisual candidate regions found: "
            f"{len(regions)}"
        )

        if regions:

            print()

            for index, region in enumerate(
                regions,
                start=1,
            ):

                start = format_time(
                    region["start"]
                )

                end = format_time(
                    region["end"]
                )

                print(
                    f"#{index:02}  "
                    f"{start} -> {end}  "
                    f"Peak: "
                    f"{region['peak_motion']:.3f}"
                )

        else:
            print(
                "\nNo unusually active "
                "visual regions detected."
            )

    except Exception as error:
        print(
            f"\nVisual analysis failed: {error}"
        )

        def run_speech_analysis(video_path: Path):
            """Run local Whisper speech transcription."""

    print("\n" + "=" * 60)
    print("SPEECH ANALYSIS")
    print("=" * 60)

    try:

        analyzer = SpeechAnalyzer(
            model_size="small",
            device="cuda",
            compute_type="float16",
        )

        result = analyzer.analyze(
            video_path
        )

        segments = result["segments"]

        print(
            f"\nDetected language: "
            f"{result['language']}"
        )

        probability = result[
            "language_probability"
        ]

        if probability is not None:

            print(
                f"Language confidence: "
                f"{probability:.2%}"
            )

        print(
            f"\nSpeech segments found: "
            f"{len(segments)}"
        )

        if segments:

            print()

            for index, segment in enumerate(
                segments,
                start=1,
            ):

                start = format_time(
                    segment["start"]
                )

                end = format_time(
                    segment["end"]
                )

                print(
                    f"#{index:03}  "
                    f"{start} -> {end}"
                )

                print(
                    f"      {segment['text']}"
                )

        else:

            print(
                "\nNo speech detected."
            )

    except Exception as error:

        print(
            f"\nSpeech analysis failed: "
            f"{error}"
        )

def run_speech_analysis(video_path: Path):
    """Run local Whisper speech transcription."""

    print("\n" + "=" * 60)
    print("SPEECH ANALYSIS")
    print("=" * 60)

    try:
        analyzer = SpeechAnalyzer(
            model_size="small",
            device="cuda",
            compute_type="float16",
        )

        result = analyzer.analyze(video_path)

        segments = result["segments"]

        print(
            f"\nDetected language: "
            f"{result['language']}"
        )

        probability = result["language_probability"]

        if probability is not None:
            print(
                f"Language confidence: "
                f"{probability:.2%}"
            )

        print(
            f"\nSpeech segments found: "
            f"{len(segments)}"
        )

        if segments:
            print()

            for index, segment in enumerate(
                segments,
                start=1,
            ):
                start = format_time(
                    segment["start"]
                )

                end = format_time(
                    segment["end"]
                )

                print(
                    f"#{index:03}  "
                    f"{start} -> {end}"
                )

                print(
                    f"      {segment['text']}"
                )

        else:
            print("\nNo speech detected.")

    except Exception as error:
        print(
            f"\nSpeech analysis failed: "
            f"{error}"
        )
        
def main():

    print_system_info()

    if len(sys.argv) < 2:

        print("\nNo recording provided.")

        print("\nUsage:")

        print(
            'python main.py '
            '"path\\to\\recording.mp4"'
        )

        return

    # This is where video_path is created.
    video_path = Path(sys.argv[1])

    if not video_path.exists():

        print("\nERROR: File not found:")
        print(video_path)

        return

    print("\n" + "=" * 60)
    print("RECORDING")
    print("=" * 60)

    try:

        info = get_video_info(video_path)

    except subprocess.CalledProcessError:

        print(
            "\nERROR: FFprobe could not "
            "analyze this recording."
        )

        return

    except Exception as error:

        print(f"\nERROR: {error}")

        return

    print(f"\nFile:       {info['file_name']}")
    print(f"Duration:   {info['duration']}")

    print(
        f"Resolution: "
        f"{info['width']}x{info['height']}"
    )

    print(f"FPS:        {info['fps']}")
    print(f"Video:      {info['video_codec']}")

    if info["has_audio"]:

        print(
            f"Audio:      "
            f"{info['audio_codec']}"
        )

    else:

        print("Audio:      NONE")

    print("\nTesting video decoding...")

    if test_opencv(video_path):

        print("OpenCV:     OK")

    else:

        print("OpenCV:     FAILED")
        return

        # Run audio analysis if audio exists.
    if info["has_audio"]:

        run_audio_analysis(video_path)

    else:

        print(
        "\nSkipping audio analysis: "
        "recording has no audio."
    )

        print("\nStarting visual analyzer...")
        run_video_analysis(video_path)

    if info["has_audio"]:

        run_speech_analysis(video_path)

    else:

        print(
        "\nSkipping speech analysis: "
        "recording has no audio."
    )

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)

if __name__ == "__main__":
    main()