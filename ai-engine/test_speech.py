from pathlib import Path

from analyzers.audio_analyzer import format_time
from analyzers.speech_analyzer import SpeechAnalyzer


video_path = Path(
    r"..\recordings\apex-test.mp4"
)

analyzer = SpeechAnalyzer(
    model_size="small",
    device="cuda",
    compute_type="float16",
)

result = analyzer.analyze(video_path)

print("\n" + "=" * 60)
print("FULL SPEECH TRANSCRIPT")
print("=" * 60)

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

segments = result["segments"]

print(
    f"\nSpeech segments found: "
    f"{len(segments)}"
)

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