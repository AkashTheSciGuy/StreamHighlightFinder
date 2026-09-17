from pathlib import Path

from faster_whisper import WhisperModel


class SpeechAnalyzer:
    """
    Generic speech transcription analyzer.

    Uses faster-whisper locally to detect and transcribe
    streamer speech with timestamps.

    This analyzer is game-independent.
    """

    def __init__(
        self,
        model_size="small",
        device="cuda",
        compute_type="float16",
        language=None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language

        self.model = None

    def load_model(self):
        """
        Load the Whisper model only when needed.
        """

        if self.model is not None:
            return

        print(
            f"Loading Whisper model: "
            f"{self.model_size}"
        )

        print(
            f"Whisper device:       "
            f"{self.device}"
        )

        print(
            f"Whisper compute:      "
            f"{self.compute_type}"
        )

        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, video_path):
        """
        Transcribe speech directly from the recording.

        faster-whisper/FFmpeg handles the audio decoding.
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        self.load_model()

        print("\nTranscribing speech...")

        segments, info = self.model.transcribe(
            str(video_path),

            # Automatically determine language unless
            # one was explicitly configured.
            language=self.language,

            # Remove long silent sections before
            # transcription.
            vad_filter=True,

            vad_parameters={
                "min_silence_duration_ms": 500,
            },

            # Conservative beam search.
            beam_size=5,

            # Don't condition every segment too strongly
            # on previous text. This is useful for streams
            # where topics/dialogue change quickly.
            condition_on_previous_text=False,
        )

        results = []

        # segments is a generator, so transcription
        # actually happens while iterating here.
        for segment in segments:

            text = segment.text.strip()

            if not text:
                continue

            results.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                }
            )

        detected_language = getattr(
            info,
            "language",
            None,
        )

        language_probability = getattr(
            info,
            "language_probability",
            None,
        )

        return {
            "language": detected_language,
            "language_probability": language_probability,
            "segments": results,
        }

    def analyze(self, video_path):
        """
        Complete speech analysis pipeline.
        """

        return self.transcribe(video_path)