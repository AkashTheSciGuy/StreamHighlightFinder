import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np


class AudioAnalyzer:
    """
    Detects unusually active/loud sections of a recording.

    These regions are CANDIDATES only.
    They are not final highlights.
    """

    def __init__(
        self,
        window_seconds=1.0,
        threshold_std=1.25,
        min_activity_seconds=2.0,
        merge_gap_seconds=3.0,
    ):
        self.window_seconds = window_seconds
        self.threshold_std = threshold_std
        self.min_activity_seconds = min_activity_seconds
        self.merge_gap_seconds = merge_gap_seconds

    def extract_audio(self, video_path, output_path):
        """
        Extract mono 16 kHz PCM audio using FFmpeg.
        """

        command = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(output_path),
        ]

        subprocess.run(command, check=True)

    def load_wav(self, wav_path):
        """
        Load extracted PCM WAV into NumPy.
        """

        with wave.open(str(wav_path), "rb") as wav_file:

            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()

            if channels != 1:
                raise RuntimeError("Expected mono audio.")

            if sample_width != 2:
                raise RuntimeError("Expected 16-bit PCM audio.")

            raw_audio = wav_file.readframes(frame_count)

        samples = np.frombuffer(
            raw_audio,
            dtype=np.int16
        ).astype(np.float32)

        samples /= 32768.0

        return samples, sample_rate

    def calculate_energy(self, samples, sample_rate):
        """
        Calculate RMS audio energy for consecutive windows.
        """

        window_size = int(
            sample_rate * self.window_seconds
        )

        energies = []

        for start in range(0, len(samples), window_size):

            chunk = samples[start:start + window_size]

            if len(chunk) == 0:
                continue

            rms = np.sqrt(
                np.mean(np.square(chunk))
            )

            timestamp = start / sample_rate

            energies.append(
                {
                    "time": timestamp,
                    "energy": float(rms),
                }
            )

        return energies

    def detect_activity(self, energies):
        """
        Find windows whose energy is significantly above
        the recording's normal audio level.
        """

        if not energies:
            return []

        values = np.array(
            [item["energy"] for item in energies],
            dtype=np.float32,
        )

        median = float(np.median(values))
        std = float(np.std(values))

        threshold = median + (
            self.threshold_std * std
        )

        active_windows = []

        for item in energies:

            if item["energy"] >= threshold:

                active_windows.append(
                    {
                        "start": item["time"],
                        "end": (
                            item["time"]
                            + self.window_seconds
                        ),
                        "energy": item["energy"],
                    }
                )

        return active_windows, threshold, median

    def merge_activity(self, windows):
        """
        Merge neighboring active windows into larger regions.
        """

        if not windows:
            return []

        merged = []

        current = {
            "start": windows[0]["start"],
            "end": windows[0]["end"],
            "peak_energy": windows[0]["energy"],
        }

        for window in windows[1:]:

            gap = window["start"] - current["end"]

            if gap <= self.merge_gap_seconds:

                current["end"] = max(
                    current["end"],
                    window["end"],
                )

                current["peak_energy"] = max(
                    current["peak_energy"],
                    window["energy"],
                )

            else:

                duration = (
                    current["end"]
                    - current["start"]
                )

                if duration >= self.min_activity_seconds:
                    merged.append(current)

                current = {
                    "start": window["start"],
                    "end": window["end"],
                    "peak_energy": window["energy"],
                }

        duration = current["end"] - current["start"]

        if duration >= self.min_activity_seconds:
            merged.append(current)

        return merged

    def analyze(self, video_path):
        """
        Complete audio-analysis pipeline.
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        print("\nExtracting analysis audio...")

        with tempfile.TemporaryDirectory() as temp_dir:

            wav_path = (
                Path(temp_dir)
                / "analysis_audio.wav"
            )

            self.extract_audio(
                video_path,
                wav_path,
            )

            samples, sample_rate = self.load_wav(
                wav_path
            )

            print(
                f"Audio samples: {len(samples):,}"
            )

            print(
                f"Sample rate:   {sample_rate} Hz"
            )

            print("Calculating audio activity...")

            energies = self.calculate_energy(
                samples,
                sample_rate,
            )

            result = self.detect_activity(
                energies
            )

            if not result:
                return {
                    "threshold": 0,
                    "median_energy": 0,
                    "regions": [],
                }

            active_windows, threshold, median = result

            regions = self.merge_activity(
                active_windows
            )

        return {
            "threshold": threshold,
            "median_energy": median,
            "regions": regions,
        }


def format_time(seconds):
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02}:{minutes:02}:{secs:02}"