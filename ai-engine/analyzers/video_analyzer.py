from pathlib import Path

import cv2
import numpy as np


class VideoAnalyzer:
    """
    Fast visual activity detector.

    Samples the recording at a low rate and measures
    changes between consecutive frames.

    The resulting regions are CANDIDATES, not final highlights.
    """

    def __init__(
        self,
        sample_fps=2.0,
        threshold_std=1.25,
        min_activity_seconds=2.0,
        merge_gap_seconds=3.0,
        analysis_width=320,
    ):
        self.sample_fps = sample_fps
        self.threshold_std = threshold_std
        self.min_activity_seconds = min_activity_seconds
        self.merge_gap_seconds = merge_gap_seconds
        self.analysis_width = analysis_width

    def prepare_frame(self, frame):
        """
        Resize and convert a frame to grayscale.

        Smaller grayscale frames are much faster to compare
        than full-resolution color frames.
        """

        height, width = frame.shape[:2]

        if width > self.analysis_width:
            scale = self.analysis_width / width

            new_height = int(height * scale)

            frame = cv2.resize(
                frame,
                (self.analysis_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        # Slight blur reduces tiny compression/noise changes.
        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0,
        )

        return gray

    def calculate_motion(self, previous_frame, current_frame):
        """
        Calculate average pixel difference between frames.

        Higher value = more visual change.
        """

        difference = cv2.absdiff(
            previous_frame,
            current_frame,
        )

        motion_score = float(
            np.mean(difference)
        )

        return motion_score

    def scan_video(self, video_path):
        """
        Sample the entire recording and calculate
        visual motion scores over time.
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():
            raise RuntimeError(
                "OpenCV could not open the video."
            )

        source_fps = capture.get(
            cv2.CAP_PROP_FPS
        )

        frame_count = capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )

        if source_fps <= 0:
            capture.release()

            raise RuntimeError(
                "Invalid source FPS."
            )

        duration = frame_count / source_fps

        # Example:
        # 25 FPS source / 2 FPS analysis
        # ≈ sample every 12 frames.
        sample_interval = max(
            1,
            int(round(
                source_fps / self.sample_fps
            )),
        )

        actual_sample_fps = (
            source_fps / sample_interval
        )

        print(
            f"Source FPS:          "
            f"{source_fps:.2f}"
        )

        print(
            f"Visual sample FPS:   "
            f"{actual_sample_fps:.2f}"
        )

        print(
            f"Frame interval:      "
            f"{sample_interval}"
        )

        print(
            f"Recording duration:  "
            f"{duration:.2f} seconds"
        )

        motion_scores = []

        previous_frame = None

        frame_number = 0

        while True:

            success, frame = capture.read()

            if not success:
                break

            if frame_number % sample_interval != 0:
                frame_number += 1
                continue

            timestamp = (
                frame_number / source_fps
            )

            prepared = self.prepare_frame(
                frame
            )

            if previous_frame is not None:

                score = self.calculate_motion(
                    previous_frame,
                    prepared,
                )

                motion_scores.append(
                    {
                        "time": timestamp,
                        "motion": score,
                    }
                )

            previous_frame = prepared

            frame_number += 1

        capture.release()

        return motion_scores, actual_sample_fps

    def detect_activity(
        self,
        motion_scores,
        sample_fps,
    ):
        """
        Detect unusually high visual activity.
        """

        if not motion_scores:
            return [], 0.0, 0.0

        values = np.array(
            [
                item["motion"]
                for item in motion_scores
            ],
            dtype=np.float32,
        )

        median = float(
            np.median(values)
        )

        std = float(
            np.std(values)
        )

        threshold = median + (
            self.threshold_std * std
        )

        window_seconds = 1.0 / sample_fps

        active_windows = []

        for item in motion_scores:

            if item["motion"] >= threshold:

                active_windows.append(
                    {
                        "start": item["time"],
                        "end": (
                            item["time"]
                            + window_seconds
                        ),
                        "motion": item["motion"],
                    }
                )

        return (
            active_windows,
            threshold,
            median,
        )

    def merge_activity(self, windows):
        """
        Merge nearby visual activity windows.
        """

        if not windows:
            return []

        merged = []

        current = {
            "start": windows[0]["start"],
            "end": windows[0]["end"],
            "peak_motion": windows[0]["motion"],
        }

        for window in windows[1:]:

            gap = (
                window["start"]
                - current["end"]
            )

            if gap <= self.merge_gap_seconds:

                current["end"] = max(
                    current["end"],
                    window["end"],
                )

                current["peak_motion"] = max(
                    current["peak_motion"],
                    window["motion"],
                )

            else:

                duration = (
                    current["end"]
                    - current["start"]
                )

                if (
                    duration
                    >= self.min_activity_seconds
                ):
                    merged.append(current)

                current = {
                    "start": window["start"],
                    "end": window["end"],
                    "peak_motion": window["motion"],
                }

        duration = (
            current["end"]
            - current["start"]
        )

        if (
            duration
            >= self.min_activity_seconds
        ):
            merged.append(current)

        return merged

    def analyze(self, video_path):
        """
        Complete visual activity pipeline.
        """

        print("\nScanning visual activity...")

        motion_scores, sample_fps = (
            self.scan_video(video_path)
        )

        print(
            f"Visual samples:      "
            f"{len(motion_scores):,}"
        )

        print(
            "Calculating visual activity..."
        )

        (
            active_windows,
            threshold,
            median,
        ) = self.detect_activity(
            motion_scores,
            sample_fps,
        )

        regions = self.merge_activity(
            active_windows
        )

        return {
            "median_motion": median,
            "threshold": threshold,
            "sample_fps": sample_fps,
            "samples": len(motion_scores),
            "regions": regions,
        }