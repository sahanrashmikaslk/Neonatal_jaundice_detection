from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PresenceResult:
    is_present: bool
    face_count: int
    max_face_area_ratio: float
    skin_ratio: float
    reason: str = ""


class BabyPresenceDetector:
    """
    Lightweight presence gate to avoid jaundice inference when no baby/face is visible.
    Combines a Haar face detector with a simple skin-exposure heuristic.
    """

    def __init__(
        self,
        min_face_area_ratio: float = 0.015,
        min_skin_ratio: float = 0.01,
        face_scale_factor: float = 1.1,
        face_min_neighbors: int = 4,
    ) -> None:
        self.min_face_area_ratio = min_face_area_ratio
        self.min_skin_ratio = min_skin_ratio
        self.face_scale_factor = face_scale_factor
        self.face_min_neighbors = face_min_neighbors

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _compute_skin_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Simple skin segmentation in YCrCb space with a small cleanup kernel."""
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array((0, 133, 77), dtype=np.uint8)
        upper = np.array((255, 173, 127), dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _detect_faces(self, frame_gray: np.ndarray, min_area: float) -> Tuple[Tuple[int, int, int, int], ...]:
        faces = self.face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=self.face_scale_factor,
            minNeighbors=self.face_min_neighbors,
        )
        filtered = tuple(face for face in faces if face[2] * face[3] >= min_area)
        return filtered

    def is_baby_present(self, frame_bgr: np.ndarray) -> PresenceResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return PresenceResult(False, 0, 0.0, 0.0, "empty frame")

        h, w = frame_bgr.shape[:2]
        min_face_area = self.min_face_area_ratio * h * w

        skin_mask = self._compute_skin_mask(frame_bgr)
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(h * w)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray, min_face_area)
        max_face_area_ratio = max([(fw * fh) / (h * w) for (_, _, fw, fh) in faces], default=0.0)

        is_present = bool(faces) and skin_ratio >= self.min_skin_ratio
        if not faces:
            reason = "no face detected"
        elif skin_ratio < self.min_skin_ratio:
            reason = "low exposed skin"
        else:
            reason = ""

        return PresenceResult(
            is_present=is_present,
            face_count=len(faces),
            max_face_area_ratio=max_face_area_ratio,
            skin_ratio=skin_ratio,
            reason=reason,
        )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Quick baby presence/skin exposure check.")
    parser.add_argument("image", type=Path, help="Path to image to evaluate.")
    parser.add_argument("--save-mask", type=Path, default=None, help="Optional path to save the skin mask PNG.")
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Could not read image at {args.image}")

    detector = BabyPresenceDetector()
    result = detector.is_baby_present(frame)

    print(f"is_present={result.is_present}")
    print(f"face_count={result.face_count}, max_face_area_ratio={result.max_face_area_ratio:.4f}")
    print(f"skin_ratio={result.skin_ratio:.4f}, reason='{result.reason}'")

    if args.save_mask:
        mask = detector._compute_skin_mask(frame)
        cv2.imwrite(str(args.save_mask), mask)
