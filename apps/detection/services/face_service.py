import base64
import threading
import uuid
from datetime import datetime, timedelta
from typing import Optional

import cv2
import numpy as np


PERSON_TTL_SECONDS = 3600
COSINE_THRESHOLD = 0.40
THUMBNAIL_SIZE = 160
MIN_CROP_PIXELS = 40


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


class FaceService:
    """
    In-memory face registry. Extracts a Facenet embedding for each new
    person track_id, deduplicates by cosine distance against previously
    seen embeddings, and expires entries after PERSON_TTL_SECONDS.
    """

    _instance = None
    _singleton_lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        from deepface import DeepFace
        self._deepface = DeepFace

        # Pre-build the Facenet graph so the first real inference is fast.
        dummy = np.zeros((THUMBNAIL_SIZE, THUMBNAIL_SIZE, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                img_path=dummy,
                model_name='Facenet',
                detector_backend='skip',
                enforce_detection=False,
            )
            print('[FaceService] Facenet model pre-loaded.')
        except Exception as exc:
            print(f'[FaceService] Pre-load warning: {exc}')

        self._persons: dict[str, dict] = {}
        self._seen_track_ids: set[int] = set()
        self._store_lock = threading.Lock()

    def process_person(self, frame: np.ndarray, bbox: dict, track_id: Optional[int]) -> Optional[dict]:
        """
        For a YOLO 'person' detection, try to register a new face entry.
        Returns the new person dict if added, otherwise None.
        """
        if track_id is None:
            return None

        with self._store_lock:
            self._cleanup_expired_locked()
            if track_id in self._seen_track_ids:
                return None

        h, w = frame.shape[:2]
        x = max(0, int(bbox['x']))
        y = max(0, int(bbox['y']))
        x2 = min(w, x + int(bbox['width']))
        y2 = min(h, y + int(bbox['height']))
        if x2 - x < MIN_CROP_PIXELS or y2 - y < MIN_CROP_PIXELS:
            return None

        crop = frame[y:y2, x:x2]

        try:
            results = self._deepface.represent(
                img_path=crop,
                model_name='Facenet',
                detector_backend='opencv',
                enforce_detection=True,
                align=True,
            )
        except Exception:
            # No face detected inside this person crop. Don't mark the
            # track_id as seen so a future, clearer frame can succeed.
            return None

        if not results:
            return None

        embedding = np.asarray(results[0].get('embedding', []), dtype=np.float32)
        if embedding.size == 0:
            return None

        photo_b64 = self._build_thumbnail(crop, results[0].get('facial_area'))
        if photo_b64 is None:
            return None

        with self._store_lock:
            for person in self._persons.values():
                if _cosine_distance(embedding, person['embedding']) < COSINE_THRESHOLD:
                    self._seen_track_ids.add(track_id)
                    return None

            person_id = str(uuid.uuid4())
            entry_time = datetime.utcnow()
            self._persons[person_id] = {
                'embedding': embedding,
                'photo': photo_b64,
                'entry_time': entry_time,
            }
            self._seen_track_ids.add(track_id)
            return {
                'id': person_id,
                'photo': photo_b64,
                'entryTime': entry_time.isoformat() + 'Z',
            }

    def get_persons(self) -> list[dict]:
        with self._store_lock:
            self._cleanup_expired_locked()
            ordered = sorted(
                self._persons.items(),
                key=lambda kv: kv[1]['entry_time'],
                reverse=True,
            )
            return [
                {
                    'id': pid,
                    'photo': p['photo'],
                    'entryTime': p['entry_time'].isoformat() + 'Z',
                }
                for pid, p in ordered
            ]

    def _cleanup_expired_locked(self):
        threshold = datetime.utcnow() - timedelta(seconds=PERSON_TTL_SECONDS)
        expired = [pid for pid, p in self._persons.items() if p['entry_time'] < threshold]
        for pid in expired:
            del self._persons[pid]

    def _build_thumbnail(self, crop: np.ndarray, facial_area: Optional[dict]) -> Optional[str]:
        face_crop = crop
        if facial_area:
            fx = max(0, int(facial_area.get('x', 0)))
            fy = max(0, int(facial_area.get('y', 0)))
            fw = int(facial_area.get('w', crop.shape[1]))
            fh = int(facial_area.get('h', crop.shape[0]))
            candidate = crop[fy:fy + fh, fx:fx + fw]
            if candidate.size > 0:
                face_crop = candidate

        try:
            face_crop = cv2.resize(face_crop, (THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        except cv2.error:
            return None

        ok, buf = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return None
        return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')
