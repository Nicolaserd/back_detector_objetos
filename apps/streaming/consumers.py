import json
import cv2
import numpy as np
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from apps.detection.services.yolo_service import YOLOService
from apps.detection.services.face_service import FaceService
from datetime import datetime

# Eager singleton instantiation: forces Facenet to load when the ASGI app
# imports this module at server startup, instead of on the first frame.
_face_service = FaceService()


class DetectionConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yolo_service = YOLOService()
        self.face_service = _face_service
        self.frame_count = 0

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        """
        Receives a frame, processes it, and returns detections + persons.
        Expects binary data (preferred) or JSON with base64.
        """
        frame = None

        if bytes_data:
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif text_data:
            data = json.loads(text_data)
            if 'image' in data:
                img_data = base64.b64decode(data['image'].split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            self.frame_count += 1
            detections = self.yolo_service.predict(frame)
            stats = self.yolo_service.get_current_stats()

            for detection in detections:
                if detection['label'] == 'person':
                    self.face_service.process_person(
                        frame, detection['bbox'], detection['trackId']
                    )

            persons = self.face_service.get_persons()

            if self.frame_count % 10 == 0:
                print(
                    f"[AI] Frame {self.frame_count} procesado. "
                    f"Detecciones: {len(detections)} | Total 15m: {stats['total']} | "
                    f"Personas registradas: {len(persons)}"
                )

            response = {
                "frameId": self.frame_count,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "detections": detections,
                "stats": stats,
                "persons": persons,
            }

            await self.send(text_data=json.dumps(response))
        else:
            if self.frame_count % 10 == 0:
                print("[ERROR] No se pudo decodificar el frame recibido.")
