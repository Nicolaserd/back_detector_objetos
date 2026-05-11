from collections import deque
from datetime import datetime, timedelta
import threading

class StatsService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(StatsService, cls).__new__(cls)
                # Almacena tuplas de (timestamp, track_id, label)
                cls._instance.detections = deque()
                # Para evitar duplicados en el mismo conteo de 15 min, 
                # pero permitir que un objeto que se fue y volvió se cuente de nuevo si tiene nuevo ID
                cls._instance.seen_tracks = set()
            return cls._instance

    def add_detection(self, track_id, label):
        """
        Registra una detección si el track_id no ha sido procesado recientemente.
        """
        now = datetime.utcnow()
        
        # Solo agregamos si tiene track_id (es un objeto rastreado)
        if track_id is not None:
            # Verificamos si ya lo vimos en la ventana actual de la deque
            # Nota: Esto es simplificado. En un sistema real podrías querer 
            # persistencia de track_ids por un tiempo específico.
            if track_id not in self.seen_tracks:
                self.detections.append((now, track_id, label))
                self.seen_tracks.add(track_id)

    def _clean_old_data(self):
        """
        Elimina detecciones con más de 15 minutos de antigüedad.
        """
        now = datetime.utcnow()
        threshold = now - timedelta(minutes=15)
        
        while self.detections and self.detections[0][0] < threshold:
            _, old_track_id, _ = self.detections.popleft()
            # Opcional: Solo remover de seen_tracks si ya no está en la deque
            # para permitir que se vuelva a contar si re-aparece después de 15 min.
            self._rebuild_seen_tracks()

    def _rebuild_seen_tracks(self):
        """Reconstruye el set de IDs vistos basado en lo que queda en la deque."""
        self.seen_tracks = {d[1] for d in self.detections}

    def get_stats(self):
        """
        Retorna el conteo de objetos por clase en los últimos 15 minutos.
        """
        self._clean_old_data()
        
        summary = {
            "total": len(self.detections),
            "byClass": {},
            "windowMinutes": 15
        }
        
        for _, _, label in self.detections:
            summary["byClass"][label] = summary["byClass"].get(label, 0) + 1
            
        return summary
