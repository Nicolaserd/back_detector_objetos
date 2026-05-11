from django.urls import path
from apps.streaming.consumers import DetectionConsumer

websocket_urlpatterns = [
    path('ws/detections/', DetectionConsumer.as_asgi()),
]
