import os
from dotenv import load_dotenv

load_dotenv()

class Env:
    DJANGO_SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-insecure-default-key')
    DEBUG = os.getenv('DEBUG', 'True') == 'True'
    ALLOWED_HOSTS = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    
    YOLO_MODEL_PATH = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.40))
    IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.50))
    CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', '0')
