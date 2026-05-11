from django.contrib import admin
from django.urls import path, include
from apps.core.views import health_check

urlpatterns = [
    path('admin/', admin.site.admin_site.urls if hasattr(admin.site, 'admin_site') else admin.site.urls),
    path('api/health/', health_check),
    # Add other app URLs as needed
]
