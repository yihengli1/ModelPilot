"""Top-level URL configuration for ModelPilot."""

from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView

urlpatterns = [
    path(
        "",
        TemplateView.as_view(template_name="home.html"),
        name="home",
    ),
    path("admin/", admin.site.urls),
    path("api/", include("api.urls", namespace="api")),
]
