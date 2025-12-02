from django.urls import include, path

from rest_framework.routers import DefaultRouter

from .views import CreateRunView, SampleDataView

router = DefaultRouter()

app_name = "api"

urlpatterns = [
    path("sample/", SampleDataView.as_view(), name="sample"),
    path("run-prompt/", CreateRunView.as_view(), name="run-prompt"),
    path("", include(router.urls)),
]
