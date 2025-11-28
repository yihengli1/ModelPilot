"""WSGI config for ModelPilot."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "modelpilot.settings")

application = get_wsgi_application()
