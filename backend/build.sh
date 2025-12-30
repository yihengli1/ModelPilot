#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt
python manage.py migrate --noinput
python manage.py seed_examples
