FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

CMD ["sh", "-c", "gunicorn modelpilot.wsgi:application --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 900"]