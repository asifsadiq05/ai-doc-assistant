FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-ben \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt requirements-ocr.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ocr.txt
COPY . .
RUN useradd -m app && mkdir -p data && chown app data
USER app
ENV OCR=1 OCR_LANG=eng+ben COOKIE_SECURE=0 DB_PATH=/srv/data/app.db
VOLUME /srv/data
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
