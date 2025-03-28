FROM python:3.9

WORKDIR /app

COPY api/ api/
COPY models/ models/
COPY scripts/ scripts/
COPY api/requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn

EXPOSE 8080

CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]