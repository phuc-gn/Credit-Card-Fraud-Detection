FROM python:3.10.14-slim-bullseye

WORKDIR /app

COPY models/model.pkl .

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]