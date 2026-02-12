FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY app.py .
COPY train_model.py .

# Train model on startup
RUN python train_model.py

CMD python app.py