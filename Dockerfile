FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml .
COPY learning_studio ./learning_studio
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["uvicorn", "learning_studio.main:app", "--host", "0.0.0.0", "--port", "8000"]
