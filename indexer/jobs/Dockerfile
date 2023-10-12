FROM python:3.11-slim

# Install dependencies
RUN pip install --no-cache-dir \
    'fastapi>=0.68.0,<0.69.0' \
    'pydantic>=1.8.0,<2.0.0' \
    'uvicorn>=0.15.0,<0.16.0' \
    numpy

WORKDIR /app

COPY ./server.py /app/server.py

ENTRYPOINT ["/bin/sh", "-c"]

CMD ["python server.py"]