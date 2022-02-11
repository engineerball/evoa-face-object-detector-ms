# FROM engineerball/python37-opencv453:1.0.0
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

EXPOSE 5000
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r  requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]