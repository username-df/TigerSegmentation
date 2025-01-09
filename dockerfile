FROM python:3.10-slim

WORKDIR /flaskapp
ADD . /flaskapp/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "./app.py"]