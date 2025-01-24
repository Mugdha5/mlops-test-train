FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements.txt
COPY zoo.csv zoo.csv
RUN pip install -r requirements.txt 
COPY . .
EXPOSE 5000
CMD ["python", "animal_classification_app.py"]
