FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MONGODBUSER=applicationAdminUser
ENV MONGODBPASSWORD=tourAdvisor@2026

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]