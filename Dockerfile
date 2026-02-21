FROM python:3.10-bullseye

WORKDIR /app

COPY . /app

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.app:app"]
