FROM python:3.8-slim-bullseye

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    bash \
    nano \
    build-essential

WORKDIR /app
COPY . /app

RUN mkdir /data \
    && chmod 777 /data

RUN mkdir /models \
    && chmod 777 /models

RUN pip3 install --no-cache-dir -r requirements.txt


ENV LEAD_SCORE_PORT=5001
ENV FLASK_RUN_PORT=${LEAD_SCORE_PORT}
EXPOSE ${LEAD_SCORE_PORT}

CMD ["python", "-m" , "flask", "run", "--host=0.0.0.0"]
