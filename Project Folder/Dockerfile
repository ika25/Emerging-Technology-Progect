FROM python:3

# project File Location within the container
WORKDIR /usr/src/app

COPY requirements.txt ./

# build env from requirments.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_APP=webService.py

CMD flask run --host=0.0.0.0