FROM python:3.7.2-alpine

WORKDIR /project

ADD . /project

RUN pip install --upgrade pip

RUN pip install --ignore-installed -r requirements.txt

CMD ["python","main.py"]