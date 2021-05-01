FROM python:3.8-alpine

RUN apk add --update --no-cache py3-pandas
ENV PYTHONPATH=/usr/lib/python3.8/site-packages

RUN pip install --upgrade pip


COPY . .

#RUN python3 decisiontree.py --data car.csv --output car.xml

CMD ["python3", "./decisiontree.py"]