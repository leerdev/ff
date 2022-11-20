FROM python:3.9

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install numpy && pip install pandas && pip install sklearn
RUN pip install -r requirements.txt

ENTRYPOINT ["python"]

CMD ["app.py"]
