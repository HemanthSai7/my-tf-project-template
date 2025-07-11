FROM tensorflow/tensorflow:2.15.0.post1-gpu

WORKDIR /usr/src/app

# Install dependencies
COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt

COPY . .
ENTRYPOINT ["python3", "trainer.py"]