# syntax=docker/dockerfile:1

# base python image for custom image
FROM python:3.9.13-slim-buster

# create working directory and install pip dependencies
WORKDIR /mistral_docker
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# copy python project files from local to /hello-py image working directory
COPY . .

# run the flask server  
CMD [ "bash", "./run_model.sh"]

