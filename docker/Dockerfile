FROM ubuntu
ENV DEBIAN_FRONTEND noninteractive

# Installing build dependencies
RUN apt-get update && apt-get install -y build-essential automake make cmake g++ wget git mercurial python3-pip curl

# Preparing Python build environment
RUN pip3 install cython future scipy nltk requests xmltodict nose2

# Prepare environment UTF-8
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Prepare cube

RUN mkdir /work && \
    cd /work && \
    git clone https://github.com/adobe/NLP-Cube.git
    cd NLP-Cube
    pip install -r requirements.txt

# Prepare notebook
RUN pip3 install jupyter
RUN pip3 install Flask
RUN pip3 install bs4

# Start webserver
#CMD cd /work/NLP-Cube/cube/ && python3 webserver.py --port 8080 --lang=en --lang=fr --lang=de
# Start notebook
CMD cd /work/NLP-Cube/ && python3 -m "notebook" --allow-root --ip=0.0.0.0 --no-browser
