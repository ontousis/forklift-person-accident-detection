FROM ubuntu:24.04

ADD best_full_integer_quant_edgetpu.tflite .

RUN apt-get update && apt -y upgrade

RUN apt install -y software-properties-common
RUN  add-apt-repository ppa:deadsnakes/ppa
RUN  apt update
RUN  apt install -y python3.9
RUN  apt install -y python3.9-distutils
RUN  apt install -y python3.9-venv
RUN python3.9 -m venv --system-site-packages /opt/forklift_env
ENV PATH="/opt/forklift_env/bin:$PATH"

RUN apt-get install -y dpkg

RUN apt-get install -y python3-pip

COPY libedgetpu1-std_16.0tf2.17.1-1.ubuntu24.04_arm64.deb /tmp/

RUN dpkg -i /tmp/libedgetpu1-std_16.0tf2.17.1-1.ubuntu24.04_arm64.deb || apt-get install -f -y

RUN apt-get clean

RUN pip install -U tflite-runtime

RUN pip install ultralytics

RUN pip install numpy==1.26.4

RUN pip install lgpio

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless

RUN pip3 install opencv-contrib-python==4.5.5.62

ADD forklifts_humans.py .

CMD ["python3", "./forklifts_humans.py"]
