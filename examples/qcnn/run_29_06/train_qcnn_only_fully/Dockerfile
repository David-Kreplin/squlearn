FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-distutils python3-pip python3-apt

RUN pip install scipy
RUN pip install matplotlib
RUN pip install qiskit==0.43.0
RUN pip install qiskit_ibm_runtime==0.9.4
RUN pip install numpy
RUN pip install iteration-utilities
RUN pip install pylatexenc
RUN pip install pandas

WORKDIR .
COPY . .

RUN pip install ./squlearn/.

CMD ["python3", "train_qcnn_only_fully_64_data_points.py"]
