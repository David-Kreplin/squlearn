FROM ubuntu:22.04

RUN apt-get update && apt-get install -y python3 python3-distutils python3-pip python3-apt

RUN pip install matplotlib
RUN pip install scikit_optimize
RUN pip install qiskit==0.45.0
RUN pip install qiskit_machine_learning==0.7.0 


WORKDIR .
COPY . .

RUN pip install -e ./squlearn/.


CMD ["python3", "optimization.py"]
