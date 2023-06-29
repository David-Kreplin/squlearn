#!/bin/bash
rm -rf squlearn
cp -r ../squlearn .

docker build -t dkr/train_qcnn_7_qubits_example1 .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --memory 128g --rm -it --name dkr_train_qcnn_7_qubits_example1 -v $(pwd):/data dkr/train_qcnn_7_qubits_example1 2>&1 | tee output.out

rm -rf squlearn