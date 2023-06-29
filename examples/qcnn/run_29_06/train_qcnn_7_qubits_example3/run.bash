#!/bin/bash
rm -rf squlearn
cp -r ../squlearn .

docker build -t dkr/train_qcnn_7_qubits_example3 .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --memory 128g --rm -it --name dkr_train_qcnn_7_qubits_example3 -v $(pwd):/data dkr/train_qcnn_7_qubits_example3 2>&1 | tee output.out

rm -rf squlearn