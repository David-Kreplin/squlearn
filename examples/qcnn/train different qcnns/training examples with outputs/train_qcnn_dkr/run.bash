#!/bin/bash
rm -rf squlearn
cp -r ../squlearn .

docker build -t dkr/train_qcnn_dkr .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --rm -it --name dkr_train_qcnn_dkr -v $(pwd):/data dkr/train_qcnn_dkr 2>&1 | tee output.out

rm -rf squlearn