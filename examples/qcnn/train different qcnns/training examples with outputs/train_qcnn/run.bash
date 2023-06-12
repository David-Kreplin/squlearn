#!/bin/bash
rm -rf squlearn
cp -r ../squlearn .

docker build -t dkr/train_qcnn .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --rm -it --name dkr_train_qcnn -v $(pwd):/data dkr/train_qcnn 2>&1 | tee output.out

rm -rf squlearn