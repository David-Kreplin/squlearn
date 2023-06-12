#!/bin/bash
rm -rf squlearn
cp -r ../squlearn .

docker build -t dkr/train_qcnn_2d-3d-var_param-gates .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --rm -it --name dkr_train_qcnn_2d-3d-var_param-gates -v $(pwd):/data dkr/train_qcnn_2d-3d-var_param-gates 2>&1 | tee output.out

rm -rf squlearn