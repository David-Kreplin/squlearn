#!/bin/bash
rm -rf squlearn
cp -r ../squlearn .

docker build -t dkr/train_qcnn_diff_rot_var_in_entang .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --rm -it --name dkr_train_qcnn_diff_rot_var_in_entang -v $(pwd):/data dkr/train_qcnn_diff_rot_var_in_entang 2>&1 | tee output.out

rm -rf squlearn