#!/bin/bash
rm -rf squlearn
cp -r ../squlearn .

docker build -t dkr/train_qcnn_pseudo_aa_entangling_conv .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --memory 128g --rm -it --name dkr_train_qcnn_pseudo_aa_entangling_conv -v $(pwd):/data dkr/train_qcnn_pseudo_aa_entangling_conv 2>&1 | tee output.out

rm -rf squlearn