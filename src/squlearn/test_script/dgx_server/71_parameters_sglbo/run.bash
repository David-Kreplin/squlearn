#!/bin/bash

rm -rf squlearn
rm -f *.out *.log

cp -r ../squlearn .

docker build -t dkr/sglbo_71_lr_1 .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --memory 10g --rm -it --name dkr_sglbo_71_lr_1 -v $(pwd):/data dkr/sglbo_71_lr_1 2>&1 | tee output.out

rm -rf squlearn