#!/bin/bash

rm -rf squlearn
rm -f *.out *.log

cp -r ../squlearn .

docker build -t dkr/adam_lr_2 .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --memory 10g --rm -it --name dkr_adam_lr_2 -v $(pwd):/data dkr/adam_lr_2 2>&1 | tee output.out

rm -rf squlearn