#!/bin/bash

rm -rf squlearn
rm -f *.out *.log

cp -r ../squlearn .

docker build -t dkr/noisy_log_adam_0x1 .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --memory 10g --rm -it --name dkr_noisy_log_adam_0x1 -v $(pwd):/data dkr/noisy_log_adam_0x1 2>&1 | tee output.out

rm -rf squlearn