#!/bin/bash

rm -rf squlearn
rm -f *.out *.log

cp -r ../squlearn .

docker build -t dkr/noisy_sigmoid_sglbo_sur_avrg .
rm output.out
echo "RUNNING JOB"
docker run --cpus 4 --memory 10g --rm -it --name dkr_noisy_sigmoid_sglbo_sur_avrg -v $(pwd):/data dkr/noisy_sigmoid_sglbo_sur_avrg 2>&1 | tee output.out

rm -rf squlearn