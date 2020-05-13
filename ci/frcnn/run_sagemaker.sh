#!/bin/bash


echo start launching
#python launch_sagemaker.py | tee log.txt
python launch_sagemaker.py > log.txt
echo start parsing
python ci/parse_and_submit.py log.txt 8 2 p3.16xlarge Sagemaker
