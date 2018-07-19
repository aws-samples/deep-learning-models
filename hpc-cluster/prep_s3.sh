#!/bin/bash
mkdir -p keys
ssh-keygen -f keys/id_rsa -t rsa -N ""
aws s3 sync . s3://{$1}/
