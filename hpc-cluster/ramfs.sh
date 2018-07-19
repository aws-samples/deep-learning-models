#!/bin/sh

mkdir -p $1
sudo mount -t tmpfs -o size=$2G,nodiratime,noatime,nodev,nosuid tmpfs $1
