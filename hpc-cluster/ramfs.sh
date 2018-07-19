#!/bin/sh
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

mkdir -p $1
sudo mount -t tmpfs -o size=$2G,nodiratime,noatime,nodev,nosuid tmpfs $1
