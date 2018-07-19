#!/bin/bash
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem='4096 87380 16777216'
sysctl -w net.ipv4.tcp_wmem='4096 65536 16777216'
sysctl -w net.core.netdev_max_backlog=30000
sysctl -w net.core.rmem_default=16777216
sysctl -w net.core.wmem_default=16777216
sysctl -w net.ipv4.tcp_mem='16777216 16777216 16777216'
sysctl -w net.ipv4.route.flush=1

echo "deb [arch=amd64] http://www.beegfs.io/release/beegfs_7 deb9 non-free" | sudo tee -a /etc/apt/sources.list.d/beegfs.list
wget -q https://www.beegfs.io/release/latest-stable/gpg/DEB-GPG-KEY-beegfs -O- | sudo apt-key add -
sudo apt update
sudo apt install beegfs-meta beegfs-storage beegfs-utils mdadm sysstat iftop htop preload ulatency ulatencyd -y

sudo /opt/beegfs/sbin/beegfs-setup-meta -p /data/beegfs/beegfs_meta -m $1 -f
sudo /opt/beegfs/sbin/beegfs-setup-storage -p $2 -m $1 -f

sudo echo "tuneNumWorkers=64" >> /etc/beegfs/beegfs-storage.conf
sudo echo "connMaxInternodeNum=256" >> /etc/beegfs/beegfs-meta.conf
sudo echo "tuneTargetChooser=roundrobin" >> /etc/beegfs/beegfs-meta.conf

sudo systemctl enable beegfs-meta
sudo systemctl start beegfs-meta
sudo systemctl enable beegfs-storage
sudo systemctl start beegfs-storage
