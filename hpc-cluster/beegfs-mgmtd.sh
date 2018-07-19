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
sudo apt install beegfs-mgmtd beegfs-helperd beegfs-client beegfs-mon beegfs-utils htop iftop preload ulatency ulatencyd -y
sudo /opt/beegfs/sbin/beegfs-setup-mgmtd -p /data/beegfs/beegfs_mgmtd

sudo echo "tuneStorageSpaceLowLimit=100G" >> /etc/beegfs/beegfs-mgmtd.conf
sudo echo "tuneStorageSpaceEmergencyLimit=10G" >> /etc/beegfs/beegfs-mgmtd.conf

sudo systemctl enable beegfs-mgmtd
sudo systemctl start beegfs-mgmtd

sudo /opt/beegfs/sbin/beegfs-setup-client -m $HOSTNAME

sudo sed -i "s|/mnt/beegfs|$1|g" /etc/beegfs/beegfs-mounts.conf
sudo mkdir -p $1

sudo echo "connMaxInternodeNum=128" >> /etc/beegfs/beegfs-client.conf
sudo echo "sysMountSanityCheckMS=0" >> /etc/beegfs/beegfs-client.conf

sudo systemctl enable beegfs-helperd
sudo systemctl enable beegfs-client
