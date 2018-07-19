#!/bin/bash
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

sudo apt update
sudo apt install build-essential automake mpich wget openmpi-* -y
sudo git clone https://github.com/ThinkParQ/ior.git /home/ubuntu/ior
cd /home/ubuntu/ior
sudo /home/ubuntu/ior/bootstrap
sudo /home/ubuntu/ior/configure --prefix=/opt/ior
sudo make
sudo make install

sysctl -w net.core.rmem_max=16777216
sysctl -w net.core.wmem_max=16777216
sysctl -w net.ipv4.tcp_rmem='4096 87380 16777216'
sysctl -w net.ipv4.tcp_wmem='4096 65536 16777216'
sysctl -w net.core.netdev_max_backlog=30000
sysctl -w net.core.rmem_default=16777216
sysctl -w net.core.wmem_default=16777216
sysctl -w net.ipv4.tcp_mem='16777216 16777216 16777216'
sysctl -w net.ipv4.route.flush=1

echo "deb [arch=amd64] http://www.beegfs.io/release/beegfs_7 deb9 non-free" | sudo tee -a /etc/apt/sources.list.d/bee.list
wget -q https://www.beegfs.io/release/latest-stable/gpg/DEB-GPG-KEY-beegfs -O- | sudo apt-key add -
sudo apt update
sudo apt install beegfs-helperd beegfs-client -y
sudo /opt/beegfs/sbin/beegfs-setup-client -m $1
sudo sed -i "s|/mnt/beegfs|/mnt/parallel|g" /etc/beegfs/beegfs-mounts.conf
sudo mkdir -p /mnt/parallel
sudo systemctl start beegfs-helperd
sudo systemctl start beegfs-client
