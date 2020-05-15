#! /bin/bash

###################################################
# Set EFA version
###################################################

EFA_VERSION=1.8.4

###################################################
# Download and install EFA driver
###################################################
cd

curl -O  https://s3-us-west-2.amazonaws.com/aws-efa-installer/aws-efa-installer-${EFA_VERSION}.tar.gz

tar -xf aws-efa-installer-${EFA_VERSION}.tar.gz

cd aws-efa-installer

sudo ./efa_installer.sh -y

sudo sed -i 's/kernel.yama.ptrace_scope = 1/kernel.yama.ptrace_scope = 0/g' \
	/etc/sysctl.d/10-ptrace.conf
