#!/bin/sh

echo "Staring LWPU Fabric Manager installation"

echo "Checking for running Fabric Manager service"

STATUS=`systemctl is-active lwpu-fabricmanager`
  if [ ${STATUS} = 'active' ]; then
    echo "Fabric Manager service is running, stopping the same....."
    systemctl stop lwpu-fabricmanager
  else 
    echo "Fabric Manager service is not running....."
  fi

# copy all the files
echo "Copying files to desired location"
# choose lib location based on system arch
ARCH_TYPE_CMD=`uname -m`
LIB_LOC="/usr/lib/x86_64-linux-gnu"
if [ ${ARCH_TYPE_CMD} = 'aarch64' ]; then
    echo "detected aarch64 and changing lib location to /usr/lib/aarch64-linux-gnu...."
    LIB_LOC="/usr/lib/aarch64-linux-gnu"
fi

cp ${PWD}/liblwfm.so.1 ${LIB_LOC}
cp -P ${PWD}/liblwfm.so   ${LIB_LOC}

cp ${PWD}/lw-fabricmanager  /usr/bin
cp ${PWD}/lwswitch-audit  /usr/bin
cp ${PWD}/lwpu-fabricmanager.service  /lib/systemd/system

mkdir /usr/share/lwpu  > /dev/null 2>&1
mkdir /usr/share/lwpu/lwswitch/  > /dev/null 2>&1
cp ${PWD}/dgx2_hgx2_topology    /usr/share/lwpu/lwswitch/
cp ${PWD}/dgxa100_hgxa100_topology    /usr/share/lwpu/lwswitch/
cp ${PWD}/fabricmanager.cfg  /usr/share/lwpu/lwswitch/

cp ${PWD}/lw_fm_agent.h     /usr/include
cp ${PWD}/lw_fm_types.h     /usr/include

mkdir /usr/share/doc/lwpu-fabricmanager > /dev/null 2>&1
cp ${PWD}/LICENSE  /usr/share/doc/lwpu-fabricmanager
cp ${PWD}/third-party-notices.txt  /usr/share/doc/lwpu-fabricmanager

# enable Fabric Manager service
systemctl enable lwpu-fabricmanager

# let the user start FM service manually.
echo "Fabric Manager installation completed."
echo "Note: Fabric Manager service is not started. Start it using systemctl commands (like systemctl start lwpu-fabricmanager)"
