#!/bin/sh

echo "Staring LWPU Fabric Manager LWLink multinode extension package installation"

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
cp ${PWD}/dgxa100_all_to_all_9node_topology   /usr/share/lwpu/lwswitch/
cp ${PWD}/dgxa100_all_to_all_9node_trunk.csv  /usr/share/lwpu/lwswitch/

# enable Fabric Manager service
systemctl enable lwpu-fabricmanager

# let the user start FM service manually.
echo "Fabric Manager LWLink multinode extension package installation completed."
echo "Note: Fabric Manager service is not started. Start it using systemctl commands (like systemctl start lwpu-fabricmanager)"
