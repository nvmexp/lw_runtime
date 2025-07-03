#!/bin/sh

# Example shell script to launch LWSwitch test suite on all devices available in the system

devices=$(lspci | grep "Bridge: LWPU" | wc -l);
i=0;

while [ $i -lt $devices ]; do
        echo
        echo "Running tests on LWSwitch "$i;
        echo
        sudo ./lwswitch -i $i $@;
        i=$((i+1))
done
