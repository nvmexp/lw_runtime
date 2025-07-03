#!/bin/bash

printf "\n******** lwPHYTools ********\n"

mkdir -p install
rm -rf install/*

cp scripts/setup_system.sh install

printf "===> Building the Receiver...\n"
make -f Makefile.recv cclean
make -f Makefile.recv
cp build/lwPHYTools_receiver install
cp scripts/receiver.sh install

printf "\n===> Building the Generator...\n"
make -f Makefile.gen cclean
make -f Makefile.gen
cp build/lwPHYTools_generator install
cp scripts/generator.sh install

printf "\n===> Files installed in $PWD/install\n"