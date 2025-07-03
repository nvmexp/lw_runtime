#!/bin/bash

/usr/local/lwca/bin/lwcc -ptx -m64 -o DcgmProfTesterKernels.ptx DcgmProfTesterKernels.lw || die "Failed to compile DcgmProfTesterKernels.lw"
