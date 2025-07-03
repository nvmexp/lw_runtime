#!/bin/ksh
##########################################################################################
# Copyright (c) 2020-2021, LWPU CORPORATION. All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.
##########################################################################################
#
# TEST SCRIPT for INSTRUMENTING TEST
#

RM=/tmp/rm
TOUCH=/tmp/touch

# create initial VectorCAST DAT files
$TOUCH /tmp/LwSciIpc_io_lwsciipc.DAT
$TOUCH /tmp/LwSciIpc_lwsciipc.DAT
$TOUCH /tmp/LwSciIpc_lwscievent.DAT
$TOUCH /tmp/LwSciIpc_lwsciipc_init.DAT
# change permission of DAT files w/ 0666 to remove DAT file access permission issue
chmod 666 /tmp/*.DAT

# run requirement-based test script
# use /tmp folder for unit binary path
. /tmp/lwsciipc_int_tests.ksh inst /proc/boot skipinit

ls -al /tmp/*.DAT
