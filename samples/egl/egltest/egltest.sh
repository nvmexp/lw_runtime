#
# Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.

# set Fifo length (default 4)
fifoLen=${FIFO_LEN:-4}

# Test eglstream single-process mailbox mode
# and non-eglstream interfaces
./egltest

# Test eglstream single-process fifo mode
./egltest -t stream -l $fifoLen

# Test eglstream cross-process mailbox mode
./egltest -t stream -p consumer -n 1 &
./egltest -t stream -p producer -n 1
wait $!

./egltest -t stream -p consumer -n 2 &
./egltest -t stream -p producer -n 2
wait $!

./egltest -t stream -p consumer -n 3 &
./egltest -t stream -p producer -n 3
wait $!

./egltest -t stream -p consumer -n 3 &
./egltest -t stream -p producer -n 3
wait $!

./egltest -t stream -p consumer -n 4 &
./egltest -t stream -p producer -n 4
wait $!

./egltest -t stream -p consumer -n 5 &
./egltest -t stream -p producer -n 5
wait $!

# Test eglstream cross-process fifo mode
./egltest -t stream -l $fifoLen -p consumer -n 1 &
./egltest -t stream -l $fifoLen -p producer -n 1
wait $!

./egltest -t stream -l $fifoLen -p consumer -n 2 &
./egltest -t stream -l $fifoLen -p producer -n 2
wait $!

./egltest -t stream -l $fifoLen -p consumer -n 3 &
./egltest -t stream -l $fifoLen -p producer -n 3
wait $!

./egltest -t stream -l $fifoLen -p consumer -n 3 &
./egltest -t stream -l $fifoLen -p producer -n 3
wait $!

./egltest -t stream -l $fifoLen -p consumer -n 4 &
./egltest -t stream -l $fifoLen -p producer -n 4
wait $!

./egltest -t stream -l $fifoLen -p consumer -n 5 &
./egltest -t stream -l $fifoLen -p producer -n 5
wait $!

# Test eglstream2 single-process mailbox mode
./egltest -t stream2 -n 1
./egltest -t stream2 -n 5

# Test eglstream2 single-process fifo mode
./egltest -t stream2 -l $fifoLen -n 1
./egltest -t stream2 -l $fifoLen -n 5

# Test eglstream2 cross-process mailbox mode
./egltest -t stream2 -p consumer -n 1 &
./egltest -t stream2 -p producer -n 1
wait $!

./egltest -t stream2 -p consumer -n 5 &
./egltest -t stream2 -p producer -n 5
wait $!

# Test eglstream2 cross-process fifo mode
./egltest -t stream2 -l $fifoLen -p consumer -n 1 &
./egltest -t stream2 -l $fifoLen -p producer -n 1
wait $!

./egltest -t stream2 -l $fifoLen -p consumer -n 5 &
./egltest -t stream2 -l $fifoLen -p producer -n 5
wait $!

#
# Instructions to test eglstream cross-partition
#
# on VM0: ifconfig hv0 12.0.0.1 up
# on VM1: ifconfig hv0 12.0.0.2 up
#

# on VM0: ./egltest -t stream -p consumer -n 1 -c 1
# on VM1: ./egltest -t stream -p producer -n 1 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -p consumer -n 2 -c 1
# on VM1: ./egltest -t stream -p producer -n 2 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -p consumer -n 3 -c 1
# on VM1: ./egltest -t stream -p producer -n 3 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -p consumer -n 4 -c 1
# on VM1: ./egltest -t stream -p producer -n 4 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -p consumer -n 5 -c 1
# on VM1: ./egltest -t stream -p producer -n 5 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -l $fifoLen -p consumer -n 1 -c 1
# on VM1: ./egltest -t stream -l $fifoLen -p producer -n 1 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -l $fifoLen -p consumer -n 2 -c 1
# on VM1: ./egltest -t stream -l $fifoLen -p producer -n 2 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -l $fifoLen -p consumer -n 3 -c 1
# on VM1: ./egltest -t stream -l $fifoLen -p producer -n 3 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -l $fifoLen -p consumer -n 4 -c 1
# on VM1: ./egltest -t stream -l $fifoLen -p producer -n 4 -c 1 -v 0 -i 12.0.0.1

# on VM0: ./egltest -t stream -l $fifoLen -p consumer -n 5 -c 1
# on VM1: ./egltest -t stream -l $fifoLen -p producer -n 5 -c 1 -v 0 -i 12.0.0.1
