#
# Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#
NCCL_HOME:=../../build/
LWDA_HOME:=/usr/local/lwca
INC:= -I$(NCCL_HOME)/include -I$(LWDA_HOME)/include
PLUGIN_SO:=libnccl-net.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): plugin.c
	$(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^

clean:
	rm -f $(PLUGIN_SO)
