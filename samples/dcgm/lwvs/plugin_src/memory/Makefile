LWDA_DIR = /usr/local/lwca
LWML_DIR = ../../../../lwml/_out/Linux_amd64_release/
COMMON_DIR = ../../../../lwml/common
LWCC     = $(LWDA_DIR)/bin/lwcc
CXX      = g++
LDFLAGS  = -shared -L $(LWDA_DIR)/lib64 -llwda
CFLAGS   = -I $(LWDA_DIR)/include -I ../../include/ -I ../../../../../sdk/lwpu/inc -I $(LWML_DIR) -I$(COMMON_DIR) -L $(LWML_DIR) -D_UNIX -DLW_UNIX 
DEBUG_CFLAGS = -g -D_DEBUG
ifdef DEBUG
  CFLAGS += $(DEBUG_CFLAGS)
else
  CFLAGS += -O3
endif

LWCFLAGS = $(CFLAGS) -Xcompiler -fpic
CCCFLAGS = $(CFLAGS) -fPIC

libMemory.so : memory.o Memory_wrapper.o Plugin_common.o
	$(CXX) -o $@ -Wl,-soname,$@ memory.o Memory_wrapper.o Plugin_common.o $(LDFLAGS)
	cp libMemory.so ../../plugins

memory.o:
	$(LWCC) -c memory.cpp $(LWCFLAGS)

Memory_wrapper.o:
	$(CXX) -c Memory_wrapper.cpp $(CCCFLAGS) 

Plugin_common.o:
	$(CXX) -c ../Plugin_common.cpp $(CCCFLAGS) 

clean:
	rm -f *.o
	rm -f *.so


