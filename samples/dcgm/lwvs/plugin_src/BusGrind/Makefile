LWDA_DIR = ../../lwca/
UNAME = $(shell uname -m)

ifeq ($(UNAME), x86_64)
ARCH = amd64
else
ARCH = $(UNAME)
endif

LWML_DIR = ../../../../lwml/_out/Linux_$(ARCH)_release/
COMMON_DIR = ../../../../lwml/common
CXX      = g++
LDFLAGS  = -shared -L $(LWDA_DIR)/lib/$(ARCH) -llwdart -lgomp 
CFLAGS   = -I $(LWDA_DIR)/include -I ../../include/ -I ../../../../../sdk/lwpu/inc -I $(LWML_DIR) -I$(COMMON_DIR) -L $(LWML_DIR) -D_UNIX -DLW_UNIX 
DEBUG_CFLAGS = -g -D_DEBUG
ifdef DEBUG
  CFLAGS += $(DEBUG_CFLAGS)
else
  CFLAGS += -O3
endif

CCCFLAGS = $(CFLAGS) -fPIC

libPcie.so : BusGrindMain.o BusGrind.o Plugin_common.o
	$(CXX) -o $@ -Wl,-soname,$@ BusGrindMain.o BusGrind.o Plugin_common.o $(LDFLAGS)
	cp libPcie.so ../../plugins/libPcie.so

BusGrindMain.o:
	$(CXX) -c BusGrindMain.cpp $(CCCFLAGS)

BusGrind.o:
	$(CXX) -c BusGrind.cpp $(CCCFLAGS) 

Plugin_common.o:
	$(CXX) -c ../Plugin_common.cpp $(CCCFLAGS) 

clean:
	rm -f *.o
	rm -f *.so


