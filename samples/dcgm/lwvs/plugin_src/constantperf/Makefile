LWDA_DIR = /usr/local/lwca
LWML_DIR = ../../../../lwml/_out/Linux_amd64_release/
COMMON_DIR = ../../../../lwml/common
LWCC     = $(LWDA_DIR)/bin/lwcc
CXX      = g++
LDFLAGS  = -shared -L $(LWDA_DIR)/lib64 -llwdart -llwblas
CFLAGS   = -I $(LWDA_DIR)/include -I ../../include/ -I ../../../../../sdk/lwpu/inc -I $(LWML_DIR) -I$(COMMON_DIR) -L $(LWML_DIR) -D_UNIX -DLW_UNIX 
DEBUG_CFLAGS = -g -D_DEBUG
ifdef DEBUG
  CFLAGS += $(DEBUG_CFLAGS)
else
  CFLAGS += -O3
endif

LWCFLAGS = $(CFLAGS) -Xcompiler -fpic
CCCFLAGS = $(CFLAGS) -fPIC

libPerformance.so : constantperf.o ConstantPerf_wrapper.o Plugin_common.o
	$(CXX) -o $@ -Wl,-soname,$@ constantperf.o ConstantPerf_wrapper.o Plugin_common.o $(LDFLAGS)
	cp libPerformance.so ../../plugins/libPerformance.so

constantperf.o:
	$(LWCC) -c constantperf.cpp $(LWCFLAGS)

ConstantPerf_wrapper.o:
	$(CXX) -c ConstantPerf_wrapper.cpp $(CCCFLAGS) 

Plugin_common.o:
	$(CXX) -c ../Plugin_common.cpp $(CCCFLAGS)

clean:
	rm -f *.o
	rm -f *.so
