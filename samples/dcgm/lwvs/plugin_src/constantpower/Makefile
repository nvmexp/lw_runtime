#/usr/local/lwca/bin/lwcc -c constantpower.c -I ../../../../lwml/_out/Linux_amd64_release/ -llwblas -L ../../../../lwml/_out/Linux_amd64_release/ -llwidia-ml -Xcompiler -fpic
#g++ -c ConstantPower_wrapper.cpp -I ../ -I ../../include/ -fPIC
#g++ -o libConstantPower.so -Wl,-soname,libConstantPower.so constantpower.o ConstantPower_wrapper.o -shared

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

libPower.so : constantpower.o ConstantPower_wrapper.o Plugin_common.o
	$(CXX) -o $@ -Wl,-soname,$@ constantpower.o ConstantPower_wrapper.o Plugin_common.o $(LDFLAGS)
	cp libPower.so ../../plugins/libPower.so

constantpower.o:
	$(LWCC) -c constantpower.cpp $(LWCFLAGS)

ConstantPower_wrapper.o:
	$(CXX) -c ConstantPower_wrapper.cpp $(CCCFLAGS) 

Plugin_common.o:
	$(CXX) -c ../Plugin_common.cpp $(CCCFLAGS)


clean:
	rm -f *.o
	rm -f *.so
