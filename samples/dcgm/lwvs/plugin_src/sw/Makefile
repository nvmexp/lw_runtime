LWML = ../../../../lwml
COMMON_DIR = ../../../../lwml/common
CPP = g++
CFLAGS = -I $(LWML)/_out/Linux_amd64_release/ -I ../../include -I$(COMMON_DIR) -fPIC -D_UNIX -DLW_UNIX 
DEBUG_CFLAGS = -g -D_DEBUG
ifdef DEBUG
  CFLAGS += $(DEBUG_CFLAGS)
else
  CFLAGS += -O3
endif

LDFLAGS = -shared

OBJ = Software.o Plugin_common.o

all: libDeployment.so
		cp libDeployment.so ../../plugins/libDeployment.so

%.o: %.cpp 
		$(CPP) -c -o $@ $< $(CFLAGS)

libDeployment.so: $(OBJ)
		$(CPP) -o $@ -Wl,-soname,$@ Software.o Plugin_common.o $(LDFLAGS)

Plugin_common.o:
	$(CXX) -c ../Plugin_common.cpp $(CFLAGS)

clean:
		rm -rf *.o
		rm -rf *.so


