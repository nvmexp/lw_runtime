LWML = /home/rtodd/p4/sw/dev/gpu_drv/chips_a/apps/lwml
CPP = g++
CFLAGS = -I $(LWML)/_out/Linux_amd64_release/ -I ./ext_includes
LDFLAGS = -L $(LWML)/_out/Linux_amd64_release/  
LIBS = -llwidia-ml 

OBJ = LwidiaValidationSuite.o \
	  Gpu.o \
      main.o

all: lwpu-vs

%.o: %.cpp 
		$(CPP) -c -o $@ $< $(CFLAGS)

lwpu-vs: $(OBJ)
		$(CPP) -o $@ $^ $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
		rm -rf *.o
		rm -rf lwpu-vs
