
default: test

test:   test.c ../inc/lwdspstr.h
	cc -g -o test -I../inc -I$(LWSDK_ROOT)/inc test.c

clean:
	-rm -rf test test.o

FORCE:
