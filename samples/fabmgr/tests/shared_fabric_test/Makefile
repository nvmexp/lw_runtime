IDIR := /usr/include
CXXFLAGS = -I $(IDIR) -I $(PWD) -I $(PWD)/utils/jsoncpp-0.10.6/include -I $(PWD)/utils/commandline
CC = g++
LDIR := /usr/lib
LDFLAGS= -L$(LDIR) -llwfm

shared_fabric_test: platformModelDelta.o sharedFabricTest.o sharedFabricTestParser.o $(PWD)/utils/jsoncpp-0.10.6/src/jsoncpp.o $(PWD)/utils/commandline/commandline.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	-@rm -f *.o
	-@rm -f shared_fabric_test
