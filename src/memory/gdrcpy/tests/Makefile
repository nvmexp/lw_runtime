DESTBIN ?= 
LWCA ?= /usr/local/lwca

# note that pkg-config respects PKG_CONFIG_PATH, which can be used
# to pick libcheck from a user-defined path
CHECK_INC  := $(shell pkg-config --cflags-only-I check)
CHECK_LIBS := $(shell pkg-config --libs check)

GDRAPI_INC := ../include
GDRAPI_SRC := ../src

LWDA_LIB := -L $(LWCA)/lib64 -L $(LWCA)/lib -L /usr/lib64/lwpu -L /usr/lib/lwpu
LWDA_INC += -I $(LWCA)/include

CPPFLAGS := $(LWDA_INC) -I $(GDRAPI_INC) -I $(GDRAPI_SRC) -I $(LWCA)/include $(CHECK_INC)
LDFLAGS  := $(LWDA_LIB) -L $(LWCA)/lib64 -L $(GDRAPI_SRC)
COMMONCFLAGS := -O2
CFLAGS   += $(COMMONCFLAGS)
CXXFLAGS += $(COMMONCFLAGS)
LIBS     := -llwda -lpthread -ldl -lgdrapi

SRCS := copybw.cpp sanity.cpp copylat.cpp apiperf.cpp
EXES := $(SRCS:.cpp=)

all: exes

exes: $(EXES)

common.o: common.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
copybw.o: copybw.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
sanity.o: sanity.cpp $(GDRAPI_INC)/gdrapi.h $(GDRAPI_SRC)/gdrapi_internal.h common.hpp
copylat.o: copylat.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
apiperf.o: apiperf.cpp $(GDRAPI_INC)/gdrapi.h common.hpp

copybw: copybw.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) -lrt

sanity: sanity.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) $(CHECK_LIBS)

copylat: copylat.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) -lrt

apiperf: apiperf.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) -lrt

clean:
	rm -f *.o $(EXES) *~ core.*

install: exes
	@ echo "installing exes in $(DESTBIN)..." && \
	mkdir -p $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx copybw -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx copylat -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx apiperf -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx sanity -t $(DESTBIN)

.PHONY: clean all exes install
