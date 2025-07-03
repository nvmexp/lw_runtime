#
# Copyright (c) 2015-2019, LWPU CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

LWDA_HOME ?= /usr/local/lwca
PREFIX ?= /usr/local
VERBOSE ?= 0
KEEP ?= 0
DEBUG ?= 0
TRACE ?= 0
PROFAPI ?= 0

LWCC = $(LWDA_HOME)/bin/lwcc

LWDA_LIB ?= $(LWDA_HOME)/lib64
LWDA_INC ?= $(LWDA_HOME)/include
LWDA_VERSION = $(strip $(shell which $(LWCC) >/dev/null && $(LWCC) --version | grep release | sed 's/.*release //' | sed 's/\,.*//'))
#LWDA_VERSION ?= $(shell ls $(LWDA_LIB)/liblwdart.so.* | head -1 | rev | cut -d "." -f -2 | rev)
LWDA_MAJOR = $(shell echo $(LWDA_VERSION) | cut -d "." -f 1)
LWDA_MINOR = $(shell echo $(LWDA_VERSION) | cut -d "." -f 2)
#$(info LWDA_VERSION ${LWDA_MAJOR}.${LWDA_MINOR})


# Better define LWCC_GENCODE in your environment to the minimal set
# of archs to reduce compile time.
LWDA8_GENCODE = -gencode=arch=compute_30,code=sm_30 \
                -gencode=arch=compute_35,code=sm_35 \
                -gencode=arch=compute_50,code=sm_50 \
                -gencode=arch=compute_60,code=sm_60 \
                -gencode=arch=compute_61,code=sm_61
LWDA9_GENCODE = -gencode=arch=compute_70,code=sm_70

LWDA8_PTX     = -gencode=arch=compute_61,code=compute_61
LWDA9_PTX     = -gencode=arch=compute_70,code=compute_70

# Include Volta support if we're using LWDA9 or above
ifeq ($(shell test "0$(LWDA_MAJOR)" -gt 8; echo $$?),0)
  LWCC_GENCODE ?= $(LWDA8_GENCODE) $(LWDA9_GENCODE) $(LWDA9_PTX)
else
  LWCC_GENCODE ?= $(LWDA8_GENCODE) $(LWDA8_PTX)
endif
#$(info LWCC_GENCODE is ${LWCC_GENCODE})

CXXFLAGS   := -DLWDA_MAJOR=$(LWDA_MAJOR) -DLWDA_MINOR=$(LWDA_MINOR) -fPIC -fvisibility=hidden
CXXFLAGS   += -Wall -Wno-unused-function -Wno-sign-compare -std=c++11 -Wvla
CXXFLAGS   += -I $(LWDA_INC)
LWLWFLAGS  := -ccbin $(CXX) $(LWCC_GENCODE) -lineinfo -std=c++11 -Xptxas -maxrregcount=96 -Xfatbin -compress-all
# Use addprefix so that we can specify more than one path
LWLDFLAGS  := -L${LWDA_LIB} -llwdart -lrt

########## GCOV ##########
GCOV ?= 0 # disable by default.
GCOV_FLAGS := $(if $(filter 0,${GCOV} ${DEBUG}),,--coverage) # only gcov=1 and debug =1
CXXFLAGS  += ${GCOV_FLAGS}
LWLWFLAGS += ${GCOV_FLAGS:%=-Xcompiler %}
LDFLAGS   += ${GCOV_FLAGS}
LWLDFLAGS   += ${GCOV_FLAGS:%=-Xcompiler %}
# $(warning GCOV_FLAGS=${GCOV_FLAGS})
########## GCOV ##########

ifeq ($(DEBUG), 0)
LWLWFLAGS += -O3
CXXFLAGS  += -O3 -g
else
LWLWFLAGS += -O0 -G -g
CXXFLAGS  += -O0 -g -ggdb3
endif

ifneq ($(VERBOSE), 0)
LWLWFLAGS += -Xptxas -v -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
CXXFLAGS  += -Wall -Wextra
else
.SILENT:
endif

ifneq ($(TRACE), 0)
CXXFLAGS  += -DENABLE_TRACE
endif

ifneq ($(KEEP), 0)
LWLWFLAGS += -keep
endif

ifneq ($(PROFAPI), 0)
CXXFLAGS += -DPROFAPI
endif
