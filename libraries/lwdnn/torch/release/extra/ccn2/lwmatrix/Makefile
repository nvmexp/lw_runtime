################################################################################
#
# Copyright 1993-2012 LWPU Corporation.  All rights reserved.
#
# NOTICE TO USER:   
#
# This source code is subject to LWPU ownership rights under U.S. and 
# international Copyright laws.  
#
# LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
# CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
# IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH 
# REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
# MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
# IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
# OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
# OR PERFORMANCE OF THIS SOURCE CODE.  
#
# U.S. Government End Users.  This source code is a "commercial item" as 
# that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
# "commercial computer software" and "commercial computer software 
# documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
# and is provided to the U.S. Government only as a commercial end item.  
# Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
# 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
# source code with only those rights set forth herein.
#
################################################################################

# Location of the LWCA Toolkit binaries and libraries
LWDA_INC_PATH  = $(LWDA_INSTALL_PATH)/include
LWDA_BIN_PATH  = $(LWDA_INSTALL_PATH)/bin
LWDA_LIB_PATH  = $(LWDA_INSTALL_PATH)/lib64

# Common binaries
LWCC            = $(LWDA_BIN_PATH)/lwcc
GCC             = g++
AR				= ar

# LWCA code generation flags
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM35)

LDFLAGS   := -L$(LWDA_LIB_PATH) -llwdart
CCFLAGS   := -m64
LWCCFLAGS := -m64

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      LWCCFLAGS += -g -G
      DBG := debug
else
      DBG := release
      LWCCFLAGS += -O3
      CCFLAGS += -O3
endif

# Add profiler output
ifeq ($(prof),1)
	LWCCFLAGS += --ptxas-options=-v
endif

TARGETDIR := ./bin/$(DBG)
OBJDIR := ./obj/$(DBG)

########## USER STUFF ###########
LDFLAGS   		+= -L../util -lutilpy -llwblas
INCLUDES      	:= -I$(LWDA_INC_PATH) -I $(LWDA_SDK_PATH)/common/inc -I./include

LWFILES	:= $(shell find . -name "*.lw")
LW_DEPS	:= $(shell find . -name "*.lwh")
CCFILES	:= $(shell find . -name "*.cpp")
C_DEPS	:= $(shell find . -name "*.h")

LWCCFLAGS += --compiler-options '-fPIC'
LDFLAGS += -shared
CCFLAGS += -fPIC
TARGET := $(TARGETDIR)/liblwmatrix.so

################################################################################
# Set up target and object files
################################################################################
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.o,$(CCFILES))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.o,$(CFILES))
OBJS +=  $(patsubst %.lw,$(OBJDIR)/%.lw.o,$(LWFILES))

# Target rules
all: makedirs $(TARGET)

$(OBJDIR)/%.lw.o : %.lw $(LW_DEPS)
	$(LWCC) $(LWCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<

$(OBJDIR)/%.cpp.o : %.cpp $(C_DEPS)
	$(GCC) $(CCFLAGS) $(INCLUDES) -o $@ -c $<

$(TARGET): $(OBJS)
	$(GCC) $(CCFLAGS) -o $@ $+ $(LDFLAGS)
	ln -sf $(TARGET) .

makedirs:
	mkdir -p $(TARGETDIR)
	mkdir -p $(OBJDIR)/src

clean:
	rm -rf ./obj
