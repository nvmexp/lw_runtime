###############################################################################
#
# Copyright 1993-2017 LWPU Corporation.  All rights reserved.
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
###############################################################################
#
# LWCA Samples: per samples dynamic makefile generator
#
###############################################################################

# To update copyright year for all samples sources execute command below
# from parent directory.
# find . -type f -exec sed -i 's/2018, LWPU CORPORATION/2019, LWPU CORPORATION/g' {} +

ifeq ($(USE_P4),1)
	USE_NEW_PROJECT_MK := 1
	include ../../../build/common.mk
	REMOVE_SMS := 21
	ALL_COMPLIANT_SMS_ARCH := $(filter-out $(REMOVE_SMS), $(SASS_ARCHITECTURES))
	ADDITIONAL_OPTIONS += --lwca-version=$(DEFAULT_COMPATIBILITY_VERSION)
else # This is the git path - for this we need to now hard-code supported SMs as no lwconfig here
	ALL_COMPLIANT_SMS_ARCH := 30 35 37 50 52 60 61 70 72 75
	ADDITIONAL_OPTIONS += --lwca-version="10.2"
endif

# Project folders that contain LWCA samples
PROJECTS ?= $(shell find ../Samples -name info.xml | sed s/info.xml//)

ifeq ($(TARGET_GEN_OS),windows)
	GEN_BUILD_FILE ?= vs2012, vs2013, vs2015, vs2017, vs2019
else
	GEN_BUILD_FILE ?= makefile, vs2012, vs2013, vs2015, vs2017, vs2019
endif

AVAILABLE_DEPENDENCIES ?= LWBLAS,LWSPARSE,LWSOLVER,LWRAND,LWFFT,NPP,LWRTC,LWGRAPH,LWJPEG,X11,screen,OpenMP,IPC,MPI,GL,GLES,DirectX,DirectX12,UVM,FreeImage,CDP,callback,only-64-bit,Stream-Priorities,EGLOutput,FP16,CPP11,EGL,MBCG,MDCG,EGLSync,VULKAN,LWSCI

ifneq (,$(AVAILABLE_DEPENDENCIES))
ADDITIONAL_OPTIONS += --available-dependencies=$(AVAILABLE_DEPENDENCIES)
endif

ifneq (,$(TARGET_GEN_OS))
ADDITIONAL_OPTIONS += --target-os=$(TARGET_GEN_OS)
endif

ifneq (,$(TARGET_GEN_ARCH))
ADDITIONAL_OPTIONS += --target-arch=$(TARGET_GEN_ARCH)
endif

PERL ?= perl

all: generate_makefiles generate_nsight_projects generate_readmes

generate_makefiles:
	$(PERL) ./generate_builders.pl --info=../Samples --type="$(GEN_BUILD_FILE)" --supported-sms="$(value ALL_COMPLIANT_SMS_ARCH)" --generate-readme-md $(ADDITIONAL_OPTIONS)

generate_nsight_projects:
ifneq ($(TARGET_GEN_OS),win32)
	$(PERL) ./InfoXMLToNsightEclipseXML_Gen.pl --samples-list="./dvs/samples_list.txt" --samples-rootdir="../Samples" --supported-sms="$(value ALL_COMPLIANT_SMS_ARCH)"
endif

install_samples_script:
ifeq ($(VULCAN),1)
ifneq ($(OS),win32)
	cp -f ./build/scripts/lwca-install-samples.sh $(VULCAN_BUILD_DIR)/lwca-install-samples-test.sh
	sed -i -e 's,<VERSION>,test,g' $(VULCAN_BUILD_DIR)/lwca-install-samples-test.sh
endif
endif

generate_readmes:
	$(PERL) ./generate_builders.pl --info=../Samples --supported-sms="$(value ALL_COMPLIANT_SMS_ARCH)" --generate-readme-md $(ADDITIONAL_OPTIONS)

# Bug 200099870: create samples.vlct separately
generate_eris_vlct:
	$(PERL) ./build/scripts/generate_builders.pl --info=./ --type=eris_$(VULCAN_ARCH) --available-dependencies="$(AVAILABLE_DEPENDENCIES)"

cross_build_samples_ppc64le:
ifeq ($(VULCAN),1)
ifneq ($(OS),win32)
	$(MAKE) -C $(VULCAN_INSTALL_DIR)/lwca/samples LWDA_PATH=$(VULCAN_INSTALL_DIR)/lwca TARGET_ARCH=ppc64le CPLUS_INCLUDE_PATH=$(VULCAN_INSTALL_DIR)/lwca/targets/ppc64le-linux/include clean
	$(MAKE) -C $(VULCAN_INSTALL_DIR)/lwca/samples LWDA_PATH=$(VULCAN_INSTALL_DIR)/lwca TARGET_ARCH=ppc64le CPLUS_INCLUDE_PATH=$(VULCAN_INSTALL_DIR)/lwca/targets/ppc64le-linux/include
endif
endif

samples_tests_ppc64le: generate_eris_vlct cross_build_samples_ppc64le

%.ph_clean : 
	@rm -rf $*Makefile
	@rm -rf $*NsightEclipse.xml
	@rm -rf $*README.md
	@rm -rf $*findgllib.mk
	@rm -rf $*findvulkan.mk
	@rm -rf $*findegl.mk
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2010.sln
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2010.vcxproj
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2012.sln
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2012.vcxproj
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2013.sln
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2013.vcxproj
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2015.sln
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2015.vcxproj
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2017.sln
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2017.vcxproj
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2019.sln
	@rm -rf $*$(shell echo $* | cut -d '/' -f3)_vs2019.vcxproj

tidy:
	@find ../* | egrep "#" | xargs rm -f
	@find ../* | egrep "*~" | xargs rm -f

clean: tidy $(addsuffix .ph_clean,$(PROJECTS))

DONOTHING:
