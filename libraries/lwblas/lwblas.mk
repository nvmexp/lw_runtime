
I_AM_SLOPPY:=1

#MIN_ARCH := 60
#USE_AGNOSTIC_TOOLCHAIN := 1
#LWDA_TOOLCHAIN_VER := gcc5.3-glibc2.12

#Bug 731791: force to always compile libray with ICC off
#Need to be overidden before include config/$(PROFILE).mk = config/Linux.mk
override USEICC=
#Bug 200494750: update the ROOTDIR
override ROOTDIR := $(subst %/,%,$(realpath $(or \
        ,${ROOTDIR}\
        ,${VULCAN_TOOLKIT_BASE}\
        ,../../..)))


ifndef PROFILE
  include $(ROOTDIR)/build/getprofile.mk
  include $(ROOTDIR)/build/config/${PROFILE}.mk
endif

SHARED_LIBRARY := lwtensor

COMPATIBILITY_VERSION := ${GET_DSO_MAJOR}


ifdef VULCAN
  INCLUDES_ABSPATH += $(VULCAN_INSTALL_DIR)/lwca/include
  SYSTEMLIBDIRS += $(VULCAN_INSTALL_DIR)/lwca/$(VULCAN_LIBSUBDIR)
  # TBD add libs here once needed
else
  INCLUDES_ABSPATH += ${ROOTDIR}/lwca/common

  INCLUDES += include
  INCLUDES += include/lwtensor
  INCLUDES += include/lwtensor/internal
  INCLUDES_ABSPATH += ${ROOTDIR}/lwblas/src
  INCLUDES_ABSPATH += ${ROOTDIR}/lwca/tools/lwdart
  INCLUDES_ABSPATH += ${ROOTDIR}/thrust
  INCLUDES_ABSPATH += ${ROOTDIR}/lwtx/headers/interface
endif

ifeq ($(RELEASE), 1)
  CFLAGS += -D__RELEASE__ -DNDEBUG
endif


ifeq ($(OS),Linux)
LIBRARIES += m \
	$(if $(filter QNX,${TARGET_OS}),c)
else ifeq (${OS},win32)
    ifndef USEVC7
        DEFINES += _CRT_SELWRE_NO_WARNINGS
        DEFINES += _CRT_NONSTDC_NO_WARNINGS
    endif
endif

CXX_STD:=c++11
LWCC_LW_OPTIONS += -Xfatbin -compress-all
CFLAGS+= -O3 -DTENSOR_CONTRACTIONS

ARCH_NEG_FILTER = 21 30 35 37 50 52 82

FILES += src/lwtensor.cpp
FILES += src/util.cpp
FILES += src/utilPLC3.cpp
FILES += src/types.cpp
FILES += src/typesPLC3.cpp
FILES += src/elementwise.cpp
FILES += src/elementwisePLC3.cpp

<<<<<<< HEAD
LW_FILES += src/tensorElementwise_cccc.lw
=======
LW_FILES += src/tensorElementwise_dispatcher.lw
>>>>>>> core
LW_FILES += src/tensorElementwise_dddd.lw
LW_FILES += src/tensorElementwise_ddds.lw
LW_FILES += src/tensorElementwise_ddss.lw
LW_FILES += src/tensorElementwise_dispatcher.lw
LW_FILES += src/tensorElementwise_hhhh.lw
LW_FILES += src/tensorElementwise_hhss.lw
LW_FILES += src/tensorElementwise_hshs.lw
LW_FILES += src/tensorElementwise_hsi8s.lw
LW_FILES += src/tensorElementwise_hsss.lw
LW_FILES += src/tensorElementwise_i8shs.lw
LW_FILES += src/tensorElementwise_i8si8s.lw
LW_FILES += src/tensorElementwise_i8sss.lw
LW_FILES += src/tensorElementwise_iiii.lw
LW_FILES += src/tensorElementwise_shss.lw
LW_FILES += src/tensorElementwise_ssdd.lw
LW_FILES += src/tensorElementwise_sshh.lw
LW_FILES += src/tensorElementwise_sshs.lw
LW_FILES += src/tensorElementwise_ssi8s.lw
LW_FILES += src/tensorElementwise_ssss.lw
LW_FILES += src/tensorElementwise_u8u8u8u8.lw
LW_FILES += src/tensorElementwise_u8u8uu.lw
LW_FILES += src/tensorElementwise_uu8uu.lw

FILES += src/typesEx.cpp
FILES += src/utilEx.cpp

LW_FILES += src/tensorContraction_auto.lw
LW_FILES += src/tensorContraction_cccc.lw
LW_FILES += src/tensorContraction.lw
LW_FILES += src/tensorContraction_dddd.lw
LW_FILES += src/tensorContraction_ddds.lw
LW_FILES += src/tensorContraction_hhhs.lw
LW_FILES += src/tensorContraction_sssd.lw
LW_FILES += src/tensorContraction_ssss.lw
LW_FILES += src/tensorContraction_zzzz.lw



RESOURCE ?= lwtensor

DEF_MASTER   = lwtensor.def
LD_DEF_FILE ?= lwtensor

include $(ROOTDIR)/build/common.mk
