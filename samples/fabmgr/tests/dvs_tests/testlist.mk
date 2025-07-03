################ Tests go here ################

TARGETS_MAKEFILE ?=


ifeq ($(LW_TARGET_OS_FAMILY),Unix)
  ifdef LWCFG_INITIALIZED
    ifeq ($(LWCFG_GLOBAL_FEATURE_RID72837_KT_MULTINODE),1)
        TARGETS_MAKEFILE += imex/imex.lwmk
    endif
  endif
endif

# This isn't a normal FM test, it's an exelwtable that will display various
# info about the GPU.  It is meant to be used at runtime in conjunction with
# scripts to report which tests are/aren't available on the current
# platform.
TARGETS_MAKEFILE += utils/testList/testListHelper.lwmk
