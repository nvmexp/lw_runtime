
TOP              = $(_HERE_)/..

LWVMIR_LIBRARY_DIR = $(TOP)/$(_LWVM_BRANCH_)/libdevice

PATH            += $(TOP)/$(_LWVM_BRANCH_)/bin;$(_HERE_);$(TOP)/lib;

INCLUDES        +=  "-I$(TOP)/include" $(_SPACE_)

LIBRARIES        =+ $(_SPACE_) "/LIBPATH:$(TOP)/lib/$(_WIN_PLATFORM_)"

LWDAFE_FLAGS    +=
PTXAS_FLAGS     +=
