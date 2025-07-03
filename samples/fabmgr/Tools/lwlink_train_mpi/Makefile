MODULE_NAME="lwlink_train_multi_node"

SOURCE_DIR=../../../..
FM_DIR = $(SOURCE_DIR)/apps/fabricmanager
LWRRENT_DIR=$(FM_DIR)/Tools/lwlink_train_mpi
LWMLCOMMON_DIR = $(SOURCE_DIR)/apps/lwml/common
OUTPUT_DIR := $(LWRRENT_DIR)/_out

CXX=mpicc

MKDIR_P = mkdir -p
CREATE_BUILD_DIR := $(shell mkdir -p $(OUTPUT_DIR))

INCLUDES := $(LWRRENT_DIR)
INCLUDES += $(FM_DIR)/Tools/libs/jsoncpp-0.10.6/include
INCLUDES += $(SOURCE_DIR)/drivers/resman/inc/libraries 
INCLUDES += $(SOURCE_DIR)/drivers/lwswitch/user/Linux
INCLUDES += $(SOURCE_DIR)/drivers/lwswitch/user/
INCLUDES += $(SOURCE_DIR)/drivers/lwlink/user/lwlink
INCLUDES += $(SOURCE_DIR)/drivers/lwlink/interface

INCLUDES += $(SOURCE_DIR)/drivers/common/inc
INCLUDES += $(SOURCE_DIR)/sdk/lwpu/inc
INCLUDES += $(LWMLCOMMON_DIR)
INCLUDES += $(SOURCE_DIR)/sdk/lwpu/inc/ctrl/
INCLUDES += $(SOURCE_DIR)/sdk/lwpu/inc/ctrl/ctrl0000/
INCLUDES += $(SOURCE_DIR)/sdk/lwpu/inc/ctrl/ctrl2080

INCLUDES += $(SOURCE_DIR)/drivers/lwswitch/interface
INCLUDES += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/utils
INCLUDES += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/include
INCLUDES += $(SOURCE_DIR)/drivers/unix/common/inc

INCLUDES += $(SOURCE_DIR)/apps/fabricmanager/common/

CPPFLAGS := $(addprefix -I,$(INCLUDES))
CPPFLAGS += -DLW_LINUX

LOCAL_SRC := $(LWRRENT_DIR)/lwlink_train_main.cpp
LOCAL_SRC += $(LWRRENT_DIR)/lwlink_train_steps.cpp
LOCAL_SRC += $(LWRRENT_DIR)/lwlink_train_cmd_parser.cpp
LOCAL_SRC += $(SOURCE_DIR)/drivers/resman/src/libraries/commandline/commandline.c
LOCAL_SRC += $(FM_DIR)/Tools/libs/jsoncpp-0.10.6/src/jsoncpp.cpp
LOCAL_SRC += $(LWRRENT_DIR)/LocalFMGpuMgr.cpp
LOCAL_SRC += $(LWRRENT_DIR)/helper.cpp
LOCAL_SRC += $(LWRRENT_DIR)/master.cpp
LOCAL_SRC += $(LWRRENT_DIR)/slave.cpp


LOCAL_SRC_EXT = $(LWMLCOMMON_DIR)/Linux/lwosLinux.c
LOCAL_SRC_EXT += $(LWMLCOMMON_DIR)/lwos_common_posix.c
LOCAL_SRC_EXT += $(LWMLCOMMON_DIR)/simple_encryption.c

# #include lwswitch src code
LOCAL_SRC_EXT  += $(SOURCE_DIR)/drivers/lwswitch/user/Linux/lwswitch_user_linux.c
LOCAL_SRC_EXT  += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/utils/lwpu-modprobe-utils.c
LOCAL_SRC_EXT  += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/utils/lwpu-modprobe-client-utils.c
LOCAL_SRC_EXT  += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/utils/pci-sysfs.c

LOCAL_SRC += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/rmapi.c
LOCAL_SRC += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/rmapi-numa.c
LOCAL_SRC += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/rmapi-mmap.c
LOCAL_SRC += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/rmapi-control-device.c
LOCAL_SRC += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/rmapi-control-device-unix.c
LOCAL_SRC += $(SOURCE_DIR)/drivers/resman/arch/lwalloc/unix/lib/utils/mem-sysfs.c
LOCAL_SRC += $(SOURCE_DIR)/drivers/common/shared/lwstatus/lwstatus.c

LOCAL_OBJ := $(LOCAL_SRC:.cpp=.o) $(LOCAL_SRC_EXT:.c=.o)

LDFLAGS =  -static-libstdc++ -static-libgcc -rdynamic -ldl -lpthread -lmpi_cxx

$(OUTPUT_DIR)/lwlink_train: $(LOCAL_OBJ)
	@$(CXX) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	@find ./../../../../ -name '*.o' -delete
	@rm -rf myprog *.o ../../../../drivers/resman/arch/lwalloc/unix/lib/utils/*.o $(OUTPUT_DIR)
