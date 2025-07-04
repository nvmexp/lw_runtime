LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_CPP_EXTENSION := .cc .cpp .cxx
LOCAL_MODULE:=shaderc
LOCAL_EXPORT_C_INCLUDES:=$(LOCAL_PATH)/include
LOCAL_SRC_FILES:=src/shaderc.cc
# The Shaderc third_party/Android.mk deduces SPVHEADERS_LOCAL_PATH,
# or delegates that responsibility to SPIRV-Tools' Android.mk.
LOCAL_C_INCLUDES:=$(LOCAL_PATH)/include $(SPVHEADERS_LOCAL_PATH)/include
LOCAL_STATIC_LIBRARIES:=shaderc_util SPIRV-Tools-opt
LOCAL_CXXFLAGS:=-std=c++11 -fno-exceptions -fno-rtti -DENABLE_HLSL=1
LOCAL_EXPORT_CPPFLAGS:=-std=c++11
LOCAL_EXPORT_LDFLAGS:=-latomic
include $(BUILD_STATIC_LIBRARY)
