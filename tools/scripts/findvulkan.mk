################################################################################
# Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of LWPU CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
################################################################################
#
#  findvulkan.mk is used to find the necessary Vulkan Libraries for specific distributions
#               this is supported on Linux
#
################################################################################

# Determine OS platform and unix distribution
ifeq ("$(TARGET_OS)","linux")
   # first search lsb_release
   DISTRO  = $(shell lsb_release -i -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
   ifeq ("$(DISTRO)","")
     # second search and parse /etc/issue
     DISTRO = $(shell more /etc/issue | awk '{print $$1}' | sed '1!d' | sed -e "/^$$/d" 2>/dev/null | tr "[:upper:]" "[:lower:]")
     # ensure data from /etc/issue is valid
     ifneq (,$(filter-out $(DISTRO),ubuntu fedora red rhel centos suse))
       DISTRO = 
     endif
     ifeq ("$(DISTRO)","")
       # third, we can search in /etc/os-release or /etc/{distro}-release
       DISTRO = $(shell awk '/ID/' /etc/*-release | sed 's/ID=//' | grep -v "VERSION" | grep -v "ID" | grep -v "DISTRIB")
     endif
   endif
endif

ifeq ("$(TARGET_OS)","linux")
    # Each set of Linux Distros have different paths for where to find their GLM/GLFW3 libraries reside
    UBUNTU = $(shell echo $(DISTRO) | grep -i ubuntu      >/dev/null 2>&1; echo $$?)
    FEDORA = $(shell echo $(DISTRO) | grep -i fedora      >/dev/null 2>&1; echo $$?)
    RHEL   = $(shell echo $(DISTRO) | grep -i 'red\|rhel' >/dev/null 2>&1; echo $$?)
    CENTOS = $(shell echo $(DISTRO) | grep -i centos      >/dev/null 2>&1; echo $$?)
    SUSE   = $(shell echo $(DISTRO) | grep -i 'suse\|sles' >/dev/null 2>&1; echo $$?)
    ifeq ("$(UBUNTU)","0")
      ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-armv7l)
        GLPATH := /usr/arm-linux-gnueabihf/lib
        GLLINK := -L/usr/arm-linux-gnueabihf/lib
        ifneq ($(TARGET_FS),) 
          GLPATH += $(TARGET_FS)/usr/lib/arm-linux-gnueabihf
          GLLINK += -L$(TARGET_FS)/usr/lib/arm-linux-gnueabihf
        endif 
      else ifeq ($(HOST_ARCH)-$(TARGET_ARCH),x86_64-aarch64)
        GLPATH := /usr/aarch64-linux-gnu/lib
        GLLINK := -L/usr/aarch64-linux-gnu/lib
        ifneq ($(TARGET_FS),) 
          GLPATH += $(TARGET_FS)/usr/lib
          GLPATH += $(TARGET_FS)/usr/lib/aarch64-linux-gnu
          GLLINK += -L$(TARGET_FS)/usr/lib/aarch64-linux-gnu
        endif 
      else
        UBUNTU_PKG_NAME = $(shell which dpkg >/dev/null 2>&1 && dpkg -l 'lwpu-*' | grep '^ii' | awk '{print $$2}' | head -1)
        ifneq ("$(UBUNTU_PKG_NAME)","")
          GLPATH    ?= /usr/lib/$(UBUNTU_PKG_NAME)
          GLLINK    ?= -L/usr/lib/$(UBUNTU_PKG_NAME)
        endif

        DFLT_PATH ?= /usr/lib
      endif
    endif
    ifeq ("$(SUSE)","0")
      GLPATH    ?= /usr/X11R6/lib64
      GLLINK    ?= -L/usr/X11R6/lib64
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(FEDORA)","0")
      GLPATH    ?= /usr/lib64/lwpu
      GLLINK    ?= -L/usr/lib64/lwpu
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(RHEL)","0")
      GLPATH    ?= /usr/lib64/lwpu
      GLLINK    ?= -L/usr/lib64/lwpu
      DFLT_PATH ?= /usr/lib64
    endif
    ifeq ("$(CENTOS)","0")
      GLPATH    ?= /usr/lib64/lwpu
      GLLINK    ?= -L/usr/lib64/lwpu
      DFLT_PATH ?= /usr/lib64
    endif

  VULKAN_SDK_PATH ?= ${VULKAN_SDK}

  ifeq ("$(VULKAN_SDK_PATH)","")
      $(info >>> WARNING - Vulkan SDK not found, please install Vulkan SDK <<<)
      SAMPLE_ENABLED := 0
  endif

  VULKAN_SDK_LIB  := $(shell find -L $(VULKAN_SDK_PATH)/lib -name libvulkan.so    -print 2>/dev/null)
  X11LIB          := $(shell find -L $(GLPATH) $(DFLT_PATH) -name libX11.so    -print 2>/dev/null)

  ifeq ("$(VULKAN_SDK_LIB)","")
      $(info >>> WARNING - libvulkan.so not found, please install libvulkan.so <<<)
      SAMPLE_ENABLED := 0
  endif

  ifeq ("$(X11LIB)","")
      $(info >>> WARNING - libX11.so not found, please install libX11.so <<<)
      SAMPLE_ENABLED := 0
  endif

  HEADER_SEARCH_PATH ?= $(TARGET_FS)/usr/include
  HEADER_SEARCH_PATH += $(TARGET_FS)/usr/local/include
  ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
      HEADER_SEARCH_PATH += /usr/arm-linux-gnueabihf/include
  else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-aarch64-linux)
      HEADER_SEARCH_PATH += /usr/aarch64-linux-gnu/include
  endif

  VULKANHEADER  := $(shell find -L $(VULKAN_SDK_PATH)/include -name vulkan.h -print 2>/dev/null)

  ifeq ("$(VULKANHEADER)","")
      $(info >>> WARNING - vulkan.h not found, please install vulkan.h <<<)
      SAMPLE_ENABLED := 0
  endif
else
endif

