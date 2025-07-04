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
#  findegl.mk is used to find the necessary EGL Libraries for specific distributions
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
     ifeq (,$(filter $(DISTRO),ubuntu fedora red rhel centos suse))
       DISTRO = 
     endif
     ifeq ("$(DISTRO)","")
       # third, we can search in /etc/os-release or /etc/{distro}-release
       DISTRO = $(shell awk '/ID/' /etc/*-release | sed 's/ID=//' | grep -v "VERSION" | grep -v "ID" | grep -v "DISTRIB")
     endif
   endif
endif

ifeq ("$(TARGET_OS)","linux")
    # $(info) >> findegl.mk -> LINUX path <<<)
    # Each set of Linux Distros have different paths for where to find their OpenGL libraries reside
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

  EGLLIB  := $(shell find -L $(GLPATH) $(DFLT_PATH) -name libEGL.so    -print 2>/dev/null)

  ifeq ("$(EGLLIB)","")
      $(info >>> WARNING - libEGL.so not found, please install libEGL.so <<<)
      SAMPLE_ENABLED := 0
  endif

  HEADER_SEARCH_PATH ?= $(TARGET_FS)/usr/include
  ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-armv7l-linux)
      HEADER_SEARCH_PATH += /usr/arm-linux-gnueabihf/include
  else ifeq ($(HOST_ARCH)-$(TARGET_ARCH)-$(TARGET_OS),x86_64-aarch64-linux)
      HEADER_SEARCH_PATH += /usr/aarch64-linux-gnu/include
  endif

  EGLHEADER  := $(shell find -L $(HEADER_SEARCH_PATH) -name egl.h -print 2>/dev/null)
  EGLEXTHEADER  := $(shell find -L $(HEADER_SEARCH_PATH) -name eglext.h -print 2>/dev/null)

  ifeq ("$(EGLHEADER)","")
      $(info >>> WARNING - egl.h not found, please install egl.h <<<)
      SAMPLE_ENABLED := 0
  endif
  ifeq ("$(EGLEXTHEADER)","")
      $(info >>> WARNING - eglext.h not found, please install eglext.h <<<)
      SAMPLE_ENABLED := 0
  endif
else
endif

# Attempt to compile a minimal EGL application and run to check if EGL_SUPPORT_REUSE_LW is supported in the EGL headers available.
ifneq ($(SAMPLE_ENABLED), 0)
      $(shell printf "#include <EGL/egl.h>\n#include <EGL/eglext.h>\nint main() {\n#ifdef EGL_SUPPORT_REUSE_LW \n #error \"Compatible EGL header found\" \n  return 0;\n#endif \n return 1;\n}"  > test.c; )
      EGL_DEFINES := $(shell $(HOST_COMPILER) $(CCFLAGS) $(EXTRA_CCFLAGS) -lEGL test.c -c 2>&1 | grep -ic "Compatible EGL header found";)
      SHOULD_WAIVE := 0
      ifeq ($(EGL_DEFINES),0)
        SHOULD_WAIVE := 1
      endif
      ifeq ($(SHOULD_WAIVE),1)
          $(info -----------------------------------------------------------------------------------------------)
          $(info WARNING - LWPU EGL EXTENSIONS are not available in the present EGL headers)
          $(info -----------------------------------------------------------------------------------------------)
          $(info   This LWCA Sample cannot be built if the EGL LWPU EXTENSIONS like EGL_SUPPORT_REUSE_LW are not supported in EGL headers.)
          $(info   This will be a dry-run of the Makefile.)
          $(info   Please install the latest khronos EGL headers and libs to build this sample)
          $(info -----------------------------------------------------------------------------------------------)
          SAMPLE_ENABLED := 0
      endif
      $(shell rm test.o test.c 2>/dev/null)
endif

