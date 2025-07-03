#/*
# * LWIDIA_COPYRIGHT_BEGIN
# *
# * Copyright 2019 by LWPU Corporation.  All rights reserved.  All
# * information contained herein is proprietary and confidential to LWPU
# * Corporation.  Any use, reproduction, or disclosure without the written
# * permission of LWPU Corporation is prohibited.
# *
# * LWIDIA_COPYRIGHT_END
# */

#-------------------------------------------------------------------------------
# JavaScript make file.
# See //sw/<tree>/diag/mods/makedefs.inc for more information about the MODS
# makefiles.
#-------------------------------------------------------------------------------

MODS_DIR = ../mods

dll_build = false
base_name = libyaml

include $(MODS_DIR)/makedefs.inc

YAML_PARENT_DIR = $(BUILD_TOOLS_DIR)/mods/yaml-cpp-0.6.2
ifeq "$(base_os)" "win32"
    ifdef CYGWIN_MAKE
        YAML_PARENT_DIR := $(shell $(MODS_DIR)/tools/abs2rel.sh '$(YAML_PARENT_DIR)')
    endif

    # This is needed because MSVC removed the definition of _NOEXCEPT since VS2017 15.8
    # This is a known bug (https://jira.mongodb.org/browse/SERVER-38098) which
    # has been fixed in yaml-cpp master
    # (https://github.com/jbeder/yaml-cpp/commit/0f9a586ca1dc29c2ecb8dd715a315b93e3f40f79)
    # but has not made it into any release yet.
    # Hence, we should be able to remove this line whenever we update yaml-cpp to a
    # newer version (>0.6.2)
    defines += -D_NOEXCEPT=noexcept
endif

YAML_DIR = $(YAML_PARENT_DIR)/src
include_dirs += $(YAML_PARENT_DIR)/include

# Source files
yaml_files += binary.cpp
yaml_files += colwert.cpp
yaml_files += directives.cpp
yaml_files += emit.cpp
yaml_files += emitfromevents.cpp
yaml_files += emitter.cpp
yaml_files += emitterstate.cpp
yaml_files += emitterutils.cpp
yaml_files += exceptions.cpp
yaml_files += exp.cpp
yaml_files += memory.cpp
yaml_files += nodebuilder.cpp
yaml_files += node.cpp
yaml_files += node_data.cpp
yaml_files += nodeevents.cpp
yaml_files += null.cpp
yaml_files += ostream_wrapper.cpp
yaml_files += parse.cpp
yaml_files += parser.cpp
yaml_files += regex_yaml.cpp
yaml_files += scanner.cpp
yaml_files += scanscalar.cpp
yaml_files += scantag.cpp
yaml_files += scantoken.cpp
yaml_files += simplekey.cpp
yaml_files += singledocparser.cpp
yaml_files += stream.cpp
yaml_files += tag.cpp
cpp_files += $(addprefix $(YAML_DIR)/, $(yaml_files))

#-------------------------------------------------------------------------------
# Includes for standard functionality
#-------------------------------------------------------------------------------
include $(MODS_DIR_UNIX)/makelib.inc
include $(MODS_DIR_UNIX)/makerule.inc

# Include dependency information.
-include $(depend_files)
