# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2014-2019 by LWPU Corporation. All rights reserved. All information
# contained herein is proprietary and confidential to LWPU Corporation. Any
# use, reproduction, or disclosure without the written permission of LWPU
# Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END

ifeq "$(base_os)" "win32"
    defines += -DXP_WIN
    JS17_DIR = .
else
    defines += -DXP_UNIX
endif

defines += -DJS_HAS_FILE_OBJECT
defines += -DJS_C_STRINGS_ARE_UTF8

cpp_files += $(JS17_DIR)/jsapi.cpp
cpp_files += $(JS17_DIR)/jsarena.cpp
cpp_files += $(JS17_DIR)/jsarray.cpp
cpp_files += $(JS17_DIR)/jsatom.cpp
cpp_files += $(JS17_DIR)/jsbool.cpp
cpp_files += $(JS17_DIR)/jscntxt.cpp
cpp_files += $(JS17_DIR)/jsdate.cpp
cpp_files += $(JS17_DIR)/jsdbgapi.cpp
cpp_files += $(JS17_DIR)/jsdhash.cpp
cpp_files += $(JS17_DIR)/jsdtoa.cpp
cpp_files += $(JS17_DIR)/jsemit.cpp
cpp_files += $(JS17_DIR)/jsexn.cpp
cpp_files += $(JS17_DIR)/jsfile.cpp
cpp_files += $(JS17_DIR)/jsfun.cpp
cpp_files += $(JS17_DIR)/jsgc.cpp
cpp_files += $(JS17_DIR)/jshash.cpp
cpp_files += $(JS17_DIR)/jsinterp.cpp
cpp_files += $(JS17_DIR)/jsiter.cpp
cpp_files += $(JS17_DIR)/jslog2.cpp
cpp_files += $(JS17_DIR)/jslong.cpp
cpp_files += $(JS17_DIR)/jsmath.cpp
cpp_files += $(JS17_DIR)/jsnum.cpp
cpp_files += $(JS17_DIR)/jsobj.cpp
cpp_files += $(JS17_DIR)/jsopcode.cpp
cpp_files += $(JS17_DIR)/jsparse.cpp
cpp_files += $(JS17_DIR)/jsprf.cpp
cpp_files += $(JS17_DIR)/jsregexp.cpp
cpp_files += $(JS17_DIR)/jsscan.cpp
cpp_files += $(JS17_DIR)/jsscope.cpp
cpp_files += $(JS17_DIR)/jsscript.cpp
cpp_files += $(JS17_DIR)/jsstr.cpp
cpp_files += $(JS17_DIR)/jsutil.cpp
cpp_files += $(JS17_DIR)/jsxdrapi.cpp
cpp_files += $(JS17_DIR)/jsxml.cpp
cpp_files += $(JS17_DIR)/prmjtime.cpp
ifeq "$(base_os)" "win32"
    cpp_files += $(JS17_DIR)/prio_win.cpp
else
    cpp_files += $(JS17_DIR)/prio_psx.cpp
endif
