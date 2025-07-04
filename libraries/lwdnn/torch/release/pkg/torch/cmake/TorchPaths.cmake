# workaround another annoying cmake bug
# http://public.kitware.com/Bug/view.php?id=14462
# https://awesome.naquadah.org/bugs/index.php?do=details&task_id=869
MACRO(NORMALIZE_PATH _path_)
  get_filename_component(${_path_}_abs "${${_path_}}" ABSOLUTE)
  SET(${_path_} "${${_path_}_abs}")
ENDMACRO()

NORMALIZE_PATH(LUA_BINDIR)
NORMALIZE_PATH(LUA_LIBDIR)
NORMALIZE_PATH(LUA_INCDIR)
NORMALIZE_PATH(LUADIR)
NORMALIZE_PATH(LIBDIR)

GET_FILENAME_COMPONENT(CMAKE_INSTALL_PREFIX "${LUA_BINDIR}" PATH)

SET(Torch_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
FILE(RELATIVE_PATH Torch_INSTALL_BIN_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUA_BINDIR}")
FILE(RELATIVE_PATH Torch_INSTALL_LIB_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUA_LIBDIR}")
FILE(RELATIVE_PATH Torch_INSTALL_INCLUDE_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUA_INCDIR}")

SET(Torch_INSTALL_MAN_SUBDIR "share/man" CACHE PATH
  "Install dir for man pages (relative to Torch_INSTALL_PREFIX)")

SET(Torch_INSTALL_SHARE_SUBDIR "share" CACHE PATH
  "Install dir for data (relative to Torch_INSTALL_PREFIX)")

SET(Torch_INSTALL_CMAKE_SUBDIR "share/cmake/torch" CACHE PATH
  "Install dir for .cmake files (relative to Torch_INSTALL_PREFIX)")

FILE(RELATIVE_PATH Torch_INSTALL_LUA_PATH_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUADIR}")
FILE(RELATIVE_PATH Torch_INSTALL_LUA_CPATH_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LIBDIR}")
