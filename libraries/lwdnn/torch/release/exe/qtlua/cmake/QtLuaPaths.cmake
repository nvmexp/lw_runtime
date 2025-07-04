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
NORMALIZE_PATH(CONFDIR)

# work-around luarocks *ugly* limitations those guys believe that only few
# directories in their PREFIX should be moved around. i really do not know
# what the hell they are thinking. you know what? it is sad.
GET_FILENAME_COMPONENT(CMAKE_INSTALL_PREFIX "${LUA_BINDIR}" PATH)
SET(QtLua_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
FILE(RELATIVE_PATH QtLua_INSTALL_BIN_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUA_BINDIR}")
FILE(RELATIVE_PATH QtLua_INSTALL_LIB_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUA_LIBDIR}")
FILE(RELATIVE_PATH QtLua_INSTALL_INCLUDE_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUA_INCDIR}")
FILE(RELATIVE_PATH QtLua_INSTALL_CMAKE_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${CONFDIR}/cmake")
FILE(RELATIVE_PATH QtLua_INSTALL_LUA_PATH_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LUADIR}")
FILE(RELATIVE_PATH QtLua_INSTALL_LUA_CPATH_SUBDIR "${CMAKE_INSTALL_PREFIX}" "${LIBDIR}")
####

SET(QtLua_INSTALL_FINDLUA_DIR "${QtLua_BINARY_DIR}/cmake")
SET(QtLua_INSTALL_BIN "${QtLua_INSTALL_PREFIX}/${QtLua_INSTALL_BIN_SUBDIR}")
SET(QtLua_INSTALL_LIB "${QtLua_INSTALL_PREFIX}/${QtLua_INSTALL_LIB_SUBDIR}")
SET(QtLua_INSTALL_INCLUDE "${QtLua_INSTALL_PREFIX}/${QtLua_INSTALL_INCLUDE_SUBDIR}")
SET(QtLua_INSTALL_CMAKE "${QtLua_INSTALL_PREFIX}/${QtLua_INSTALL_CMAKE_SUBDIR}")
SET(QtLua_INSTALL_LUA_PATH "${QtLua_INSTALL_PREFIX}/${QtLua_INSTALL_LUA_PATH_SUBDIR}")
SET(QtLua_INSTALL_LUA_CPATH "${QtLua_INSTALL_PREFIX}/${QtLua_INSTALL_LUA_CPATH_SUBDIR}")

# reverse relative path to prefix (ridbus is the palindrom of subdir)
FILE(RELATIVE_PATH QtLua_INSTALL_BIN_RIDBUS "${QtLua_INSTALL_BIN}" "${QtLua_INSTALL_PREFIX}/.")
FILE(RELATIVE_PATH QtLua_INSTALL_CMAKE_RIDBUS "${QtLua_INSTALL_CMAKE}" "${QtLua_INSTALL_PREFIX}/.")
GET_FILENAME_COMPONENT(QtLua_INSTALL_BIN_RIDBUS "${QtLua_INSTALL_BIN_RIDBUS}" PATH)
GET_FILENAME_COMPONENT(QtLua_INSTALL_CMAKE_RIDBUS "${QtLua_INSTALL_CMAKE_RIDBUS}" PATH)

FILE(RELATIVE_PATH QtLua_INSTALL_BIN2LIB "${QtLua_INSTALL_BIN}" "${QtLua_INSTALL_LIB}")

IF(UNIX)
  OPTION(QtLua_BUILD_WITH_RPATH "Build libraries with rpaths" ON)

  IF(QtLua_BUILD_WITH_RPATH)
    FILE(RELATIVE_PATH QtLua_INSTALL_BIN2LIB 
      "${QtLua_INSTALL_BIN}" "${QtLua_INSTALL_LIB}")
    IF(APPLE)
      SET(CMAKE_MACOSX_RPATH ON) # @rpath in libs
      SET(CMAKE_INSTALL_RPATH "@loader_path/${QtLua_INSTALL_BIN2LIB}") # exec
    ELSE()
      SET(CMAKE_INSTALL_RPATH "\$ORIGIN/${QtLua_INSTALL_BIN2LIB}")
    ENDIF()
  ENDIF(QtLua_BUILD_WITH_RPATH)

ENDIF(UNIX)

IF (WIN32)
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
ENDIF (WIN32)
