SET(Torch_INSTALL_BIN "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_BIN_SUBDIR}")
SET(Torch_INSTALL_MAN "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_MAN_SUBDIR}")
SET(Torch_INSTALL_LIB "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_LIB_SUBDIR}")
SET(Torch_INSTALL_SHARE "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_SHARE_SUBDIR}")
SET(Torch_INSTALL_INCLUDE "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_INCLUDE_SUBDIR}")
#SET(Torch_INSTALL_DOK "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_DOK_SUBDIR}")
#SET(Torch_INSTALL_HTML "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_HTML_SUBDIR}")
SET(Torch_INSTALL_CMAKE "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_CMAKE_SUBDIR}")
SET(Torch_INSTALL_LUA_PATH "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_LUA_PATH_SUBDIR}")
#SET(Torch_INSTALL_LUA_PKG_PATH "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_LUA_PKG_PATH_SUBDIR}")
SET(Torch_INSTALL_LUA_CPATH "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_LUA_CPATH_SUBDIR}")
#SET(Torch_INSTALL_LUAROCKS_SYSCONF "${Torch_INSTALL_PREFIX}/${Torch_INSTALL_LUAROCKS_SYSCONF_SUBDIR}")

# reverse relative path to prefix (ridbus is the palindrom of subdir)
FILE(RELATIVE_PATH Torch_INSTALL_BIN_RIDBUS "${Torch_INSTALL_BIN}" "${Torch_INSTALL_PREFIX}/.")
FILE(RELATIVE_PATH Torch_INSTALL_CMAKE_RIDBUS "${Torch_INSTALL_CMAKE}" "${Torch_INSTALL_PREFIX}/.")
GET_FILENAME_COMPONENT(Torch_INSTALL_BIN_RIDBUS "${Torch_INSTALL_BIN_RIDBUS}" PATH)
GET_FILENAME_COMPONENT(Torch_INSTALL_CMAKE_RIDBUS "${Torch_INSTALL_CMAKE_RIDBUS}" PATH)

IF(UNIX)
  OPTION(WITH_RPATH "Build libraries with exelwtable rpaths" ON)

  IF(WITH_RPATH)
    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    FILE(RELATIVE_PATH Torch_INSTALL_BIN2LIB
      "${Torch_INSTALL_BIN}" "${Torch_INSTALL_LIB}")
    IF(APPLE)
      SET(CMAKE_MACOSX_RPATH TRUE) # @rpath in libs
      SET(CMAKE_INSTALL_RPATH "@exelwtable_path/${Torch_INSTALL_BIN2LIB}") # exec
    ELSE()
      SET(CMAKE_INSTALL_RPATH "\$ORIGIN/${Torch_INSTALL_BIN2LIB}")
    ENDIF()
  ELSE()
    SET(CMAKE_MACOSX_RPATH FALSE) # no @rpath in libs
  ENDIF()

ENDIF(UNIX)

IF (WIN32)
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
ENDIF (WIN32)
