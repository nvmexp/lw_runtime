CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)

IF(APPLE)
  # work around for mac os x bug:
  # http://stackoverflow.com/questions/16286588/lwca-5-0-cmake-and-make-failing-on-osx-10-8-3
  if (NOT DEFINED LWDA_HOST_COMPILER AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND EXISTS /usr/bin/gcc)
    set(LWDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by LWCC")
    message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.")
  endif()
ENDIF()

FIND_PACKAGE(LWCA 6.5 REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
    if(LWDA_VERSION VERSION_LESS "8.0")
      MESSAGE(STATUS "Found gcc >=5 and LWCA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")
    endif(LWDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")

# Detect LWCA architecture and get best LWCC flags
IF(COMMAND LWDA_SELECT_LWCC_ARCH_FLAGS)
  # New CMake function, comes with FindLWDA
  LWDA_SELECT_LWCC_ARCH_FLAGS(LWCC_FLAGS_EXTRA $ELW{TORCH_LWDA_ARCH_LIST})
ELSE()
  # Old way of getting the flags, with extra include file
  INCLUDE(${CMAKE_LWRRENT_SOURCE_DIR}/cmake/FindLwdaArch.cmake)
  SELECT_LWCC_ARCH_FLAGS(LWCC_FLAGS_EXTRA)
ENDIF()
LIST(APPEND LWDA_LWCC_FLAGS ${LWCC_FLAGS_EXTRA})


INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src-lwca init.lw)
FILE(GLOB luasrc *.lua)
SET(luasrc ${luasrc} test/test.lua)
#ADD_TORCH_PACKAGE(lwnnx "${src-lwca}" "${luasrc}" "Experiment lwca nn library")

LWDA_ADD_LIBRARY(lwnnx MODULE ${src-lwca})
LWDA_ADD_LWBLAS_TO_TARGET(lwnnx)
TARGET_LINK_LIBRARIES(lwnnx luaT THC TH)
IF(APPLE)
  SET_TARGET_PROPERTIES(lwnnx PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
ENDIF()
IF(LUALIB)
  TARGET_LINK_LIBRARIES(lwnnx ${LUALIB})
ENDIF()

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(lwnnx PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

INSTALL(TARGETS lwnnx
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/lwnnx")
