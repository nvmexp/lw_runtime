CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(LUAROCKS_PREFIX)
  MESSAGE(STATUS "Installing Torch-Signal through Luarocks")
  STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
  MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()
FIND_PACKAGE(Torch REQUIRED)

FILE(GLOB luasrc *.lua)
#FILE(GLOB src *.c)
SET(src "init.c")
ADD_TORCH_PACKAGE(signal "${src}" "${luasrc}" "Signal Processing Toolbox")

