CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

FIND_PACKAGE(Torch REQUIRED)

SET(src sys.c)
SET(luasrc init.lua colors.lua fpath.lua)

ADD_TORCH_PACKAGE(sys "${src}" "${luasrc}")

TARGET_LINK_LIBRARIES(sys luaT TH)
IF(LUALIB)
  TARGET_LINK_LIBRARIES(sys ${LUALIB})
ENDIF()
