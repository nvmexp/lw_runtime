CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

FIND_PACKAGE(Torch REQUIRED)
FIND_PACKAGE(LWCA 7.5 REQUIRED)

IF(COMMAND LWDA_SELECT_LWCC_ARCH_FLAGS)
# New CMake function, comes with FindLWDA
  LWDA_SELECT_LWCC_ARCH_FLAGS(LWCC_FLAGS_EXTRA $ELW{TORCH_LWDA_ARCH_LIST})
ELSE()
  SET(LWCC_FLAGS_EXTRA "-gencode=arch=compute_30,code=sm_30;-gencode=arch=compute_35,code=sm_35;-gencode=arch=compute_50,code=sm_50")
ENDIF()

SET(LWDA_LWCC_FLAGS "${LWDA_LWCC_FLAGS};${LWCC_FLAGS_EXTRA};-O3 -llwblas -llwdart")

MESSAGE("LWCC FLAGS: " ${LWDA_LWCC_FLAGS})

IF(APPLE)
  LIST(APPEND LWDA_LWCC_FLAGS "-Xcompiler -stdlib=libstdc++ -Xlinker -stdlib=libstdc++")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
ENDIF()

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/TH" "${Torch_INSTALL_INCLUDE}/THC"
"${CMAKE_LWRRENT_SOURCE_DIR}/lwda_helpers/")

LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

SET(src
  lwdacolw3/src/colw_util.lw
  lwdacolw3/src/filter_acts.lw
  lwdacolw3/src/img_acts.lw
  lwdacolw3/src/weight_acts.lw
  bias_kernels.lw)

FILE(GLOB luasrc *.lua)

LWDA_ADD_LIBRARY(ccn2 MODULE ${src})
TARGET_LINK_LIBRARIES(ccn2 THC TH)

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(ccn2 PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

INSTALL(TARGETS ccn2
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/ccn2")
