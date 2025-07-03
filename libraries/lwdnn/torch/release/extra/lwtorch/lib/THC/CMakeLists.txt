CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)

SET(CMAKE_MODULE_PATH ${CMAKE_LWRRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

SET(LWDA_ATTACH_VS_BUILD_RULE_TO_LWDA_FILE OFF)
OPTION(NDEBUG "disable asserts (WARNING: this may result in invalid memory accesses)")
IF(NOT NDEBUG)
  MESSAGE(STATUS "Removing -DNDEBUG from compile flags")
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS "" ${CMAKE_C_FLAGS})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_DEBUG "" ${CMAKE_C_FLAGS_DEBUG})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "" ${CMAKE_C_FLAGS_RELEASE})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS "" ${CMAKE_CXX_FLAGS})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_DEBUG "" ${CMAKE_CXX_FLAGS_DEBUG})
  STRING(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "" ${CMAKE_CXX_FLAGS_RELEASE})
ENDIF()

IF(NOT Torch_FOUND)
  FIND_PACKAGE(Torch)
ENDIF()

IF(NOT TH_LIBRARIES)
  SET(TH_LIBRARIES "TH")
ENDIF(NOT TH_LIBRARIES)
MESSAGE(STATUS "TH_LIBRARIES: ${TH_LIBRARIES}")

IF(NOT LWDA_FOUND)
  FIND_PACKAGE(LWCA 5.5 REQUIRED)
ENDIF()

IF(NOT MAGMA_FOUND)
  FIND_PACKAGE(MAGMA)
ENDIF()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
    if(LWDA_VERSION VERSION_LESS "8.0")
      MESSAGE(STATUS "Found gcc >=5 and LWCA <= 7.5, adding workaround C++ flags")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED -D__STRICT_ANSI__")
    endif(LWDA_VERSION VERSION_LESS "8.0")
  endif(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.9.3")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")


if(LWDA_VERSION VERSION_GREATER "8.0")
  LIST(APPEND LWDA_LWCC_FLAGS "-D__LWDA_NO_HALF_OPERATORS__")
endif(LWDA_VERSION VERSION_GREATER "8.0")

IF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  IF(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "4.7" OR CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL "4.7" )
    SET(CXX_VERSION "c++11")
  ELSE()
    SET(CXX_VERSION "c++0x")
  ENDIF()
  SET_SOURCE_FILES_PROPERTIES(
    THCTensorRandom.cpp
    THCCachingAllocator.cpp
    THCCachingHostAllocator.cpp
    THCStream.cpp
    PROPERTIES COMPILE_FLAGS -std=${CXX_VERSION})
ELSE()
  SET(CMAKE_CXX_STANDARD 11)
ENDIF()


INCLUDE_DIRECTORIES(${LWDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${LWDA_SDK_ROOT_DIR}/common/inc")

IF ("$ELW{STATIC_TH}" STREQUAL "YES")
LIST(APPEND LWDA_LWCC_FLAGS "-Xcompiler -fPIC")
ENDIF()

IF(MAGMA_FOUND)
  INCLUDE_DIRECTORIES(${MAGMA_INCLUDE_DIR})
  SET(CMAKE_REQUIRED_INCLUDES "${MAGMA_INCLUDE_DIR};${LWDA_INCLUDE_DIRS}")
  INCLUDE(CheckPrototypeDefinition)
  check_prototype_definition(magma_get_sgeqrf_nb
   "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
   "0"
   "magma.h"
    MAGMA_V2)
  IF (MAGMA_V2)
    add_definitions(-DMAGMA_V2)
  ENDIF (MAGMA_V2)

  SET(USE_MAGMA 1)
  MESSAGE(STATUS "Compiling with MAGMA support")
  MESSAGE(STATUS "MAGMA INCLUDE DIRECTORIES: ${MAGMA_INCLUDE_DIR}")
  MESSAGE(STATUS "MAGMA LIBRARIES: ${MAGMA_LIBRARIES}")
  MESSAGE(STATUS "MAGMA V2 check: ${MAGMA_V2}")
ELSE(MAGMA_FOUND)
  MESSAGE(STATUS "MAGMA not found. Compiling without MAGMA support")
ENDIF(MAGMA_FOUND)

IF ($ELW{TH_BINARY_BUILD})
  MESSAGE(STATUS "TH_BINARY_BUILD detected. Statically linking libstdc++")
  SET(CMAKE_CXX_FLAGS "-static-libstdc++ ${CMAKE_CXX_FLAGS}")
  IF (UNIX AND NOT APPLE)
    # hiding statically linked library symbols, this flag is not available for the linker under MACOSX
    SET(CMAKE_CXX_FLAGS "-Wl,--exclude-libs,libstdc++.a ${CMAKE_CXX_FLAGS}")
  ENDIF(UNIX AND NOT APPLE)
ENDIF()

IF(APPLE)
  IF(${LWDA_VERSION} LESS 6.0)
    # work around for mac os x bug:
    # http://stackoverflow.com/questions/16286588/lwca-5-0-cmake-and-make-failing-on-osx-10-8-3
    if (NOT DEFINED LWDA_HOST_COMPILER AND CMAKE_C_COMPILER_ID STREQUAL "Clang" AND EXISTS /usr/bin/gcc)
      set(LWDA_HOST_COMPILER /usr/bin/gcc CACHE FILEPATH "Host side compiler used by LWCC")
      message(STATUS "Setting CMAKE_HOST_COMPILER to /usr/bin/gcc instead of ${CMAKE_C_COMPILER}.")
    endif()

    # bug on Apple
    LINK_DIRECTORIES("/usr/local/lwca/lib/")
  ELSEIF(${LWDA_VERSION} LESS 7.0)
    SET(LWDA_HOST_COMPILER clang)
    LIST(APPEND LWDA_LWCC_FLAGS "-Xcompiler -stdlib=libstdc++ -Xlinker -stdlib=libstdc++")
    IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
    ENDIF()
  ENDIF()
  # LWCA 7 supports clang and libc++ so no need to change anything
ENDIF(APPLE)

# Detect LWCA architecture and get best LWCC flags
IF(NOT COMMAND LWDA_SELECT_LWCC_ARCH_FLAGS OR MSVC)
  INCLUDE(${CMAKE_LWRRENT_SOURCE_DIR}/cmake/select_compute_arch.cmake)
ENDIF()
LIST(APPEND LWDA_LWCC_FLAGS $ELW{TORCH_LWCC_FLAGS})
LWDA_SELECT_LWCC_ARCH_FLAGS(LWCC_FLAGS_EXTRA $ELW{TORCH_LWDA_ARCH_LIST})
LIST(APPEND LWDA_LWCC_FLAGS ${LWCC_FLAGS_EXTRA})

IF(NOT THC_INSTALL_BIN_SUBDIR
    OR NOT THC_INSTALL_LIB_SUBDIR
    OR NOT THC_INSTALL_INCLUDE_SUBDIR
    OR NOT THC_INSTALL_CMAKE_SUBDIR)

  INCLUDE_DIRECTORIES(${TH_INCLUDE_PATH} ${TH_INCLUDE_PATH}/TH)
  LINK_DIRECTORIES(${TH_LIB_PATH})

  IF(Torch_INSTALL_BIN_SUBDIR)
    SET(THC_INSTALL_BIN_SUBDIR ${Torch_INSTALL_BIN_SUBDIR})
    SET(THC_INSTALL_LIB_SUBDIR ${Torch_INSTALL_LIB_SUBDIR})
    SET(THC_INSTALL_INCLUDE_SUBDIR ${Torch_INSTALL_INCLUDE_SUBDIR})
    SET(THC_INSTALL_CMAKE_SUBDIR ${Torch_INSTALL_CMAKE_SUBDIR})
  ELSE(Torch_INSTALL_BIN_SUBDIR)
    # not installing in a Torch context, so Torch_INSTALL_BIN_SUBDIR is not available
    SET(THC_INSTALL_BIN_SUBDIR "bin" CACHE PATH "THC install binary subdirectory")
    SET(THC_INSTALL_LIB_SUBDIR "lib" CACHE PATH "THC install library subdirectory")
    SET(THC_INSTALL_INCLUDE_SUBDIR "include" CACHE PATH "THC install include subdirectory")
    SET(THC_INSTALL_CMAKE_SUBDIR "share/cmake/THC" CACHE PATH "THC install cmake subdirectory")
  ENDIF(Torch_INSTALL_BIN_SUBDIR)

ENDIF()

INCLUDE_DIRECTORIES("${CMAKE_LWRRENT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_LWRRENT_BINARY_DIR}")
CONFIGURE_FILE(THCGeneral.h.in "${CMAKE_LWRRENT_BINARY_DIR}/THCGeneral.h")

IF(MSVC)
  LIST(APPEND LWDA_LWCC_FLAGS "-Xcompiler /wd4819")
ELSE()
  SET(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
ENDIF()

SET(src
    THCCachingAllocator.cpp
    THCCachingHostAllocator.cpp
    THCGeneral.c
    THCStorageCopy.c
    THCStream.cpp
    THCTensor.c
    THCTensorCopy.c
    THCTensorRandom.cpp
    THCThreadLocal.c
    )

SET(src-lwca
  THCReduceApplyUtils.lw
  THCBlas.lw
  THCSleep.lw
  THCStorage.lw
  THCStorageCopy.lw
  THCTensor.lw
  THCTensorCopy.lw
  THCTensorMath.lw
  THCTensorMath2.lw
  THCTensorMathBlas.lw
  THCTensorMathMagma.lw
  THCTensorMathPairwise.lw
  THCTensorMathReduce.lw
  THCTensorMathScan.lw
  THCTensorIndex.lw
  THCTensorColw.lw
  THCTensorRandom.lw
  THCTensorScatterGather.lw
  THCTensorTopK.lw
  THCTensorSort.lw
  THCTensorTypeUtils.lw
  THCSortUtils.lw
  THCTensorMode.lw
  )

# loop over all types
foreach(THC_TYPE Byte Char Short Int Long Half Float Double)
   # loop over files which need to be split between types (because of long compile times)
   foreach(THC_FILE TensorSort TensorMathCompareT TensorMathPointwise TensorMathCompare TensorMathReduce TensorMasked)
      if(NOT EXISTS "${CMAKE_LWRRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.lw")
         FILE(WRITE "${CMAKE_LWRRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.lw"
              "#include \"../THC${THC_FILE}.lwh\"\n#include \"../generic/THC${THC_FILE}.lw\"\n#include \"../THCGenerate${THC_TYPE}Type.h\"\n")
      endif()
      LIST(APPEND src-lwca "${CMAKE_LWRRENT_SOURCE_DIR}/generated/THC${THC_FILE}${THC_TYPE}.lw")
   endforeach()
endforeach()

MESSAGE(STATUS "got lwca version " ${LWDA_VERSION})

IF(LWDA_HAS_FP16 OR NOT ${LWDA_VERSION} LESS 7.5)
  MESSAGE(STATUS "Found LWCA with FP16 support, compiling with torch.LwdaHalfTensor")
  LIST(APPEND src-lwca THCHalf.lw)
  LIST(APPEND LWDA_LWCC_FLAGS "-DLWDA_HAS_FP16=1")
  SET(CMAKE_C_FLAGS "-DLWDA_HAS_FP16=1 ${CMAKE_C_FLAGS}")
ELSE(LWDA_HAS_FP16 OR NOT ${LWDA_VERSION} LESS 7.5)
  MESSAGE(STATUS "Could not find LWCA with FP16 support, compiling without torch.LwdaHalfTensor")
ENDIF(LWDA_HAS_FP16 OR NOT ${LWDA_VERSION} LESS 7.5)

MESSAGE(STATUS "LWDA_LWCC_FLAGS: ${LWDA_LWCC_FLAGS}")
IF ("$ELW{STATIC_TH}" STREQUAL "YES")
  LWDA_ADD_LIBRARY(THC STATIC ${src} ${src-lwca})
  SET_TARGET_PROPERTIES(THC PROPERTIES COMPILE_FLAGS "-fPIC")
ELSE()
  LWDA_ADD_LIBRARY(THC SHARED ${src} ${src-lwca})
  LWDA_ADD_LWBLAS_TO_TARGET(THC)
  TARGET_LINK_LIBRARIES(THC ${TH_LIBRARIES} ${LWDA_lwrand_LIBRARY} ${LWDA_lwsparse_LIBRARY})

  IF(USE_MAGMA)
    TARGET_LINK_LIBRARIES(THC ${MAGMA_LIBRARIES})
    IF ($ELW{TH_BINARY_BUILD})
      # because magma is linked statically and it wants a BLAS,
      # we need to link the BLAS lib against THC. Usually TH will
      # load a BLAS library and it's all fine, but in the binary builds,
      # TH uses static linkage to MKL, so it doesn't have all symbols that
      # magma needs. So in this case, explicitly find a BLAS and link against it
      # just like in TH
      SET(CMAKE_MODULE_PATH ${CMAKE_LWRRENT_SOURCE_DIR}/../TH/cmake ${CMAKE_MODULE_PATH})
      FIND_PACKAGE(BLAS)
      IF(BLAS_FOUND)
        TARGET_LINK_LIBRARIES(THC "${BLAS_LIBRARIES};${BLAS_LIBRARIES};${BLAS_LIBRARIES}")
      ELSE(BLAS_FOUND)
        MESSAGE(FATAL_ERROR "Binary build needs blas to be found here")
      ENDIF(BLAS_FOUND)
    ENDIF($ELW{TH_BINARY_BUILD})
  ENDIF(USE_MAGMA)

  IF(NOT THC_SO_VERSION)
    SET(THC_SO_VERSION 0)
  ENDIF(NOT THC_SO_VERSION)
  MESSAGE(STATUS "THC_SO_VERSION: ${THC_SO_VERSION}")
  SET_TARGET_PROPERTIES(THC PROPERTIES
    VERSION   ${THC_SO_VERSION}
    SOVERSION ${THC_SO_VERSION})


  INSTALL(TARGETS THC
    RUNTIME DESTINATION "${THC_INSTALL_BIN_SUBDIR}"
    LIBRARY DESTINATION "${THC_INSTALL_LIB_SUBDIR}"
    ARCHIVE DESTINATION "${THC_INSTALL_LIB_SUBDIR}")
ENDIF()


INSTALL(FILES
          THC.h
          ${CMAKE_LWRRENT_BINARY_DIR}/THCGeneral.h
          THCBlas.h
          THCSleep.h
          THCStorage.h
          THCStorageCopy.h
          THCStream.h
          THCThreadLocal.h
          THCTensor.h
          THCTensorCopy.h
          THCTensorRandom.h
          THCTensorMath.h
          THCTensorColw.h
          THCApply.lwh
          THCReduce.lwh
          THCReduceAll.lwh
          THCReduceApplyUtils.lwh
          THCAsmUtils.lwh
          THCAtomics.lwh
          THCScanUtils.lwh
          THCSortUtils.lwh
          THCAllocator.h
          THCCachingAllocator.h
          THCCachingHostAllocator.h
          THCDeviceUtils.lwh
          THCDeviceTensor.lwh
          THCDeviceTensor-inl.lwh
          THCDeviceTensorUtils.lwh
          THCDeviceTensorUtils-inl.lwh
          THCGenerateAllTypes.h
          THCGenerateByteType.h
          THCGenerateCharType.h
          THCGenerateShortType.h
          THCGenerateIntType.h
          THCGenerateLongType.h
          THCGenerateHalfType.h
          THCGenerateFloatType.h
          THCGenerateFloatTypes.h
          THCGenerateDoubleType.h
          THCHalf.h
          THCNumerics.lwh
          THCTensorSort.lwh
          THCTensorInfo.lwh
          THCTensorMathPointwise.lwh
          THCTensorTypeUtils.lwh
          THCTensorRandom.lwh
          THCTensorMathMagma.lwh
          THCThrustAllocator.lwh
          THCTensorMode.lwh
          THCTensorTopK.lwh
          DESTINATION "${THC_INSTALL_INCLUDE_SUBDIR}/THC")

INSTALL(FILES
          generic/THCStorage.c
          generic/THCStorage.lw
          generic/THCStorage.h
          generic/THCTensor.c
          generic/THCTensor.lw
          generic/THCTensor.h
          generic/THCStorageCopy.c
          generic/THCStorageCopy.lw
          generic/THCStorageCopy.h
          generic/THCTensorCopy.c
          generic/THCTensorCopy.lw
          generic/THCTensorCopy.h
          generic/THCTensorMasked.h
          generic/THCTensorMasked.lw
          generic/THCTensorMath.h
          generic/THCTensorMath.lw
          generic/THCTensorMathBlas.lw
          generic/THCTensorMathBlas.h
          generic/THCTensorMathCompare.h
          generic/THCTensorMathCompare.lw
          generic/THCTensorMathCompareT.h
          generic/THCTensorMathCompareT.lw
          generic/THCTensorMathMagma.h
          generic/THCTensorMathMagma.lw
          generic/THCTensorMathPairwise.h
          generic/THCTensorMathPairwise.lw
          generic/THCTensorMathPointwise.h
          generic/THCTensorMathPointwise.lw
          generic/THCTensorMathReduce.h
          generic/THCTensorMathReduce.lw
          generic/THCTensorMathScan.h
          generic/THCTensorMathScan.lw
          generic/THCTensorScatterGather.h
          generic/THCTensorScatterGather.lw
          generic/THCTensorIndex.h
          generic/THCTensorIndex.lw
          generic/THCTensorSort.h
          generic/THCTensorSort.lw
          generic/THCDeviceTensorUtils.lw
          generic/THCTensorRandom.h
          generic/THCTensorRandom.lw
          generic/THCTensorMode.h
          generic/THCTensorMode.lw
          generic/THCTensorTopK.h
          generic/THCTensorTopK.lw
          DESTINATION "${THC_INSTALL_INCLUDE_SUBDIR}/THC/generic")
