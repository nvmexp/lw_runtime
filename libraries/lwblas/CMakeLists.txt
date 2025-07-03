# ---[ Check cmake version.
cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ---[ Project specification.
project(lwtensor VERSION 1.3.1.3 LANGUAGES C CXX LWCA)
set(LWTENSOR_PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}" CACHE STRING "lwTENSOR version") # required for docs

# ---[ Configure dependency structure.
# set(CMAKE_RELEASE_POSTFIX "")
# set(CMAKE_DEBUG_POSTFIX "-debug")
# Set a default build type if none was specified

# ##########################################
# lwTENSOR build mode
# ##########################################

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()

# ##########################################
# lwTENSOR dependencies
# ##########################################

# in windows, don't find_package as multiple versions are pulled in due to cmake scouring
# default install dir. instead explicitly point to lwca toolkit when running cmake:
# -DCMAKE_LWDA_COMPILER:FILEPATH=c:/tools/lwca-10.1-latest/bin/lwcc.exe
if(WIN32)
    include_directories("${CMAKE_LWDA_TOOLKIT_INCLUDE_DIRECTORIES}")
else()
    find_package(LWCA 10.1 REQUIRED)
    include_directories("${LWDA_INCLUDE_DIRS}")
endif()

if(NOT DEFINED LWDA_VERSION_REVISION)
  exelwte_process (COMMAND ${CMAKE_LWDA_COMPILER} "--version" OUTPUT_VARIABLE LWCC_OUT)
  string(REGEX REPLACE ".*release +.+, *V([0-9]+)\\.([0-9]+)\\.([0-9]+).*" "\\1" LWCC_VERSION_MAJOR ${LWCC_OUT})
  string(REGEX REPLACE ".*release +.+, *V([0-9]+)\\.([0-9]+)\\.([0-9]+).*" "\\2" LWCC_VERSION_MINOR ${LWCC_OUT})
  string(REGEX REPLACE ".*release +.+, *V([0-9]+)\\.([0-9]+)\\.([0-9]+).*" "\\3" LWCC_VERSION_REVISION ${LWCC_OUT})
  message(STATUS "LWCC version detected: ${LWCC_VERSION_MAJOR}.${LWCC_VERSION_MINOR}.${LWCC_VERSION_REVISION}")

  if(DEFINED LWDA_VERSION_MAJOR AND NOT LWDA_VERSION_MAJOR STREQUAL LWCC_VERSION_MAJOR)
    message(FATAL_ERROR "LWCA toolkit major version mismatch: \"${LWDA_VERSION_MAJOR}\" != \"${LWCC_VERSION_MAJOR}\"")
  elseif(NOT DEFINED LWDA_VERSION_MAJOR)
    set(LWDA_VERSION_MAJOR "${LWCC_VERSION_MAJOR}" CACHE STRING "LWCA toolkit major version")
  endif()

  if(DEFINED LWDA_VERSION_MINOR AND NOT LWDA_VERSION_MINOR STREQUAL LWCC_VERSION_MINOR)
    message(FATAL_ERROR "LWCA toolkit minor version mismatch: \"${LWDA_VERSION_MINOR}\" != \"${LWCC_VERSION_MINOR}\"")
  elseif(NOT DEFINED LWDA_VERSION_MINOR)
    set(LWDA_VERSION_MINOR "${LWCC_VERSION_MINOR}" CACHE STRING "LWCA toolkit minor version")
  endif()

  set(LWDA_VERSION_REVISION "${LWCC_VERSION_REVISION}" CACHE STRING "LWCA toolkit revision")
  message(STATUS "LWCA toolkit revision: ${LWDA_VERSION_REVISION}")
endif()

if ((LWDA_VERSION_MAJOR GREATER 11) OR ((LWDA_VERSION_MAJOR EQUAL 11) AND (LWDA_VERSION_MINOR GREATER_EQUAL 1)))
  if (SYS_ARCH STREQUAL "ppc64le" OR WIN32)
    # disable on windows because msvc does not play well with lwte
    # disable on ppc64le because gcc is too old
    set(LWTENSOR_ENABLE_LWTE_DEFAULT OFF)
  else()
    set(LWTENSOR_ENABLE_LWTE_DEFAULT ON)
  endif()
else()
  set(LWTENSOR_ENABLE_LWTE_DEFAULT OFF)
endif()

add_compile_definitions(LWDA_VERSION_REVISION=${LWDA_VERSION_REVISION})

# Explicitly looking for python, otherwise we have issues in windows
find_package(Python REQUIRED)

set(CONFU_DEPENDENCIES_SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps CACHE PATH "Confu-style dependencies source directory")
set(CONFU_DEPENDENCIES_BINARY_DIR ${CMAKE_BINARY_DIR}/deps CACHE PATH "Confu-style dependencies binary directory")

# ##########################################
# lwTENSOR build options
# ##########################################

SET(LIBRARY_TYPE "default" CACHE STRING "Type of library (shared, static, or default) to build")
SET_PROPERTY(CACHE LIBRARY_TYPE PROPERTY STRINGS default static shared)
OPTION(BUILD_TOOLS "Build command-line tools" ON)
OPTION(DEVELOP "Build with develop options (e.g., no ptx code)" OFF)
OPTION(BUILD_UNIT_TESTS "Build lwTENSOR unit tests" ON)
OPTION(BUILD_MOCK_TESTS "Build lwTENSOR mock tests" ON)
OPTION(BUILD_BENCHMARKS "Build lwTENSOR micro-benchmarks" ON)
OPTION(BUILD_EXAMPLES "Build lwTENSOR examples" ON)
OPTION(USE_LEGACY_LWVM "Use old version of lwvm (only affects 11.2 TK and newer)" ON)
OPTION(BUILD_DOCS "Build lwTENSOR documentation" ON)
OPTION(LWDA_SDK_INSTALL "Make install path compatible with LWCA SDK" OFF)
OPTION(BUILD_DEBUG_GLIBC "Build with debug glibc" OFF)
OPTION(LWTENSOR_EXPOSE_INTERNAL "Exposes internal functionality in binaries" OFF)
OPTION(LWTENSOR_ENABLE_LWTE "Enables LwTe backedn for contractions" ${LWTENSOR_ENABLE_LWTE_DEFAULT})
OPTION(BUILD_WITH_GCOV "Enable code coverage flags, must build with debug mode" OFF)
OPTION(LWTENSOR_SPLIT_K_SWEEP "Enables users to use -partitionK as the target CTA value" OFF)


# ##########################################
# lwTENSOR building flags
# ##########################################

# Common custom flags TODO: custom flags are replicated in host and device compiler. Assign correctly.
#set(LWTENSOR_LWSTOM_FLAGS "-DLWTLASS_ENABLE_TENSOR_CORE_MMA=1 -DLWTLASS_CONTRACTION_KERNEL_RENAME -DJETFIRE_ENABLED=1")
set(LWTENSOR_LWSTOM_FLAGS "-DLWTLASS_ENABLE_TENSOR_CORE_MMA=1 -DJETFIRE_ENABLED=1 -DLWTLASS_ENABLE_EXTENDED_PTX=1 -DLWTLASS_LWDA_RP2RP_ENABLED=1 -DLWTLASS_ENABLE_INTERNAL_LWVM=1 -DLWTLASS_LWDA_SECTOR_PROMOTION_ENABLED=1 -DLWTLASS_ENABLE_COMPILER_KNOBS=1 -DLWTLASS_ENABLE_LWOPT=1 -DLWDA_PTX_KNOB_SCHED_MEM_NO_ALIAS_ENABLED=1 -DLWDA_PTX_KNOB_DISABLE_IMPLICIT_MEM_DESC_ENABLED=1 -DLWDA_PTX_KNOB_SCHED_READ_SB_BASE_LATENCY=1 -DLWTENSOR_LWDA_VERSION_MAJOR=${LWDA_VERSION_MAJOR}")
set(LWTENSOR_LWSTOM_FLAGS "${LWTENSOR_LWSTOM_FLAGS} -DLWTLASS_CONTRACTION_KERNEL_RENAME") # Comment out for contraction kernel generation
if(LWTENSOR_GIT_TAG)
    add_definitions(-DLWTENSOR_GIT_TAG=${LWTENSOR_GIT_TAG})
endif()
if(LWTENSOR_GIT_COMMIT)
    add_definitions(-DLWTENSOR_GIT_COMMIT=${LWTENSOR_GIT_COMMIT})
endif()
if(LWTENSOR_BUILD_ID)
    add_definitions(-DLWTENSOR_BUILD_ID=${LWTENSOR_BUILD_ID})
endif()

# Global CXX/LWCA flags
if(LWTENSOR_EXPOSE_INTERNAL)
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -DLWTENSOR_EXPOSE_INTERNAL=1")
endif()

if(LWTENSOR_ENABLE_LWTE)
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -DLWTENSOR_ENABLE_LWTE=1")
  set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_LWDA_STANDARD 14)
else()
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_LWDA_STANDARD 11)
endif()

if(NOT CMAKE_CROSSCOMPILING AND NOT WIN32)
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -m64")
endif()

set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -Wall")
if(NOT WIN32)
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -Wextra -Wunused")
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -O3")
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -fPIC")
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -fno-strict-aliasing")
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -Wno-sign-compare")
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -static-libstdc++")
  if(BUILD_WITH_GCOV)
    set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
  endif(BUILD_WITH_GCOV)
  set(LWTENSOR_LWML_LIBRARY_NAME "lwpu-ml")
else()
  set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} /wd5045 /wd4514")
  set(LWTENSOR_LWML_LIBRARY_NAME "lwml")
endif()

if(NOT DEVELOP)
  set(LWTENSOR_KEEP TRUE)
else()
  set(LWTENSOR_KEEP FALSE)
endif()

function(lwtensor_fixed_keep TARGET)
    if(LWTENSOR_KEEP)
        message(STATUS "Keep ${TARGET} fixed.")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/keep")
        set(KEEP_DIR "${CMAKE_BINARY_DIR}/keep/${TARGET}")
        file(MAKE_DIRECTORY "${KEEP_DIR}")
        set_property(TARGET ${TARGET} APPEND_STRING PROPERTY COMPILE_FLAGS " --keep --keep-dir ${KEEP_DIR}")
    endif()
endfunction()

set(LWTENSOR_LWDA_CXX_FLAGS "${LWTENSOR_LWDA_CXX_FLAGS} ${LWTENSOR_LWSTOM_FLAGS}")

# Global CXX flags/options
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LWTENSOR_LWDA_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${LWTENSOR_LWDA_CXX_FLAGS}")

if(BUILD_DEBUG_GLIBC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${LWTENSOR_LWDA_CXX_FLAGS} -D_GLIBCXX_DEBUG")
endif()

set(CMAKE_LWDA_FLAGS_ARCH_SM70 "-gencode arch=compute_70,code=sm_70")
set(CMAKE_LWDA_FLAGS_ARCH_SM75 "-gencode arch=compute_75,code=sm_75")

if(${LWDA_VERSION_MAJOR} GREATER_EQUAL 11)
  set(CMAKE_LWDA_FLAGS_ARCH_SM80 "-gencode arch=compute_80,code=sm_80")
  set(CMAKE_LWDA_FLAGS_ARCH_SM61 "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80")
else()
  set(CMAKE_LWDA_FLAGS_ARCH_SM80 "")
  set(CMAKE_LWDA_FLAGS_ARCH_SM61 "-gencode arch=compute_60,code=sm_60 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75")
endif()

set(CMAKE_LWDA_FLAGS_ARCH_ALL  "${CMAKE_LWDA_FLAGS_ARCH_SM61} ${CMAKE_LWDA_FLAGS_ARCH_SM70} ${CMAKE_LWDA_FLAGS_ARCH_SM75} ${CMAKE_LWDA_FLAGS_ARCH_SM80}")

# Global LWCA CXX flags/options
set(LWDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_LWDA_STANDARD_REQUIRED ON)
set(CMAKE_LWDA_EXTENSIONS OFF)
set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} -fmad=true")

set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} -Xfatbin -compress-all")
if((${LWDA_VERSION_MAJOR} GREATER_EQUAL 11) OR ((${LWDA_VERSION_MAJOR} EQUAL 10) AND (${LWDA_VERSION_MINOR} GREATER_EQUAL 2)))
  set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} -Xcicc --uumn -Xptxas -uumn")
endif()

if(NOT WIN32)
    if (USE_LEGACY_LWVM AND ((${LWDA_VERSION_MAJOR} EQUAL 11) AND (${LWDA_VERSION_MINOR} GREATER_EQUAL 2) AND (${LWDA_VERSION_MINOR} LESS_EQUAL 3)))
      set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} --lwvm-version=lwvm-prev")
    endif()
endif()

if (NOT WIN32 AND (LWDA_VERSION_MAJOR EQUAL 11) AND (LWDA_VERSION_MINOR GREATER_EQUAL 2))
  set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} --threads 8")
endif()

# DEVICE LINK CMAKE ISSUE FIX #1: Compile everything for everything. Uncomment below.
#set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} ${CMAKE_LWDA_FLAGS_ARCH_ALL}")
set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} -Xlwdafe --display_error_number") # Show error/warning numbers
set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} -Xcompiler \"${LWTENSOR_LWDA_CXX_FLAGS}\"")
set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} ${LWTENSOR_LWSTOM_FLAGS}")
set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} --expt-relaxed-constexpr") # avoid warnings for 10.1 w.r.t std::isnan
#set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS}  --ptxas-options=-v") # prints resource usage infos

# Debug LWCA flags
set(CMAKE_LWDA_FLAGS_DEBUG "${CMAKE_LWDA_FLAGS_DEBUG} -DDEVELOP -Xcompiler=-DDEVELOP")
if(BUILD_DEBUG_GLIBC)
  set(CMAKE_LWDA_FLAGS_DEBUG "${CMAKE_LWDA_FLAGS_DEBUG} -Xcompiler=-D_GLIBCXX_DEBUG")
endif()

# Release LWCA flags
# For now, there is no target based way of passing flags to linker. Using global flags.
set(CMAKE_LWDA_FLAGS_RELEASE "${CMAKE_LWDA_FLAGS_RELEASE} -DNDEBUG -Xcompiler=-DNDEBUG")
if (NOT WIN32)
  set(CMAKE_LWDA_FLAGS_RELEASE "${CMAKE_LWDA_FLAGS_RELEASE} -Xlinker -Bsymbolic,--default-symver")

  if(LWTENSOR_EXPOSE_INTERNAL)
    set(CMAKE_LWDA_FLAGS_RELEASE "${CMAKE_LWDA_FLAGS_RELEASE} -Xlinker --version-script=${PROJECT_SOURCE_DIR}/cmake/liblwtensor-internal.map") #redundant jic
  else()
    set(CMAKE_LWDA_FLAGS_RELEASE "${CMAKE_LWDA_FLAGS_RELEASE} -Xlinker --version-script=${PROJECT_SOURCE_DIR}/cmake/liblwtensor.map") #redundant jic
  endif()
endif()


# On release, hide symbols
if(CMAKE_BUILD_TYPE MATCHES Release)
  set(CMAKE_CXX_VISIBILITY_PRESET hidden)
  set(CMAKE_LWDA_VISIBILITY_PRESET hidden)
  set(CMAKE_C_VISIBILITY_PRESET hidden)
endif()

# ##########################################
# lwTENSOR sources and headers
# ##########################################

include(cmake/sources.cmake)

# ##########################################
# lwTENSOR directories
# ##########################################

# By default put libraries in build/lib (pre-install)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Installation directories
set(LWTENSOR_INSTALL_DIR "liblwtensor")
set(LWTENSOR_INCLUDES_INSTALL_DIR "liblwtensor/include")
set(LWTENSOR_LIBRARY_INSTALL_DIR "liblwtensor/lib")
set(LWTENSOR_BINARY_INSTALL_DIR "liblwtensor/bin")
set(LWTENSOR_TEST_INSTALL_DIR "liblwtensor/test")
set(LWTENSOR_CMAKECONFIG_INSTALL_DIR "liblwtensor/lib/cmake/lwtensor")
set(LWTENSOR_EXAMPLES_INSTALL_DIR "liblwtensor/samples")

if(CMAKE_BUILD_TYPE MATCHES Release)
  if(${LWDA_VERSION_STRING} VERSION_GREATER 11.0)
    set(LWTENSOR_LIBRARY_INSTALL_DIR ${LWTENSOR_LIBRARY_INSTALL_DIR}/${LWDA_VERSION_MAJOR})
  else()
    set(LWTENSOR_LIBRARY_INSTALL_DIR ${LWTENSOR_LIBRARY_INSTALL_DIR}/${LWDA_VERSION_STRING})
  endif()
endif()

# ##########################################
# lwTENSOR library target
# ##########################################

if(WIN32)
  string(REPLACE "/Od" "/Ot /O2" CMAKE_LWDA_FLAGS_DEBUG "${CMAKE_LWDA_FLAGS_DEBUG}")
  string(REPLACE "/RTC1" "" CMAKE_LWDA_FLAGS_DEBUG "${CMAKE_LWDA_FLAGS_DEBUG}")
  string(REPLACE "/Od" "/Ot /O2" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "/RTC1" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "/Ob0" "/Ob1" CMAKE_LWDA_FLAGS_DEBUG "${CMAKE_LWDA_FLAGS_DEBUG}")
  string(REPLACE "/Ob0" "/Ob1" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

message(STATUS "CMAKE_LWDA_FLAGS ${CMAKE_LWDA_FLAGS}")
message(STATUS "CMAKE_LWDA_FLAGS_DEBUG ${CMAKE_LWDA_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")

# Build the shared library
# ---------------------------

# lwTensor public headers
add_library(lwtensor_public_headers INTERFACE)
target_include_directories(lwtensor_public_headers BEFORE
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include/lwtensor>
        $<INSTALL_INTERFACE:${LWTENSOR_INCLUDES_INSTALL_DIR}>
)

#lwTensor private files
add_library(lwtensor_private_includes INTERFACE)
target_include_directories(lwtensor_private_includes BEFORE
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/src>
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/lwtlass/include>
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/lwtlass/tools/util/include>
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/LwTe/examples/>
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/LwTe/include/>
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/dnnheuristic/>
        $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/dnnheuristic/arch>
)

target_link_libraries(lwtensor_private_includes INTERFACE lwtensor_public_headers)

add_library(lwtensor_dnn_heuristic OBJECT
                ${LWTENSOR_DNN_HEURISTIC_SRC})
set_target_properties(lwtensor_dnn_heuristic PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lwtensor_dnn_heuristic
  PRIVATE
    lwtensor_private_includes
)


add_library(lwtensor_base_objs_sm61 OBJECT
              ${LWTENSOR_SRC_SM61})
set_target_properties(lwtensor_base_objs_sm61 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_SM61}"
)
lwtensor_fixed_keep(lwtensor_base_objs_sm61)
target_link_libraries(lwtensor_base_objs_sm61
  PRIVATE
    lwtensor_private_includes
)

add_library(lwtensor_base_objs_sm70 OBJECT
              ${LWTENSOR_SRC_SM70})
set_target_properties(lwtensor_base_objs_sm70 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_SM70}"
)
lwtensor_fixed_keep(lwtensor_base_objs_sm70)
target_link_libraries(lwtensor_base_objs_sm70
  PRIVATE
    lwtensor_private_includes
)

add_library(lwtensor_base_objs_sm75 OBJECT
              ${LWTENSOR_SRC_SM75})
set_target_properties(lwtensor_base_objs_sm75 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_SM75}"
)
lwtensor_fixed_keep(lwtensor_base_objs_sm75)
target_link_libraries(lwtensor_base_objs_sm75
  PRIVATE
    lwtensor_private_includes
)

if(${LWDA_VERSION_MAJOR} GREATER_EQUAL 11)

add_library(lwtensor_base_objs_sm80 OBJECT
              ${LWTENSOR_SRC_SM80})
set_target_properties(lwtensor_base_objs_sm80 PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_SM80}"
)
lwtensor_fixed_keep(lwtensor_base_objs_sm80)
target_link_libraries(lwtensor_base_objs_sm80
  PRIVATE
    lwtensor_private_includes
)

else()

add_library(lwtensor_base_objs_sm80 ALIAS lwtensor_base_objs_sm70)

endif()



# Base library
add_library(lwtensor_base_objs_lwda OBJECT
              ${LWTENSOR_CONTRACTION_SRC}
              ${LWTENSOR_ELEMENTWISE_SRC})
set_target_properties(lwtensor_base_objs_lwda PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_ALL}"
)
lwtensor_fixed_keep(lwtensor_base_objs_lwda)
target_link_libraries(lwtensor_base_objs_lwda
  PRIVATE
    lwtensor_private_includes
)

add_library(lwtensor_base_objs_cpp OBJECT
              ${LWTENSOR_UTILS_SRC})
set_target_properties(lwtensor_base_objs_cpp PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(lwtensor_base_objs_cpp
  PRIVATE
    lwtensor_private_includes
)

add_library(lwtensor_base INTERFACE)
target_sources(lwtensor_base INTERFACE
    $<TARGET_OBJECTS:lwtensor_base_objs_sm61>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm70>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm75>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm80>
    $<TARGET_OBJECTS:lwtensor_base_objs_cpp>
    $<TARGET_OBJECTS:lwtensor_base_objs_lwda>
    $<TARGET_OBJECTS:lwtensor_dnn_heuristic>)

if(${LWDA_VERSION_MAJOR} GREATER_EQUAL 11)
SET(LWTENSOR_LWBLASLT_LIBRARY "lwblasLt")
else()
SET(LWTENSOR_LWBLASLT_LIBRARY "")
endif()

# Set flags to export files, which expose API functions in release build
if(WIN32)
  set(DYNAMIC_SYMBOL_EXPORT_FLAG "/DEF:\"${PROJECT_SOURCE_DIR}/cmake/liblwtensor.def\"")
else()
  if(LWTENSOR_EXPOSE_INTERNAL)
    set(DYNAMIC_SYMBOL_EXPORT_FLAG "-Wl,--version-script=${PROJECT_SOURCE_DIR}/cmake/liblwtensor-internal.map")
  else()
    set(DYNAMIC_SYMBOL_EXPORT_FLAG "-Wl,--version-script=${PROJECT_SOURCE_DIR}/cmake/liblwtensor.map")
  endif()
endif()

if(WIN32)
else()
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-rpath,'$ORIGIN'")
endif()

IF(LIBRARY_TYPE STREQUAL "default")
  add_library(lwtensor SHARED
    $<TARGET_OBJECTS:lwtensor_base_objs_sm61>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm70>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm75>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm80>
    $<TARGET_OBJECTS:lwtensor_base_objs_cpp>
    $<TARGET_OBJECTS:lwtensor_base_objs_lwda>
    $<TARGET_OBJECTS:lwtensor_dnn_heuristic>)
  # DEVICE LINK CMAKE ISSUE FIX #2: Patch link file. Probably not portable, smaller binaries.
  set_target_properties(lwtensor
    PROPERTIES
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    LINK_FLAGS_RELEASE ${DYNAMIC_SYMBOL_EXPORT_FLAG})
  add_library(lwtensor_static STATIC
    $<TARGET_OBJECTS:lwtensor_base_objs_sm61>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm70>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm75>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm80>
    $<TARGET_OBJECTS:lwtensor_base_objs_cpp>
    $<TARGET_OBJECTS:lwtensor_base_objs_lwda>
    $<TARGET_OBJECTS:lwtensor_dnn_heuristic>)
  set_target_properties(lwtensor_static
    PROPERTIES
    LINK_FLAGS_RELEASE "-Wl,--exclude-libs=ALL")
  target_link_libraries(lwtensor_static ${LWTENSOR_LWBLASLT_LIBRARY})
  target_link_libraries(lwtensor ${LWTENSOR_LWBLASLT_LIBRARY})
  add_library(lwtensor::lwtensor ALIAS lwtensor)
ELSEIF(LIBRARY_TYPE STREQUAL "shared")
  add_library(lwtensor SHARED
    $<TARGET_OBJECTS:lwtensor_base_objs_sm61>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm70>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm75>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm80>
    $<TARGET_OBJECTS:lwtensor_base_objs_cpp>
    $<TARGET_OBJECTS:lwtensor_base_objs_lwda>
    $<TARGET_OBJECTS:lwtensor_dnn_heuristic>)
  # DEVICE LINK CMAKE ISSUE FIX #2: Patch link file. Probably not portable, smaller binaries.
  set_target_properties(lwtensor
    PROPERTIES
    VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    LINK_FLAGS_RELEASE ${DYNAMIC_SYMBOL_EXPORT_FLAG})
  target_link_libraries(lwtensor ${LWTENSOR_LWBLASLT_LIBRARY})
  add_library(lwtensor::lwtensor ALIAS lwtensor)
ELSEIF(LIBRARY_TYPE STREQUAL "static")
  add_library(lwtensor_static STATIC
    $<TARGET_OBJECTS:lwtensor_base_objs_sm61>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm70>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm75>
    $<TARGET_OBJECTS:lwtensor_base_objs_sm80>
    $<TARGET_OBJECTS:lwtensor_base_objs_cpp>
    $<TARGET_OBJECTS:lwtensor_base_objs_lwda>
    $<TARGET_OBJECTS:lwtensor_dnn_heuristic>)
  set_target_properties(lwtensor_static
    PROPERTIES
    LINK_FLAGS_RELEASE "-Wl,--exclude-libs=ALL")
  target_link_libraries(lwtensor_static ${LWTENSOR_LWBLASLT_LIBRARY})
  add_library(lwtensor::lwtensor ALIAS lwtensor_static)
ELSE()
 MESSAGE(FATAL_ERROR "Unsupported library type ${LIBRARY_TYPE}")
ENDIF()

# ##########################################
# Install library and examples
# ##########################################

IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
ENDIF()

# Install library
IF(LIBRARY_TYPE STREQUAL "default")
  install(
      TARGETS lwtensor_public_headers lwtensor
      EXPORT lwtensor-targets
      LIBRARY DESTINATION ${LWTENSOR_LIBRARY_INSTALL_DIR}
      COMPONENT LWTENSOR_PACKAGE
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
      ARCHIVE DESTINATION ${LWTENSOR_LIBRARY_INSTALL_DIR}
      COMPONENT LWTENSOR_PACKAGE
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
      RUNTIME DESTINATION ${LWTENSOR_LIBRARY_INSTALL_DIR}
      COMPONENT LWTENSOR_PACKAGE
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
      INCLUDES DESTINATION ${LWTENSOR_INCLUDES_INSTALL_DIR}
  )
  install(
      TARGETS lwtensor_static
      EXPORT lwtensor-targets
      ARCHIVE DESTINATION ${LWTENSOR_LIBRARY_INSTALL_DIR}
      COMPONENT LWTENSOR_PACKAGE
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
      INCLUDES DESTINATION ${LWTENSOR_INCLUDES_INSTALL_DIR}
  )
ELSEIF(LIBRARY_TYPE STREQUAL "shared")
  install(
      TARGETS lwtensor_public_headers lwtensor
      EXPORT lwtensor-targets
      LIBRARY DESTINATION ${LWTENSOR_LIBRARY_INSTALL_DIR}
      COMPONENT LWTENSOR_PACKAGE
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
      ARCHIVE DESTINATION ${LWTENSOR_LIBRARY_INSTALL_DIR}
      COMPONENT LWTENSOR_PACKAGE
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
      RUNTIME DESTINATION ${LWTENSOR_LIBRARY_INSTALL_DIR}
      COMPONENT LWTENSOR_PACKAGE
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
      INCLUDES DESTINATION ${LWTENSOR_INCLUDES_INSTALL_DIR}
  )
ELSEIF(LIBRARY_TYPE STREQUAL "static")
  install(
      TARGETS lwtensor_public_headers lwtensor_static
      EXPORT lwtensor-targets
      ARCHIVE DESTINATION ${LWTENSOR_LIBRARY_INSTALL_DIR}
      COMPONENT LWTENSOR_PACKAGE
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
      INCLUDES DESTINATION ${LWTENSOR_INCLUDES_INSTALL_DIR}
  )
ELSE()
 MESSAGE(FATAL_ERROR "Unsupported library type ${LIBRARY_TYPE}")
ENDIF()

# Install headers
install(
    FILES
        ${CMAKE_LWRRENT_SOURCE_DIR}/include/lwtensor.h
    DESTINATION ${LWTENSOR_INCLUDES_INSTALL_DIR}
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
    COMPONENT LWTENSOR_PACKAGE
)

if(LWTENSOR_EXPOSE_INTERNAL)
  install(
      FILES
          ${CMAKE_LWRRENT_SOURCE_DIR}/include/lwtensor/internal/lwtensor.h
      DESTINATION ${LWTENSOR_INCLUDES_INSTALL_DIR}/lwtensor/internal
      PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
      COMPONENT LWTENSOR_PACKAGE
  )
endif()

if(CMAKE_BUILD_TYPE MATCHES Release)

   # Generate the correct release types.h header
   add_lwstom_command(TARGET lwtensor
       POST_BUILD
       COMMAND ${Python_EXELWTABLE}
       ARGS ${CMAKE_LWRRENT_SOURCE_DIR}/misc/release.py ${CMAKE_LWRRENT_SOURCE_DIR} ${PROJECT_BINARY_DIR}
       BYPRODUCTS ${PROJECT_BINARY_DIR}/types.h ${PROJECT_BINARY_DIR}/lwtensor.h
       USES_TERMINAL VERBATIM
       COMMENT "Generating release-version header files"
       )

  install(
      FILES
          ${PROJECT_BINARY_DIR}/types.h
      DESTINATION ${LWTENSOR_INCLUDES_INSTALL_DIR}/lwtensor
      PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
      COMPONENT LWTENSOR_PACKAGE
  )
  install(
      FILES
          ${CMAKE_LWRRENT_SOURCE_DIR}/release/license.txt
      DESTINATION ${LWTENSOR_INSTALL_DIR}
      PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
      COMPONENT LWTENSOR_PACKAGE
  )
else()
install(
    FILES
        ${CMAKE_LWRRENT_SOURCE_DIR}/include/lwtensor/types.h
    DESTINATION ${LWTENSOR_INCLUDES_INSTALL_DIR}/lwtensor
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
    COMPONENT LWTENSOR_PACKAGE
)
endif()

# Install examples
install(
    FILES
        ${CMAKE_LWRRENT_SOURCE_DIR}/external/lwdalibrarysamples/lwtensor/contraction.lw
        ${CMAKE_LWRRENT_SOURCE_DIR}/external/lwdalibrarysamples/lwtensor/contraction_simple.lw
        ${CMAKE_LWRRENT_SOURCE_DIR}/external/lwdalibrarysamples/lwtensor/contraction_autotuning.lw
        ${CMAKE_LWRRENT_SOURCE_DIR}/external/lwdalibrarysamples/lwtensor/elementwise_binary.lw
        ${CMAKE_LWRRENT_SOURCE_DIR}/external/lwdalibrarysamples/lwtensor/elementwise_permute.lw
        ${CMAKE_LWRRENT_SOURCE_DIR}/external/lwdalibrarysamples/lwtensor/elementwise_trinary.lw
        ${CMAKE_LWRRENT_SOURCE_DIR}/external/lwdalibrarysamples/lwtensor/reduction.lw
        ${CMAKE_LWRRENT_SOURCE_DIR}/external/lwdalibrarysamples/lwtensor/Makefile
    DESTINATION ${LWTENSOR_EXAMPLES_INSTALL_DIR}
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
    COMPONENT LWTENSOR_EXAMPLES
)

# ##########################################
# lwTENSOR package
# ##########################################

# Detect architecture if not set
if(NOT CMAKE_CROSSCOMPILING)
  exelwte_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE SYS_ARCH )
endif()

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYS_NAME)

# Set name and attributes
set(CPACK_PACKAGE_NAME "liblwtensor")
set(CPACK_PACKAGE_VENDOR "LWPU CORPORATION")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "lwTENSOR library")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME ${CPACK_PACKAGE_NAME}-${SYS_NAME}-${SYS_ARCH}-${CPACK_PACKAGE_VERSION})
set(CPACK_ARCHIVE_LWTENSOR_PACKAGE_FILE_NAME ${CPACK_PACKAGE_FILE_NAME})
set(CPACK_STRIP_FILES TRUE)

set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)

IF(CMAKE_BUILD_TYPE MATCHES Release)
  set(CPACK_COMPONENTS_ALL LWTENSOR_PACKAGE LWTENSOR_TEST)
  set(CPACK_COMPONENT_LWTENSOR_PACKAGE_GROUP LWTENSOR_PACKAGE)
  set(CPACK_COMPONENT_LWTENSOR_TEST_GROUP LWTENSOR_TEST)
ELSE()
  set(CPACK_COMPONENTS_GROUPING "ALL_COMPONENTS_IN_ONE")
  set(CPACK_COMPONENTS_ALL LWTENSOR_PACKAGE LWTENSOR_TEST LWTENSOR_EXAMPLES)
ENDIF()

set(CPACK_GENERATOR "TGZ;ZIP")

# Include CPack to introduce the appropriate targets
include(CPack)

# ##########################################
# lwTENSOR testing
# ##########################################
IF(BUILD_UNIT_TESTS OR BUILD_MOCK_TESTS)
  ENABLE_TESTING()
ENDIF()

#set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} ${CMAKE_LWDA_FLAGS_ARCH_ALL}")

# ##########################################
# lwTENSOR examples
# ##########################################
if(BUILD_EXAMPLES AND NOT CMAKE_CROSSCOMPILING) # examples not working in cross-compilation
  if(NOT DEFINED LWTENSOR_ROOT)
    set(LWTENSOR_ROOT ${CMAKE_LWRRENT_SOURCE_DIR})
  endif()
  if(NOT DEFINED LWTENSOR_EXAMPLE_BINARY_INSTALL_DIR)
    set(LWTENSOR_EXAMPLE_BINARY_INSTALL_DIR "liblwtensor/bin")
  endif()
  if(WIN32)
    add_subdirectory(external/lwdalibrarysamples/lwtensor)
  else()
      if(NOT BUILD_WITH_GCOV)
        add_subdirectory(samples)
      endif()
  endif()
endif()

# ##########################################
# lwTENSOR documentation
# ##########################################
if(BUILD_DOCS)
    add_subdirectory(doc)
endif()

# ##########################################
# lwTENSOR google test
# ##########################################

# ---[ Set GOOGLETEST_SOURCE_DIR to avoid downloading the source everytime.
set(GOOGLETEST_SOURCE_DIR ${CMAKE_LWRRENT_SOURCE_DIR}/test/googletest)

IF(BUILD_MOCK_TESTS OR BUILD_UNIT_TESTS)

  # ---[ Download googletest.
  IF(NOT DEFINED GOOGLETEST_SOURCE_DIR)
    message(STATUS "Downloading Google Test to ${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest (define GOOGLETEST_SOURCE_DIR to avoid it)")
    configure_file(cmake/DownloadGoogleTest.cmake "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download/CMakeLists.txt")
    exelwte_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    exelwte_process(COMMAND "${CMAKE_COMMAND}" --build .
      WORKING_DIRECTORY "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest-download")
    set(GOOGLETEST_SOURCE_DIR "${CONFU_DEPENDENCIES_SOURCE_DIR}/googletest" CACHE STRING "Google Test source directory")
  ENDIF()

  # Software dependencies.
  # ---------------------------
  find_package(Python)
  link_directories(${CMAKE_LWDA_IMPLICIT_LINK_DIRECTORIES})

  IF(NOT TARGET gtest)
    ADD_SUBDIRECTORY(
      "${GOOGLETEST_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/googletest")
    INCLUDE_DIRECTORIES("${GOOGLETEST_SOURCE_DIR}/googletest/include")
  ENDIF()

  FILE(GLOB TEST_CXX_SRC
    ${lwtensor_SOURCE_DIR}/test/src/*.cpp
    ${lwtensor_SOURCE_DIR}/test/src/*.lw)

  # apiTest, unitTest
  # ---------------------------
  if(CMAKE_BUILD_TYPE MATCHES Debug)
    LIST(REMOVE_ITEM TEST_CXX_SRC ${CMAKE_LWRRENT_SOURCE_DIR}/test/src/lwtensorTest.lw)

    #---[ Loop over each test routine.
    IF(LIBRARY_TYPE STREQUAL "default" OR LIBRARY_TYPE STREQUAL "shared")
      IF(NOT WIN32)
      FOREACH(src ${TEST_CXX_SRC})
        GET_FILENAME_COMPONENT(exe ${src} NAME_WE)
        add_exelwtable(${exe} ${src})
        target_include_directories(${exe}
            PUBLIC
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include>
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include/lwtensor>
            PRIVATE
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/lwtensor/internal>
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/lwtlass/include>
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/dnnheuristic>
        )

        set_target_properties(${exe} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_ALL}"
        )
        lwtensor_fixed_keep(${exe})

        target_link_libraries(${exe}
            PUBLIC
                lwtensor
            PRIVATE
                lwdart
                lwblas
                lwblasLt
                lwToolsExt
                gtest
                ${CMAKE_DL_LIBS}
                rt
        )
        add_test(NAME ${exe} COMMAND ./bin/${exe})
        INSTALL(TARGETS ${exe}
          DESTINATION ${LWTENSOR_BINARY_INSTALL_DIR}
          COMPONENT LWTENSOR_TEST)
      ENDFOREACH()
      ENDIF()
    ENDIF()



    IF(LIBRARY_TYPE STREQUAL "default" OR LIBRARY_TYPE STREQUAL "static")
      FOREACH(src ${TEST_CXX_SRC})
        GET_FILENAME_COMPONENT(exe ${src} NAME_WE)
        add_exelwtable(${exe}_static ${src})
        target_include_directories(${exe}_static
            PUBLIC
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include>
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include/lwtensor>
            PRIVATE
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/lwtensor/internal>
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/lwtlass/include>
              $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/dnnheuristic>
        )
        set_target_properties(${exe}_static PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_ALL}"
        )
        lwtensor_fixed_keep(${exe}_static)
        IF(WIN32)
          target_link_libraries(${exe}_static
              PUBLIC
                  lwtensor_static
              PRIVATE
                  lwdart_static
                  lwblas
                  lwblasLt
                  gtest
                  ${CMAKE_DL_LIBS}
          )
          target_compile_definitions(${exe}_static PRIVATE -DGTEST_LINKED_AS_SHARED_LIBRARY)
        ELSE()
          # FIXME: link lwblasLt statically
          target_link_libraries(${exe}_static
              PUBLIC
                  lwtensor_static
              PRIVATE
                  lwdart_static
                  lwblas
                  lwblasLt
                  lwlibos
                  lwToolsExt
                  gtest
                  ${CMAKE_DL_LIBS}
                  rt
          )
        ENDIF()
        add_test(NAME ${exe}_static COMMAND ./bin/${exe}_static)
        INSTALL(TARGETS ${exe}_static
          DESTINATION ${LWTENSOR_BINARY_INSTALL_DIR}
          COMPONENT LWTENSOR_TEST)
      ENDFOREACH()
    ENDIF()
  endif()

  # lwtensorTest
  # ---------------------------
  FILE(GLOB TEST_CXX_SRC ${lwtensor_SOURCE_DIR}/test/src/*utensorTest.lw)

  IF(LIBRARY_TYPE STREQUAL "default" OR LIBRARY_TYPE STREQUAL "shared")
    FOREACH(src ${TEST_CXX_SRC})
      GET_FILENAME_COMPONENT(exe ${src} NAME_WE)
      add_exelwtable(${exe} ${src})
      target_include_directories(${exe}
          PUBLIC
            $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include/lwtensor>
          PRIVATE
            $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/lwtensor/internal>
            $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/lwtlass/include>
      )

      if(NOT CMAKE_CROSSCOMPILING)
        set_target_properties(${exe} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_ALL} -Xcompiler -fopenmp"
            LINK_FLAGS "-fopenmp"
        )
      else()
        set_target_properties(${exe} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_ALL}"
        )
      endif()
      lwtensor_fixed_keep(${exe})
      if(NOT WIN32)
        target_link_libraries(${exe}
            PUBLIC
                lwtensor
            PRIVATE
                lwdart
                lwblas
                lwblasLt
                lwToolsExt
                gtest
                ${CMAKE_DL_LIBS}
                ${LWTENSOR_LWML_LIBRARY_NAME}
                rt
        )
      else()
        target_link_libraries(${exe}
              PUBLIC
                  lwtensor
              PRIVATE
                  lwdart
                  lwblas
                  lwblasLt
                  gtest
                  ${CMAKE_DL_LIBS}
                  ${LWTENSOR_LWML_LIBRARY_NAME}
          )
      endif()
      MESSAGE(STATUS "${exe}")
      MESSAGE(STATUS "${PYTHON_EXELWTABLE}")
      IF(EXE STREQUAL "lwtensorTest")
        ADD_TEST(NAME lwtensorTestL0 COMMAND ${PYTHON_EXELWTABLE} lwtensor_run.py -binname lwtensorTest -config b,pw,L0 WORKING_DIRECTORY scripts)
      ENDIF()
      INSTALL(TARGETS ${exe}
        DESTINATION ${LWTENSOR_BINARY_INSTALL_DIR}
        COMPONENT LWTENSOR_TEST)
    ENDFOREACH()
  ENDIF()

  IF(LIBRARY_TYPE STREQUAL "default" OR LIBRARY_TYPE STREQUAL "static")
    FOREACH(src ${TEST_CXX_SRC})
      GET_FILENAME_COMPONENT(exe ${src} NAME_WE)
      add_exelwtable(${exe}_static ${src})
      target_include_directories(${exe}_static
          PUBLIC
            $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/include/lwtensor>
          PRIVATE
            $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/lwtensor/internal>
            $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/external/lwtlass/include>
      )

      if(NOT CMAKE_CROSSCOMPILING)
        set_target_properties(${exe}_static PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_ALL} -Xcompiler -fopenmp"
            LINK_FLAGS "-fopenmp"
        )
      else()
        set_target_properties(${exe}_static PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            COMPILE_FLAGS "${CMAKE_LWDA_FLAGS_ARCH_ALL}"
        )
      endif()
      lwtensor_fixed_keep(${exe}_static)

      IF(WIN32)
        target_link_libraries(${exe}_static
            PUBLIC
                lwtensor_static
            PRIVATE
                lwdart_static
                lwblas
                lwblasLt
                gtest
                ${CMAKE_DL_LIBS}
                ${LWTENSOR_LWML_LIBRARY_NAME}
        )
        target_compile_definitions(${exe}_static PRIVATE -DGTEST_LINKED_AS_SHARED_LIBRARY)
      ELSE()
        # FIXME: link lwblasLt statically
        target_link_libraries(${exe}_static
            PUBLIC
                lwtensor_static
            PRIVATE
                lwdart_static
                lwblas
                lwblasLt
                lwlibos
                lwToolsExt
                gtest
                ${CMAKE_DL_LIBS}
                ${LWTENSOR_LWML_LIBRARY_NAME}
                rt
        )
      ENDIF()
      MESSAGE(STATUS "${exe}_static")
      INSTALL(TARGETS ${exe}_static
        DESTINATION ${LWTENSOR_BINARY_INSTALL_DIR}
        COMPONENT LWTENSOR_TEST)
    ENDFOREACH()
  ENDIF()

if (WIN32 AND CMAKE_BUILD_TYPE MATCHES Release)
  install(
      TARGETS gtest gmock gtest_main
      EXPORT lwtensor-targets
      RUNTIME DESTINATION ${LWTENSOR_BINARY_INSTALL_DIR}
      COMPONENT LWTENSOR_TEST
      PERMISSIONS OWNER_EXELWTE OWNER_WRITE OWNER_READ GROUP_EXELWTE GROUP_READ WORLD_EXELWTE WORLD_READ
  )
endif()

IF(NOT WIN32)
  ADD_TEST(NAME DontLinkGOMP COMMAND bash -c "! readelf -d $<TARGET_FILE:lwtensor> | grep gomp")
ENDIF()

  # ---[ L0 TESTING
  # ---[ Add buggy script (removed as of 10/15/2019, added to other test cases)
  # CONFIGURE_FILE(test/bug.sh ${CMAKE_BINARY_DIR}/test/bug.sh COPYONLY)
  # ADD_TEST(NAME bugTest COMMAND ./bin/lwtensorTest -file test/bug.sh)
  # ---[ Add coreTest script
  CONFIGURE_FILE(test/correctness/coretest_testsuits.sh ${CMAKE_BINARY_DIR}/test/coretest_testsuits.sh COPYONLY)
  ADD_TEST(NAME coreTest COMMAND ./bin/lwtensorTest -showFailureOnly -file test/coretest_testsuits.sh)
  # ---[ Add L0
  CONFIGURE_FILE(test/correctness/lwtensorElementwiseL0.sh ${CMAKE_BINARY_DIR}/test/lwtensorElementwiseL0.sh COPYONLY)
  ADD_TEST(NAME L0 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorElementwiseL0.sh)
  # ---[ Add contraction L0
  CONFIGURE_FILE(test/correctness/lwtensorContractionL0.sh ${CMAKE_BINARY_DIR}/test/lwtensorContractionL0.sh COPYONLY)
  ADD_TEST(NAME ContractionL0 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorContractionL0.sh)
  # ---[ Add contraction with cache L0
  CONFIGURE_FILE(test/correctness/lwtensorContractionCacheL0.sh ${CMAKE_BINARY_DIR}/test/lwtensorContractionCacheL0.sh COPYONLY)
  ADD_TEST(NAME ContractionCacheL0 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorContractionCacheL0.sh)
  CONFIGURE_FILE(test/correctness/lwtensorContractionCacheRepeat.sh ${CMAKE_BINARY_DIR}/test/lwtensorContractionCacheRepeat.sh COPYONLY)
  ADD_TEST(NAME ContractionCacheRepeat0 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorContractionCacheRepeat.sh)
  # run the same test twice to make sure the reading from cache is also working
  CONFIGURE_FILE(test/correctness/lwtensorContractionCacheRepeat.sh ${CMAKE_BINARY_DIR}/test/lwtensorContractionCacheRepeat.sh COPYONLY)
  ADD_TEST(NAME ContractionCacheRepeat1 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorContractionCacheRepeat.sh)
  # ---[ Add reduction L0
  CONFIGURE_FILE(test/correctness/lwtensorReductionL0.sh ${CMAKE_BINARY_DIR}/test/lwtensorReductionL0.sh COPYONLY)
  ADD_TEST(NAME ReductionL0 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorReductionL0.sh)
  # ---[ Add high dimensional contraction L0
  CONFIGURE_FILE(test/correctness/lwtensorContractionHighDimensionalL0.sh ${CMAKE_BINARY_DIR}/test/lwtensorContractionHighDimensionalL0.sh COPYONLY)
  ADD_TEST(NAME HighDimensionalContractionL0 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorContractionHighDimensionalL0.sh)
  # ---[ Add high dimensional reduction L0
  CONFIGURE_FILE(test/correctness/lwtensorReductionHighDimensionalL0.sh ${CMAKE_BINARY_DIR}/test/lwtensorReductionHighDimensionalL0.sh COPYONLY)
  ADD_TEST(NAME HighDimensionalReductionL0 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorReductionHighDimensionalL0.sh)
  # ---[ Add high dimensional elementwise L0
  CONFIGURE_FILE(test/correctness/lwtensorElementwiseHighDimensionalL0.sh ${CMAKE_BINARY_DIR}/test/lwtensorElementwiseHighDimensionalL0.sh COPYONLY)
  ADD_TEST(NAME HighDimensionalElementwiseL0 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorElementwiseHighDimensionalL0.sh)

  # ---[ L1 TESTING
  IF($ELW{LWTENSOR_CI_TEST_L1} MATCHES 1)
    # ---[ Add L1
    CONFIGURE_FILE(test/correctness/lwtensorElementwiseL1.sh ${CMAKE_BINARY_DIR}/test/lwtensorElementwiseL1.sh COPYONLY)
    ADD_TEST(NAME L1 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorElementwiseL1.sh)
    # ---[ Add contraction L1
    CONFIGURE_FILE(test/correctness/lwtensorContractionL1.sh ${CMAKE_BINARY_DIR}/test/lwtensorContractionL1.sh COPYONLY)
    ADD_TEST(NAME ContractionL1 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorContractionL1.sh)
    # ---[ Add reduciton L1
    CONFIGURE_FILE(test/correctness/lwtensorReductionL1.sh ${CMAKE_BINARY_DIR}/test/lwtensorReductionL1.sh COPYONLY)
    ADD_TEST(NAME ReductionL1 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorReductionL1.sh)
    # ---[ Add high dimensional contraction L1
    CONFIGURE_FILE(test/correctness/lwtensorContractionHighDimensionalL1.sh ${CMAKE_BINARY_DIR}/test/lwtensorContractionHighDimensionalL1.sh COPYONLY)
    ADD_TEST(NAME HighDimensionalContractionL1 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorContractionHighDimensionalL1.sh)
  ENDIF()

  # ---[ L2 TESTING
  IF($ELW{LWTENSOR_CI_TEST_L2} MATCHES 1)
    # ---[ Add L2
    CONFIGURE_FILE(test/correctness/lwtensorElementwiseL2.sh ${CMAKE_BINARY_DIR}/test/lwtensorElementwiseL2.sh COPYONLY)
    ADD_TEST(NAME L2 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorElementwiseL2.sh)
    # ---[ Add contraction L2
    CONFIGURE_FILE(test/correctness/lwtensorContractionL2.sh ${CMAKE_BINARY_DIR}/test/lwtensorContractionL2.sh COPYONLY)
    ADD_TEST(NAME ContractionL2 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorContractionL2.sh)
    # ---[ Add reduciton L2
    CONFIGURE_FILE(test/correctness/lwtensorReductionL2.sh ${CMAKE_BINARY_DIR}/test/lwtensorReductionL2.sh COPYONLY)
    ADD_TEST(NAME ReductionL2 COMMAND ./bin/lwtensorTest -showFailureOnly -file test/lwtensorReductionL2.sh)
  ENDIF()

  # ---[ Copy testing infrastrcture.
  FILE(COPY ${lwtensor_SOURCE_DIR}/test/scripts DESTINATION ${CMAKE_BINARY_DIR})
  # ---[ Deal with lwtensorTest here.
ENDIF()

# Install correctness test suites
# --------------------------------
file(GLOB lwtensor_correctness_SCRIPTS test/correctness/*.sh)
install(FILES ${lwtensor_correctness_SCRIPTS}
 DESTINATION ${LWTENSOR_TEST_INSTALL_DIR}
 COMPONENT LWTENSOR_TEST)

# Install regression test suites
# --------------------------------
list(APPEND lwtensor_regression_contraction
  test/regression/contraction/rand1000.sh
  test/regression/contraction/gemm_sq_nt.sh
  test/regression/contraction/gemm_sq_nn.sh
  test/regression/contraction/mxnet.sh)
install(FILES ${lwtensor_regression_contraction}
 DESTINATION ${LWTENSOR_TEST_INSTALL_DIR}/regression/contraction
 COMPONENT LWTENSOR_TEST)

list(APPEND lwtensor_regression_elementwise
  test/regression/elementwise/easy_ew.sh
  test/regression/elementwise/rand300_ew.sh
  test/regression/elementwise/permute.sh)
install(FILES ${lwtensor_regression_elementwise}
 DESTINATION ${LWTENSOR_TEST_INSTALL_DIR}/regression/elementwise
 COMPONENT LWTENSOR_TEST)

list(APPEND lwtensor_regression_reduction
  test/regression/reduction/gemv_largeK.sh
  test/regression/reduction/reduction_largeK.sh
  test/regression/reduction/gemvBenchmark.sh)
install(FILES ${lwtensor_regression_reduction}
 DESTINATION ${LWTENSOR_TEST_INSTALL_DIR}/regression/reduction
 COMPONENT LWTENSOR_TEST)
