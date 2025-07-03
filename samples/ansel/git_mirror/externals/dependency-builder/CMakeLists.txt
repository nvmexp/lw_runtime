cmake_minimum_required(VERSION 2.8.4)
include(ExternalProject)

set(variables
  CMAKE_C_FLAGS_DEBUG
  CMAKE_C_FLAGS_MINSIZEREL
  CMAKE_C_FLAGS_RELEASE
  CMAKE_C_FLAGS_RELWITHDEBINFO
  CMAKE_CXX_FLAGS_DEBUG
  CMAKE_CXX_FLAGS_MINSIZEREL
  CMAKE_CXX_FLAGS_RELEASE
  CMAKE_CXX_FLAGS_RELWITHDEBINFO
)

macro(configure_msvc_runtime)
  if(MSVC)
    # Default to statically-linked runtime.
    if("${MSVC_RUNTIME}" STREQUAL "")
      set(MSVC_RUNTIME "static")
    endif()
    # Set compiler options.
    if(${MSVC_RUNTIME} STREQUAL "static")
      message(STATUS
        "MSVC -> forcing use of statically-linked runtime."
      )
      foreach(variable ${variables})
        if(${variable} MATCHES "/MD")
          string(REGEX REPLACE "/MD" "/MT" ${variable} ${${variable}})
        endif()
      endforeach()
    else()
      message(STATUS
        "MSVC -> forcing use of dynamically-linked runtime."
      )
      foreach(variable ${variables})
        if(${variable} MATCHES "/MT")
          string(REGEX REPLACE "/MT" "/MD" ${variable} ${${variable}})
        endif()
      endforeach()
    endif()
  endif()
endmacro()

configure_msvc_runtime()
foreach(variable ${variables})
set(opt "${${variable}} /Qspectre /Dssize_t=int")
list(APPEND CMAKE_ARGS -D${variable}:string=${opt})
endforeach()

ExternalProject_Add(
  zlib-ng

  GIT_REPOSITORY "https://github.com/Dead2/zlib-ng"
  GIT_TAG "develop"
  
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  
  SOURCE_DIR "${SRC_DIR}/source/zlib-ng"
  CMAKE_ARGS ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/libs/zlib-ng/1.9.9 -DZLIB_ENABLE_TESTS=OFF -DZLIB_COMPAT=ON
  TEST_COMMAND ""
)

ExternalProject_Add(
  yaml-cpp

  GIT_REPOSITORY "https://github.com/jbeder/yaml-cpp.git"
  GIT_TAG "release-0.5.3"
  
  UPDATE_COMMAND ""
  PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/yaml/release-0.5.3/ordering.patch
  
  SOURCE_DIR "${SRC_DIR}/source/yaml-cpp"
  CMAKE_ARGS ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/libs/yaml-cpp/0.5.3 -DMSVC_SHARED_RT=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_CPP_BUILD_CONTRIB=OFF -DBoost_INCLUDE_DIR=${CMAKE_SOURCE_DIR}/.deps/boost/include
  TEST_COMMAND ""
)

ExternalProject_Add(
  openexr-ilmbase

  GIT_REPOSITORY "https://github.com/openexr/openexr.git"
  GIT_TAG "v2.2.1"
  
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  
  SOURCE_DIR "${SRC_DIR}/source/openexr"
  SOURCE_SUBDIR "IlmBase"
  CMAKE_ARGS ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/libs/openexr/2.2.1 -DBUILD_SHARED_LIBS=OFF
  TEST_COMMAND ""
)

ExternalProject_Add(
  openexr-openexr

  GIT_REPOSITORY "https://github.com/openexr/openexr.git"
  GIT_TAG "v2.2.1"
  
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  
  SOURCE_DIR "${SRC_DIR}/source/openexr"
  SOURCE_SUBDIR "OpenEXR"
  CMAKE_ARGS ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/libs/openexr/2.2.1 -DZLIB_ROOT=${TARGET_DIR}/libs/zlib-ng/1.9.9 -DILMBASE_PACKAGE_PREFIX=${TARGET_DIR}/libs/openexr/2.2.1 -DBUILD_SHARED_LIBS=OFF
  TEST_COMMAND ""
)

ExternalProject_Add(
  libjpeg-turbo

  GIT_REPOSITORY "https://github.com/libjpeg-turbo/libjpeg-turbo.git"
  GIT_TAG "1.5.3"
  
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  
  SOURCE_DIR "${SRC_DIR}/source/libjpeg-turbo"
  CMAKE_ARGS ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/libs/libjpeg-turbo/1.5.3 -DNASM=${CMAKE_SOURCE_DIR}/.deps/nasm/nasm.exe
  TEST_COMMAND ""
)

ExternalProject_Add(
  kissfft

  GIT_REPOSITORY "https://github.com/AchimTuran/KissFFT.git"
  GIT_TAG "V1.30"
  
  UPDATE_COMMAND ""
  PATCH_COMMAND git apply ${CMAKE_SOURCE_DIR}/patches/kissfft/V1.30/kissfft.patch
  
  SOURCE_DIR "${SRC_DIR}/source/kissfft"
  CMAKE_ARGS ${CMAKE_ARGS} -DCMAKE_INSTALL_PREFIX=${TARGET_DIR}/libs/kissfft/1.30
  TEST_COMMAND ""
)

add_dependencies(openexr-openexr openexr-ilmbase zlib-ng)
