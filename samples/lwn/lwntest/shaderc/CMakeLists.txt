cmake_minimum_required(VERSION 2.8.12)
project(shaderc)
enable_testing()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "")
  message(STATUS "No build type selected, default to Debug")
  set(CMAKE_BUILD_TYPE "Debug")
endif()

message(STATUS "Shaderc: build type is \"${CMAKE_BUILD_TYPE}\".")

option(SHADERC_ENABLE_SPVC "Enable libshaderc_spvc" OFF)

option(SHADERC_SKIP_INSTALL "Skip installation" ${SHADERC_SKIP_INSTALL})
if(NOT ${SHADERC_SKIP_INSTALL})
  set(SHADERC_ENABLE_INSTALL ON)
endif()

option(SHADERC_SKIP_TESTS "Skip building tests" ${SHADERC_SKIP_TESTS})
if(NOT ${SHADERC_SKIP_TESTS})
  set(SHADERC_ENABLE_TESTS ON)
endif()
if(${SHADERC_ENABLE_TESTS})
  message(STATUS "Configuring Shaderc to build tests.")
else()
  message(STATUS "Configuring Shaderc to avoid building tests.")
endif()

option(SHADERC_SPVC_ENABLE_DIRECT_LOGGING "Send logging messages from spvc directly to stdout/err" OFF)
if (${SHADERC_SPVC_ENABLE_DIRECT_LOGGING})
  add_definitions(-DSHADERC_SPVC_ENABLE_DIRECT_LOGGING)
endif()

option(SHADERC_SPVC_DISABLE_CONTEXT_LOGGING "Send logging messages from spvc to context object" OFF)
if (${SHADERC_SPVC_DISABLE_CONTEXT_LOGGING})
  add_definitions(-DSHADERC_SPVC_DISABLE_CONTEXT_LOGGING)
endif()

option(SHADERC_ENABLE_WERROR_COMPILE "Enable passing -Werror to compiler, if available" ON)

set (CMAKE_CXX_STANDARD 11)

include(GNUInstallDirs)
include(cmake/setup_build.cmake)
include(cmake/utils.cmake)
include(CheckCXXCompilerFlag)

set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS ${DISABLE_EXCEPTIONS} CACHE BOOL "Coupling SPIRV-Cross exception colwersion to DISABLE_EXCEPTIONS" FORCE)
if(DISABLE_EXCEPTIONS)
  # Need to set additional values here, since some of the wrapped code oclwrs in
  # .h/.hpp files, so maybe included outside of the library.
 add_definitions(-DSPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS)
endif()

# These flags are not supported on Windows and some older version of GCC
# that our bots use.
# Warning about implicit fallthrough in switch blocks
check_cxx_compiler_flag(-Wimplicit-fallthrough COMPILER_SUPPORTS_FALLTHROUGH_WARNING)
if (COMPILER_SUPPORTS_FALLTHROUGH_WARNING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wimplicit-fallthrough")
endif()

# Warning about extra semi-colons
check_cxx_compiler_flag(-Wextra-semi COMPILER_SUPPORTS_EXTRA_SEMI_WARNING)
if (COMPILER_SUPPORTS_EXTRA_SEMI_WARNING)
    add_compile_options("-Wextra-semi")
endif()

find_host_package(PythonInterp 3 REQUIRED)

add_lwstom_target(check-copyright ALL
  ${PYTHON_EXELWTABLE} ${CMAKE_LWRRENT_SOURCE_DIR}/utils/add_copyright.py
  --check
  WORKING_DIRECTORY ${CMAKE_LWRRENT_SOURCE_DIR}
  COMMENT "Check copyright")

add_lwstom_target(add-copyright
  ${PYTHON_EXELWTABLE} ${CMAKE_LWRRENT_SOURCE_DIR}/utils/add_copyright.py
  WORKING_DIRECTORY ${CMAKE_LWRRENT_SOURCE_DIR}
  COMMENT "Add copyright")

if(MSVC)
  option(SHADERC_ENABLE_SHARED_CRT
          "Use the shared CRT instead of the static CRT"
          ${SHADERC_ENABLE_SHARED_CRT})
  if (NOT SHADERC_ENABLE_SHARED_CRT)
    # Link exelwtables statically by replacing /MD with /MT everywhere.
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif(NOT SHADERC_ENABLE_SHARED_CRT)
endif(MSVC)


# Configure subdirectories.
# We depend on these for later projects, so they should come first.
add_subdirectory(third_party)

if(SHADERC_ENABLE_SPVC)
add_subdirectory(libshaderc_spvc)
add_subdirectory(spvc)
endif()
add_subdirectory(libshaderc_util)
add_subdirectory(libshaderc)
add_subdirectory(glslc)
add_subdirectory(examples)

add_lwstom_target(build-version
  ${PYTHON_EXELWTABLE}
  ${CMAKE_LWRRENT_SOURCE_DIR}/utils/update_build_version.py
  ${shaderc_SOURCE_DIR} ${spirv-tools_SOURCE_DIR} ${glslang_SOURCE_DIR} ${CMAKE_LWRRENT_BINARY_DIR}/build-version.inc
  COMMENT "Update build-version.inc in the Shaderc build directory (if necessary).")

function(define_pkg_config_file NAME LIBS)
  add_lwstom_target(${NAME}-pkg-config ALL
    COMMAND ${CMAKE_COMMAND}
      -DCHANGES_FILE=${CMAKE_LWRRENT_SOURCE_DIR}/CHANGES
      -DTEMPLATE_FILE=${CMAKE_LWRRENT_SOURCE_DIR}/cmake/${NAME}.pc.in
      -DOUT_FILE=${CMAKE_LWRRENT_BINARY_DIR}/${NAME}.pc
      -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
      -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
      -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
      -DLIBS=${LIBS}
      -P ${CMAKE_LWRRENT_SOURCE_DIR}/cmake/write_pkg_config.cmake
  DEPENDS "CHANGES" "cmake/${NAME}.pc.in" "cmake/write_pkg_config.cmake")

  if (SHADERC_ENABLE_INSTALL)
    install(
      FILES
        ${CMAKE_LWRRENT_BINARY_DIR}/${NAME}.pc
      DESTINATION
        ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
  endif()
endfunction()

if(SHADERC_ENABLE_SPVC)
  set(EXTRA_STATIC_PKGCONFIG_LIBS -lshaderc_spvc)
endif()
define_pkg_config_file(shaderc -lshaderc_shared)
define_pkg_config_file(shaderc_static "-lshaderc ${EXTRA_STATIC_PKGCONFIG_LIBS} -lshaderc_util")
define_pkg_config_file(shaderc_combined -lshaderc_combined)
