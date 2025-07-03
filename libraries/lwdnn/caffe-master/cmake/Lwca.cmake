# Known LWPU GPU achitectures Caffe can be compiled for.
# This list will be used for LWDA_ARCH_NAME = All option
set(Caffe_known_gpu_archs "30 35 50 52 60 61 70 75")

################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   caffe_detect_installed_gpus(out_variable)
function(caffe_detect_installed_gpus out_variable)
  if(NOT LWDA_gpu_detect_output)
    set(__lwfile ${PROJECT_BINARY_DIR}/detect_lwda_archs.lw)

    file(WRITE ${__lwfile} ""
      "#include <cstdio>\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (lwdaSuccess != lwdaGetDeviceCount(&count)) return -1;\n"
      "  if (count == 0) return -1;\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    lwdaDeviceProp prop;\n"
      "    if (lwdaSuccess == lwdaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")

    exelwte_process(COMMAND "${LWDA_LWCC_EXELWTABLE}" "--run" "${__lwfile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __lwcc_res OUTPUT_VARIABLE __lwcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(__lwcc_res EQUAL 0)
      set(LWDA_gpu_detect_output ${__lwcc_out} CACHE INTERNAL "Returned GPU architectures from caffe_detect_gpus tool" FORCE)
    endif()
  endif()

  if(NOT LWDA_gpu_detect_output)
    message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
    set(${out_variable} ${Caffe_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${LWDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()


################################################################################################
# Function for selecting GPU arch flags for lwcc based on LWDA_ARCH_NAME
# Usage:
#   caffe_select_lwcc_arch_flags(out_variable)
function(caffe_select_lwcc_arch_flags out_variable)
  # List of arch names
  set(__archs_names "Kepler" "Maxwell" "Pascal" "Volta" "Turing" "All" "Manual")
  set(__archs_name_default "All")
  if(NOT CMAKE_CROSSCOMPILING)
    list(APPEND __archs_names "Auto")
    set(__archs_name_default "Auto")
  endif()

  # set LWDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(LWDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target LWPU GPU architecture.")
  set_property( CACHE LWDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
  mark_as_advanced(LWDA_ARCH_NAME)

  # verify LWDA_ARCH_NAME value
  if(NOT ";${__archs_names};" MATCHES ";${LWDA_ARCH_NAME};")
    string(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Only ${__archs_names} architecture names are supported.")
  endif()

  if(${LWDA_ARCH_NAME} STREQUAL "Manual")
    set(LWDA_ARCH_BIN ${Caffe_known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
    set(LWDA_ARCH_PTX "50"                     CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
    mark_as_advanced(LWDA_ARCH_BIN LWDA_ARCH_PTX)
  else()
    unset(LWDA_ARCH_BIN CACHE)
    unset(LWDA_ARCH_PTX CACHE)
  endif()

  if(${LWDA_ARCH_NAME} STREQUAL "Kepler")
    set(__lwda_arch_bin "30 35")
  elseif(${LWDA_ARCH_NAME} STREQUAL "Maxwell")
    set(__lwda_arch_bin "50 52")
  elseif(${LWDA_ARCH_NAME} STREQUAL "Pascal")
    set(__lwda_arch_bin "60 61")
  elseif(${LWDA_ARCH_NAME} STREQUAL "Volta")
    set(__lwda_arch_bin "70")
  elseif(${LWDA_ARCH_NAME} STREQUAL "Turing")
    set(__lwda_arch_bin "75")
  elseif(${LWDA_ARCH_NAME} STREQUAL "All")
    set(__lwda_arch_bin ${Caffe_known_gpu_archs})
  elseif(${LWDA_ARCH_NAME} STREQUAL "Auto")
    caffe_detect_installed_gpus(__lwda_arch_bin)
  else()  # (${LWDA_ARCH_NAME} STREQUAL "Manual")
    set(__lwda_arch_bin ${LWDA_ARCH_BIN})
  endif()

  # remove dots and colwert to lists
  string(REGEX REPLACE "\\." "" __lwda_arch_bin "${__lwda_arch_bin}")
  string(REGEX REPLACE "\\." "" __lwda_arch_ptx "${LWDA_ARCH_PTX}")
  string(REGEX MATCHALL "[0-9()]+" __lwda_arch_bin "${__lwda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   __lwda_arch_ptx "${__lwda_arch_ptx}")
  caffe_list_unique(__lwda_arch_bin __lwda_arch_ptx)

  set(__lwcc_flags "")
  set(__lwcc_archs_readable "")

  # Tell LWCC to add binaries for the specified GPUs
  foreach(__arch ${__lwda_arch_bin})
    if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      list(APPEND __lwcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND __lwcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      list(APPEND __lwcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
      list(APPEND __lwcc_archs_readable sm_${__arch})
    endif()
  endforeach()

  # Tell LWCC to add PTX intermediate code for the specified architectures
  foreach(__arch ${__lwda_arch_ptx})
    list(APPEND __lwcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
    list(APPEND __lwcc_archs_readable compute_${__arch})
  endforeach()

  string(REPLACE ";" " " __lwcc_archs_readable "${__lwcc_archs_readable}")
  set(${out_variable}          ${__lwcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${__lwcc_archs_readable} PARENT_SCOPE)
endfunction()

################################################################################################
# Short command for lwca compilation
# Usage:
#   caffe_lwda_compile(<objlist_variable> <lwda_files>)
macro(caffe_lwda_compile objlist_variable)
  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var}_backup_in_lwda_compile_ "${${var}}")

    # we remove /EHa as it generates warnings under windows
    string(REPLACE "/EHa" "" ${var} "${${var}}")

  endforeach()

  lwda_compile(lwda_objcs ${ARGN})

  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var} "${${var}_backup_in_lwda_compile_}")
    unset(${var}_backup_in_lwda_compile_)
  endforeach()

  set(${objlist_variable} ${lwda_objcs})
endmacro()

################################################################################################
# Short command for lwDNN detection. Believe it soon will be a part of LWCA toolkit distribution.
# That's why not FindlwDNN.cmake file, but just the macro
# Usage:
#   detect_lwDNN()
function(detect_lwDNN)
  set(LWDNN_ROOT "" CACHE PATH "LWDNN root folder")

  find_path(LWDNN_INCLUDE lwdnn.h
            PATHS ${LWDNN_ROOT} $ELW{LWDNN_ROOT} ${LWDA_TOOLKIT_INCLUDE}
            DOC "Path to lwDNN include directory." )

  get_filename_component(__libpath_hist ${LWDA_LWDART_LIBRARY} PATH)
  find_library(LWDNN_LIBRARY NAMES liblwdnn.so # liblwdnn_static.a
                             PATHS ${LWDNN_ROOT} $ELW{LWDNN_ROOT} ${LWDNN_INCLUDE} ${__libpath_hist}
                             DOC "Path to lwDNN library.")

  if(LWDNN_INCLUDE AND LWDNN_LIBRARY)
    set(HAVE_LWDNN  TRUE PARENT_SCOPE)
    set(LWDNN_FOUND TRUE PARENT_SCOPE)

    file(READ ${LWDNN_INCLUDE}/lwdnn.h LWDNN_VERSION_FILE_CONTENTS)

    # lwDNN v3 and beyond
    string(REGEX MATCH "define LWDNN_MAJOR * +([0-9]+)"
           LWDNN_VERSION_MAJOR "${LWDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define LWDNN_MAJOR * +([0-9]+)" "\\1"
           LWDNN_VERSION_MAJOR "${LWDNN_VERSION_MAJOR}")
    string(REGEX MATCH "define LWDNN_MINOR * +([0-9]+)"
           LWDNN_VERSION_MINOR "${LWDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define LWDNN_MINOR * +([0-9]+)" "\\1"
           LWDNN_VERSION_MINOR "${LWDNN_VERSION_MINOR}")
    string(REGEX MATCH "define LWDNN_PATCHLEVEL * +([0-9]+)"
           LWDNN_VERSION_PATCH "${LWDNN_VERSION_FILE_CONTENTS}")
    string(REGEX REPLACE "define LWDNN_PATCHLEVEL * +([0-9]+)" "\\1"
           LWDNN_VERSION_PATCH "${LWDNN_VERSION_PATCH}")

    if(NOT LWDNN_VERSION_MAJOR)
      set(LWDNN_VERSION "???")
    else()
      set(LWDNN_VERSION "${LWDNN_VERSION_MAJOR}.${LWDNN_VERSION_MINOR}.${LWDNN_VERSION_PATCH}")
    endif()

    message(STATUS "Found lwDNN: ver. ${LWDNN_VERSION} found (include: ${LWDNN_INCLUDE}, library: ${LWDNN_LIBRARY})")

    string(COMPARE LESS "${LWDNN_VERSION_MAJOR}" 3 lwDNLWersionIncompatible)
    if(lwDNLWersionIncompatible)
      message(FATAL_ERROR "lwDNN version >3 is required.")
    endif()

    set(LWDNN_VERSION "${LWDNN_VERSION}" PARENT_SCOPE)
    mark_as_advanced(LWDNN_INCLUDE LWDNN_LIBRARY LWDNN_ROOT)

  endif()
endfunction()

################################################################################################
###  Non macro section
################################################################################################

find_package(LWCA 8.0 QUIET)
find_lwda_helper_libs(lwrand)  # cmake 2.8.7 compartibility which doesn't search for lwrand

if(NOT LWDA_FOUND)
  return()
endif()

set(HAVE_LWDA TRUE)
message(STATUS "LWCA detected: " ${LWDA_VERSION})

if(NOT LWDA_VERSION VERSION_LESS "10.0")
  STRING(REGEX REPLACE "LWDA_lwblas_device_LIBRARY-NOTFOUND" "" LWDA_LWBLAS_LIBRARIES ${LWDA_LWBLAS_LIBRARIES})
endif()

include_directories(SYSTEM ${LWDA_INCLUDE_DIRS})
list(APPEND Caffe_LINKER_LIBS ${LWDA_LWDART_LIBRARY}
                              ${LWDA_lwrand_LIBRARY} ${LWDA_LWBLAS_LIBRARIES})

# lwdnn detection
if(USE_LWDNN)
#  detect_lwDNN()

#  FIND_PACKAGE(LWDNN 6.0 EXACT)
  FIND_PACKAGE(LWDNN)

#  IF(NOT LWDNN_FOUND)
#    LWDNN_INSTALL(6.0-rc ${CMAKE_INSTALL_PREFIX})
#  ENDIF()

  if(LWDNN_FOUND)
   set(HAVE_LWDNN ${LWDNN_FOUND})
  endif()

  if(HAVE_LWDNN)
    add_definitions(-DUSE_LWDNN)
    include_directories(SYSTEM ${LWDNN_INCLUDE})
    list(APPEND Caffe_LINKER_LIBS ${LWDNN_LIBRARY})
  endif()
endif()

if(UNIX OR APPLE)
  list(APPEND LWDA_LWCC_FLAGS -std=c++11;-Xcompiler;-fPIC)
endif()

if(APPLE)
  list(APPEND LWDA_LWCC_FLAGS -Xcompiler;-Wno-unused-function)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  list(APPEND LWDA_LWCC_FLAGS -G -g)
endif()

SET(LWDA_PROPAGATE_HOST_FLAGS OFF)

# setting lwcc arch flags
caffe_select_lwcc_arch_flags(LWCC_FLAGS_EXTRA)
list(APPEND LWDA_LWCC_FLAGS ${LWCC_FLAGS_EXTRA})
message(STATUS "Added LWCA LWCC flags for: ${LWCC_FLAGS_EXTRA_readable}")

# Boost 1.55 workaround, see https://svn.boost.org/trac/boost/ticket/9392 or
# https://github.com/ComputationalRadiationPhysics/picongpu/blob/master/src/picongpu/CMakeLists.txt
if(Boost_VERSION EQUAL 105500)
  message(STATUS "Lwca + Boost 1.55: Applying noinline work around")
  # avoid warning for CMake >= 2.8.12
  set(LWDA_LWCC_FLAGS "${LWDA_LWCC_FLAGS} \"-DBOOST_NOINLINE=__attribute__((noinline))\" ")
endif()

# disable some lwcc diagnostic that apears in boost, glog, glags, opencv, etc.
foreach(diag cc_clobber_ignored integer_sign_change useless_using_declaration set_but_not_used)
  list(APPEND LWDA_LWCC_FLAGS -Xlwdafe --diag_suppress=${diag})
endforeach()

# setting default testing device
if(NOT LWDA_TEST_DEVICE)
  set(LWDA_TEST_DEVICE -1)
endif()

mark_as_advanced(LWDA_BUILD_LWBIN LWDA_BUILD_EMULATION LWDA_VERBOSE_BUILD)
mark_as_advanced(LWDA_SDK_ROOT_DIR LWDA_SEPARABLE_COMPILATION)

# Handle clang/libc++ issue
if(APPLE)
  caffe_detect_darwin_version(OSX_VERSION)

  # OSX 10.9 and higher uses clang/libc++ by default which is incompartible with old LWCA toolkits
  if(OSX_VERSION VERSION_GREATER 10.8)
    # enabled by default if and only if LWCA version is less than 7.0
    caffe_option(USE_libstdcpp "Use libstdc++ instead of libc++" (LWDA_VERSION VERSION_LESS 7.0))
  endif()
endif()
