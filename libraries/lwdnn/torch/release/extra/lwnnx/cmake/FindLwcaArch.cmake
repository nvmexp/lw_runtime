# Borrowed from caffe
# https://github.com/BVLC/caffe/blob/master/cmake/Lwca.cmake

# Known LWPU GPU achitectures Torch can be compiled for.
# This list will be used for LWDA_ARCH_NAME = All option
SET(KNOWN_GPU_ARCHITECTURES "2.0 2.1(2.0) 3.0 3.5 5.0")

IF (LWDA_VERSION VERSION_GREATER "6.5")
   SET(KNOWN_GPU_ARCHITECTURES "${KNOWN_GPU_ARCHITECTURES} 5.2")
ENDIF (LWDA_VERSION VERSION_GREATER "6.5")

################################################################################################
# Removes duplicates from LIST(s)
# Usage:
#   LIST_UNIQUE(<list_variable> [<list_variable>] [...])
MACRO(LIST_UNIQUE)
  FOREACH(__lst ${ARGN})
    IF(${__lst})
      LIST(REMOVE_DUPLICATES ${__lst})
    ENDIF()
  ENDFOREACH()
ENDMACRO()

################################################################################################
# A function for automatic detection of GPUs installed  (IF autodetection is enabled)
# Usage:
#   DETECT_INSTALLED_GPUS(OUT_VARIABLE)
FUNCTION(DETECT_INSTALLED_GPUS OUT_VARIABLE)
  IF(NOT LWDA_GPU_DETECT_OUTPUT)
    SET(__lwfile ${PROJECT_BINARY_DIR}/detect_lwda_archs.lw)

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

    EXELWTE_PROCESS(COMMAND "${LWDA_LWCC_EXELWTABLE}" "--run" "${__lwfile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __lwcc_res OUTPUT_VARIABLE __lwcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    IF(__lwcc_res EQUAL 0)
      STRING(REPLACE "2.1" "2.1(2.0)" __lwcc_out "${__lwcc_out}")
      SET(LWDA_GPU_DETECT_OUTPUT ${__lwcc_out} CACHE INTERNAL "Returned GPU architetures from detect_gpus tool" FORCE)
    ENDIF()
  ENDIF()

  IF(NOT LWDA_GPU_DETECT_OUTPUT)
    message(STATUS "Automatic GPU detection failed. Building for all known architectures.")
    SET(${OUT_VARIABLE} ${KNOWN_GPU_ARCHITECTURES} PARENT_SCOPE)
  ELSE()
    SET(${OUT_VARIABLE} ${LWDA_GPU_DETECT_OUTPUT} PARENT_SCOPE)
  ENDIF()
ENDFUNCTION()


################################################################################################
# Function for selecting GPU arch flags for lwcc based on LWDA_ARCH_NAME
# Usage:
#   SELECT_LWCC_ARCH_FLAGS(out_variable)
FUNCTION(SELECT_LWCC_ARCH_FLAGS out_variable)
  # List of arch names
  SET(__archs_names "Fermi" "Kepler" "Maxwell" "All" "Manual")
  SET(__archs_name_default "All")
  IF(NOT CMAKE_CROSSCOMPILING)
    LIST(APPEND __archs_names "Auto")
    SET(__archs_name_default "Auto")
  ENDIF()

  # SET LWDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  SET(LWDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target LWPU GPU achitecture.")
  SET_property( CACHE LWDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
  mark_as_advanced(LWDA_ARCH_NAME)

  # verIFy LWDA_ARCH_NAME value
  IF(NOT ";${__archs_names};" MATCHES ";${LWDA_ARCH_NAME};")
    STRING(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Only ${__archs_names} architeture names are supported.")
  ENDIF()

  IF(${LWDA_ARCH_NAME} STREQUAL "Manual")
    SET(LWDA_ARCH_BIN ${KNOWN_GPU_ARCHITECTURES} CACHE STRING "SpecIFy 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
    SET(LWDA_ARCH_PTX "50"                     CACHE STRING "SpecIFy 'virtual' PTX architectures to build PTX intermediate code for")
    mark_as_advanced(LWDA_ARCH_BIN LWDA_ARCH_PTX)
  else()
    unSET(LWDA_ARCH_BIN CACHE)
    unSET(LWDA_ARCH_PTX CACHE)
  ENDIF()

  IF(${LWDA_ARCH_NAME} STREQUAL "Fermi")
    SET(__lwda_arch_bin "2.0 2.1(2.0)")
  elseIF(${LWDA_ARCH_NAME} STREQUAL "Kepler")
    SET(__lwda_arch_bin "3.0 3.5")
  elseIF(${LWDA_ARCH_NAME} STREQUAL "Maxwell")
    SET(__lwda_arch_bin "5.0 5.2")
  elseIF(${LWDA_ARCH_NAME} STREQUAL "All")
    SET(__lwda_arch_bin ${KNOWN_GPU_ARCHITECTURES})
  elseIF(${LWDA_ARCH_NAME} STREQUAL "Auto")
    DETECT_INSTALLED_GPUS(__lwda_arch_bin)
  else()  # (${LWDA_ARCH_NAME} STREQUAL "Manual")
    SET(__lwda_arch_bin ${LWDA_ARCH_BIN})
  ENDIF()

  MESSAGE(STATUS "Compiling for LWCA architecture: ${__lwda_arch_bin}")

  # remove dots and colwert to lists
  STRING(REGEX REPLACE "\\." "" __lwda_arch_bin "${__lwda_arch_bin}")
  STRING(REGEX REPLACE "\\." "" __lwda_arch_ptx "${LWDA_ARCH_PTX}")
  STRING(REGEX MATCHALL "[0-9()]+" __lwda_arch_bin "${__lwda_arch_bin}")
  STRING(REGEX MATCHALL "[0-9]+"   __lwda_arch_ptx "${__lwda_arch_ptx}")
  LIST_UNIQUE(__lwda_arch_bin __lwda_arch_ptx)

  SET(__lwcc_flags "")
  SET(__lwcc_archs_readable "")

  # Tell LWCC to add binaries for the specIFied GPUs
  FOREACH(__arch ${__lwda_arch_bin})
    IF(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specIFied PTX for the concrete BIN
      LIST(APPEND __lwcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      LIST(APPEND __lwcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specIFy PTX for the concrete BIN, we assume PTX=BIN
      LIST(APPEND __lwcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
      LIST(APPEND __lwcc_archs_readable sm_${__arch})
    ENDIF()
  ENDFOREACH()

  # Tell LWCC to add PTX intermediate code for the specIFied architectures
  FOREACH(__arch ${__lwda_arch_ptx})
    LIST(APPEND __lwcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
    LIST(APPEND __lwcc_archs_readable compute_${__arch})
  ENDFOREACH()

  STRING(REPLACE ";" " " __lwcc_archs_readable "${__lwcc_archs_readable}")
  SET(${out_variable}          ${__lwcc_flags}          PARENT_SCOPE)
  SET(${out_variable}_readable ${__lwcc_archs_readable} PARENT_SCOPE)
ENDFUNCTION()
