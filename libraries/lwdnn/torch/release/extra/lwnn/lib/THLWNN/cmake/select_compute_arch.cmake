# Synopsis:
#   LWDA_SELECT_LWCC_ARCH_FLAGS(out_variable [target_LWDA_architectures])
#   -- Selects GPU arch flags for lwcc based on target_LWDA_architectures
#      target_LWDA_architectures : Auto | Common | All | LIST(ARCH_AND_PTX ...)
#       - "Auto" detects local machine GPU compute arch at runtime.
#       - "Common" and "All" cover common and entire subsets of architectures
#      ARCH_AND_PTX : NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
#      NAME: Fermi Kepler Maxwell Kepler+CheetAh Kepler+Tesla Maxwell+CheetAh Pascal
#      NUM: Any number. Only those pairs are lwrrently accepted by LWCC though:
#            2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2
#      Returns LIST of flags to be added to LWDA_LWCC_FLAGS in ${out_variable}
#      Additionally, sets ${out_variable}_readable to the resulting numeric list
#      Example:
#       LWDA_SELECT_LWCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.2(5.0) Maxwell)
#        LIST(APPEND LWDA_LWCC_FLAGS ${ARCH_FLAGS})
#
#      More info on LWCA architectures: https://en.wikipedia.org/wiki/LWCA
#

# This list will be used for LWDA_ARCH_NAME = All option
set(LWDA_KNOWN_GPU_ARCHITECTURES  "Fermi" "Kepler" "Maxwell")

# This list will be used for LWDA_ARCH_NAME = Common option (enabled by default)
set(LWDA_COMMON_GPU_ARCHITECTURES "3.0" "3.5" "5.0")

if (LWDA_VERSION VERSION_GREATER "6.5")
  list(APPEND LWDA_KNOWN_GPU_ARCHITECTURES "Kepler+CheetAh" "Kepler+Tesla" "Maxwell+CheetAh")
  list(APPEND LWDA_COMMON_GPU_ARCHITECTURES "5.2")
endif ()

if (LWDA_VERSION VERSION_GREATER "7.5")
  list(APPEND LWDA_KNOWN_GPU_ARCHITECTURES "Pascal")
  list(APPEND LWDA_COMMON_GPU_ARCHITECTURES "6.0" "6.1" "6.1+PTX")
else()
  list(APPEND LWDA_COMMON_GPU_ARCHITECTURES "5.2+PTX")
endif ()



################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   LWDA_DETECT_INSTALLED_GPUS(OUT_VARIABLE)
#
function(LWDA_DETECT_INSTALLED_GPUS OUT_VARIABLE)
  if(NOT LWDA_GPU_DETECT_OUTPUT)
    set(lwfile ${PROJECT_BINARY_DIR}/detect_lwda_archs.lw)

    file(WRITE ${lwfile} ""
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

    exelwte_process(COMMAND "${LWDA_LWCC_EXELWTABLE}" "--run" "${lwfile}"
                    "-ccbin" ${CMAKE_CXX_COMPILER}
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE lwcc_res OUTPUT_VARIABLE lwcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(lwcc_res EQUAL 0)
      # only keep the last line of lwcc_out
      string(REGEX REPLACE ";" "\\\\;" lwcc_out "${lwcc_out}")
      string(REGEX REPLACE "\n" ";" lwcc_out "${lwcc_out}")
      list(GET lwcc_out -1 lwcc_out)
      string(REPLACE "2.1" "2.1(2.0)" lwcc_out "${lwcc_out}")
      set(LWDA_GPU_DETECT_OUTPUT ${lwcc_out} CACHE INTERNAL "Returned GPU architetures from detect_gpus tool" FORCE)
    endif()
  endif()

  if(NOT LWDA_GPU_DETECT_OUTPUT)
    message(STATUS "Automatic GPU detection failed. Building for common architectures.")
    set(${OUT_VARIABLE} ${LWDA_COMMON_GPU_ARCHITECTURES} PARENT_SCOPE)
  else()
    set(${OUT_VARIABLE} ${LWDA_GPU_DETECT_OUTPUT} PARENT_SCOPE)
  endif()
endfunction()


################################################################################################
# Function for selecting GPU arch flags for lwcc based on LWCA architectures from parameter list
# Usage:
#   SELECT_LWCC_ARCH_FLAGS(out_variable [list of LWCA compute archs])
function(LWDA_SELECT_LWCC_ARCH_FLAGS out_variable)
  set(LWDA_ARCH_LIST "${ARGN}")

  if("X${LWDA_ARCH_LIST}" STREQUAL "X" )
    set(LWDA_ARCH_LIST "Auto")
  endif()

  set(lwda_arch_bin)
  set(lwda_arch_ptx)

  if("${LWDA_ARCH_LIST}" STREQUAL "All")
    set(LWDA_ARCH_LIST ${LWDA_KNOWN_GPU_ARCHITECTURES})
  elseif("${LWDA_ARCH_LIST}" STREQUAL "Common")
    set(LWDA_ARCH_LIST ${LWDA_COMMON_GPU_ARCHITECTURES})
  elseif("${LWDA_ARCH_LIST}" STREQUAL "Auto")
    LWDA_DETECT_INSTALLED_GPUS(LWDA_ARCH_LIST)
    message(STATUS "Autodetected LWCA architecture(s): ${LWDA_ARCH_LIST}")
  endif()

  # Now process the list and look for names
  string(REGEX REPLACE "[ \t]+" ";" LWDA_ARCH_LIST "${LWDA_ARCH_LIST}")
  list(REMOVE_DUPLICATES LWDA_ARCH_LIST)
  foreach(arch_name ${LWDA_ARCH_LIST})
    set(arch_bin)
    set(add_ptx FALSE)
    # Check to see if we are compiling PTX
    if(arch_name MATCHES "(.*)\\+PTX$")
      set(add_ptx TRUE)
      set(arch_name ${CMAKE_MATCH_1})
    endif()
    if(arch_name MATCHES "(^[0-9]\\.[0-9](\\([0-9]\\.[0-9]\\))?)$")
      set(arch_bin ${CMAKE_MATCH_1})
      set(arch_ptx ${arch_bin})
    else()
      # Look for it in our list of known architectures
      if(${arch_name} STREQUAL "Fermi")
        set(arch_bin "2.0 2.1(2.0)")
      elseif(${arch_name} STREQUAL "Kepler+CheetAh")
        set(arch_bin 3.2)
      elseif(${arch_name} STREQUAL "Kepler+Tesla")
        set(arch_bin 3.7)
      elseif(${arch_name} STREQUAL "Kepler")
        set(arch_bin 3.0 3.5)
        set(arch_ptx 3.5)
      elseif(${arch_name} STREQUAL "Maxwell+CheetAh")
        set(arch_bin 5.3)
      elseif(${arch_name} STREQUAL "Maxwell")
        set(arch_bin 5.0 5.2)
        set(arch_ptx 5.2)
      elseif(${arch_name} STREQUAL "Pascal")
        set(arch_bin 6.0 6.1)
        set(arch_ptx 6.1)
      else()
        message(SEND_ERROR "Unknown LWCA Architecture Name ${arch_name} in LWDA_SELECT_LWCC_ARCH_FLAGS")
      endif()
    endif()
    if(NOT arch_bin)
      message(SEND_ERROR "arch_bin wasn't set for some reason")
    endif()
    list(APPEND lwda_arch_bin ${arch_bin})
    if(add_ptx)
      if (NOT arch_ptx)
        set(arch_ptx ${arch_bin})
      endif()
      list(APPEND lwda_arch_ptx ${arch_ptx})
    endif()
  endforeach()

  # remove dots and colwert to lists
  string(REGEX REPLACE "\\." "" lwda_arch_bin "${lwda_arch_bin}")
  string(REGEX REPLACE "\\." "" lwda_arch_ptx "${lwda_arch_ptx}")
  string(REGEX MATCHALL "[0-9()]+" lwda_arch_bin "${lwda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   lwda_arch_ptx "${lwda_arch_ptx}")

  if(lwda_arch_bin)
    list(REMOVE_DUPLICATES lwda_arch_bin)
  endif()
  if(lwda_arch_ptx)
    list(REMOVE_DUPLICATES lwda_arch_ptx)
  endif()

  set(lwcc_flags "")
  set(lwcc_archs_readable "")

  # Tell LWCC to add binaries for the specified GPUs
  foreach(arch ${lwda_arch_bin})
    if(arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified ARCH for the concrete CODE
      list(APPEND lwcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND lwcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify ARCH for the concrete CODE, we assume ARCH=CODE
      list(APPEND lwcc_flags -gencode arch=compute_${arch},code=sm_${arch})
      list(APPEND lwcc_archs_readable sm_${arch})
    endif()
  endforeach()

  # Tell LWCC to add PTX intermediate code for the specified architectures
  foreach(arch ${lwda_arch_ptx})
    list(APPEND lwcc_flags -gencode arch=compute_${arch},code=compute_${arch})
    list(APPEND lwcc_archs_readable compute_${arch})
  endforeach()

  string(REPLACE ";" " " lwcc_archs_readable "${lwcc_archs_readable}")
  set(${out_variable}          ${lwcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${lwcc_archs_readable} PARENT_SCOPE)
endfunction()
