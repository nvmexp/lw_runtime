# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.
#.rst:
# FindLWDNN
# -------
#
# Find LWDNN library
#
# Valiables that affect result:
# <VERSION>, <REQUIRED>, <QUIETLY>: as usual
#
# <EXACT> : as usual, plus we do find '5.1' version if you wanted '5' 
#           (not if you wanted '5.0', as usual)   
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``LWDNN_INCLUDE``
#   where to find lwdnn.h.
# ``LWDNN_LIBRARY``
#   the libraries to link against to use LWDNN.
# ``LWDNN_FOUND``
#   If false, do not try to use LWDNN.
# ``LWDNN_VERSION``
#   Version of the LWDNN library we looked for 
#
# Exported functions
# ^^^^^^^^^^^^^^^^
# function(LWDNN_INSTALL version dest_dir)
#  This function will try to download and install LWDNN.
#
#

function(LWDNN_INSTALL version dest_dir)
  string(REGEX REPLACE "-rc$" "" version_base "${version}")

  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
      set(__url_arch_name linux-x64 )
    elseif("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "ppc")
      set(__url_arch_name linux-ppc64le ) 
      #  TX1 has to be installed via JetPack
    endif()
  elseif  (APPLE)
    set(__url_arch_name osx-x64)
  elseif(WIN32)
    if(CMAKE_SYSTEM_VERSION MATCHES "10")
      set(__url_arch_name windows10)
    else()
      set(__url_arch_name windows7)
    endif()
  endif()
  
  # Download and install LWDNN locally if not found on the system
  if(__url_arch_name) 
    set(__download_dir ${CMAKE_LWRRENT_BINARY_DIR}/downloads)
    file(MAKE_DIRECTORY ${__download_dir})
    set(__lwdnn_filename lwdnn-${LWDA_VERSION}-${__url_arch_name}-v${version}.tgz)
    set(__base_url http://developer.download.lwpu.com/compute/redist/lwdnn)
    set(__lwdnn_url ${__base_url}/v${version_base}/${__lwdnn_filename})
    set(__lwdnn_tgz ${__download_dir}/${__lwdnn_filename})
    
    if(NOT EXISTS ${__lwdnn_tgz})
      message("Downloading LWDNN library from LWPU...")
      file(DOWNLOAD ${__lwdnn_url} ${__lwdnn_tgz}
	SHOW_PROGRESS STATUS LWDNN_STATUS
	)
      if("${LWDNN_STATUS}" MATCHES "0")
	exelwte_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${__lwdnn_tgz}" WORKING_DIRECTORY "${__download_dir}")
      else()
	message("Was not able to download LWDNN from ${__lwdnn_url}. Please install LwDNN manually from https://developer.lwpu.com/lwDNN")
	file(REMOVE ${__lwdnn_tgz})
      endif()
    endif()
    
    if(WIN32)
      file(GLOB __lwdnn_binfiles ${__download_dir}/lwca/bin*/*)
      install(FILES ${__lwdnn_binfiles} 
	DESTINATION  "${dest_dir}/bin")
    endif()
    
    file(GLOB __lwdnn_incfiles ${__download_dir}/lwca/include/*)
    install(FILES ${__lwdnn_incfiles} 
      DESTINATION  "${dest_dir}/include")

    file(GLOB __lwdnn_libfiles ${__download_dir}/lwca/lib*/*)
    install(FILES ${__lwdnn_libfiles} 
      DESTINATION  "${dest_dir}/lib")

  endif(__url_arch_name)
endfunction()

#####################################################

get_filename_component(__libpath_lwdart ${LWDA_LWDART_LIBRARY} PATH)
unset(LWDNN_LIBRARY CACHE)

find_path(LWDNN_INCLUDE lwdnn.h PATHS
    ${LWDNN_DIR}/lwca/include $ELW{LWDNN_DIR}/lwca/include 
    ${LWDNN_DIR}/include $ELW{LWDNN_DIR}/include 
    ${LWDNN_DIR} $ELW{LWDNN_DIR} 
    ${LWDNN_PATH} $ELW{LWDNN_PATH} ${LWDA_TOOLKIT_INCLUDE} ELW{CMAKE_INCLUDE_PATH}
  DOC "Path to LWDNN include directory." )
# We use major only in library search as major/minor is not entirely consistent among platforms.
# Also, looking for exact minor version of .so is in general not a good idea.
# More strict enforcement of minor/patch version is done if/when the header file is examined.
if(LWDNN_FIND_VERSION_EXACT)
  SET(__lwdnn_ver_suffix ".${LWDNN_FIND_VERSION_MAJOR}")
  SET(__lwdnn_lib_win_name lwdnn64_${LWDNN_FIND_VERSION_MAJOR}.dll)
  SET(LWDNN_MAJOR_VERSION ${LWDNN_FIND_MAJOR_VERSION})
else()
  SET(__lwdnn_lib_win_name lwdnn64.dll)
endif()

find_library(LWDNN_LIBRARY NAMES liblwdnn.so${__lwdnn_ver_suffix} liblwdnn${__lwdnn_ver_suffix}.dylib ${__lwdnn_lib_win_name}
  PATHS ${LWDNN_DIR}/lwca/lib64 $ELW{LWDNN_DIR}/lwca/lib64 ${LWDNN_DIR}/lib64 $ELW{LWDNN_DIR}/lib64 ${LWDNN_DIR} $ELW{LWDNN_DIR} 
  ${LWDNN_PATH} $ELW{LWDNN_PATH} $ELW{LD_LIBRARY_PATH} ${__libpath_lwdart}
  DOC "LWDNN library." )

mark_as_advanced(LWDNN_INCLUDE LWDNN_LIBRARY )

if(LWDNN_INCLUDE)
  file(READ ${LWDNN_INCLUDE}/lwdnn.h LWDNN_VERSION_FILE_CONTENTS)
  string(REGEX MATCH "define LWDNN_MAJOR * +([0-9]+)"
    LWDNN_MAJOR_VERSION "${LWDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define LWDNN_MAJOR * +([0-9]+)" "\\1"
    LWDNN_MAJOR_VERSION "${LWDNN_MAJOR_VERSION}")
  string(REGEX MATCH "define LWDNN_MINOR * +([0-9]+)"
    LWDNN_MINOR_VERSION "${LWDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define LWDNN_MINOR * +([0-9]+)" "\\1"
    LWDNN_MINOR_VERSION "${LWDNN_MINOR_VERSION}")
  string(REGEX MATCH "define LWDNN_PATCHLEVEL * +([0-9]+)"
    LWDNN_PATCH_VERSION "${LWDNN_VERSION_FILE_CONTENTS}")
  string(REGEX REPLACE "define LWDNN_PATCHLEVEL * +([0-9]+)" "\\1"
    LWDNN_PATCH_VERSION "${LWDNN_PATCH_VERSION}")  
endif()

if(NOT LWDNN_MAJOR_VERSION)
  set(LWDNN_VERSION "???")
else()
## Fixing the case where 5.1 does not fit 'exact' 5.
  set(LWDNN_VERSION ${LWDNN_MAJOR_VERSION}.${LWDNN_MINOR_VERSION})
  if(LWDNN_FIND_VERSION_EXACT AND "x${LWDNN_FIND_VERSION_MINOR}" STREQUAL "x")
    if(LWDNN_MAJOR_VERSION EQUAL LWDNN_FIND_VERSION_MAJOR)
      set(LWDNN_VERSION ${LWDNN_FIND_VERSION})
    endif()
  endif()
    math(EXPR LWDNN_VERSION_NUM "${LWDNN_MAJOR_VERSION} * 1000 + ${LWDNN_MINOR_VERSION} * 100 + ${LWDNN_PATCH_VERSION}")
endif()


  
find_package_handle_standard_args(LWDNN
                                  REQUIRED_VARS LWDNN_LIBRARY 
                                  VERSION_VAR   LWDNN_VERSION)


