# Find the LWML library
#
# The following variables are optionally searched for defaults
#  LWML_ROOT_DIR:    Base directory where all LWML components are found
#
# The following are set after configuration is done:
#  LWML_FOUND
#  LWML_INCLUDE_DIR
#  LWML_LIBRARY

file (GLOB MLPATH /usr/lib/lwpu-???)
find_path(LWML_INCLUDE_DIR NAMES lwml.h PATHS  ${LWDA_INCLUDE_DIRS} ${LWML_ROOT_DIR}/include)
find_library(LWML_LIBRARY lwpu-ml PATHS ${MLPATH} /usr/local/lwca/lib64/stubs ${LWML_ROOT_DIR}/lib ${LWML_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LWML DEFAULT_MSG LWML_INCLUDE_DIR LWML_LIBRARY)

if(LWML_FOUND)
  message(STATUS "Found LWML (include: ${LWML_INCLUDE_DIR}, library: ${LWML_LIBRARY})")
  mark_as_advanced(LWML_INCLUDE_DIR LWML_LIBRARY)
endif()

