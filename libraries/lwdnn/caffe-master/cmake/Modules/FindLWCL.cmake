# Find the LWCL libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR:    Base directory where all LWCL components are found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIR
#  NCCL_LIBRARY

find_path(NCCL_INCLUDE_DIR NAMES lwcl.h
    PATHS ${NCCL_ROOT_DIR}/include
    )

find_library(NCCL_LIBRARY NAMES lwcl
    PATHS ${NCCL_ROOT_DIR}/lib ${NCCL_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LWCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if(NCCL_FOUND)
  message(STATUS "Found LWCL (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARY})")
  mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARY)
  caffe_parse_header(${NCCL_INCLUDE_DIR}/lwcl.h
          NCCL_VERION_LINES NCCL_MAJOR NCCL_MINOR NCCL_PATCH)
  set(NCCL_VERSION "${NCCL_MAJOR}.${NCCL_MINOR}.${NCCL_PATCH}")

endif()

