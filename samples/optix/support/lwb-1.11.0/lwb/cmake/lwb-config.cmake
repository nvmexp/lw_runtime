#
# find_package(LWB) config file.
#
# Defines a LWB::LWB target that may be linked from user projects to include
# LWB.

if (TARGET LWB::LWB)
  return()
endif()

function(_lwb_declare_interface_alias alias_name ugly_name)
  # 1) Only IMPORTED and ALIAS targets can be placed in a namespace.
  # 2) When an IMPORTED library is linked to another target, its include
  #    directories are treated as SYSTEM includes.
  # 3) lwcc will automatically check the LWCA Toolkit include path *before* the
  #    system includes. This means that the Toolkit LWB will *always* be used
  #    during compilation, and the include paths of an IMPORTED LWB::LWB
  #    target will never have any effect.
  # 4) This behavior can be fixed by setting the property NO_SYSTEM_FROM_IMPORTED
  #    on EVERY target that links to LWB::LWB. This would be a burden and a
  #    footgun for our users. Forgetting this would silently pull in the wrong LWB!
  # 5) A workaround is to make a non-IMPORTED library outside of the namespace,
  #    configure it, and then ALIAS it into the namespace (or ALIAS and then
  #    configure, that seems to work too).
  add_library(${ugly_name} INTERFACE)
  add_library(${alias_name} ALIAS ${ugly_name})
endfunction()

#
# Setup targets
#

_lwb_declare_interface_alias(LWB::LWB _LWB_LWB)
# Pull in the include dir detected by lwb-config-version.cmake
set(_LWB_INCLUDE_DIR "${_LWB_VERSION_INCLUDE_DIR}"
  CACHE INTERNAL "Location of LWB headers."
)
unset(_LWB_VERSION_INCLUDE_DIR CACHE) # Clear tmp variable from cache
target_include_directories(_LWB_LWB INTERFACE "${_LWB_INCLUDE_DIR}")

if (LWB_IGNORE_DEPRECATED_CPP_DIALECT OR
    THRUST_IGNORE_DEPRECATED_CPP_DIALECT)
  target_compile_definitions(_LWB_LWB INTERFACE "LWB_IGNORE_DEPRECATED_CPP_DIALECT")
endif()

if (LWB_IGNORE_DEPRECATED_CPP_11 OR
    THRUST_IGNORE_DEPRECATED_CPP_11)
  target_compile_definitions(_LWB_LWB INTERFACE "LWB_IGNORE_DEPRECATED_CPP_11")
endif()

if (LWB_IGNORE_DEPRECATED_COMPILER OR
    THRUST_IGNORE_DEPRECATED_COMPILER)
  target_compile_definitions(_LWB_LWB INTERFACE "LWB_IGNORE_DEPRECATED_COMPILER")
endif()

#
# Standardize version info
#

set(LWB_VERSION ${${CMAKE_FIND_PACKAGE_NAME}_VERSION} CACHE INTERNAL "")
set(LWB_VERSION_MAJOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MAJOR} CACHE INTERNAL "")
set(LWB_VERSION_MINOR ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_MINOR} CACHE INTERNAL "")
set(LWB_VERSION_PATCH ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_PATCH} CACHE INTERNAL "")
set(LWB_VERSION_TWEAK ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_TWEAK} CACHE INTERNAL "")
set(LWB_VERSION_COUNT ${${CMAKE_FIND_PACKAGE_NAME}_VERSION_COUNT} CACHE INTERNAL "")

include(FindPackageHandleStandardArgs)
if (NOT LWB_CONFIG)
  set(LWB_CONFIG "${CMAKE_LWRRENT_LIST_FILE}")
endif()
find_package_handle_standard_args(LWB CONFIG_MODE)
