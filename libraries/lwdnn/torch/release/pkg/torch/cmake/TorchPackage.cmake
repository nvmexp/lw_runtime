# -*- cmake -*-

MACRO(ADD_TORCH_LIBRARY package type src)
  IF ("${type}" STREQUAL "STATIC")
    if ("${src}" MATCHES "lw$" OR "${src}" MATCHES "lw;")
      LWDA_ADD_LIBRARY(${package} STATIC ${src})
    else()
      ADD_LIBRARY(${package} STATIC ${src})
    endif()
  ELSE()
    if ("${src}" MATCHES "lw$" OR "${src}" MATCHES "lw;")
      LWDA_ADD_LIBRARY(${package} ${type} ${src})
    else()
      ADD_LIBRARY(${package} ${type} ${src})
    endif()
  ENDIF()
ENDMACRO()

MACRO(ADD_TORCH_PACKAGE package src luasrc)
  INCLUDE_DIRECTORIES(${CMAKE_LWRRENT_SOURCE_DIR})
  INCLUDE_DIRECTORIES(${Torch_LUA_INCLUDE_DIR})

 ### C/C++ sources
 # As per CMake doc, macro arguments are not variables, so simple test syntax not working
  IF(NOT "${src}" STREQUAL "")

    ADD_TORCH_LIBRARY(${package} MODULE "${src}")

    ### Torch packages supposes libraries prefix is "lib"
    SET_TARGET_PROPERTIES(${package} PROPERTIES
      PREFIX "lib"
      IMPORT_PREFIX "lib"
      INSTALL_NAME_DIR "@exelwtable_path/${Torch_INSTALL_BIN2CPATH}")

    IF(APPLE)
      SET_TARGET_PROPERTIES(${package} PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup")
    ENDIF()

    IF (BUILD_STATIC OR "$ELW{STATIC_TH}" STREQUAL "YES")
      ADD_TORCH_LIBRARY(${package}_static STATIC "${src}")
      SET_TARGET_PROPERTIES(${package}_static PROPERTIES
        COMPILE_FLAGS "-fPIC")
      SET_TARGET_PROPERTIES(${package}_static PROPERTIES
        PREFIX "lib" IMPORT_PREFIX "lib" OUTPUT_NAME "${package}")
    ENDIF()

    INSTALL(TARGETS ${package}
      RUNTIME DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR}
      LIBRARY DESTINATION ${Torch_INSTALL_LUA_CPATH_SUBDIR})

  ENDIF(NOT "${src}" STREQUAL "")

  ### lua sources
  IF(NOT "${luasrc}" STREQUAL "")
    INSTALL(FILES ${luasrc}
      DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${package})
  ENDIF(NOT "${luasrc}" STREQUAL "")

ENDMACRO(ADD_TORCH_PACKAGE)
