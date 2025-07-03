find_program(DOXYGEN doxygen)
find_program(SPHINX sphinx-build)

if((NOT DOXYGEN) OR (NOT SPHINX))
    message(STATUS "Target `doc` is disabled (doxygen, sphinx are required)")
else()
    set_directory_properties(PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES "lwtensor/_build")
    add_subdirectory(lwtensor)
    add_lwstom_target(doc DEPENDS lwtensor_doc)
    add_lwstom_command(TARGET doc
       POST_BUILD
       COMMAND ${Python_EXELWTABLE}
       ARGS ${CMAKE_LWRRENT_SOURCE_DIR}/../misc/release.py ${CMAKE_LWRRENT_SOURCE_DIR}/..  ${PROJECT_BINARY_DIR}/../doc/lwtensor/
       BYPRODUCTS ${PROJECT_BINARY_DIR}../doc/lwtensor/types.h ${PROJECT_BINARY_DIR}../doc/lwtensor/lwtensor.h
       USES_TERMINAL VERBATIM
       COMMENT "Generating release-version header files for docs"
       )
endif()


