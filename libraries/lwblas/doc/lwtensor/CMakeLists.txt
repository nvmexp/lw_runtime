#set(LWTENSOR_DOCS_TAG "internal") Internal release
set(LWTENSOR_DOCS_TAG "public")

add_lwstom_target(lwtensor_doc
    COMMAND ${DOXYGEN} 
    COMMAND ${CMAKE_MAKE_PROGRAM} html
            SPHINXOPTS=\"-t ${LWTENSOR_DOCS_TAG} -Q -Dversion=${LWTENSOR_PROJECT_VERSION} -Drelease=${LWTENSOR_PROJECT_VERSION}\" 
    SOURCES Doxyfile conf.py index.rst
    WORKING_DIRECTORY ${CMAKE_LWRRENT_SOURCE_DIR}
)

