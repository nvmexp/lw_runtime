add_library(glslang-default-resource-limits
            ${CMAKE_LWRRENT_SOURCE_DIR}/ResourceLimits.cpp
            ${CMAKE_LWRRENT_SOURCE_DIR}/resource_limits_c.cpp)
set_property(TARGET glslang-default-resource-limits PROPERTY FOLDER glslang)
set_property(TARGET glslang-default-resource-limits PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(glslang-default-resource-limits
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}>
                           PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)


set(SOURCES StandAlone.cpp DirStackFileIncluder.h)

add_exelwtable(glslangValidator ${SOURCES})
set_property(TARGET glslangValidator PROPERTY FOLDER tools)
glslang_set_link_args(glslangValidator)

set(LIBRARIES
    glslang
    SPIRV
    glslang-default-resource-limits)

if(ENABLE_SPVREMAPPER)
    set(LIBRARIES ${LIBRARIES} SPVRemapper)
endif()

if(WIN32)
    set(LIBRARIES ${LIBRARIES} psapi)
elseif(UNIX)
    if(NOT ANDROID)
        set(LIBRARIES ${LIBRARIES} pthread)
    endif()
endif(WIN32)

target_link_libraries(glslangValidator ${LIBRARIES})
target_include_directories(glslangValidator PUBLIC
    $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}/../External>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/External>)

if(ENABLE_SPVREMAPPER)
    set(REMAPPER_SOURCES spirv-remap.cpp)
    add_exelwtable(spirv-remap ${REMAPPER_SOURCES})
    set_property(TARGET spirv-remap PROPERTY FOLDER tools)
    glslang_set_link_args(spirv-remap)
    target_link_libraries(spirv-remap ${LIBRARIES})
endif()

if(WIN32)
    source_group("Source" FILES ${SOURCES})
endif(WIN32)

if(ENABLE_GLSLANG_INSTALL)
    install(TARGETS glslangValidator EXPORT glslangValidatorTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(EXPORT glslangValidatorTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)

    if(ENABLE_SPVREMAPPER)
        install(TARGETS spirv-remap EXPORT spirv-remapTargets
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
        install(EXPORT spirv-remapTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
    endif()

    if(BUILD_SHARED_LIBS)
        install(TARGETS glslang-default-resource-limits EXPORT glslang-default-resource-limitsTargets
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
        install(EXPORT glslang-default-resource-limitsTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake)
    endif()
endif(ENABLE_GLSLANG_INSTALL)
