CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)

INCLUDE_DIRECTORIES(${CMAKE_LWRRENT_SOURCE_DIR}/src)
SET(src 
	${CMAKE_LWRRENT_SOURCE_DIR}/src/autolink.c 
	${CMAKE_LWRRENT_SOURCE_DIR}/src/buffer.c 
	${CMAKE_LWRRENT_SOURCE_DIR}/src/markdown.c 
	${CMAKE_LWRRENT_SOURCE_DIR}/src/stack.c 
	${CMAKE_LWRRENT_SOURCE_DIR}/html/houdini_href_e.c 
	${CMAKE_LWRRENT_SOURCE_DIR}/html/houdini_html_e.c 
	${CMAKE_LWRRENT_SOURCE_DIR}/html/html.c 
	${CMAKE_LWRRENT_SOURCE_DIR}/html/html_smartypants.c)
SET(luasrc 
	   ${CMAKE_LWRRENT_SOURCE_DIR}/elw.lua 
	   ${CMAKE_LWRRENT_SOURCE_DIR}/init.lua 
	   ${CMAKE_LWRRENT_SOURCE_DIR}/sdcdefs.lua 
	   ${CMAKE_LWRRENT_SOURCE_DIR}/htmlcdefs.lua 
	   ${CMAKE_LWRRENT_SOURCE_DIR}/html.lua 
	   ${CMAKE_LWRRENT_SOURCE_DIR}/ascii.lua)

ADD_LIBRARY("sundown" MODULE ${src})
SET_TARGET_PROPERTIES("sundown" PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

IF(APPLE)
  SET_TARGET_PROPERTIES("sundown" PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
ENDIF()

INSTALL(FILES ${luasrc} DESTINATION ${LUADIR}/sundown)
INSTALL(TARGETS sundown
  LIBRARY DESTINATION ${LIBDIR}
  RUNTIME DESTINATION ${LIBDIR})

IF(LUALIB)
  TARGET_LINK_LIBRARIES(sundown ${LUALIB})
ENDIF()
