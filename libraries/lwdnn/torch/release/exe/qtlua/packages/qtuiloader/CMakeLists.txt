# -*- cmake -*-

SET(QT_USE_QTXML 1)
SET(QT_USE_QTUITOOLS 1)
INCLUDE(${QT_USE_FILE})		     
INCLUDE_DIRECTORIES("${CMAKE_LWRRENT_SOURCE_DIR}" "${CMAKE_LWRRENT_BINARY_DIR}")

SET(qtuiloader_SRC uiloader.h uiloader.cpp)

MACRO_INSTALL_QTLUA_FILES(qtuiloader init.lua test.lua test.ui)

MACRO_ADD_QTLUA_MODULE(libqtuiloader ${qtuiloader_SRC})

#ADD_TORCH_DOK(dok qtuiloader "QT Interface" "QT Designer UI files loader" 6.)
