# -*- cmake -*-

INCLUDE_DIRECTORIES("${CMAKE_LWRRENT_SOURCE_DIR}" "${CMAKE_LWRRENT_BINARY_DIR}")
INCLUDE(${QT_USE_FILE})		     

SET(qtwidget_SRC 
  qtluapainter.h qtluapainter.cpp
  qtluaprinter.h qtluaprinter.cpp
  qtlualistener.h qtlualistener.cpp
  qtwidget.h qtwidget.cpp )

MACRO_INSTALL_QTLUA_FILES(qtwidget init.lua)

MACRO_QT4_AUTOGEN(qtwidget_GEN ${qtwidget_SRC})
MACRO_ADD_QTLUA_MODULE(libqtwidget ${qtwidget_SRC} ${qtwidget_GEN})

#ADD_TORCH_DOK(dok qtwidget "QT Interface" "Widgets and events" 6.)

