
#
#  Copyright (c) 2008 - 2010 LWPU Corporation.  All rights reserved.
#
#  LWPU Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from LWPU Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

# Colwert binary files to const char * strings and add them to
# ${output}.cpp. This also generates a symbol table.
# const char *const[] const ${output}_symbols with pairs of
# source, symbolName. The symbol table is being terminated by
# two zeros.

# This script either provides the bin2cpp function or
# accepts the following variable to be passed in like
# -DVAR:TYPE=VALUE
#
# LWDA_BIN2C_EXELWTABLE
# CPP_FILE
# H_FILE
# CPP_SYMBOL
# SOURCE_BASE
# SOURCES

function(filename_to_variable var filename)
  string(REGEX REPLACE "\\.|/|-" "_" ${var} ${filename})
  set(${var} ${${var}} PARENT_SCOPE)
endfunction()

function(bin2h outputInclude exportSymbol)
  set(temp_file "${outputInclude}.tmp")
  file(WRITE "${temp_file}"
    "//Generated with bin2cpp\n"
    "#pragma once\n"
    "#include <stddef.h>\n"
    "#ifdef __cplusplus\n"
    "extern \"C\" {\n"
    "#endif\n"
    )

  foreach(source ${ARGN})
    # message(STATUS "embedding file ${source}")

    if (IS_ABSOLUTE "${source}")
      message(FATAL_ERROR "Source input is absolute path.  This will generate variable names with the path in it, which isn't want you probably want.\n\tSOURCE = ${source}\n")
    endif()

    filename_to_variable(symbol "${source}")
    file(APPEND "${temp_file}" "  extern const char *${symbol};\n")
  endforeach()

  file(APPEND "${temp_file}"
    "  extern const char *${exportSymbol}[];\n"
    "  extern const size_t *${exportSymbol}_sizes[];\n"
    "#ifdef __cplusplus\n"
    "}\n"
    "#endif\n"
    )
  exelwte_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different "${temp_file}" "${outputInclude}")
  exelwte_process(COMMAND ${CMAKE_COMMAND} -E remove "${temp_file}")
endfunction()


function(bin2cpp outputSource exportSymbol sourceBase sources)
  file(WRITE "${outputSource}" "//Generated with bin2cpp\n"
    "#include <stddef.h>\n"
    )

  foreach(source ${sources})
    message(STATUS "embedding file ${source}")

    if (IS_ABSOLUTE "${source}")
      message(FATAL_ERROR "Source input is absolute path.  This will generate variable names with the path in it, which isn't want you probably want.\n\tSOURCE = ${source}\n")
    endif()

    filename_to_variable(symbol "${source}")
    
    exelwte_process( COMMAND ${LWDA_BIN2C_EXELWTABLE} -p 0 -c -n ${symbol}_uchar "${sourceBase}/${source}"
      OUTPUT_VARIABLE bindata
      RESULT_VARIABLE result
      ERROR_VARIABLE error
      )
    if(result)
      exelwte_process(COMMAND ${CMAKE_COMMAND} -E remove -f "${outputSource}")
      message("bin2c out:\n" ${bindata})
      message(FATAL_ERROR "bin2c error when running " ${LWDA_BIN2C_EXELWTABLE} -p 0 -c -n ${symbol}_uchar "${sourceBase}/${source}" " :\n" ${result} ${error})
    endif()
    file(APPEND "${outputSource}" "${bindata}\n")

    set(castList "${castList}  const char *${symbol} = (const char *)${symbol}_uchar;\n")

    if(symbolTable)
      set(symbolTable "${symbolTable}\n   ,\"${source}\", ${symbol}")
    else()
      set(symbolTable "\"${source}\", ${symbol}")
    endif()

    if(symbolTable_sizes)
      set(symbolTable_sizes "${symbolTable_sizes}\n   ,sizeof(${symbol}_uchar) - 1")
    else()
      set(symbolTable_sizes "sizeof(${symbol}_uchar) - 1")
    endif()

  endforeach()
  file(APPEND "${outputSource}"
    "#ifdef __cplusplus\n"
    "extern \"C\" {\n"
    "#endif\n"
    "  // Cast the unsigned char arrays to const char* to match header file\n"
    "${castList}\n"
    "  const char *${exportSymbol}[] =\n"
    "  { ${symbolTable}\n"
    "   ,0,0\n"
    "  };\n"
    "\n"
    "  // it's not possible to export const size_t symbol[], so export const size_t *symbol[]\n"
    "  const size_t ${exportSymbol}_sizes_tmp[] =\n"
    "  { ${symbolTable_sizes}, static_cast<const size_t>(~0)};\n"
    "\n"
    "  const size_t *${exportSymbol}_sizes[] = {${exportSymbol}_sizes_tmp};\n"
    "#ifdef __cplusplus\n"
    "}\n"
    "#endif\n"
    )
endfunction()


separate_arguments(SOURCES)

set (SOURCES ${SOURCES})
if (CPP_FILE)
  bin2cpp("${CPP_FILE}" "${CPP_SYMBOL}" "${SOURCE_BASE}" "${SOURCES}")
endif()
if (H_FILE)
  bin2h  ("${H_FILE}"   "${CPP_SYMBOL}" "${SOURCES}")
endif()
