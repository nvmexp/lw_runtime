#  James Bigler, LWPU Corp (lwpu.com - jbigler)
#
#  Copyright (c) 2008 - 2009 LWPU Corporation.  All rights reserved.
#
#  This code is licensed under the MIT License.  See the FindLWDA.cmake script
#  for the text of the license.

# The MIT License
#
# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


##########################################################################
# This file runs the lwcc commands to produce the desired output file along with
# the dependency file needed by CMake to compute dependencies.  In addition the
# file checks the output of each command and if the command fails it deletes the
# output files.

# Input variables
#
# verbose:BOOL=<>          OFF: Be as quiet as possible (default)
#                          ON : Describe each step
#
# build_configuration:STRING=<> Typically one of Debug, MinSizeRel, Release, or
#                               RelWithDebInfo, but it should match one of the
#                               entries in LWDA_HOST_FLAGS. This is the build
#                               configuration used when compiling the code.  If
#                               blank or unspecified Debug is assumed as this is
#                               what CMake does.
#
# generated_file:STRING=<> File to generate.  This argument must be passed in.
#
# generated_lwbin_file:STRING=<> File to generate.  This argument must be passed
#                                                   in if build_lwbin is true.

if(NOT generated_file)
  message(FATAL_ERROR "You must specify generated_file on the command line")
endif()

# Set these up as variables to make reading the generated file easier
set(CMAKE_COMMAND "@CMAKE_COMMAND@") # path
set(source_file "@source_file@") # path
set(LWCC_generated_dependency_file "@LWCC_generated_dependency_file@") # path
set(cmake_dependency_file "@cmake_dependency_file@") # path
set(LWDA_make2cmake "@LWDA_make2cmake@") # path
set(LWDA_parse_lwbin "@LWDA_parse_lwbin@") # path
set(build_lwbin @build_lwbin@) # bool
set(LWDA_HOST_COMPILER "@LWDA_HOST_COMPILER@") # path
# We won't actually use these variables for now, but we need to set this, in
# order to force this file to be run again if it changes.
set(generated_file_path "@generated_file_path@") # path
set(generated_file_internal "@generated_file@") # path
set(generated_lwbin_file_internal "@generated_lwbin_file@") # path

set(LWDA_LWCC_EXELWTABLE "@LWDA_LWCC_EXELWTABLE@") # path
set(LWDA_LWCC_FLAGS @LWDA_LWCC_FLAGS@ ;; @LWDA_WRAP_OPTION_LWCC_FLAGS@) # list
@LWDA_LWCC_FLAGS_CONFIG@
set(lwcc_flags @lwcc_flags@) # list
set(LWDA_LWCC_INCLUDE_ARGS "@LWDA_LWCC_INCLUDE_ARGS@") # list (needs to be in quotes to handle spaces properly).
set(format_flag "@format_flag@") # string
set(lwda_language_flag @lwda_language_flag@) # list

if(build_lwbin AND NOT generated_lwbin_file)
  message(FATAL_ERROR "You must specify generated_lwbin_file on the command line")
endif()

# This is the list of host compilation flags.  It C or CXX should already have
# been chosen by FindLWDA.cmake.
@LWDA_HOST_FLAGS@

# Take the compiler flags and package them up to be sent to the compiler via -Xcompiler
set(lwcc_host_compiler_flags "")
# If we weren't given a build_configuration, use Debug.
if(NOT build_configuration)
  set(build_configuration Debug)
endif()
string(TOUPPER "${build_configuration}" build_configuration)
#message("LWDA_LWCC_HOST_COMPILER_FLAGS = ${LWDA_LWCC_HOST_COMPILER_FLAGS}")
foreach(flag ${CMAKE_HOST_FLAGS} ${CMAKE_HOST_FLAGS_${build_configuration}})
  # Extra quotes are added around each flag to help lwcc parse out flags with spaces.
  set(lwcc_host_compiler_flags "${lwcc_host_compiler_flags},\"${flag}\"")
endforeach()
if (lwcc_host_compiler_flags)
  set(lwcc_host_compiler_flags "-Xcompiler" ${lwcc_host_compiler_flags})
endif()
#message("lwcc_host_compiler_flags = \"${lwcc_host_compiler_flags}\"")
# Add the build specific configuration flags
list(APPEND LWDA_LWCC_FLAGS ${LWDA_LWCC_FLAGS_${build_configuration}})

# Any -ccbin existing in LWDA_LWCC_FLAGS gets highest priority
list( FIND LWDA_LWCC_FLAGS "-ccbin" ccbin_found0 )
list( FIND LWDA_LWCC_FLAGS "--compiler-bindir" ccbin_found1 )
if( ccbin_found0 LESS 0 AND ccbin_found1 LESS 0 AND LWDA_HOST_COMPILER )
  if (LWDA_HOST_COMPILER STREQUAL "$(VCInstallDir)bin" AND DEFINED CCBIN)
    set(CCBIN -ccbin "${CCBIN}")
  else()
    set(CCBIN -ccbin "${LWDA_HOST_COMPILER}")
  endif()
endif()

# lwda_exelwte_process - Exelwtes a command with optional command echo and status message.
#
#   status  - Status message to print if verbose is true
#   command - COMMAND argument from the usual exelwte_process argument structure
#   ARGN    - Remaining arguments are the command with arguments
#
#   LWDA_result - return value from running the command
#
# Make this a macro instead of a function, so that things like RESULT_VARIABLE
# and other return variables are present after exelwting the process.
macro(lwda_exelwte_process status command)
  set(_command ${command})
  if(NOT "x${_command}" STREQUAL "xCOMMAND")
    message(FATAL_ERROR "Malformed call to lwda_exelwte_process.  Missing COMMAND as second argument. (command = ${command})")
  endif()
  if(verbose)
    exelwte_process(COMMAND "${CMAKE_COMMAND}" -E echo -- ${status})
    # Now we need to build up our command string.  We are accounting for quotes
    # and spaces, anything else is left up to the user to fix if they want to
    # copy and paste a runnable command line.
    set(lwda_exelwte_process_string)
    foreach(arg ${ARGN})
      # If there are quotes, excape them, so they come through.
      string(REPLACE "\"" "\\\"" arg ${arg})
      # Args with spaces need quotes around them to get them to be parsed as a single argument.
      if(arg MATCHES " ")
        list(APPEND lwda_exelwte_process_string "\"${arg}\"")
      else()
        list(APPEND lwda_exelwte_process_string ${arg})
      endif()
    endforeach()
    # Echo the command
    exelwte_process(COMMAND ${CMAKE_COMMAND} -E echo ${lwda_exelwte_process_string})
  endif()
  # Run the command
  exelwte_process(COMMAND ${ARGN} RESULT_VARIABLE LWDA_result )
endmacro()

# Delete the target file
lwda_exelwte_process(
  "Removing ${generated_file}"
  COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
  )

# For LWCA 2.3 and below, -G -M doesn't work, so remove the -G flag
# for dependency generation and hope for the best.
set(depends_LWDA_LWCC_FLAGS "${LWDA_LWCC_FLAGS}")
set(LWDA_VERSION @LWDA_VERSION@)
if(LWDA_VERSION VERSION_LESS "3.0")
  cmake_policy(PUSH)
  # CMake policy 0007 NEW states that empty list elements are not
  # ignored.  I'm just setting it to avoid the warning that's printed.
  cmake_policy(SET CMP0007 NEW)
  # Note that this will remove all oclwrances of -G.
  list(REMOVE_ITEM depends_LWDA_LWCC_FLAGS "-G")
  cmake_policy(POP)
endif()

# lwcc doesn't define __LWDACC__ for some reason when generating dependency files.  This
# can cause incorrect dependencies when #including files based on this macro which is
# defined in the generating passes of lwcc ilwokation.  We will go ahead and manually
# define this for now until a future version fixes this bug.
set(LWDACC_DEFINE -D__LWDACC__)

# Generate the dependency file
lwda_exelwte_process(
  "Generating dependency file: ${LWCC_generated_dependency_file}"
  COMMAND "${LWDA_LWCC_EXELWTABLE}"
  -M
  ${LWDACC_DEFINE}
  "${source_file}"
  -o "${LWCC_generated_dependency_file}"
  ${CCBIN}
  ${lwcc_flags}
  ${lwcc_host_compiler_flags}
  ${depends_LWDA_LWCC_FLAGS}
  -DLWCC
  ${LWDA_LWCC_INCLUDE_ARGS}
  )

if(LWDA_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Generate the cmake readable dependency file to a temp file.  Don't put the
# quotes just around the filenames for the input_file and output_file variables.
# CMake will pass the quotes through and not be able to find the file.
lwda_exelwte_process(
  "Generating temporary cmake readable file: ${cmake_dependency_file}.tmp"
  COMMAND "${CMAKE_COMMAND}"
  -D "input_file:FILEPATH=${LWCC_generated_dependency_file}"
  -D "output_file:FILEPATH=${cmake_dependency_file}.tmp"
  -D "verbose=${verbose}"
  -P "${LWDA_make2cmake}"
  )

if(LWDA_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Copy the file if it is different
lwda_exelwte_process(
  "Copy if different ${cmake_dependency_file}.tmp to ${cmake_dependency_file}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${cmake_dependency_file}.tmp" "${cmake_dependency_file}"
  )

if(LWDA_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Delete the temporary file
lwda_exelwte_process(
  "Removing ${cmake_dependency_file}.tmp and ${LWCC_generated_dependency_file}"
  COMMAND "${CMAKE_COMMAND}" -E remove "${cmake_dependency_file}.tmp" "${LWCC_generated_dependency_file}"
  )

if(LWDA_result)
  message(FATAL_ERROR "Error generating ${generated_file}")
endif()

# Generate the code
lwda_exelwte_process(
  "Generating ${generated_file}"
  COMMAND "${LWDA_LWCC_EXELWTABLE}"
  "${source_file}"
  ${lwda_language_flag}
  ${format_flag} -o "${generated_file}"
  ${CCBIN}
  ${lwcc_flags}
  ${lwcc_host_compiler_flags}
  ${LWDA_LWCC_FLAGS}
  -DLWCC
  ${LWDA_LWCC_INCLUDE_ARGS}
  )

if(LWDA_result)
  # Since lwcc can sometimes leave half done files make sure that we delete the output file.
  lwda_exelwte_process(
    "Removing ${generated_file}"
    COMMAND "${CMAKE_COMMAND}" -E remove "${generated_file}"
    )
  message(FATAL_ERROR "Error generating file ${generated_file}")
else()
  if(verbose)
    message("Generated ${generated_file} successfully.")
  endif()
endif()

# Lwbin resource report commands.
if( build_lwbin )
  # Run with -lwbin to produce resource usage report.
  lwda_exelwte_process(
    "Generating ${generated_lwbin_file}"
    COMMAND "${LWDA_LWCC_EXELWTABLE}"
    "${source_file}"
    ${LWDA_LWCC_FLAGS}
    ${lwcc_flags}
    ${CCBIN}
    ${lwcc_host_compiler_flags}
    -DLWCC
    -lwbin
    -o "${generated_lwbin_file}"
    ${LWDA_LWCC_INCLUDE_ARGS}
    )

  # Execute the parser script.
  lwda_exelwte_process(
    "Exelwting the parser script"
    COMMAND  "${CMAKE_COMMAND}"
    -D "input_file:STRING=${generated_lwbin_file}"
    -P "${LWDA_parse_lwbin}"
    )

endif()
