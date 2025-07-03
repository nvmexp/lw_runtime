#.rst:
# FindLWDA
# --------
#
# Tools for building LWCA C files: libraries and build dependencies.
#
# This script locates the LWPU LWCA C tools.  It should work on linux,
# windows, and mac and should be reasonably up to date with LWCA C
# releases.
#
# This script makes use of the standard find_package arguments of
# <VERSION>, REQUIRED and QUIET.  LWDA_FOUND will report if an
# acceptable version of LWCA was found.
#
# The script will prompt the user to specify LWDA_TOOLKIT_ROOT_DIR if
# the prefix cannot be determined by the location of lwcc in the system
# path and REQUIRED is specified to find_package().  To use a different
# installed version of the toolkit set the environment variable
# LWDA_BIN_PATH before running cmake (e.g.
# LWDA_BIN_PATH=/usr/local/lwda1.0 instead of the default
# /usr/local/lwca) or set LWDA_TOOLKIT_ROOT_DIR after configuring.  If
# you change the value of LWDA_TOOLKIT_ROOT_DIR, various components that
# depend on the path will be relocated.
#
# It might be necessary to set LWDA_TOOLKIT_ROOT_DIR manually on certain
# platforms, or to use a lwca runtime not installed in the default
# location.  In newer versions of the toolkit the lwca library is
# included with the graphics driver- be sure that the driver version
# matches what is needed by the lwca runtime version.
#
# The following variables affect the behavior of the macros in the
# script (in alphebetical order).  Note that any of these flags can be
# changed multiple times in the same directory before calling
# LWDA_ADD_EXELWTABLE, LWDA_ADD_LIBRARY, LWDA_COMPILE, LWDA_COMPILE_PTX,
# LWDA_COMPILE_FATBIN, LWDA_COMPILE_LWBIN or LWDA_WRAP_SRCS::
#
#   LWDA_64_BIT_DEVICE_CODE (Default matches host bit size)
#   -- Set to ON to compile for 64 bit device code, OFF for 32 bit device code.
#      Note that making this different from the host code when generating object
#      or C files from LWCA code just won't work, because size_t gets defined by
#      lwcc in the generated source.  If you compile to PTX and then load the
#      file yourself, you can mix bit sizes between device and host.
#
#   LWDA_ATTACH_VS_BUILD_RULE_TO_LWDA_FILE (Default ON)
#   -- Set to ON if you want the custom build rule to be attached to the source
#      file in Visual Studio.  Turn OFF if you add the same lwca file to multiple
#      targets.
#
#      This allows the user to build the target from the LWCA file; however, bad
#      things can happen if the LWCA source file is added to multiple targets.
#      When performing parallel builds it is possible for the custom build
#      command to be run more than once and in parallel causing cryptic build
#      errors.  VS runs the rules for every source file in the target, and a
#      source can have only one rule no matter how many projects it is added to.
#      When the rule is run from multiple targets race conditions can occur on
#      the generated file.  Eventually everything will get built, but if the user
#      is unaware of this behavior, there may be confusion.  It would be nice if
#      this script could detect the reuse of source files across multiple targets
#      and turn the option off for the user, but no good solution could be found.
#
#   LWDA_BUILD_LWBIN (Default OFF)
#   -- Set to ON to enable and extra compilation pass with the -lwbin option in
#      Device mode. The output is parsed and register, shared memory usage is
#      printed during build.
#
#   LWDA_BUILD_EMULATION (Default OFF for device mode)
#   -- Set to ON for Emulation mode. -D_DEVICEEMU is defined for LWCA C files
#      when LWDA_BUILD_EMULATION is TRUE.
#
#   LWDA_GENERATED_OUTPUT_DIR (Default CMAKE_LWRRENT_BINARY_DIR)
#   -- Set to the path you wish to have the generated files placed.  If it is
#      blank output files will be placed in CMAKE_LWRRENT_BINARY_DIR.
#      Intermediate files will always be placed in
#      CMAKE_LWRRENT_BINARY_DIR/CMakeFiles.
#
#   LWDA_HOST_COMPILATION_CPP (Default ON)
#   -- Set to OFF for C compilation of host code.
#
#   LWDA_HOST_COMPILER (Default CMAKE_C_COMPILER, $(VCInstallDir)/bin for VS)
#   -- Set the host compiler to be used by lwcc.  Ignored if -ccbin or
#      --compiler-bindir is already present in the LWDA_LWCC_FLAGS or
#      LWDA_LWCC_FLAGS_<CONFIG> variables.  For Visual Studio targets
#      $(VCInstallDir)/bin is a special value that expands out to the path when
#      the command is run from within VS.
#
#   LWDA_LWCC_FLAGS
#   LWDA_LWCC_FLAGS_<CONFIG>
#   -- Additional LWCC command line arguments.  NOTE: multiple arguments must be
#      semi-colon delimited (e.g. --compiler-options;-Wall)
#
#   LWDA_PROPAGATE_HOST_FLAGS (Default ON)
#   -- Set to ON to propagate CMAKE_{C,CXX}_FLAGS and their configuration
#      dependent counterparts (e.g. CMAKE_C_FLAGS_DEBUG) automatically to the
#      host compiler through lwcc's -Xcompiler flag.  This helps make the
#      generated host code match the rest of the system better.  Sometimes
#      certain flags give lwcc problems, and this will help you turn the flag
#      propagation off.  This does not affect the flags supplied directly to lwcc
#      via LWDA_LWCC_FLAGS or through the OPTION flags specified through
#      LWDA_ADD_LIBRARY, LWDA_ADD_EXELWTABLE, or LWDA_WRAP_SRCS.  Flags used for
#      shared library compilation are not affected by this flag.
#
#   LWDA_SEPARABLE_COMPILATION (Default OFF)
#   -- If set this will enable separable compilation for all LWCA runtime object
#      files.  If used outside of LWDA_ADD_EXELWTABLE and LWDA_ADD_LIBRARY
#      (e.g. calling LWDA_WRAP_SRCS directly),
#      LWDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME and
#      LWDA_LINK_SEPARABLE_COMPILATION_OBJECTS should be called.
#
#   LWDA_SOURCE_PROPERTY_FORMAT
#   -- If this source file property is set, it can override the format specified
#      to LWDA_WRAP_SRCS (OBJ, PTX, LWBIN, or FATBIN).  If an input source file
#      is not a .lw file, setting this file will cause it to be treated as a .lw
#      file. See documentation for set_source_files_properties on how to set
#      this property.
#
#   LWDA_USE_STATIC_LWDA_RUNTIME (Default ON)
#   -- When enabled the static version of the LWCA runtime library will be used
#      in LWDA_LIBRARIES.  If the version of LWCA configured doesn't support
#      this option, then it will be silently disabled.
#
#   LWDA_VERBOSE_BUILD (Default OFF)
#   -- Set to ON to see all the commands used when building the LWCA file.  When
#      using a Makefile generator the value defaults to VERBOSE (run make
#      VERBOSE=1 to see output), although setting LWDA_VERBOSE_BUILD to ON will
#      always print the output.
#
# The script creates the following macros (in alphebetical order)::
#
#   LWDA_ADD_LWFFT_TO_TARGET( lwda_target )
#   -- Adds the lwfft library to the target (can be any target).  Handles whether
#      you are in emulation mode or not.
#
#   LWDA_ADD_LWBLAS_TO_TARGET( lwda_target )
#   -- Adds the lwblas library to the target (can be any target).  Handles
#      whether you are in emulation mode or not.
#
#   LWDA_ADD_EXELWTABLE( lwda_target file0 file1 ...
#                        [WIN32] [MACOSX_BUNDLE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#   -- Creates an exelwtable "lwda_target" which is made up of the files
#      specified.  All of the non LWCA C files are compiled using the standard
#      build rules specified by CMAKE and the lwca files are compiled to object
#      files using lwcc and the host compiler.  In addition LWDA_INCLUDE_DIRS is
#      added automatically to include_directories().  Some standard CMake target
#      calls can be used on the target after calling this macro
#      (e.g. set_target_properties and target_link_libraries), but setting
#      properties that adjust compilation flags will not affect code compiled by
#      lwcc.  Such flags should be modified before calling LWDA_ADD_EXELWTABLE,
#      LWDA_ADD_LIBRARY or LWDA_WRAP_SRCS.
#
#   LWDA_ADD_LIBRARY( lwda_target file0 file1 ...
#                     [STATIC | SHARED | MODULE] [EXCLUDE_FROM_ALL] [OPTIONS ...] )
#   -- Same as LWDA_ADD_EXELWTABLE except that a library is created.
#
#   LWDA_BUILD_CLEAN_TARGET()
#   -- Creates a colwience target that deletes all the dependency files
#      generated.  You should make clean after running this target to ensure the
#      dependency files get regenerated.
#
#   LWDA_COMPILE( generated_files file0 file1 ... [STATIC | SHARED | MODULE]
#                 [OPTIONS ...] )
#   -- Returns a list of generated files from the input source files to be used
#      with ADD_LIBRARY or ADD_EXELWTABLE.
#
#   LWDA_COMPILE_PTX( generated_files file0 file1 ... [OPTIONS ...] )
#   -- Returns a list of PTX files generated from the input source files.
#
#   LWDA_COMPILE_FATBIN( generated_files file0 file1 ... [OPTIONS ...] )
#   -- Returns a list of FATBIN files generated from the input source files.
#
#   LWDA_COMPILE_LWBIN( generated_files file0 file1 ... [OPTIONS ...] )
#   -- Returns a list of LWBIN files generated from the input source files.
#
#   LWDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME( output_file_var
#                                                        lwda_target
#                                                        object_files )
#   -- Compute the name of the intermediate link file used for separable
#      compilation.  This file name is typically passed into
#      LWDA_LINK_SEPARABLE_COMPILATION_OBJECTS.  output_file_var is produced
#      based on lwda_target the list of objects files that need separable
#      compilation as specified by object_files.  If the object_files list is
#      empty, then output_file_var will be empty.  This function is called
#      automatically for LWDA_ADD_LIBRARY and LWDA_ADD_EXELWTABLE.  Note that
#      this is a function and not a macro.
#
#   LWDA_INCLUDE_DIRECTORIES( path0 path1 ... )
#   -- Sets the directories that should be passed to lwcc
#      (e.g. lwcc -Ipath0 -Ipath1 ... ). These paths usually contain other .lw
#      files.
#
#
#   LWDA_LINK_SEPARABLE_COMPILATION_OBJECTS( output_file_var lwda_target
#                                            lwcc_flags object_files)
#   -- Generates the link object required by separable compilation from the given
#      object files.  This is called automatically for LWDA_ADD_EXELWTABLE and
#      LWDA_ADD_LIBRARY, but can be called manually when using LWDA_WRAP_SRCS
#      directly.  When called from LWDA_ADD_LIBRARY or LWDA_ADD_EXELWTABLE the
#      lwcc_flags passed in are the same as the flags passed in via the OPTIONS
#      argument.  The only lwcc flag added automatically is the bitness flag as
#      specified by LWDA_64_BIT_DEVICE_CODE.  Note that this is a function
#      instead of a macro.
#
#   LWDA_SELECT_LWCC_ARCH_FLAGS(out_variable [target_LWDA_architectures])
#   -- Selects GPU arch flags for lwcc based on target_LWDA_architectures
#      target_LWDA_architectures : Auto | Common | All | LIST(ARCH_AND_PTX ...)
#       - "Auto" detects local machine GPU compute arch at runtime.
#       - "Common" and "All" cover common and entire subsets of architectures
#      ARCH_AND_PTX : NAME | NUM.NUM | NUM.NUM(NUM.NUM) | NUM.NUM+PTX
#      NAME: Fermi Kepler Maxwell Kepler+CheetAh Kepler+Tesla Maxwell+CheetAh Pascal
#      NUM: Any number. Only those pairs are lwrrently accepted by LWCC though:
#            2.0 2.1 3.0 3.2 3.5 3.7 5.0 5.2 5.3 6.0 6.2
#      Returns LIST of flags to be added to LWDA_LWCC_FLAGS in ${out_variable}
#      Additionally, sets ${out_variable}_readable to the resulting numeric list
#      Example:
#       LWDA_SELECT_LWCC_ARCH_FLAGS(ARCH_FLAGS 3.0 3.5+PTX 5.2(5.0) Maxwell)
#        LIST(APPEND LWDA_LWCC_FLAGS ${ARCH_FLAGS})
#
#      More info on LWCA architectures: https://en.wikipedia.org/wiki/LWCA
#      Note that this is a function instead of a macro.
#
#   LWDA_WRAP_SRCS ( lwda_target format generated_files file0 file1 ...
#                    [STATIC | SHARED | MODULE] [OPTIONS ...] )
#   -- This is where all the magic happens.  LWDA_ADD_EXELWTABLE,
#      LWDA_ADD_LIBRARY, LWDA_COMPILE, and LWDA_COMPILE_PTX all call this
#      function under the hood.
#
#      Given the list of files (file0 file1 ... fileN) this macro generates
#      custom commands that generate either PTX or linkable objects (use "PTX" or
#      "OBJ" for the format argument to switch).  Files that don't end with .lw
#      or have the HEADER_FILE_ONLY property are ignored.
#
#      The arguments passed in after OPTIONS are extra command line options to
#      give to lwcc.  You can also specify per configuration options by
#      specifying the name of the configuration followed by the options.  General
#      options must precede configuration specific options.  Not all
#      configurations need to be specified, only the ones provided will be used.
#
#         OPTIONS -DFLAG=2 "-DFLAG_OTHER=space in flag"
#         DEBUG -g
#         RELEASE --use_fast_math
#         RELWITHDEBINFO --use_fast_math;-g
#         MINSIZEREL --use_fast_math
#
#      For certain configurations (namely VS generating object files with
#      LWDA_ATTACH_VS_BUILD_RULE_TO_LWDA_FILE set to ON), no generated file will
#      be produced for the given lwca file.  This is because when you add the
#      lwca file to Visual Studio it knows that this file produces an object file
#      and will link in the resulting object file automatically.
#
#      This script will also generate a separate cmake script that is used at
#      build time to ilwoke lwcc.  This is for several reasons.
#
#        1. lwcc can return negative numbers as return values which confuses
#        Visual Studio into thinking that the command succeeded.  The script now
#        checks the error codes and produces errors when there was a problem.
#
#        2. lwcc has been known to not delete incomplete results when it
#        encounters problems.  This confuses build systems into thinking the
#        target was generated when in fact an unusable file exists.  The script
#        now deletes the output files if there was an error.
#
#        3. By putting all the options that affect the build into a file and then
#        make the build rule dependent on the file, the output files will be
#        regenerated when the options change.
#
#      This script also looks at optional arguments STATIC, SHARED, or MODULE to
#      determine when to target the object compilation for a shared library.
#      BUILD_SHARED_LIBS is ignored in LWDA_WRAP_SRCS, but it is respected in
#      LWDA_ADD_LIBRARY.  On some systems special flags are added for building
#      objects intended for shared libraries.  A preprocessor macro,
#      <target_name>_EXPORTS is defined when a shared library compilation is
#      detected.
#
#      Flags passed into add_definitions with -D or /D are passed along to lwcc.
#
#
#
# The script defines the following variables::
#
#   LWDA_VERSION_MAJOR    -- The major version of lwca as reported by lwcc.
#   LWDA_VERSION_MINOR    -- The minor version.
#   LWDA_VERSION
#   LWDA_VERSION_STRING   -- LWDA_VERSION_MAJOR.LWDA_VERSION_MINOR
#   LWDA_HAS_FP16         -- Whether a short float (float16,fp16) is supported.
#
#   LWDA_TOOLKIT_ROOT_DIR -- Path to the LWCA Toolkit (defined if not set).
#   LWDA_SDK_ROOT_DIR     -- Path to the LWCA SDK.  Use this to find files in the
#                            SDK.  This script will not directly support finding
#                            specific libraries or headers, as that isn't
#                            supported by LWPU.  If you want to change
#                            libraries when the path changes see the
#                            FindLWDA.cmake script for an example of how to clear
#                            these variables.  There are also examples of how to
#                            use the LWDA_SDK_ROOT_DIR to locate headers or
#                            libraries, if you so choose (at your own risk).
#   LWDA_INCLUDE_DIRS     -- Include directory for lwca headers.  Added automatically
#                            for LWDA_ADD_EXELWTABLE and LWDA_ADD_LIBRARY.
#   LWDA_LIBRARIES        -- Lwca RT library.
#   LWDA_LWFFT_LIBRARIES  -- Device or emulation library for the Lwca FFT
#                            implementation (alternative to:
#                            LWDA_ADD_LWFFT_TO_TARGET macro)
#   LWDA_LWBLAS_LIBRARIES -- Device or emulation library for the Lwca BLAS
#                            implementation (alterative to:
#                            LWDA_ADD_LWBLAS_TO_TARGET macro).
#   LWDA_lwdart_static_LIBRARY -- Statically linkable lwca runtime library.
#                                 Only available for LWCA version 5.5+
#   LWDA_lwpti_LIBRARY    -- LWCA Profiling Tools Interface library.
#                            Only available for LWCA version 4.0+.
#   LWDA_lwrand_LIBRARY   -- LWCA Random Number Generation library.
#                            Only available for LWCA version 3.2+.
#   LWDA_lwsolver_LIBRARY -- LWCA Direct Solver library.
#                            Only available for LWCA version 7.0+.
#   LWDA_lwsparse_LIBRARY -- LWCA Sparse Matrix library.
#                            Only available for LWCA version 3.2+.
#   LWDA_npp_LIBRARY      -- LWPU Performance Primitives lib.
#                            Only available for LWCA version 4.0+.
#   LWDA_nppc_LIBRARY     -- LWPU Performance Primitives lib (core).
#                            Only available for LWCA version 5.5+.
#   LWDA_nppi_LIBRARY     -- LWPU Performance Primitives lib (image processing).
#                            Only available for LWCA version 5.5+.
#   LWDA_npps_LIBRARY     -- LWPU Performance Primitives lib (signal processing).
#                            Only available for LWCA version 5.5+.
#   LWDA_lwlwvenc_LIBRARY -- LWCA Video Encoder library.
#                            Only available for LWCA version 3.2+.
#                            Windows only.
#   LWDA_lwlwvid_LIBRARY  -- LWCA Video Decoder library.
#                            Only available for LWCA version 3.2+.
#                            Windows only.
#

#   James Bigler, LWPU Corp (lwpu.com - jbigler)
#   Abe Stephens, SCI Institute -- http://www.sci.utah.edu/~abe/FindLwda.html
#
#   Copyright (c) 2008 - 2009 LWPU Corporation.  All rights reserved.
#
#   Copyright (c) 2007-2009
#   Scientific Computing and Imaging Institute, University of Utah
#
#   This code is licensed under the MIT License.  See the FindLWDA.cmake script
#   for the text of the license.

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
#
###############################################################################

# FindLWDA.cmake

# We need to have at least this version to support the VERSION_LESS argument to 'if' (2.6.2) and unset (2.6.3)
cmake_policy(PUSH)
cmake_minimum_required(VERSION 2.6.3)
cmake_policy(POP)

# This macro helps us find the location of helper files we will need the full path to
macro(LWDA_FIND_HELPER_FILE _name _extension)
  set(_full_name "${_name}.${_extension}")
  # CMAKE_LWRRENT_LIST_FILE contains the full path to the file lwrrently being
  # processed.  Using this variable, we can pull out the current path, and
  # provide a way to get access to the other files we need local to here.
  get_filename_component(CMAKE_LWRRENT_LIST_DIR "${CMAKE_LWRRENT_LIST_FILE}" PATH)
  set(LWDA_${_name} "${CMAKE_LWRRENT_LIST_DIR}/FindLWDA/${_full_name}")
  if(NOT EXISTS "${LWDA_${_name}}")
    set(error_message "${_full_name} not found in ${CMAKE_LWRRENT_LIST_DIR}/FindLWDA")
    if(LWDA_FIND_REQUIRED)
      message(FATAL_ERROR "${error_message}")
    else()
      if(NOT LWDA_FIND_QUIETLY)
        message(STATUS "${error_message}")
      endif()
    endif()
  endif()
  # Set this variable as internal, so the user isn't bugged with it.
  set(LWDA_${_name} ${LWDA_${_name}} CACHE INTERNAL "Location of ${_full_name}" FORCE)
endmacro()

#####################################################################
## LWDA_INCLUDE_LWCC_DEPENDENCIES
##

# So we want to try and include the dependency file if it exists.  If
# it doesn't exist then we need to create an empty one, so we can
# include it.

# If it does exist, then we need to check to see if all the files it
# depends on exist.  If they don't then we should clear the dependency
# file and regenerate it later.  This covers the case where a header
# file has disappeared or moved.

macro(LWDA_INCLUDE_LWCC_DEPENDENCIES dependency_file)
  set(LWDA_LWCC_DEPEND)
  set(LWDA_LWCC_DEPEND_REGENERATE FALSE)


  # Include the dependency file.  Create it first if it doesn't exist .  The
  # INCLUDE puts a dependency that will force CMake to rerun and bring in the
  # new info when it changes.  DO NOT REMOVE THIS (as I did and spent a few
  # hours figuring out why it didn't work.
  if(NOT EXISTS ${dependency_file})
    file(WRITE ${dependency_file} "#FindLWDA.cmake generated file.  Do not edit.\n")
  endif()
  # Always include this file to force CMake to run again next
  # invocation and rebuild the dependencies.
  #message("including dependency_file = ${dependency_file}")
  include(${dependency_file})

  # Now we need to verify the existence of all the included files
  # here.  If they aren't there we need to just blank this variable and
  # make the file regenerate again.
#   if(DEFINED LWDA_LWCC_DEPEND)
#     message("LWDA_LWCC_DEPEND set")
#   else()
#     message("LWDA_LWCC_DEPEND NOT set")
#   endif()
  if(LWDA_LWCC_DEPEND)
    #message("LWDA_LWCC_DEPEND found")
    foreach(f ${LWDA_LWCC_DEPEND})
      # message("searching for ${f}")
      if(NOT EXISTS ${f})
        #message("file ${f} not found")
        set(LWDA_LWCC_DEPEND_REGENERATE TRUE)
      endif()
    endforeach()
  else()
    #message("LWDA_LWCC_DEPEND false")
    # No dependencies, so regenerate the file.
    set(LWDA_LWCC_DEPEND_REGENERATE TRUE)
  endif()

  #message("LWDA_LWCC_DEPEND_REGENERATE = ${LWDA_LWCC_DEPEND_REGENERATE}")
  # No incoming dependencies, so we need to generate them.  Make the
  # output depend on the dependency file itself, which should cause the
  # rule to re-run.
  if(LWDA_LWCC_DEPEND_REGENERATE)
    set(LWDA_LWCC_DEPEND ${dependency_file})
    #message("Generating an empty dependency_file: ${dependency_file}")
    file(WRITE ${dependency_file} "#FindLWDA.cmake generated file.  Do not edit.\n")
  endif()

endmacro()

###############################################################################
###############################################################################
# Setup variables' defaults
###############################################################################
###############################################################################

# Allow the user to specify if the device code is supposed to be 32 or 64 bit.
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(LWDA_64_BIT_DEVICE_CODE_DEFAULT ON)
else()
  set(LWDA_64_BIT_DEVICE_CODE_DEFAULT OFF)
endif()
option(LWDA_64_BIT_DEVICE_CODE "Compile device code in 64 bit mode" ${LWDA_64_BIT_DEVICE_CODE_DEFAULT})

# Attach the build rule to the source file in VS.  This option
option(LWDA_ATTACH_VS_BUILD_RULE_TO_LWDA_FILE "Attach the build rule to the LWCA source file.  Enable only when the LWCA source file is added to at most one target." ON)

# Prints out extra information about the lwca file during compilation
option(LWDA_BUILD_LWBIN "Generate and parse .lwbin files in Device mode." OFF)

# Set whether we are using emulation or device mode.
option(LWDA_BUILD_EMULATION "Build in Emulation mode" OFF)

# Where to put the generated output.
set(LWDA_GENERATED_OUTPUT_DIR "" CACHE PATH "Directory to put all the output files.  If blank it will default to the CMAKE_LWRRENT_BINARY_DIR")

# Parse HOST_COMPILATION mode.
option(LWDA_HOST_COMPILATION_CPP "Generated file extension" ON)

# Extra user settable flags
set(LWDA_LWCC_FLAGS "" CACHE STRING "Semi-colon delimit multiple arguments.")

if(CMAKE_GENERATOR MATCHES "Visual Studio")
  set(LWDA_HOST_COMPILER "$(VCInstallDir)bin" CACHE FILEPATH "Host side compiler used by LWCC")
else()
  if(APPLE
      AND "${CMAKE_C_COMPILER_ID}" MATCHES "Clang"
      AND "${CMAKE_C_COMPILER}" MATCHES "/cc$")
    # Using cc which is symlink to clang may let LWCC think it is GCC and issue
    # unhandled -dumpspecs option to clang. Also in case neither
    # CMAKE_C_COMPILER is defined (project does not use C language) nor
    # LWDA_HOST_COMPILER is specified manually we should skip -ccbin and let
    # lwcc use its own default C compiler.
    # Only care about this on APPLE with clang to avoid
    # following symlinks to things like ccache
    if(DEFINED CMAKE_C_COMPILER AND NOT DEFINED LWDA_HOST_COMPILER)
      get_filename_component(c_compiler_realpath "${CMAKE_C_COMPILER}" REALPATH)
      # if the real path does not end up being clang then
      # go back to using CMAKE_C_COMPILER
      if(NOT "${c_compiler_realpath}" MATCHES "/clang$")
        set(c_compiler_realpath "${CMAKE_C_COMPILER}")
      endif()
    else()
      set(c_compiler_realpath "")
    endif()
    set(LWDA_HOST_COMPILER "${c_compiler_realpath}" CACHE FILEPATH "Host side compiler used by LWCC")
  else()
    set(LWDA_HOST_COMPILER "${CMAKE_C_COMPILER}"
      CACHE FILEPATH "Host side compiler used by LWCC")
  endif()
endif()

# Propagate the host flags to the host compiler via -Xcompiler
option(LWDA_PROPAGATE_HOST_FLAGS "Propage C/CXX_FLAGS and friends to the host compiler via -Xcompile" ON)

# Enable LWDA_SEPARABLE_COMPILATION
option(LWDA_SEPARABLE_COMPILATION "Compile LWCA objects with separable compilation enabled.  Requires LWCA 5.0+" OFF)

# Specifies whether the commands used when compiling the .lw file will be printed out.
option(LWDA_VERBOSE_BUILD "Print out the commands run while compiling the LWCA source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option." OFF)

mark_as_advanced(
  LWDA_64_BIT_DEVICE_CODE
  LWDA_ATTACH_VS_BUILD_RULE_TO_LWDA_FILE
  LWDA_GENERATED_OUTPUT_DIR
  LWDA_HOST_COMPILATION_CPP
  LWDA_LWCC_FLAGS
  LWDA_PROPAGATE_HOST_FLAGS
  LWDA_BUILD_LWBIN
  LWDA_BUILD_EMULATION
  LWDA_VERBOSE_BUILD
  LWDA_SEPARABLE_COMPILATION
  )

# Makefile and similar generators don't define CMAKE_CONFIGURATION_TYPES, so we
# need to add another entry for the CMAKE_BUILD_TYPE.  We also need to add the
# standerd set of 4 build types (Debug, MinSizeRel, Release, and RelWithDebInfo)
# for completeness.  We need run this loop in order to accomodate the addition
# of extra configuration types.  Duplicate entries will be removed by
# REMOVE_DUPLICATES.
set(LWDA_configuration_types ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE} Debug MinSizeRel Release RelWithDebInfo)
list(REMOVE_DUPLICATES LWDA_configuration_types)
foreach(config ${LWDA_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(LWDA_LWCC_FLAGS_${config_upper} "" CACHE STRING "Semi-colon delimit multiple arguments.")
    mark_as_advanced(LWDA_LWCC_FLAGS_${config_upper})
endforeach()

###############################################################################
###############################################################################
# Locate LWCA, Set Build Type, etc.
###############################################################################
###############################################################################

macro(lwda_unset_include_and_libraries)
  unset(LWDA_TOOLKIT_INCLUDE CACHE)
  unset(LWDA_LWDART_LIBRARY CACHE)
  unset(LWDA_LWDA_LIBRARY CACHE)
  # Make sure you run this before you unset LWDA_VERSION.
  if(LWDA_VERSION VERSION_EQUAL "3.0")
    # This only existed in the 3.0 version of the LWCA toolkit
    unset(LWDA_LWDARTEMU_LIBRARY CACHE)
  endif()
  unset(LWDA_lwdart_static_LIBRARY CACHE)
  unset(LWDA_lwblas_LIBRARY CACHE)
  unset(LWDA_lwblas_device_LIBRARY CACHE)
  unset(LWDA_lwblasemu_LIBRARY CACHE)
  unset(LWDA_lwfft_LIBRARY CACHE)
  unset(LWDA_lwfftemu_LIBRARY CACHE)
  unset(LWDA_lwpti_LIBRARY CACHE)
  unset(LWDA_lwrand_LIBRARY CACHE)
  unset(LWDA_lwsolver_LIBRARY CACHE)
  unset(LWDA_lwsparse_LIBRARY CACHE)
  unset(LWDA_npp_LIBRARY CACHE)
  unset(LWDA_nppc_LIBRARY CACHE)
  unset(LWDA_nppi_LIBRARY CACHE)
  unset(LWDA_npps_LIBRARY CACHE)
  unset(LWDA_lwlwvenc_LIBRARY CACHE)
  unset(LWDA_lwlwvid_LIBRARY CACHE)
  unset(LWDA_USE_STATIC_LWDA_RUNTIME CACHE)
  unset(LWDA_GPU_DETECT_OUTPUT CACHE)
endmacro()

# Check to see if the LWDA_TOOLKIT_ROOT_DIR and LWDA_SDK_ROOT_DIR have changed,
# if they have then clear the cache variables, so that will be detected again.
if(NOT "${LWDA_TOOLKIT_ROOT_DIR}" STREQUAL "${LWDA_TOOLKIT_ROOT_DIR_INTERNAL}")
  unset(LWDA_TOOLKIT_TARGET_DIR CACHE)
  unset(LWDA_LWCC_EXELWTABLE CACHE)
  lwda_unset_include_and_libraries()
  unset(LWDA_VERSION CACHE)
endif()

if(NOT "${LWDA_TOOLKIT_TARGET_DIR}" STREQUAL "${LWDA_TOOLKIT_TARGET_DIR_INTERNAL}")
  lwda_unset_include_and_libraries()
endif()

#
#  End of unset()
#

#
#  Start looking for things
#

# Search for the lwca distribution.
if(NOT LWDA_TOOLKIT_ROOT_DIR AND NOT CMAKE_CROSSCOMPILING)
  # Search in the LWDA_BIN_PATH first.
  find_path(LWDA_TOOLKIT_ROOT_DIR
    NAMES lwcc lwcc.exe
    PATHS
      ELW LWDA_TOOLKIT_ROOT
      ELW LWDA_PATH
      ELW LWDA_BIN_PATH
    PATH_SUFFIXES bin bin64
    DOC "Toolkit location."
    NO_DEFAULT_PATH
    )

  # Now search default paths
  find_path(LWDA_TOOLKIT_ROOT_DIR
    NAMES lwcc lwcc.exe
    PATHS /usr/local/bin
          /usr/local/lwca/bin
    DOC "Toolkit location."
    )

  if (LWDA_TOOLKIT_ROOT_DIR)
    string(REGEX REPLACE "[/\\\\]?bin[64]*[/\\\\]?$" "" LWDA_TOOLKIT_ROOT_DIR ${LWDA_TOOLKIT_ROOT_DIR})
    # We need to force this back into the cache.
    set(LWDA_TOOLKIT_ROOT_DIR ${LWDA_TOOLKIT_ROOT_DIR} CACHE PATH "Toolkit location." FORCE)
    set(LWDA_TOOLKIT_TARGET_DIR ${LWDA_TOOLKIT_ROOT_DIR})
  endif()

  if (NOT EXISTS ${LWDA_TOOLKIT_ROOT_DIR})
    if(LWDA_FIND_REQUIRED)
      message(FATAL_ERROR "Specify LWDA_TOOLKIT_ROOT_DIR")
    elseif(NOT LWDA_FIND_QUIETLY)
      message("LWDA_TOOLKIT_ROOT_DIR not found or specified")
    endif()
  endif ()
endif ()

if(CMAKE_CROSSCOMPILING)
  SET (LWDA_TOOLKIT_ROOT $ELW{LWDA_TOOLKIT_ROOT})
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
    # Support for LWPACK
    set (LWDA_TOOLKIT_TARGET_NAME "armv7-linux-androideabi")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    # Support for arm cross compilation
    set(LWDA_TOOLKIT_TARGET_NAME "armv7-linux-gnueabihf")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    # Support for aarch64 cross compilation
    if (ANDROID_ARCH_NAME STREQUAL "arm64")
      set(LWDA_TOOLKIT_TARGET_NAME "aarch64-linux-androideabi")
    else()
      set(LWDA_TOOLKIT_TARGET_NAME "aarch64-linux")
    endif (ANDROID_ARCH_NAME STREQUAL "arm64")
  endif()

  if (EXISTS "${LWDA_TOOLKIT_ROOT}/targets/${LWDA_TOOLKIT_TARGET_NAME}")
    set(LWDA_TOOLKIT_TARGET_DIR "${LWDA_TOOLKIT_ROOT}/targets/${LWDA_TOOLKIT_TARGET_NAME}" CACHE PATH "LWCA Toolkit target location.")
    SET (LWDA_TOOLKIT_ROOT_DIR ${LWDA_TOOLKIT_ROOT})
    mark_as_advanced(LWDA_TOOLKIT_TARGET_DIR)
  endif()

  # add known LWCA targetr root path to the set of directories we search for programs, libraries and headers
  set( CMAKE_FIND_ROOT_PATH "${LWDA_TOOLKIT_TARGET_DIR};${CMAKE_FIND_ROOT_PATH}")
  macro( lwda_find_host_program )
    find_host_program( ${ARGN} )
  endmacro()
else()
  # for non-cross-compile, find_host_program == find_program and LWDA_TOOLKIT_TARGET_DIR == LWDA_TOOLKIT_ROOT_DIR
  macro( lwda_find_host_program )
    find_program( ${ARGN} )
  endmacro()
  SET (LWDA_TOOLKIT_TARGET_DIR ${LWDA_TOOLKIT_ROOT_DIR})
endif()


# LWDA_LWCC_EXELWTABLE
lwda_find_host_program(LWDA_LWCC_EXELWTABLE
  NAMES lwcc
  PATHS "${LWDA_TOOLKIT_ROOT_DIR}"
  ELW LWDA_PATH
  ELW LWDA_BIN_PATH
  PATH_SUFFIXES bin bin64
  NO_DEFAULT_PATH
  )
# Search default search paths, after we search our own set of paths.
lwda_find_host_program(LWDA_LWCC_EXELWTABLE lwcc)
mark_as_advanced(LWDA_LWCC_EXELWTABLE)

if(LWDA_LWCC_EXELWTABLE AND NOT LWDA_VERSION)
  # Compute the version.
  exelwte_process (COMMAND ${LWDA_LWCC_EXELWTABLE} "--version" OUTPUT_VARIABLE LWCC_OUT)
  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\1" LWDA_VERSION_MAJOR ${LWCC_OUT})
  string(REGEX REPLACE ".*release ([0-9]+)\\.([0-9]+).*" "\\2" LWDA_VERSION_MINOR ${LWCC_OUT})
  set(LWDA_VERSION "${LWDA_VERSION_MAJOR}.${LWDA_VERSION_MINOR}" CACHE STRING "Version of LWCA as computed from lwcc.")
  mark_as_advanced(LWDA_VERSION)
else()
  # Need to set these based off of the cached value
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" LWDA_VERSION_MAJOR "${LWDA_VERSION}")
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" LWDA_VERSION_MINOR "${LWDA_VERSION}")
endif()


# Always set this colwenience variable
set(LWDA_VERSION_STRING "${LWDA_VERSION}")

# LWDA_TOOLKIT_INCLUDE
find_path(LWDA_TOOLKIT_INCLUDE
  device_functions.h # Header included in toolkit
  PATHS ${LWDA_TOOLKIT_TARGET_DIR}
  ELW LWDA_PATH
  ELW LWDA_INC_PATH
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  )
# Search default search paths, after we search our own set of paths.
find_path(LWDA_TOOLKIT_INCLUDE device_functions.h)
mark_as_advanced(LWDA_TOOLKIT_INCLUDE)

if (LWDA_VERSION VERSION_GREATER "7.0" OR EXISTS "${LWDA_TOOLKIT_INCLUDE}/lwda_fp16.h")
  set(LWDA_HAS_FP16 TRUE)
else()
  set(LWDA_HAS_FP16 FALSE)
endif()

# Set the user list of include dir to nothing to initialize it.
set (LWDA_LWCC_INCLUDE_ARGS_USER "")
set (LWDA_INCLUDE_DIRS ${LWDA_TOOLKIT_INCLUDE})

macro(lwda_find_library_local_first_with_path_ext _var _names _doc _path_ext )
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    # LWCA 3.2+ on Windows moved the library directories, so we need the new
    # and old paths.
    set(_lwda_64bit_lib_dir "${_path_ext}lib/x64" "${_path_ext}lib64" "${_path_ext}libx64" )
  endif()
  # LWCA 3.2+ on Windows moved the library directories, so we need to new
  # (lib/Win32) and the old path (lib).
  find_library(${_var}
    NAMES ${_names}
    PATHS "${LWDA_TOOLKIT_TARGET_DIR}"
    ELW LWDA_PATH
    ELW LWDA_LIB_PATH
    PATH_SUFFIXES ${_lwda_64bit_lib_dir} "${_path_ext}lib/Win32" "${_path_ext}lib" "${_path_ext}libWin32"
    DOC ${_doc}
    NO_DEFAULT_PATH
    )
  if (NOT CMAKE_CROSSCOMPILING)
    # Search default search paths, after we search our own set of paths.
    find_library(${_var}
      NAMES ${_names}
      PATHS "/usr/lib/lwpu-current"
      DOC ${_doc}
      )
  endif()
endmacro()

macro(lwda_find_library_local_first _var _names _doc)
  lwda_find_library_local_first_with_path_ext( "${_var}" "${_names}" "${_doc}" "" )
endmacro()

macro(find_library_local_first _var _names _doc )
  lwda_find_library_local_first( "${_var}" "${_names}" "${_doc}" "" )
endmacro()


# LWDA_LIBRARIES
lwda_find_library_local_first(LWDA_LWDART_LIBRARY lwdart "\"lwdart\" library")
if(LWDA_VERSION VERSION_EQUAL "3.0")
  # The lwdartemu library only existed for the 3.0 version of LWCA.
  lwda_find_library_local_first(LWDA_LWDARTEMU_LIBRARY lwdartemu "\"lwdartemu\" library")
  mark_as_advanced(
    LWDA_LWDARTEMU_LIBRARY
    )
endif()

if(LWDA_USE_STATIC_LWDA_RUNTIME AND NOT LWDA_VERSION VERSION_LESS "5.5")
  lwda_find_library_local_first(LWDA_lwdart_static_LIBRARY lwdart_static "static LWCA runtime library")
  mark_as_advanced(LWDA_lwdart_static_LIBRARY)
endif()


if(LWDA_lwdart_static_LIBRARY)
  # Set whether to use the static lwca runtime.
  option(LWDA_USE_STATIC_LWDA_RUNTIME "Use the static version of the LWCA runtime library if available" ON)
  set(LWDA_LWDART_LIBRARY_VAR LWDA_lwdart_static_LIBRARY)
else()
  option(LWDA_USE_STATIC_LWDA_RUNTIME "Use the static version of the LWCA runtime library if available" OFF)
  set(LWDA_LWDART_LIBRARY_VAR LWDA_LWDART_LIBRARY)
endif()

if(LWDA_USE_STATIC_LWDA_RUNTIME)
  if(UNIX)
    # Check for the dependent libraries.  Here we look for pthreads.
    if (DEFINED CMAKE_THREAD_PREFER_PTHREAD)
      set(_lwda_cmake_thread_prefer_pthread ${CMAKE_THREAD_PREFER_PTHREAD})
    endif()
    set(CMAKE_THREAD_PREFER_PTHREAD 1)

    # Many of the FindXYZ CMake comes with makes use of try_compile with int main(){return 0;}
    # as the source file.  Unfortunately this causes a warning with -Wstrict-prototypes and
    # -Werror causes the try_compile to fail.  We will just temporarily disable other flags
    # when doing the find_package command here.
    set(_lwda_cmake_c_flags ${CMAKE_C_FLAGS})
    set(CMAKE_C_FLAGS "-fPIC")
    find_package(Threads REQUIRED)
    set(CMAKE_C_FLAGS ${_lwda_cmake_c_flags})

    if (DEFINED _lwda_cmake_thread_prefer_pthread)
      set(CMAKE_THREAD_PREFER_PTHREAD ${_lwda_cmake_thread_prefer_pthread})
      unset(_lwda_cmake_thread_prefer_pthread)
    else()
      unset(CMAKE_THREAD_PREFER_PTHREAD)
    endif()
  endif()
  if (NOT APPLE AND LWDA_VERSION VERSION_LESS "7.0")
    # Before LWCA 7.0, there was librt that has things such as, clock_gettime, shm_open, and shm_unlink.
    find_library(LWDA_rt_LIBRARY rt)
    if (NOT LWDA_rt_LIBRARY)
      message(WARNING "Expecting to find librt for liblwdart_static, but didn't find it.")
    endif()
  endif()
endif()

# LWPTI library showed up in lwca toolkit 4.0
if(NOT LWDA_VERSION VERSION_LESS "4.0")
  lwda_find_library_local_first_with_path_ext(LWDA_lwpti_LIBRARY lwpti "\"lwpti\" library" "extras/LWPTI/")
  mark_as_advanced(LWDA_lwpti_LIBRARY)
endif()

# Set the LWDA_LIBRARIES variable.  This is the set of stuff to link against if you are
# using the LWCA runtime.  For the dynamic version of the runtime, most of the
# dependencies are brough in, but for the static version there are additional libraries
# and linker commands needed.
# Initialize to empty
set(LWDA_LIBRARIES)

# If we are using emulation mode and we found the lwdartemu library then use
# that one instead of lwdart.
if(LWDA_BUILD_EMULATION AND LWDA_LWDARTEMU_LIBRARY)
  list(APPEND LWDA_LIBRARIES ${LWDA_LWDARTEMU_LIBRARY})
elseif(LWDA_USE_STATIC_LWDA_RUNTIME AND LWDA_lwdart_static_LIBRARY)
  list(APPEND LWDA_LIBRARIES ${LWDA_lwdart_static_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
  if (LWDA_rt_LIBRARY)
    list(APPEND LWDA_LIBRARIES ${LWDA_rt_LIBRARY})
  endif()
  if(APPLE)
    # We need to add the default path to the driver (liblwda.dylib) as an rpath, so that
    # the static lwca runtime can find it at runtime.
    list(APPEND LWDA_LIBRARIES -Wl,-rpath,/usr/local/lwca/lib)
  endif()
else()
  list(APPEND LWDA_LIBRARIES ${LWDA_LWDART_LIBRARY})
endif()

# 1.1 toolkit on linux doesn't appear to have a separate library on
# some platforms.
lwda_find_library_local_first(LWDA_LWDA_LIBRARY lwca "\"lwca\" library (older versions only).")

mark_as_advanced(
  LWDA_LWDA_LIBRARY
  LWDA_LWDART_LIBRARY
  )

#######################
# Look for some of the toolkit helper libraries
macro(FIND_LWDA_HELPER_LIBS _name)
  lwda_find_library_local_first(LWDA_${_name}_LIBRARY ${_name} "\"${_name}\" library")
  mark_as_advanced(LWDA_${_name}_LIBRARY)
endmacro()

#######################
# Disable emulation for v3.1 onward
if(LWDA_VERSION VERSION_GREATER "3.0")
  if(LWDA_BUILD_EMULATION)
    message(FATAL_ERROR "LWDA_BUILD_EMULATION is not supported in version 3.1 and onwards.  You must disable it to proceed.  You have version ${LWDA_VERSION}.")
  endif()
endif()

# Search for additional LWCA toolkit libraries.
if(LWDA_VERSION VERSION_LESS "3.1")
  # Emulation libraries aren't available in version 3.1 onward.
  find_lwda_helper_libs(lwfftemu)
  find_lwda_helper_libs(lwblasemu)
endif()
find_lwda_helper_libs(lwfft)
find_lwda_helper_libs(lwblas)
if(NOT LWDA_VERSION VERSION_LESS "3.2")
  # lwsparse showed up in version 3.2
  find_lwda_helper_libs(lwsparse)
  find_lwda_helper_libs(lwrand)
  if (WIN32)
    find_lwda_helper_libs(lwlwvenc)
    find_lwda_helper_libs(lwlwvid)
  endif()
endif()
if(LWDA_VERSION VERSION_GREATER "5.0")
  find_lwda_helper_libs(lwblas_device)
  # In LWCA 5.5 NPP was splitted onto 3 separate libraries.
  find_lwda_helper_libs(nppc)
  find_lwda_helper_libs(nppi)
  find_lwda_helper_libs(npps)
  set(LWDA_npp_LIBRARY "${LWDA_nppc_LIBRARY};${LWDA_nppi_LIBRARY};${LWDA_npps_LIBRARY}")
elseif(NOT LWDA_VERSION VERSION_LESS "4.0")
  find_lwda_helper_libs(npp)
endif()
if(NOT LWDA_VERSION VERSION_LESS "7.0")
  # lwsolver showed up in version 7.0
  find_lwda_helper_libs(lwsolver)
endif()

if (LWDA_BUILD_EMULATION)
  set(LWDA_LWFFT_LIBRARIES ${LWDA_lwfftemu_LIBRARY})
  set(LWDA_LWBLAS_LIBRARIES ${LWDA_lwblasemu_LIBRARY})
else()
  set(LWDA_LWFFT_LIBRARIES ${LWDA_lwfft_LIBRARY})
  set(LWDA_LWBLAS_LIBRARIES ${LWDA_lwblas_LIBRARY} ${LWDA_lwblas_device_LIBRARY})
endif()

########################
# Look for the SDK stuff.  As of LWCA 3.0 LWSDKLWDA_ROOT has been replaced with
# LWSDKCOMPUTE_ROOT with the old LWCA C contents moved into the C subdirectory
find_path(LWDA_SDK_ROOT_DIR common/inc/lwtil.h
 HINTS
  "$ELW{LWSDKCOMPUTE_ROOT}/C"
  ELW LWSDKLWDA_ROOT
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\LWPU Corporation\\Installed Products\\LWPU SDK 10\\Compute;InstallDir]"
 PATHS
  "/Developer/GPU\ Computing/C"
  )

# Keep the LWDA_SDK_ROOT_DIR first in order to be able to override the
# environment variables.
set(LWDA_SDK_SEARCH_PATH
  "${LWDA_SDK_ROOT_DIR}"
  "${LWDA_TOOLKIT_ROOT_DIR}/local/LWSDK0.2"
  "${LWDA_TOOLKIT_ROOT_DIR}/LWSDK0.2"
  "${LWDA_TOOLKIT_ROOT_DIR}/LW_LWDA_SDK"
  "$ELW{HOME}/LWIDIA_LWDA_SDK"
  "$ELW{HOME}/LWIDIA_LWDA_SDK_MACOSX"
  "/Developer/LWCA"
  )

# Example of how to find an include file from the LWDA_SDK_ROOT_DIR

# find_path(LWDA_LWT_INCLUDE_DIR
#   lwtil.h
#   PATHS ${LWDA_SDK_SEARCH_PATH}
#   PATH_SUFFIXES "common/inc"
#   DOC "Location of lwtil.h"
#   NO_DEFAULT_PATH
#   )
# # Now search system paths
# find_path(LWDA_LWT_INCLUDE_DIR lwtil.h DOC "Location of lwtil.h")

# mark_as_advanced(LWDA_LWT_INCLUDE_DIR)


# Example of how to find a library in the LWDA_SDK_ROOT_DIR

# # lwtil library is called lwtil64 for 64 bit builds on windows.  We don't want
# # to get these confused, so we are setting the name based on the word size of
# # the build.

# if(CMAKE_SIZEOF_VOID_P EQUAL 8)
#   set(lwda_lwtil_name lwtil64)
# else()
#   set(lwda_lwtil_name lwtil32)
# endif()

# find_library(LWDA_LWT_LIBRARY
#   NAMES lwtil ${lwda_lwtil_name}
#   PATHS ${LWDA_SDK_SEARCH_PATH}
#   # The new version of the sdk shows up in common/lib, but the old one is in lib
#   PATH_SUFFIXES "common/lib" "lib"
#   DOC "Location of lwtil library"
#   NO_DEFAULT_PATH
#   )
# # Now search system paths
# find_library(LWDA_LWT_LIBRARY NAMES lwtil ${lwda_lwtil_name} DOC "Location of lwtil library")
# mark_as_advanced(LWDA_LWT_LIBRARY)
# set(LWDA_LWT_LIBRARIES ${LWDA_LWT_LIBRARY})



#############################
# Check for required components
set(LWDA_FOUND TRUE)

set(LWDA_TOOLKIT_ROOT_DIR_INTERNAL "${LWDA_TOOLKIT_ROOT_DIR}" CACHE INTERNAL
  "This is the value of the last time LWDA_TOOLKIT_ROOT_DIR was set successfully." FORCE)
set(LWDA_TOOLKIT_TARGET_DIR_INTERNAL "${LWDA_TOOLKIT_TARGET_DIR}" CACHE INTERNAL
  "This is the value of the last time LWDA_TOOLKIT_TARGET_DIR was set successfully." FORCE)
set(LWDA_SDK_ROOT_DIR_INTERNAL "${LWDA_SDK_ROOT_DIR}" CACHE INTERNAL
  "This is the value of the last time LWDA_SDK_ROOT_DIR was set successfully." FORCE)

#include(${CMAKE_LWRRENT_LIST_DIR}/FindPackageHandleStandardArgs.cmake)
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

find_package_handle_standard_args(LWCA
  REQUIRED_VARS
    LWDA_TOOLKIT_ROOT_DIR
    LWDA_LWCC_EXELWTABLE
    LWDA_INCLUDE_DIRS
    ${LWDA_LWDART_LIBRARY_VAR}
  VERSION_VAR
    LWDA_VERSION
  )



###############################################################################
###############################################################################
# Macros
###############################################################################
###############################################################################

###############################################################################
# Add include directories to pass to the lwcc command.
macro(LWDA_INCLUDE_DIRECTORIES)
  foreach(dir ${ARGN})
    list(APPEND LWDA_LWCC_INCLUDE_ARGS_USER -I${dir})
  endforeach()
endmacro()


##############################################################################
lwda_find_helper_file(parse_lwbin cmake)
lwda_find_helper_file(make2cmake cmake)
lwda_find_helper_file(run_lwcc cmake)
include("${CMAKE_LWRRENT_LIST_DIR}/FindLWDA/select_compute_arch.cmake")

##############################################################################
# Separate the OPTIONS out from the sources
#
macro(LWDA_GET_SOURCES_AND_OPTIONS _sources _cmake_options _options)
  set( ${_sources} )
  set( ${_cmake_options} )
  set( ${_options} )
  set( _found_options FALSE )
  foreach(arg ${ARGN})
    if("x${arg}" STREQUAL "xOPTIONS")
      set( _found_options TRUE )
    elseif(
        "x${arg}" STREQUAL "xWIN32" OR
        "x${arg}" STREQUAL "xMACOSX_BUNDLE" OR
        "x${arg}" STREQUAL "xEXCLUDE_FROM_ALL" OR
        "x${arg}" STREQUAL "xSTATIC" OR
        "x${arg}" STREQUAL "xSHARED" OR
        "x${arg}" STREQUAL "xMODULE"
        )
      list(APPEND ${_cmake_options} ${arg})
    else()
      if ( _found_options )
        list(APPEND ${_options} ${arg})
      else()
        # Assume this is a file
        list(APPEND ${_sources} ${arg})
      endif()
    endif()
  endforeach()
endmacro()

##############################################################################
# Parse the OPTIONS from ARGN and set the variables prefixed by _option_prefix
#
macro(LWDA_PARSE_LWCC_OPTIONS _option_prefix)
  set( _found_config )
  foreach(arg ${ARGN})
    # Determine if we are dealing with a perconfiguration flag
    foreach(config ${LWDA_configuration_types})
      string(TOUPPER ${config} config_upper)
      if (arg STREQUAL "${config_upper}")
        set( _found_config _${arg})
        # Set arg to nothing to keep it from being processed further
        set( arg )
      endif()
    endforeach()

    if ( arg )
      list(APPEND ${_option_prefix}${_found_config} "${arg}")
    endif()
  endforeach()
endmacro()

##############################################################################
# Helper to add the include directory for LWCA only once
function(LWDA_ADD_LWDA_INCLUDE_ONCE)
  get_directory_property(_include_directories INCLUDE_DIRECTORIES)
  set(_add TRUE)
  if(_include_directories)
    foreach(dir ${_include_directories})
      if("${dir}" STREQUAL "${LWDA_INCLUDE_DIRS}")
        set(_add FALSE)
      endif()
    endforeach()
  endif()
  if(_add)
    include_directories(${LWDA_INCLUDE_DIRS})
  endif()
endfunction()

function(LWDA_BUILD_SHARED_LIBRARY shared_flag)
  set(cmake_args ${ARGN})
  # If SHARED, MODULE, or STATIC aren't already in the list of arguments, then
  # add SHARED or STATIC based on the value of BUILD_SHARED_LIBS.
  list(FIND cmake_args SHARED _lwda_found_SHARED)
  list(FIND cmake_args MODULE _lwda_found_MODULE)
  list(FIND cmake_args STATIC _lwda_found_STATIC)
  if( _lwda_found_SHARED GREATER -1 OR
      _lwda_found_MODULE GREATER -1 OR
      _lwda_found_STATIC GREATER -1)
    set(_lwda_build_shared_libs)
  else()
    if (BUILD_SHARED_LIBS)
      set(_lwda_build_shared_libs SHARED)
    else()
      set(_lwda_build_shared_libs STATIC)
    endif()
  endif()
  set(${shared_flag} ${_lwda_build_shared_libs} PARENT_SCOPE)
endfunction()

##############################################################################
# Helper to avoid clashes of files with the same basename but different paths.
# This doesn't attempt to do exactly what CMake internals do, which is to only
# add this path when there is a conflict, since by the time a second collision
# in names is detected it's already too late to fix the first one.  For
# consistency sake the relative path will be added to all files.
function(LWDA_COMPUTE_BUILD_PATH path build_path)
  #message("LWDA_COMPUTE_BUILD_PATH([${path}] ${build_path})")
  # Only deal with CMake style paths from here on out
  file(TO_CMAKE_PATH "${path}" bpath)
  if (IS_ABSOLUTE "${bpath}")
    # Absolute paths are generally unnessary, especially if something like
    # file(GLOB_RELWRSE) is used to pick up the files.

    string(FIND "${bpath}" "${CMAKE_LWRRENT_BINARY_DIR}" _binary_dir_pos)
    if (_binary_dir_pos EQUAL 0)
      file(RELATIVE_PATH bpath "${CMAKE_LWRRENT_BINARY_DIR}" "${bpath}")
    else()
      file(RELATIVE_PATH bpath "${CMAKE_LWRRENT_SOURCE_DIR}" "${bpath}")
    endif()
  endif()

  # This recipe is from cmLocalGenerator::CreateSafeUniqueObjectFileName in the
  # CMake source.

  # Remove leading /
  string(REGEX REPLACE "^[/]+" "" bpath "${bpath}")
  # Avoid absolute paths by removing ':'
  string(REPLACE ":" "_" bpath "${bpath}")
  # Avoid relative paths that go up the tree
  string(REPLACE "../" "__/" bpath "${bpath}")
  # Avoid spaces
  string(REPLACE " " "_" bpath "${bpath}")

  # Strip off the filename.  I wait until here to do it, since removin the
  # basename can make a path that looked like path/../basename turn into
  # path/.. (notice the trailing slash).
  get_filename_component(bpath "${bpath}" PATH)

  set(${build_path} "${bpath}" PARENT_SCOPE)
  #message("${build_path} = ${bpath}")
endfunction()

##############################################################################
# This helper macro populates the following variables and setups up custom
# commands and targets to ilwoke the lwcc compiler to generate C or PTX source
# dependent upon the format parameter.  The compiler is ilwoked once with -M
# to generate a dependency file and a second time with -lwca or -ptx to generate
# a .cpp or .ptx file.
# INPUT:
#   lwda_target         - Target name
#   format              - PTX, LWBIN, FATBIN or OBJ
#   FILE1 .. FILEN      - The remaining arguments are the sources to be wrapped.
#   OPTIONS             - Extra options to LWCC
# OUTPUT:
#   generated_files     - List of generated files
##############################################################################
##############################################################################

macro(LWDA_WRAP_SRCS lwda_target format generated_files)

  # If CMake doesn't support separable compilation, complain
  if(LWDA_SEPARABLE_COMPILATION AND CMAKE_VERSION VERSION_LESS "2.8.10.1")
    message(SEND_ERROR "LWDA_SEPARABLE_COMPILATION isn't supported for CMake versions less than 2.8.10.1")
  endif()

  # Set up all the command line flags here, so that they can be overridden on a per target basis.

  set(lwcc_flags "")

  # Emulation if the card isn't present.
  if (LWDA_BUILD_EMULATION)
    # Emulation.
    set(lwcc_flags ${lwcc_flags} --device-emulation -D_DEVICEEMU -g)
  else()
    # Device mode.  No flags necessary.
  endif()

  if(LWDA_HOST_COMPILATION_CPP)
    set(LWDA_C_OR_CXX CXX)
  else()
    if(LWDA_VERSION VERSION_LESS "3.0")
      set(lwcc_flags ${lwcc_flags} --host-compilation C)
    else()
      message(WARNING "--host-compilation flag is deprecated in LWCA version >= 3.0.  Removing --host-compilation C flag" )
    endif()
    set(LWDA_C_OR_CXX C)
  endif()

  set(generated_extension ${CMAKE_${LWDA_C_OR_CXX}_OUTPUT_EXTENSION})

  if(LWDA_64_BIT_DEVICE_CODE)
    set(lwcc_flags ${lwcc_flags} -m64)
  else()
    set(lwcc_flags ${lwcc_flags} -m32)
  endif()

  if(LWDA_TARGET_CPU_ARCH)
    set(lwcc_flags ${lwcc_flags} "--target-cpu-architecture=${LWDA_TARGET_CPU_ARCH}")
  endif()

  # This needs to be passed in at this stage, because VS needs to fill out the
  # value of VCInstallDir from within VS.  Note that CCBIN is only used if
  # -ccbin or --compiler-bindir isn't used and LWDA_HOST_COMPILER matches
  # $(VCInstallDir)/bin.
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set(ccbin_flags -D "\"CCBIN:PATH=$(VCInstallDir)bin\"" )
  else()
    set(ccbin_flags)
  endif()

  # Figure out which configure we will use and pass that in as an argument to
  # the script.  We need to defer the decision until compilation time, because
  # for VS projects we won't know if we are making a debug or release build
  # until build time.
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    set( LWDA_build_configuration "$(ConfigurationName)" )
  else()
    set( LWDA_build_configuration "${CMAKE_BUILD_TYPE}")
  endif()

  # Initialize our list of includes with the user ones followed by the LWCA system ones.
  set(LWDA_LWCC_INCLUDE_ARGS ${LWDA_LWCC_INCLUDE_ARGS_USER} "-I${LWDA_INCLUDE_DIRS}")
  # Get the include directories for this directory and use them for our lwcc command.
  # Remove duplicate entries which may be present since include_directories
  # in CMake >= 2.8.8 does not remove them.
  get_directory_property(LWDA_LWCC_INCLUDE_DIRECTORIES INCLUDE_DIRECTORIES)
  list(REMOVE_DUPLICATES LWDA_LWCC_INCLUDE_DIRECTORIES)
  if(LWDA_LWCC_INCLUDE_DIRECTORIES)
    foreach(dir ${LWDA_LWCC_INCLUDE_DIRECTORIES})
      list(APPEND LWDA_LWCC_INCLUDE_ARGS -I${dir})
    endforeach()
  endif()

  # Reset these variables
  set(LWDA_WRAP_OPTION_LWCC_FLAGS)
  foreach(config ${LWDA_configuration_types})
    string(TOUPPER ${config} config_upper)
    set(LWDA_WRAP_OPTION_LWCC_FLAGS_${config_upper})
  endforeach()

  LWDA_GET_SOURCES_AND_OPTIONS(_lwda_wrap_sources _lwda_wrap_cmake_options _lwda_wrap_options ${ARGN})
  LWDA_PARSE_LWCC_OPTIONS(LWDA_WRAP_OPTION_LWCC_FLAGS ${_lwda_wrap_options})

  # Figure out if we are building a shared library.  BUILD_SHARED_LIBS is
  # respected in LWDA_ADD_LIBRARY.
  set(_lwda_build_shared_libs FALSE)
  # SHARED, MODULE
  list(FIND _lwda_wrap_cmake_options SHARED _lwda_found_SHARED)
  list(FIND _lwda_wrap_cmake_options MODULE _lwda_found_MODULE)
  if(_lwda_found_SHARED GREATER -1 OR _lwda_found_MODULE GREATER -1)
    set(_lwda_build_shared_libs TRUE)
  endif()
  # STATIC
  list(FIND _lwda_wrap_cmake_options STATIC _lwda_found_STATIC)
  if(_lwda_found_STATIC GREATER -1)
    set(_lwda_build_shared_libs FALSE)
  endif()

  # LWDA_HOST_FLAGS
  if(_lwda_build_shared_libs)
    # If we are setting up code for a shared library, then we need to add extra flags for
    # compiling objects for shared libraries.
    set(LWDA_HOST_SHARED_FLAGS ${CMAKE_SHARED_LIBRARY_${LWDA_C_OR_CXX}_FLAGS})
  else()
    set(LWDA_HOST_SHARED_FLAGS)
  endif()
  # Only add the CMAKE_{C,CXX}_FLAGS if we are propagating host flags.  We
  # always need to set the SHARED_FLAGS, though.
  if(LWDA_PROPAGATE_HOST_FLAGS)
    set(_lwda_host_flags "set(CMAKE_HOST_FLAGS ${CMAKE_${LWDA_C_OR_CXX}_FLAGS} ${LWDA_HOST_SHARED_FLAGS})")
  else()
    set(_lwda_host_flags "set(CMAKE_HOST_FLAGS ${LWDA_HOST_SHARED_FLAGS})")
  endif()

  set(_lwda_lwcc_flags_config "# Build specific configuration flags")
  # Loop over all the configuration types to generate appropriate flags for run_lwcc.cmake
  foreach(config ${LWDA_configuration_types})
    string(TOUPPER ${config} config_upper)
    # CMAKE_FLAGS are strings and not lists.  By not putting quotes around CMAKE_FLAGS
    # we colwert the strings to lists (like we want).

    if(LWDA_PROPAGATE_HOST_FLAGS)
      # lwcc chokes on -g3 in versions previous to 3.0, so replace it with -g
      set(_lwda_fix_g3 FALSE)

      if(CMAKE_COMPILER_IS_GNUCC)
        if (LWDA_VERSION VERSION_LESS  "3.0" OR
            LWDA_VERSION VERSION_EQUAL "4.1" OR
            LWDA_VERSION VERSION_EQUAL "4.2"
            )
          set(_lwda_fix_g3 TRUE)
        endif()
      endif()
      if(_lwda_fix_g3)
        string(REPLACE "-g3" "-g" _lwda_C_FLAGS "${CMAKE_${LWDA_C_OR_CXX}_FLAGS_${config_upper}}")
      else()
        set(_lwda_C_FLAGS "${CMAKE_${LWDA_C_OR_CXX}_FLAGS_${config_upper}}")
      endif()

      set(_lwda_host_flags "${_lwda_host_flags}\nset(CMAKE_HOST_FLAGS_${config_upper} ${_lwda_C_FLAGS})")
    endif()

    # Note that if we ever want LWDA_LWCC_FLAGS_<CONFIG> to be string (instead of a list
    # like it is lwrrently), we can remove the quotes around the
    # ${LWDA_LWCC_FLAGS_${config_upper}} variable like the CMAKE_HOST_FLAGS_<CONFIG> variable.
    set(_lwda_lwcc_flags_config "${_lwda_lwcc_flags_config}\nset(LWDA_LWCC_FLAGS_${config_upper} ${LWDA_LWCC_FLAGS_${config_upper}} ;; ${LWDA_WRAP_OPTION_LWCC_FLAGS_${config_upper}})")
  endforeach()

  # Process the C++11 flag.  If the host sets the flag, we need to add it to lwcc and
  # remove it from the host. This is because -Xcompile -std=c++ will choke lwcc (it uses
  # the C preprocessor).  In order to get this to work correctly, we need to use lwcc's
  # specific c++11 flag.
  if( "${_lwda_host_flags}" MATCHES "-std=c\\+\\+11")
    # Add the c++11 flag to lwcc if it isn't already present.  Note that we only look at
    # the main flag instead of the configuration specific flags.
    if( NOT "${LWDA_LWCC_FLAGS}" MATCHES "-std;c\\+\\+11" )
      list(APPEND lwcc_flags --std c++11)
    endif()
    string(REGEX REPLACE "[-]+std=c\\+\\+11" "" _lwda_host_flags "${_lwda_host_flags}")
  endif()

  # Get the list of definitions from the directory property
  get_directory_property(LWDA_LWCC_DEFINITIONS COMPILE_DEFINITIONS)
  if(LWDA_LWCC_DEFINITIONS)
    foreach(_definition ${LWDA_LWCC_DEFINITIONS})
      list(APPEND lwcc_flags "-D${_definition}")
    endforeach()
  endif()

  if(_lwda_build_shared_libs)
    list(APPEND lwcc_flags "-D${lwda_target}_EXPORTS")
  endif()

  # Reset the output variable
  set(_lwda_wrap_generated_files "")

  # Iterate over the macro arguments and create custom
  # commands for all the .lw files.
  foreach(file ${ARGN})
    # Ignore any file marked as a HEADER_FILE_ONLY
    get_source_file_property(_is_header ${file} HEADER_FILE_ONLY)
    # Allow per source file overrides of the format.  Also allows compiling non-.lw files.
    get_source_file_property(_lwda_source_format ${file} LWDA_SOURCE_PROPERTY_FORMAT)
    if((${file} MATCHES "\\.lw$" OR _lwda_source_format) AND NOT _is_header)

      if(NOT _lwda_source_format)
        set(_lwda_source_format ${format})
      endif()
      # If file isn't a .lw file, we need to tell lwcc to treat it as such.
      if(NOT ${file} MATCHES "\\.lw$")
        set(lwda_language_flag -x=lw)
      else()
        set(lwda_language_flag)
      endif()

      if( ${_lwda_source_format} MATCHES "OBJ")
        set( lwda_compile_to_external_module OFF )
      else()
        set( lwda_compile_to_external_module ON )
        if( ${_lwda_source_format} MATCHES "PTX" )
          set( lwda_compile_to_external_module_type "ptx" )
        elseif( ${_lwda_source_format} MATCHES "LWBIN")
          set( lwda_compile_to_external_module_type "lwbin" )
        elseif( ${_lwda_source_format} MATCHES "FATBIN")
          set( lwda_compile_to_external_module_type "fatbin" )
        else()
          message( FATAL_ERROR "Invalid format flag passed to LWDA_WRAP_SRCS or set with LWDA_SOURCE_PROPERTY_FORMAT file property for file '${file}': '${_lwda_source_format}'.  Use OBJ, PTX, LWBIN or FATBIN.")
        endif()
      endif()

      if(lwda_compile_to_external_module)
        # Don't use any of the host compilation flags for PTX targets.
        set(LWDA_HOST_FLAGS)
        set(LWDA_LWCC_FLAGS_CONFIG)
      else()
        set(LWDA_HOST_FLAGS ${_lwda_host_flags})
        set(LWDA_LWCC_FLAGS_CONFIG ${_lwda_lwcc_flags_config})
      endif()

      # Determine output directory
      lwda_compute_build_path("${file}" lwda_build_path)
      set(lwda_compile_intermediate_directory "${CMAKE_LWRRENT_BINARY_DIR}/CMakeFiles/${lwda_target}.dir/${lwda_build_path}")
      if(LWDA_GENERATED_OUTPUT_DIR)
        set(lwda_compile_output_dir "${LWDA_GENERATED_OUTPUT_DIR}")
      else()
        if ( lwda_compile_to_external_module )
          set(lwda_compile_output_dir "${CMAKE_LWRRENT_BINARY_DIR}")
        else()
          set(lwda_compile_output_dir "${lwda_compile_intermediate_directory}")
        endif()
      endif()

      # Add a custom target to generate a c or ptx file. ######################

      get_filename_component( basename ${file} NAME )
      if( lwda_compile_to_external_module )
        set(generated_file_path "${lwda_compile_output_dir}")
        set(generated_file_basename "${lwda_target}_generated_${basename}.${lwda_compile_to_external_module_type}")
        set(format_flag "-${lwda_compile_to_external_module_type}")
        file(MAKE_DIRECTORY "${lwda_compile_output_dir}")
      else()
        set(generated_file_path "${lwda_compile_output_dir}/${CMAKE_CFG_INTDIR}")
        set(generated_file_basename "${lwda_target}_generated_${basename}${generated_extension}")
        if(LWDA_SEPARABLE_COMPILATION)
          set(format_flag "-dc")
        else()
          set(format_flag "-c")
        endif()
      endif()

      # Set all of our file names.  Make sure that whatever filenames that have
      # generated_file_path in them get passed in through as a command line
      # argument, so that the ${CMAKE_CFG_INTDIR} gets expanded at run time
      # instead of configure time.
      set(generated_file "${generated_file_path}/${generated_file_basename}")
      set(cmake_dependency_file "${lwda_compile_intermediate_directory}/${generated_file_basename}.depend")
      set(LWCC_generated_dependency_file "${lwda_compile_intermediate_directory}/${generated_file_basename}.LWCC-depend")
      set(generated_lwbin_file "${generated_file_path}/${generated_file_basename}.lwbin.txt")
      set(lwstom_target_script "${lwda_compile_intermediate_directory}/${generated_file_basename}.cmake")

      # Setup properties for obj files:
      if( NOT lwda_compile_to_external_module )
        set_source_files_properties("${generated_file}"
          PROPERTIES
          EXTERNAL_OBJECT true # This is an object file not to be compiled, but only be linked.
          )
      endif()

      # Don't add CMAKE_LWRRENT_SOURCE_DIR if the path is already an absolute path.
      get_filename_component(file_path "${file}" PATH)
      if(IS_ABSOLUTE "${file_path}")
        set(source_file "${file}")
      else()
        set(source_file "${CMAKE_LWRRENT_SOURCE_DIR}/${file}")
      endif()

      if( NOT lwda_compile_to_external_module AND LWDA_SEPARABLE_COMPILATION)
        list(APPEND ${lwda_target}_SEPARABLE_COMPILATION_OBJECTS "${generated_file}")
      endif()

      # Bring in the dependencies.  Creates a variable LWDA_LWCC_DEPEND #######
      lwda_include_lwcc_dependencies(${cmake_dependency_file})

      # Colwience string for output ###########################################
      if(LWDA_BUILD_EMULATION)
        set(lwda_build_type "Emulation")
      else()
        set(lwda_build_type "Device")
      endif()

      # Build the LWCC made dependency file ###################################
      set(build_lwbin OFF)
      if ( NOT LWDA_BUILD_EMULATION AND LWDA_BUILD_LWBIN )
         if ( NOT lwda_compile_to_external_module )
           set ( build_lwbin ON )
         endif()
      endif()

      # Configure the build script
      configure_file("${LWDA_run_lwcc}" "${lwstom_target_script}" @ONLY)


      # So if a user specifies the same lwca file as input more than once, you
      # can have bad things happen with dependencies.  Here we check an option
      # to see if this is the behavior they want.
      if(LWDA_ATTACH_VS_BUILD_RULE_TO_LWDA_FILE)
        set(main_dep MAIN_DEPENDENCY ${source_file})
      else()
        set(main_dep DEPENDS ${source_file})
      endif()

      if(LWDA_VERBOSE_BUILD)
        set(verbose_output ON)
      elseif(CMAKE_GENERATOR MATCHES "Makefiles")
        set(verbose_output "$(VERBOSE)")
      else()
        set(verbose_output OFF)
      endif()

      # Create up the comment string
      file(RELATIVE_PATH generated_file_relative_path "${CMAKE_BINARY_DIR}" "${generated_file}")
      if(lwda_compile_to_external_module)
        set(lwda_build_comment_string "Building LWCC ${lwda_compile_to_external_module_type} file ${generated_file_relative_path}")
      else()
        set(lwda_build_comment_string "Building LWCC (${lwda_build_type}) object ${generated_file_relative_path}")
      endif()

      set(_verbatim VERBATIM)
      if(ccbin_flags MATCHES "\\$\\(VCInstallDir\\)")
        set(_verbatim "")
      endif()

      # Build the generated file and dependency file ##########################
      add_lwstom_command(
        OUTPUT ${generated_file}
        # These output files depend on the source_file and the contents of cmake_dependency_file
        ${main_dep}
        DEPENDS ${LWDA_LWCC_DEPEND}
        DEPENDS ${lwstom_target_script}
        # Make sure the output directory exists before trying to write to it.
        COMMAND ${CMAKE_COMMAND} -E make_directory "${generated_file_path}"
        COMMAND ${CMAKE_COMMAND} ARGS
          -D verbose:BOOL=${verbose_output}
          ${ccbin_flags}
          -D build_configuration:STRING=${LWDA_build_configuration}
          -D "generated_file:STRING=${generated_file}"
          -D "generated_lwbin_file:STRING=${generated_lwbin_file}"
          -P "${lwstom_target_script}"
        WORKING_DIRECTORY "${lwda_compile_intermediate_directory}"
        COMMENT "${lwda_build_comment_string}"
        ${_verbatim}
        )

      # Make sure the build system knows the file is generated.
      set_source_files_properties(${generated_file} PROPERTIES GENERATED TRUE)

      list(APPEND _lwda_wrap_generated_files ${generated_file})

      # Add the other files that we want cmake to clean on a cleanup ##########
      list(APPEND LWDA_ADDITIONAL_CLEAN_FILES "${cmake_dependency_file}")
      list(REMOVE_DUPLICATES LWDA_ADDITIONAL_CLEAN_FILES)
      set(LWDA_ADDITIONAL_CLEAN_FILES ${LWDA_ADDITIONAL_CLEAN_FILES} CACHE INTERNAL "List of intermediate files that are part of the lwca dependency scanning.")

    endif()
  endforeach()

  # Set the return parameter
  set(${generated_files} ${_lwda_wrap_generated_files})
endmacro()

function(_lwda_get_important_host_flags important_flags flag_string)
  if(CMAKE_GENERATOR MATCHES "Visual Studio")
    string(REGEX MATCHALL "/M[DT][d]?" flags "${flag_string}")
    list(APPEND ${important_flags} ${flags})
  else()
    string(REGEX MATCHALL "-fPIC" flags "${flag_string}")
    list(APPEND ${important_flags} ${flags})
  endif()
  set(${important_flags} ${${important_flags}} PARENT_SCOPE)
endfunction()

###############################################################################
###############################################################################
# Separable Compilation Link
###############################################################################
###############################################################################

# Compute the filename to be used by LWDA_LINK_SEPARABLE_COMPILATION_OBJECTS
function(LWDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME output_file_var lwda_target object_files)
  if (object_files)
    set(generated_extension ${CMAKE_${LWDA_C_OR_CXX}_OUTPUT_EXTENSION})
    set(output_file "${CMAKE_LWRRENT_BINARY_DIR}/CMakeFiles/${lwda_target}.dir/${CMAKE_CFG_INTDIR}/${lwda_target}_intermediate_link${generated_extension}")
  else()
    set(output_file)
  endif()

  set(${output_file_var} "${output_file}" PARENT_SCOPE)
endfunction()

# Setup the build rule for the separable compilation intermediate link file.
function(LWDA_LINK_SEPARABLE_COMPILATION_OBJECTS output_file lwda_target options object_files)
  if (object_files)

    set_source_files_properties("${output_file}"
      PROPERTIES
      EXTERNAL_OBJECT TRUE # This is an object file not to be compiled, but only
                           # be linked.
      GENERATED TRUE       # This file is generated during the build
      )

    # For now we are ignoring all the configuration specific flags.
    set(lwcc_flags)
    LWDA_PARSE_LWCC_OPTIONS(lwcc_flags ${options})
    if(LWDA_64_BIT_DEVICE_CODE)
      list(APPEND lwcc_flags -m64)
    else()
      list(APPEND lwcc_flags -m32)
    endif()
    # If -ccbin, --compiler-bindir has been specified, don't do anything.  Otherwise add it here.
    list( FIND lwcc_flags "-ccbin" ccbin_found0 )
    list( FIND lwcc_flags "--compiler-bindir" ccbin_found1 )
    if( ccbin_found0 LESS 0 AND ccbin_found1 LESS 0 AND LWDA_HOST_COMPILER )
      # Match VERBATIM check below.
      if(LWDA_HOST_COMPILER MATCHES "\\$\\(VCInstallDir\\)")
        list(APPEND lwcc_flags -ccbin "\"${LWDA_HOST_COMPILER}\"")
      else()
        list(APPEND lwcc_flags -ccbin "${LWDA_HOST_COMPILER}")
      endif()
    endif()

    # Create a list of flags specified by LWDA_LWCC_FLAGS_${CONFIG} and CMAKE_${LWDA_C_OR_CXX}_FLAGS*
    set(config_specific_flags)
    set(flags)
    foreach(config ${LWDA_configuration_types})
      string(TOUPPER ${config} config_upper)
      # Add config specific flags
      foreach(f ${LWDA_LWCC_FLAGS_${config_upper}})
        list(APPEND config_specific_flags $<$<CONFIG:${config}>:${f}>)
      endforeach()
      set(important_host_flags)
      _lwda_get_important_host_flags(important_host_flags "${CMAKE_${LWDA_C_OR_CXX}_FLAGS_${config_upper}}")
      foreach(f ${important_host_flags})
        list(APPEND flags $<$<CONFIG:${config}>:-Xcompiler> $<$<CONFIG:${config}>:${f}>)
      endforeach()
    endforeach()
    # Add CMAKE_${LWDA_C_OR_CXX}_FLAGS
    set(important_host_flags)
    _lwda_get_important_host_flags(important_host_flags "${CMAKE_${LWDA_C_OR_CXX}_FLAGS}")
    foreach(f ${important_host_flags})
      list(APPEND flags -Xcompiler ${f})
    endforeach()

    # Add our general LWDA_LWCC_FLAGS with the configuration specifig flags
    set(lwcc_flags ${LWDA_LWCC_FLAGS} ${config_specific_flags} ${lwcc_flags})

    file(RELATIVE_PATH output_file_relative_path "${CMAKE_BINARY_DIR}" "${output_file}")

    # Some generators don't handle the multiple levels of custom command
    # dependencies correctly (obj1 depends on file1, obj2 depends on obj1), so
    # we work around that issue by compiling the intermediate link object as a
    # pre-link custom command in that situation.
    set(do_obj_build_rule TRUE)
    if (MSVC_VERSION GREATER 1599 AND MSVC_VERSION LESS 1800)
      # VS 2010 and 2012 have this problem.
      set(do_obj_build_rule FALSE)
    endif()

    set(_verbatim VERBATIM)
    if(lwcc_flags MATCHES "\\$\\(VCInstallDir\\)")
      set(_verbatim "")
    endif()

    if (do_obj_build_rule)
      add_lwstom_command(
        OUTPUT ${output_file}
        DEPENDS ${object_files}
        COMMAND ${LWDA_LWCC_EXELWTABLE} ${lwcc_flags} -dlink ${object_files} -o ${output_file}
        ${flags}
        COMMENT "Building LWCC intermediate link file ${output_file_relative_path}"
        ${_verbatim}
        )
    else()
      get_filename_component(output_file_dir "${output_file}" DIRECTORY)
      add_lwstom_command(
        TARGET ${lwda_target}
        PRE_LINK
        COMMAND ${CMAKE_COMMAND} -E echo "Building LWCC intermediate link file ${output_file_relative_path}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${output_file_dir}"
        COMMAND ${LWDA_LWCC_EXELWTABLE} ${lwcc_flags} ${flags} -dlink ${object_files} -o "${output_file}"
        ${_verbatim}
        )
    endif()
 endif()
endfunction()

###############################################################################
###############################################################################
# ADD LIBRARY
###############################################################################
###############################################################################
macro(LWDA_ADD_LIBRARY lwda_target)

  LWDA_ADD_LWDA_INCLUDE_ONCE()

  # Separate the sources from the options
  LWDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  LWDA_BUILD_SHARED_LIBRARY(_lwda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  LWDA_WRAP_SRCS( ${lwda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_lwda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  LWDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${lwda_target} "${${lwda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${lwda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${lwda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  LWDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${lwda_target} "${_options}" "${${lwda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${lwda_target}
    ${LWDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. LWDA_C_OR_CXX is computed based on LWDA_HOST_COMPILATION_CPP.
  set_target_properties(${lwda_target}
    PROPERTIES
    LINKER_LANGUAGE ${LWDA_C_OR_CXX}
    )

endmacro()


###############################################################################
###############################################################################
# ADD EXELWTABLE
###############################################################################
###############################################################################
macro(LWDA_ADD_EXELWTABLE lwda_target)

  LWDA_ADD_LWDA_INCLUDE_ONCE()

  # Separate the sources from the options
  LWDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  LWDA_WRAP_SRCS( ${lwda_target} OBJ _generated_files ${_sources} OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  LWDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(link_file ${lwda_target} "${${lwda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_exelwtable(${lwda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${lwda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  LWDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${link_file}" ${lwda_target} "${_options}" "${${lwda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${lwda_target}
    ${LWDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. LWDA_C_OR_CXX is computed based on LWDA_HOST_COMPILATION_CPP.
  set_target_properties(${lwda_target}
    PROPERTIES
    LINKER_LANGUAGE ${LWDA_C_OR_CXX}
    )

endmacro()


###############################################################################
###############################################################################
# (Internal) helper for manually added lwca source files with specific targets
###############################################################################
###############################################################################
macro(lwda_compile_base lwda_target format generated_files)

  # Separate the sources from the options
  LWDA_GET_SOURCES_AND_OPTIONS(_sources _cmake_options _options ${ARGN})
  # Create custom commands and targets for each file.
  LWDA_WRAP_SRCS( ${lwda_target} ${format} _generated_files ${_sources} ${_cmake_options}
    OPTIONS ${_options} )

  set( ${generated_files} ${_generated_files})

endmacro()

###############################################################################
###############################################################################
# LWCA COMPILE
###############################################################################
###############################################################################
macro(LWDA_COMPILE generated_files)
  lwda_compile_base(lwda_compile OBJ ${generated_files} ${ARGN})
endmacro()

###############################################################################
###############################################################################
# LWCA COMPILE PTX
###############################################################################
###############################################################################
macro(LWDA_COMPILE_PTX generated_files)
  lwda_compile_base(lwda_compile_ptx PTX ${generated_files} ${ARGN})
endmacro()

###############################################################################
###############################################################################
# LWCA COMPILE FATBIN
###############################################################################
###############################################################################
macro(LWDA_COMPILE_FATBIN generated_files)
  lwda_compile_base(lwda_compile_fatbin FATBIN ${generated_files} ${ARGN})
endmacro()

###############################################################################
###############################################################################
# LWCA COMPILE LWBIN
###############################################################################
###############################################################################
macro(LWDA_COMPILE_LWBIN generated_files)
  lwda_compile_base(lwda_compile_lwbin LWBIN ${generated_files} ${ARGN})
endmacro()


###############################################################################
###############################################################################
# LWCA ADD LWFFT TO TARGET
###############################################################################
###############################################################################
macro(LWDA_ADD_LWFFT_TO_TARGET target)
  if (LWDA_BUILD_EMULATION)
    target_link_libraries(${target} ${LWDA_lwfftemu_LIBRARY})
  else()
    target_link_libraries(${target} ${LWDA_lwfft_LIBRARY})
  endif()
endmacro()

###############################################################################
###############################################################################
# LWCA ADD LWBLAS TO TARGET
###############################################################################
###############################################################################
macro(LWDA_ADD_LWBLAS_TO_TARGET target)
  if (LWDA_BUILD_EMULATION)
    target_link_libraries(${target} ${LWDA_lwblasemu_LIBRARY})
  else()
    target_link_libraries(${target} ${LWDA_lwblas_LIBRARY} ${LWDA_lwblas_device_LIBRARY})
  endif()
endmacro()

###############################################################################
###############################################################################
# LWCA BUILD CLEAN TARGET
###############################################################################
###############################################################################
macro(LWDA_BUILD_CLEAN_TARGET)
  # Call this after you add all your LWCA targets, and you will get a colwience
  # target.  You should also make clean after running this target to get the
  # build system to generate all the code again.

  set(lwda_clean_target_name clean_lwda_depends)
  if (CMAKE_GENERATOR MATCHES "Visual Studio")
    string(TOUPPER ${lwda_clean_target_name} lwda_clean_target_name)
  endif()
  add_lwstom_target(${lwda_clean_target_name}
    COMMAND ${CMAKE_COMMAND} -E remove ${LWDA_ADDITIONAL_CLEAN_FILES})

  # Clear out the variable, so the next time we configure it will be empty.
  # This is useful so that the files won't persist in the list after targets
  # have been removed.
  set(LWDA_ADDITIONAL_CLEAN_FILES "" CACHE INTERNAL "List of intermediate files that are part of the lwca dependency scanning.")
endmacro()
