dnl
dnl Definitions for creating shared libraries
dnl
dnl The purpose of these definitions is to provide common support for 
dnl shared libraries, with *or without* the use of the GNU Libtool package.
dnl For many of our important platforms, the Libtool approach is overkill,
dnl and can be partilwlarly painful for developers.
dnl
dnl To use libtool, you need macros that are defined by libtool for libtool
dnl Don't even think about the consequences of this for updating and for
dnl using user-versions of libtool :(
dnl 
dnl !!!!!!!!!!!!!!!!!!!!!
dnl libtool requires ac 2.50 !!!!!!!!!!!!!!!!!
dnl 
dnl builtin(include,libtool.m4)
dnl
dnl/*D
dnl PAC_ARG_SHAREDLIBS - Add --enable-sharedlibs=kind to configure.
dnl 
dnl Synopsis:
dnl PAC_ARG_SHAREDLIBS
dnl
dnl Output effects:
dnl Adds '--enable-sharedlibs=kind' to the command line.  If this is enabled,
dnl then based on the value of 'kind', programs are selected for the 
dnl names 'CC_SHL' and 'CC_LINK_SHL' that configure will substitute for in
dnl 'Makefile.in's.  These symbols are generated by 'simplemake' when
dnl shared library support is selected.
dnl The variable 'C_LINKPATH_SHL' is set to the option to specify the 
dnl path to search at runtime for libraries (-rpath in gcc/GNU ld).
dnl This can be turned off with --disable-rpath , which is appropriate
dnl for libraries and for exelwtables that may be installed in different
dnl locations.
dnl The variable 'SHLIB_EXT' is set to the extension used by shared 
dnl libraries; under most forms of Unix, this is 'so'; under Mac OS/X, this
dnl is 'dylib', and under Windows (including cygwin), this is 'dll'.
dnl
dnl Supported values of 'kind' include \:
dnl+    gcc - Use gcc to create both shared objects and libraries
dnl.    osx-gcc - Use gcc on Mac OS/X to create both shared objects and
dnl               libraries
dnl.    solaris-cc - Use native Solaris cc to create shared objects and 
dnl               libraries
dnl.    cygwin-gcc - Use gcc on Cygwin to create shared objects and libraries
dnl-    none - The same as '--disable-sharedlibs'
dnl
dnl Others will be added as experience dictates.  Likely names are
dnl + libtool - For general GNU libtool
dnl - linux-pgcc - For Portland group under Linux
dnl
dnl Notes:
dnl Shared libraries are only partially implemented.  Additional symbols
dnl will probably be defined, including symbols to specify how shared library
dnl search paths are specified and how shared library names are set.
dnl D*/
AC_DEFUN([PAC_ARG_SHAREDLIBS],[

AC_ARG_ENABLE(shared,
	AC_HELP_STRING([--enable-shared], [Enable shared library builds]),,
	enable_shared=no)

AC_ARG_ENABLE(rpath,
	AC_HELP_STRING([--enable-rpath],
		[Determine whether the rpath is set when programs are
		compiled and linked when shared libraries are built.
		The default is yes; use --disable-rpath to turn this
		feature off; in that case, shared libraries will be
		found according to the rules for your system (e.g., in
		LD_LIBRARY_PATH)]),,enable_rpath=yes)

AC_ARG_ENABLE(sharedlibs,
[  --enable-sharedlibs=kind - Enable shared libraries.  kind may be
        gcc     - Standard gcc and GNU ld options for creating shared libraries
        osx-gcc - Special options for gcc needed only on OS/X
        solaris-cc - Solaris native (SPARC) compilers for 32 bit systems
        cygwin-gcc - Special options for gcc needed only for cygwin
        none    - same as --disable-sharedlibs
      Only gcc, osx-gcc, and solaris-cc are lwrrently supported
],,enable_sharedlibs=default)

if test "$enable_sharedlibs" = "default" ; then
   if test "$enable_shared" = "yes" ; then
      AS_CASE([$host],
	      [*-*-darwin*], [enable_sharedlibs=gcc-osx],
	      [*-*-cygwin*|*-*-mingw*|*-*-pw32*|*-*-cegcc*], [enable_sharedlibs=cygwin-gcc],
	      [*-*-sunos*], [enable_sharedlibs=solaris-gcc],
	      [enable_sharedlibs=gcc])
   else
      enable_sharedlibs=none
   fi
fi

# If --enable-sharedlibs is given, but --enable-shared is not, throw
# an error
if test "$enable_sharedlibs" != "no" -a "$enable_sharedlibs" != "none" ; then
   if test "$enable_shared" = "no" ; then
      AC_MSG_ERROR([--enable-sharedlibs cannot be used without --enable-shared])
   fi
fi

CC_SHL=true
C_LINK_SHL=true
C_LINKPATH_SHL=""
SHLIB_EXT=unknown
SHLIB_FROM_LO=no
SHLIB_INSTALL='$(INSTALL_PROGRAM)'
case "$enable_sharedlibs" in 
    no|none)
    ;;
    gcc-osx|osx-gcc)
    AC_MSG_RESULT([Creating shared libraries using GNU for Mac OSX])
    C_LINK_SHL='${CC} -dynamiclib -undefined suppress -single_module -flat_namespace'
    CC_SHL='${CC} -fPIC'
    # No way in osx to specify the location of the shared libraries at link
    # time (see the code in createshlib in mpich/src/util)
    # As of 10.5, -Wl,-rpath,dirname should work .  The dirname 
    # must be a single directory, not a colon-separated list (use multiple
    # -Wl,-rpath,path for each of the paths in the list).  However, os x
    # apparently records the library full path, so rpath isn't as useful
    # as it is on other systems
    C_LINKPATH_SHL=""
    SHLIB_EXT="dylib"
    enable_sharedlibs="osx-gcc"
    ;;
    gcc)
    AC_MSG_RESULT([Creating shared libraries using GNU])
    # Not quite right yet.  See mpich/util/makesharedlib
    # Use syntax that works in both Make and the shell
    #C_LINK_SHL='${CC} -shared -Wl,-r'
    C_LINK_SHL='${CC} -shared'
    # For example, include the libname as ${LIBNAME_SHL}
    #C_LINK_SHL='${CC} -shared -Wl,-h,<finallibname>'
    # May need -fPIC .  Test to see which one works.
    for sh_arg in "-fPIC" "-fpic" "-KPIC" ; do
        PAC_C_CHECK_COMPILER_OPTION($sh_arg,works=yes,works=no)
        if test "$works" = "yes" ; then
           CC_SHL="${CC} ${sh_arg}"
	   break
	fi
    done
    if test "$works" != "yes"; then
       AC_MSG_ERROR([Cannot build shared libraries with this compiler])
    fi
    # This used to have -Wl,-rpath earlier, but that causes problems
    # on many systems.
    if test $enable_rpath = "yes" ; then
        C_LINKPATH_SHL="-Wl,-rpath,"
    fi
    SHLIB_EXT=so
    # We need to test that this isn't osx.  The following is a 
    # simple hack
    osname=`uname -s`
    case $osname in 
        *Darwin*|*darwin*)
	AC_MSG_ERROR([You must specify --enable-sharedlibs=osx-gcc for Mac OS/X])
        ;;	
        *CYGWIN*|*cygwin*)
	AC_MSG_ERROR([You must specify --enable-sharedlibs=cygwin-gcc for Cygwin])
	;;
	*SunOS*)
	AC_MSG_ERROR([You must specify --enable-sharedlibs=solaris-gcc for Solaris with gcc])
	;;
    esac
    ;;

    cygwin|cygwin-gcc|gcc-cygwin)
    AC_MSG_RESULT([Creating shared libraries using GNU under CYGWIN])
    C_LINK_SHL='${CC} -shared'
    CC_SHL='${CC}'
    # DLL Libraries need to be in the user's path (!)
    C_LINKPATH_SHL=""
    SHLIB_EXT="dll"
    enable_sharedlibs="cygwin-gcc"
    ;;	

    libtool)
    # set TRY_LIBTOOL to yes to experiment with libtool.  You are on your
    # own - only send fixes, not bug reports.
    if test "$TRY_LIBTOOL" != yes ; then
        AC_MSG_ERROR([Creating shared libraries using libtool not yet supported])
    else
    # Using libtool requires a heavy-weight process to test for 
    # various stuff that libtool needs.  Without this, you'll get a
    # bizarre error message about libtool being unable to find
    # configure.in or configure.ac (!)
        # Libtool expects to see at least enable-shared.
        if test "X$enable_shared" = "X" ; then enable_shared=yes ; fi
	# Initialize libtool
	# This works, but libtool version 2 places the object files
	# in a different place, making it harder to integrate with
	# our base approach.  Disabling for now
	dnl LT_PREREQ([2.2.6])
        dnl LT_INIT([disable-shared])
	AC_MSG_ERROR([To use this test verison, edit aclocal_shl.m4])
        # Likely to be
        # either CC or CC_SHL is libtool $cc
        CC_SHL='${LIBTOOL} --mode=compile ${CC}'
        # CC_LINK_SHL includes the final installation path
        # For many systems, the link may need to include *all* libraries
        # (since many systems don't allow any unsatisfied dependencies)
        # We need to give libtool the .lo file, not the .o files
        SHLIB_FROM_LO=yes
        # We also need to add -no-undefined when the compiler is gcc and
        # we are building under cygwin
        sysname=`uname -s | tr abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ`
        isCygwin=no
        case "$sysname" in 
 	*CYGWIN*) isCygwin=yes ;;
        esac
        if test "$isCygwin" = yes ; then
            C_LINK_SHL='${LIBTOOL} --mode=link ${CC} -no-undefined -rpath ${libdir}'
        else
            C_LINK_SHL='${LIBTOOL} --mode=link ${CC} -rpath ${libdir}'
        fi
	if test $enable_rpath = "yes" ; then
            C_LINKPATH_SHL="-rpath "
        fi
        # We also need a special install process with libtool.  Note that this
        # will also install the static libraries
        SHLIB_INSTALL='$(LIBTOOL) --mode=install $(INSTALL_PROGRAM)'
        # Note we may still need to add
        #'$(LIBTOOL) --mode=finish $(libdir)'
    fi
    ;;
dnl
dnl Other, such as solaris-cc
    solaris|solaris-cc)
    AC_MSG_RESULT([Creating shared libraries using Solaris])
    # pic32 is appropriate for both 32 and 64 bit Solaris
    C_LINK_SHL='${CC} -G -xcode=pic32'
    CC_SHL='${CC} -xcode=pic32'
    if test $enable_rpath = "yes" ; then
        C_LINKPATH_SHL="-R"
    fi
    SHLIB_EXT=so
    enable_sharedlibs="solaris-cc"
    ;;

    solaris-gcc)
    # This is the same as gcc, except for the C_LINKPATH_SHL
    AC_MSG_RESULT([Creating shared libraries using Solaris with gcc])
    C_LINK_SHL='${CC} -shared'
    CC_SHL='${CC} -fPIC'
    if test $enable_rpath = "yes" ; then
        C_LINKPATH_SHL="-R"
    fi
    SHLIB_EXT=so
    enable_sharedlibs="solaris-gcc"
    ;;

    linuxppc-xlc)
    # This is only the beginning of xlc support, thanks to andy@vpac.org
    CC_SHL='${CC} -qmkshrobj'
    # More recent versions allow multiple args, separated by commas
    if test $enable_rpath = "yes" ; then
        C_LINKPATH_SHL="-Wl,-rpath,"
    fi
    #C_LINKPATH_SHL="-Wl,-rpath -Wl,"
    C_LINK_SHL='${CC} -shared -qmkshrobj'
    SHLIB_EXT=so
    # Note that the full line should be more like
    # $CLINKER -shared -qmkshrobj -Wl,-h,$libbase.$slsuffix -o ../shared/$libbase.$slsuffix *.o $OtherLibs
    # for the appropriate values of $libbase and $slsuffix
    # The -h name sets the name of the object; this is necessary to
    # ensure that the dynamic linker can find the proper shared library.
    ;;

    *)
    AC_MSG_ERROR([Unknown value $enable_sharedlibs for enable-sharedlibs.  Values should be gcc or osx-gcc])
    enable_sharedlibs=no
    ;;  
esac
# Check for the shared-library extension
PAC_CC_SHLIB_EXT
dnl
AC_SUBST(CC_SHL)
AC_SUBST(C_LINK_SHL)
AC_SUBST(C_LINKPATH_SHL)
AC_SUBST(SHLIB_EXT)
AC_SUBST(SHLIB_FROM_LO)
AC_SUBST(SHLIB_INSTALL)
])

dnl /*D
dnl PAC_xx_SHAREDLIBS - Get compiler and linker for shared libraries
dnl These routines may be used to determine the compiler and the
dnl linker to be used in creating shared libraries
dnl Rather than set predefined variable names, they set an argument 
dnl (if provided)
dnl
dnl Synopsis
dnl PAC_CC_SHAREDLIBS(type,CCvar,CLINKvar)
dnl D*/
AC_DEFUN([PAC_CC_SHAREDLIBS],
[
pac_kinds=$1
ifelse($1,,[
    pac_prog=""
    AC_CHECK_PROG(pac_prog,gcc,yes,no)
    # If we are gcc but OS X, set the special type
    # We need a similar setting for cygwin
    if test "$pac_prog" = yes ; then 
        osname=`uname -s`
    	case $osname in 
             *Darwin*|*darwin*) pac_kinds=gcc-osx
             ;;				
	     *) pac_kinds=gcc
	     ;;
	esac
    fi
    pac_prog=""
    AC_CHECK_PROG(pac_prog,libtool,yes,no)
    if test "$pac_prog" = yes ; then pac_kinds="$pac_kinds libtool" ; fi
])
for pac_arg in $pac_kinds ; do
    case $pac_arg in 
    gcc)
    # For example, include the libname as ${LIBNAME_SHL}
    #C_LINK_SHL='${CC} -shared -Wl,-h,<finallibname>'
    pac_cc_sharedlibs='gcc -shared'
    # Make sure we select the correct fpic option
    PAC_C_CHECK_COMPILER_OPTION(-fPIC,fPIC_OK=yes,fPIC_OK=no)
    if test "$fPIC_OK" != yes ; then
        PAC_C_CHECK_COMPILER_OPTION(-fpic,fpic_ok=yes,fpic_ok=no)
        if test "$fpic_ok" != yes ; then
	     AC_MSG_ERROR([Neither -fpic nor -fPIC accepted by $CC])
        else
	     pac_cc_sharedlibs="$pac_cc_sharedlibs -fpic"
        fi
    else
        pac_cc_sharedlibs="$pac_cc_sharedlibs -fPIC"
    fi
    pac_clink_sharedlibs='gcc -shared'
    pac_type_sharedlibs=gcc
    ;;
    gcc-osx|osx-gcc)
    pac_clink_sharedlibs='${CC} -dynamiclib -undefined suppress -single_module -flat_namespace'
    pac_cc_sharedlibs='${CC} -fPIC'
    pac_type_sharedlibs=gcc-osx
    ;;
    libtool)
    AC_CHECK_PROGS(LIBTOOL,libtool,false)
    if test "$LIBTOOL" = "false" ; then
	AC_MSG_WARN([Could not find libtool])
    else
        # Likely to be
        # either CC or CC_SHL is libtool $cc
        pac_cc_sharedlibs'${LIBTOOL} -mode=compile ${CC}'
        pac_clink_sharedlibs='${LIBTOOL} -mode=link ${CC} -rpath ${libdir}'
	pac_type_sharedlibs=libtool
    fi
    ;;
    *)
    ;;
    esac
    if test -n "$pac_cc_sharedlibs" ; then break ; fi
done
if test -z "$pac_cc_sharedlibs" ; then pac_cc_sharedlibs=true ; fi
if test -z "$pac_clink_sharedlibs" ; then pac_clink_sharedlibs=true ; fi
ifelse($2,,CC_SHL=$pac_cc_sharedlibs,$2=$pac_cc_sharedlibs)
ifelse($3,,C_LINK_SHL=$pac_clink_sharedlibs,$3=$pac_clink_sharedlibs)
ifelse($4,,SHAREDLIB_TYPE=$pac_type_sharedlibs,$4=$pac_type_sharedlibs)
])

dnl This macro ensures that all of the necessary substitutions are 
dnl made by any subdirectory configure (which may simply SUBST the
dnl necessary values rather than trying to determine them from scratch)
dnl This is a more robust (and, in the case of libtool, only 
dnl managable) method.
AC_DEFUN([PAC_CC_SUBDIR_SHLIBS],[
	AC_SUBST(CC_SHL)
        AC_SUBST(C_LINK_SHL)
        AC_SUBST(LIBTOOL)
        AC_SUBST(ENABLE_SHLIB)
        AC_SUBST(SHLIB_EXT)
	if test "$ENABLE_SHLIB" = "libtool" ; then
	    if test -z "$LIBTOOL" ; then
		AC_MSG_WARN([libtool selected for shared library support but LIBTOOL is not defined])
            fi
	    # Libtool needs master_top_builddir
	    if test "X$master_top_builddir" = "X" ; then
	        AC_MSG_ERROR([Libtool requires master_top_builddir - check configure.ac sources])
	    fi
	    AC_SUBST(master_top_builddir)
	fi
])

dnl PAC_CC_SHLIB_EXT - get the extension for shared libraries
dnl Set the variable SHLIB_EXT if it is other than unknown.
AC_DEFUN([PAC_CC_SHLIB_EXT],[
# Not all systems use .so as the extension for shared libraries (cygwin
# and OSX are two important examples).  If we did not set the SHLIB_EXT,
# then try and determine it.  We need this to properly implement
# clean steps that look for libfoo.$SHLIB_EXT .
if test "$SHLIB_EXT" = "unknown" ; then
    osname=`uname -s`
    case $osname in 
        *Darwin*|*darwin*) SHLIB_EXT=dylib
        ;;	
        *CYGWIN*|*cygwin*) SHLIB_EXT=dll
        ;;
	*Linux*|*LINUX*|*SunOS*) SHLIB_EXT=so
	;;
   esac
fi
])

dnl PAC_COMPILER_SHLIB_FLAGS(compiler-var,output-file)
dnl
dnl Uses confdb/config.rpath to determine important linking flags and
dnl information.  This is mainly intended to support the compiler wrapper
dnl scripts in MPICH ("mpicc" and friends) which cannot directly use libtool to
dnl handle linking.  MPICH's compiler wrappers attempt to link exelwtables with
dnl an rpath by default.  The resulting variable assignment statements will be
dnl placed into "output-file", which is then suitable for AC_SUBST_FILE or
dnl sourcing in a shell script (including configure itself).
dnl
dnl This macro assumes that the basic tests associated with "compiler-var" have
dnl been run already, but does not AC_REQUIRE them in order to prevent problems
dnl with "expanded before required" when requiring the AC_PROG_{CC,F77,FC,CXX}
dnl macros.
dnl
dnl Example usage:
dnl
dnl ----8<----
dnl # compute flags for linking exelwtables against shared libraries when using
dnl # the modern Fortran compiler ($FC).
dnl PAC_COMPILER_SHLIB_FLAGS([FC],[src/elw/fc_shlib.conf])
dnl ----8<----
AC_DEFUN([PAC_COMPILER_SHLIB_FLAGS],[
AC_REQUIRE_AUX_FILE([config.rpath])
AC_REQUIRE([AC_CANONICAL_HOST])

# It appears that the libtool dynamic linking strategy on Darwin is this:
# 1. Link all shared libs with "-install_name /full/path/to/libfoo.dylib"
# 2. Don't do anything special when linking programs, since it seems that the
# darwin dynamic linker will always use the "install_name" field from the lib
# that was found at program link-time.  (CONFIRMED) This is in opposition to
# the way that Linux does it, where specifying a "-rpath" argument at program
# link-time is important.
#
# There is an alternative darwin strategy for versions
# >= 10.5, see this: http://www.cmake.org/pipermail/cmake/2010-August/038970.html
# (goodell@ 2011-06-17)

AC_MSG_CHECKING([for shared library (esp. rpath) characteristics of $1])

# unfortunately, config.rpath expects the compiler in question is always CC and
# uses several other environment variables as input
PAC_PUSH_FLAG([CC])
PAC_PUSH_FLAG([GCC])
PAC_PUSH_FLAG([LD])
# these two don't lwrrently get overridden, but we push/pop them for safety in
# case they do in the future
PAC_PUSH_FLAG([LDFLAGS])
PAC_PUSH_FLAG([with_gnu_ld])

# set the temporary override values (if any)
m4_case([$1],
[CC],
    [:], dnl do nothing special for CC, values are already correct
[F77],
    [CC="$$1"
     GCC="$G77"
     LD="$LD_F77"],
[FC],
    [CC="$$1"
     GCC="$GCC_FC"
     LD="$LD_FC"],
[CXX],
    [CC="$$1"
     GCC="$GXX"
     LD="$LD_CXX"],
[m4_fatal([unknown compiler argument "$1"])])

# ensure the values are available to the script
export CC
export GCC
export LDFLAGS
export LD
export with_gnu_ld

AS_IF([$ac_aux_dir/config.rpath "$host" > $2],[:],[AC_MSG_ERROR([unable to execute $ac_aux_dir/config.rpath])])

C_LINKPATH_SHL=""
AC_SUBST([C_LINKPATH_SHL])

rm -f conftest.out

# restore the old values
PAC_POP_FLAG([with_gnu_ld])
PAC_POP_FLAG([LD])
PAC_POP_FLAG([LDFLAGS])
PAC_POP_FLAG([GCC])
PAC_POP_FLAG([CC])

AC_MSG_RESULT([done (results in $2)])
])

