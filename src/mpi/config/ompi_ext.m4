dnl -*- shell-script -*-
dnl
dnl Copyright (c) 2004-2009 The Trustees of Indiana University and Indiana
dnl                         University Research and Technology
dnl                         Corporation.  All rights reserved.
dnl Copyright (c) 2009-2017 Cisco Systems, Inc.  All rights reserved
dnl Copyright (c) 2011-2012 Oak Ridge National Labs.  All rights reserved.
dnl Copyright (c) 2015-2018 Research Organization for Information Science
dnl                         and Technology (RIST).  All rights reserved.
dnl Copyright (c) 2017      The University of Tennessee and The University
dnl                         of Tennessee Research Foundation.  All rights
dnl                         reserved.
dnl Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
dnl $COPYRIGHT$
dnl
dnl Additional copyrights may follow
dnl
dnl $HEADER$
dnl

######################################################################
#
# OMPI_EXT
#
# configure the Interface Extensions [similar to MCA version].  Works hand in
# hand with Open MPI's autogen.pl, requiring it's specially formatted lists
# of frameworks, components, etc.
#
# USAGE:
#   OMPI_EXT()
#
######################################################################
AC_DEFUN([OMPI_EXT],[
    dnl for OPAL_CONFIGURE_USER elw variable
    AC_REQUIRE([OPAL_CONFIGURE_SETUP])

    # Note that we do not build DSO's here -- we *only* build colwenience
    # libraries that get slurped into higher-level libraries
    #
    # [default -- no option given] = No extensions built
    # --enable-mpi-ext=[,]*EXTENSION[,]*
    #
    AC_ARG_ENABLE(mpi-ext,
        AC_HELP_STRING([--enable-mpi-ext[=LIST]],
                       [Comma-separated list of extensions that should be built.  Possible values: ompi_mpiext_list.  Example: "--enable-mpi-ext=foo,bar" will enable building the MPI extensions "foo" and "bar".  If LIST is empty or the special value "all", then all available MPI extensions will be built (default: all).]))

    # print some nice messages about what we're about to do...
    AC_MSG_CHECKING([for available MPI Extensions])
    AC_MSG_RESULT([ompi_mpiext_list])

    AC_MSG_CHECKING([which MPI extension should be enabled])
    if test "$enable_mpi_ext" = "" || \
       test "$enable_mpi_ext" = "yes" || \
       test "$enable_mpi_ext" = "all"; then
        enable_mpi_ext=all
        msg="All Available Extensions"
        str="`echo ENABLE_EXT_ALL=1`"
        eval $str
    else
        ifs_save="$IFS"
        IFS="${IFS}$PATH_SEPARATOR,"
        msg=
        for item in $enable_mpi_ext; do
            type="`echo $item | cut -s -f1 -d-`"
            if test -z $type ; then
                type=$item
            fi
            str="`echo ENABLE_${type}=1 | sed s/-/_/g`"
            eval $str
            msg="$item $msg"
        done
        IFS="$ifs_save"
    fi
    AC_MSG_RESULT([$msg])
    unset msg

    m4_ifdef([ompi_mpiext_list], [],
             [m4_fatal([Could not find MPI Extensions list.  Aborting.])])

    EXT_CONFIGURE
])


######################################################################
#
# EXT_CONFIGURE
#
# USAGE:
#   EXT_CONFIGURE()
#
######################################################################
AC_DEFUN([EXT_CONFIGURE],[
    outdir=ompi/include

    # first create the output include directory
    mkdir -p $outdir

    ###############
    # C Bindings
    ###############

    # remove any previously generated #include files
    mpi_ext_h=$outdir/mpi-ext.h
    rm -f $mpi_ext_h

    # Create the final mpi-ext.h file.
    cat > $mpi_ext_h <<EOF
/*
 * \$HEADER\$
 */

#ifndef OMPI_MPI_EXT_H
#define OMPI_MPI_EXT_H 1

#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

#define OMPI_HAVE_MPI_EXT 1

EOF

    ###############
    # mpif.h Bindings
    ###############

    # remove any previously generated #include files
    mpif_ext_h=$outdir/mpif-ext.h
    rm -f $mpif_ext_h

    # Create the final mpif-ext.h file.
    cat > $mpif_ext_h <<EOF
! -*- fortran -*-
! \$HEADER\$
!
! *** THIS FILE IS AUTOMATICALLY GENERATED!
! *** Any manual edits will be lost!
!
      integer OMPI_HAVE_MPI_EXT
      parameter (OMPI_HAVE_MPI_EXT=1)
!
EOF

    ###############
    # "use mpi" Bindings
    ###############

    # Although Fortran module files are essentially compiled header
    # files, we don't create them in ompi/include, like we do for
    # mpi.h and mpif.h.  Instead, we build them down in ompi/fortran,
    # when we build the rest of the Fortran modules.  Indeed, in the
    # "use mpi" case, it needs some of the same internal modules that
    # the mpi_f08 module itself needs.  So the mpi_f08_ext module has
    # to be built *after* the mpi_f08 module (so that all the internal
    # modules it needs are already built).

    # remove any previously generated #include files
    outdir=ompi/mpi/fortran/mpiext-use-mpi
    mkdir -p $outdir
    mpiusempi_ext_h=$outdir/mpi-ext-module.F90
    rm -f $mpiusempi_ext_h

    # Create the final mpiusempi-ext.h file.
    cat > $mpiusempi_ext_h <<EOF
! -*- fortran -*-
! \$HEADER\$
!
! *** THIS FILE IS AUTOMATICALLY GENERATED!
! *** Any manual edits will be lost!
!
#include "ompi/mpi/fortran/configure-fortran-output.h"

module mpi_ext
!     Even though this is not a useful parameter (cannot be used as a
!     preprocessor catch) define it to keep the linker from complaining
!     during the build.
      integer OMPI_HAVE_MPI_EXT
      parameter (OMPI_HAVE_MPI_EXT=1)
!
EOF

    # Make an AM conditional to see whether we're building the mpi_ext
    # module.  Note that we only build it if we support the ignore-tkr
    # mpi module.
    AS_IF([test $OMPI_BUILD_FORTRAN_BINDINGS -ge $OMPI_FORTRAN_USEMPI_BINDINGS && \
           test $OMPI_FORTRAN_HAVE_IGNORE_TKR -eq 1],
          [OMPI_BUILD_FORTRAN_USEMPI_EXT=1],
          [OMPI_BUILD_FORTRAN_USEMPI_EXT=0])
    AM_CONDITIONAL(OMPI_BUILD_FORTRAN_USEMPI_EXT,
                   [test $OMPI_BUILD_FORTRAN_USEMPI_EXT -eq 1])

    ###############
    # "use mpi_f08" Bindings
    ###############

    # See note above: we generate the mpi_f08_ext module in
    # ompi/mpi/fortran/mpiext-use-mpi-f08

    # remove any previously generated #include files
    outdir=ompi/mpi/fortran/mpiext-use-mpi-f08
    mkdir -p $outdir
    mpiusempif08_ext_h=$outdir/mpi-f08-ext-module.F90
    rm -f $mpiusempif08_ext_h

    # Create the final mpiusempi-ext.h file.
    cat > $mpiusempif08_ext_h <<EOF
! -*- fortran -*-
! \$HEADER\$
!
! *** THIS FILE IS AUTOMATICALLY GENERATED!
! *** Any manual edits will be lost!
!
#include "ompi/mpi/fortran/configure-fortran-output.h"

module mpi_f08_ext
!     Even though this is not a useful parameter (cannot be used as a
!     preprocessor catch) define it to keep the linker from complaining
!     during the build.
      integer OMPI_HAVE_MPI_EXT
      parameter (OMPI_HAVE_MPI_EXT=1)
!
EOF

    # Only build this mpi_f08_ext module if we're building the "use
    # mpi_f08" module
    AS_IF([test $OMPI_BUILD_FORTRAN_BINDINGS -ge $OMPI_FORTRAN_USEMPIF08_BINDINGS],
          [OMPI_BUILD_FORTRAN_USEMPIF08_EXT=1],
          [OMPI_BUILD_FORTRAN_USEMPIF08_EXT=0])
    AM_CONDITIONAL(OMPI_BUILD_FORTRAN_USEMPIF08_EXT,
                   [test $OMPI_BUILD_FORTRAN_USEMPIF08_EXT -eq 1])

    # Make an AM conditional to see whether we're building either the
    # mpi_ext or mpi_f08_Ext modules.
    AM_CONDITIONAL(OMPI_BUILD_FORTRAN_USEMPI_OR_USEMPIF08_EXT,
                   [test $OMPI_BUILD_FORTRAN_USEMPI_EXT -eq 1 || \
                    test $OMPI_BUILD_FORTRAN_USEMPIF08_EXT -eq 1])

    #
    # Process each component
    #

    # remove any previously generated #include files
    outfile_real=ompi/mpiext/static-components.h
    outfile=$outfile_real.new
    rm -f $outfile $outfile.struct $outfile.extern
    $MKDIR_P ompi/mpiext
    touch $outfile.struct $outfile.extern

    m4_foreach(extension, [ompi_mpiext_list],
               [m4_ifval(extension,
               [EXT_CONFIGURE_M4_CONFIG_COMPONENT(extension,
                                                  [OMPI_MPIEXT_ALL],
                                                  [OMPI_MPIEXT_C],
                                                  [OMPI_MPIEXT_MPIFH],
                                                  [OMPI_MPIEXT_USEMPI],
                                                  [OMPI_MPIEXT_USEMPIF08])])])

    ###############
    # C Bindings
    ###############
    # Create the final mpi-ext.h file.
    cat >> $mpi_ext_h <<EOF

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif /* OMPI_MPI_EXT_H */

EOF

    ###############
    # mpif.h Bindings
    ###############
    # Create the final mpif-ext.h file.
    cat >> $mpif_ext_h <<EOF
!
EOF

    ###############
    # "use mpi" Bindings
    ###############
    # Create the final mpiusempi-ext.h file.
    cat >> $mpiusempi_ext_h <<EOF
!
end module mpi_ext
EOF

    ###############
    # "use mpi_f08" Bindings
    ###############
    # Create the final mpiusempi-ext.h file.
    cat >> $mpiusempif08_ext_h <<EOF
!
end module mpi_f08_ext
EOF

    # Create the final .h file that will be included in the type's
    # top-level glue.  This lists all the static components.  We don't
    # need to do this for "common".
    if test "$2" != "common"; then
        cat > $outfile <<EOF
/*
 * \$HEADER\$
 */
#if defined(c_plusplus) || defined(__cplusplus)
extern "C" {
#endif

`cat $outfile.extern`

const ompi_mpiext_component_t *ompi_mpiext_components[[]] = {
`cat $outfile.struct`
  NULL
};

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

EOF
        # Only replace the header file if a) it doesn't previously
        # exist, or b) the contents are different.  Do this to not
        # trigger recompilation of certain .c files just because the
        # timestamp changed on $outfile_real (similar to the way AC
        # handles AC_CONFIG_HEADER files).
        diff $outfile $outfile_real > /dev/null 2>&1
        if test "$?" != "0"; then
            mv $outfile $outfile_real
        else
            rm -f $outfile
        fi
    fi
    rm -f $outfile.struct $outfile.extern

    # We have all the results we need.  Now put them in various
    # variables/defines so that others can see the results.

    OMPI_EXT_MAKE_DIR_LIST(OMPI_MPIEXT_ALL_SUBDIRS, $OMPI_MPIEXT_ALL)

    OMPI_EXT_MAKE_LISTS(OMPI_MPIEXT_C, $OMPI_MPIEXT_C, c, c)
    OMPI_EXT_MAKE_LISTS(OMPI_MPIEXT_MPIFH, $OMPI_MPIEXT_MPIFH, mpif-h, mpifh)
    OMPI_EXT_MAKE_LISTS(OMPI_MPIEXT_USEMPI, $OMPI_MPIEXT_USEMPI, use-mpi, usempi)
    OMPI_EXT_MAKE_LISTS(OMPI_MPIEXT_USEMPIF08, $OMPI_MPIEXT_USEMPIF08, use-mpi-f08, usempif08)

    comps=`echo $OMPI_MPIEXT_C | sed -e 's/^[ \t]*//;s/[ \t]*$//;s/ /, /g'`
    AC_DEFINE_UNQUOTED([OMPI_MPIEXT_COMPONENTS], ["$comps"],
                       [MPI Extensions included in libmpi])
])


######################################################################
#
# EXT_CONFIGURE_M4_CONFIG_COMPONENT
#
#
# USAGE:
#   EXT_CONFIGURE_M4_CONFIG_COMPONENT((1) component_name,
#                                     (2) all_components_variable,
#                                     (3) c_components_variable,
#                                     (4) mpifh_components_variable,
#                                     (5) usempi_components_variable,
#                                     (6) usempif08_components_variable)
#
# - component_name is a single, naked string (no prefix)
# - all others are naked component names (e.g., "example").  If an
#   extension is named in that variable, it means that that extension
#   has bindings of that flavor.
#
######################################################################
AC_DEFUN([EXT_CONFIGURE_M4_CONFIG_COMPONENT],[
    opal_show_subsubsubtitle "MPI Extension $1"

    EXT_COMPONENT_BUILD_CHECK($1, [should_build=1], [should_build=0])

    # try to configure the component
    m4_ifdef([OMPI_MPIEXT_$1_CONFIG], [],
             [m4_fatal([Could not find OMPI_MPIEXT_]$1[_CONFIG macro for ]$1[ component])])

    OMPI_MPIEXT_$1_CONFIG([should_build=${should_build}], [should_build=0])

    AS_IF([test $should_build -eq 1],
          [EXT_PROCESS_COMPONENT([$1], [$2], [$3], [$4], [$5], [$6])],
          [EXT_PROCESS_DEAD_COMPONENT([$1], [$2])])
])

######################################################################
#
# EXT_PROCESS_COMPONENT
#
# does all setup work for given component.  It should be known before
# calling that this component can build properly (and exists)
#
# USAGE:
#   EXT_CONFIGURE_ALL_CONFIG_COMPONENTS((1) component_name
#                                       (2) all_components_variable,
#                                       (3) c_components_variable,
#                                       (4) mpifh_components_variable,
#                                       (5) usempi_components_variable,
#                                       (6) usempif08_components_variable)
#
# C bindings are mandatory.  Other bindings are optional / built if
# they are found.  Here's the files that the m4 expects:
#
#--------------------
#
# C:
# - c/mpiext_<component>_c.h: is installed to
#   <includedir>/openmpi/mpiext/mpiext_<component>_c.h and is included in
#   mpi_ext.h
# - c/libmpiext_<component>.la: colwneience library slurped into libmpi.la
#
# mpi.f.h:
# - mpif-h/mpiext_<component>_mpifh.h: is installed to
#   <includedir>openmpi/mpiext/mpiext_<component>_mpifh.h and is included mpi
#   mpif_ext.h
# - mpif-h/libmpiext_<component>_mpifh.la: colwenience library slurped
#   into libmpi_mpifh.la
#
# If the ..._mpifh.h file exists, it is assumed that "make all" will
# build the .la file.  And therefore we'll include full support for
# the mpif.h bindings for this extension in OMPI.
#
#--------------------
#
# use mpi:
# - use-mpi/mpiext_<component>_usempi.h: included in the mpi_ext module
#
# Only supported when the ignore-tkr mpi module is built (this
# lwrrently means: when you don't use gfortran).
#
# If the ..._usempi.h file exists, it is assumed that we'll include
# full support for the mpi_ext bindings for this extension in OMPI.
#
# NO LIBRARY IS SUPPORTED FOR THE mpi MODULE BINDINGS!  It is assumed
# that all required symbols will be in the
# libmpiext_<component>_mpifh.la library, and all that this set of
# bindings does it give strong type checking to those subroutines.
#
#--------------------
#
# use mpi_f08:
# - use-mpi-f08/mpiext_<component>_usempif08.h: included in the mpi_ext module
# - use-mpi-f08/libmpiext_<component>_usempif08.la: colwenience
#   library slurped into libmpi_usempif08.la
#
# Only supported when the non-descriptor-based mpi_f08 module is built
# (this lwrrently means: when you don't use gfortran).
#
# If the ..._usempif08.h file exists, it is assumed that "make all"
# will build the .la file.  And therefore we'll include full support
# for the mpi_f08 bindings for this extension in OMPI.
#
######################################################################
AC_DEFUN([EXT_PROCESS_COMPONENT],[
    component=$1

    # Output pretty results
    AC_MSG_CHECKING([if MPI Extension $component can compile])
    AC_MSG_RESULT([yes])

    tmp[=]m4_translit([$1],[a-z],[A-Z])
    component_define="OMPI_HAVE_MPI_EXT_${tmp}"

    ###############
    # C Bindings
    ###############
    test_header="${srcdir}/ompi/mpiext/${component}/c/mpiext_${component}_c.h"

    AC_MSG_CHECKING([if MPI Extension $component has C bindings])

    AS_IF([test ! -e "$test_header" && test ! -e "$test_header.in"],
          [ # There *must* be C bindings
           AC_MSG_RESULT([no])
           AC_MSG_WARN([C bindings for MPI extensions are required])
           AC_MSG_ERROR([Cannot continue])])

    AC_MSG_RESULT([yes (required)])

    # Save the list of headers and colwenience libraries that this
    # component will output
    $2="$$2 $component"
    $3="$$3 $component"

    # JMS Where is this needed?
    EXT_C_HEADERS="$EXT_C_HEADERS mpiext/c/mpiext_${component}_c.h"

    component_header="mpiext_${component}_c.h"

    cat >> $mpi_ext_h <<EOF
/* Enabled Extension: $component */
#define $component_define 1
#include "openmpi/mpiext/$component_header"

EOF

    ###############
    # mpif.h bindings
    ###############
    #
    # Test if this extension has mpif.h bindings
    # If not, skip this step.
    #
    test_header="${srcdir}/ompi/mpiext/$component/mpif-h/mpiext_${component}_mpifh.h"
    enabled_mpifh=0

    AC_MSG_CHECKING([if MPI Extension $component has mpif.h bindings])

    if test -e "$test_header" ; then
        AC_MSG_RESULT([yes])
        enabled_mpifh=1

        EXT_MPIFH_HEADERS="$EXT_MPIFH_HEADERS mpiext/mpiext_${component}_mpifh.h"
        $4="$$4 $component"

        # Per https://github.com/open-mpi/ompi/pull/6030, we will end
        # up putting a user-visible Fortran "include" statement in the
        # installed mpif-ext.h file, and we therefore have to ensure
        # that the total length of the line is <=72 characters.  Doing
        # a little math here:
        #
        # leading indent spaces: 6 chars
        # "include '": 9 chars
        # "openmpi/mpiext/mpiext_NAME_mpifh.h": without NAME, 30 chars
        # trailing "'": 1 char
        #
        # 6+9+30+1 = 46 chars overhead.
        # 72-46 = 26 characters left for NAME.
        #
        # It would be exceedingly unusual to have an MPI extension
        # name > 26 characters.  But just in case, put a check here
        # to make sure: error out if the MPI extension name is > 26
        # characters (because otherwise it'll just be a really weird /
        # hard to diagnose compile error when a user tries to compile
        # a Fortran MPI application that includes `mpif-ext.h`).
        len=`echo $component | wc -c`
        result=`expr $len \> 26`
        AS_IF([test $result -eq 1],
              [AC_MSG_WARN([MPI extension name too long: $component])
               AC_MSG_WARN([For esoteric reasons, MPI Extensions with mpif.h bindings must have a name that is <= 26 characters])
               AC_MSG_ERROR([Cannot continue])])

        component_header="mpiext_${component}_mpifh.h"

        cat >> $mpif_ext_h <<EOF
!
!     Enabled Extension: $component
!
      integer $component_define
      parameter ($component_define=1)

      include 'openmpi/mpiext/$component_header'

EOF
    else
        AC_MSG_RESULT([no])

        cat >> $mpif_ext_h <<EOF
!
!     Enabled Extension: $component
!     No mpif.h bindings available
!
      integer $component_define
      parameter ($component_define=0)

EOF
    fi

    ###############
    # "use mpi" bindings
    ###############
    #
    # Test if this extension has "use mpi" bindings
    # If not, skip this step.
    #
    test_header="${srcdir}/ompi/mpiext/$component/use-mpi/mpiext_${component}_usempi.h"

    AC_MSG_CHECKING([if MPI Extension $component has "use mpi" bindings])

    if test -e "$test_header" ; then
        AC_MSG_RESULT([yes])

        EXT_USEMPI_HEADERS="$EXT_USEMPI_HEADERS mpiext/$component/use-mpi/mpiext_${component}_usempi.h"
        $5="$$5 $component"
        component_header="mpiext_${component}_usempi.h"

        cat >> $mpiusempi_ext_h <<EOF
!
!     Enabled Extension: $component
!
EOF
        #
        # Include the mpif.h header if it is available.  Cannot do
        # this from inside the usempi.h since, for VPATH builds, the
        # srcdir is needed to find the header.
        #
        if test "$enabled_mpifh" = 1; then
            mpifh_component_header="mpiext_${component}_mpifh.h"
            cat >> $mpiusempi_ext_h <<EOF
#include "${srcdir}/ompi/mpiext/$component/mpif-h/$mpifh_component_header"
EOF
        fi

        cat >> $mpiusempi_ext_h <<EOF
#include "${srcdir}/ompi/mpiext/$component/use-mpi/$component_header"

EOF
    else
        AC_MSG_RESULT([no])

        cat >> $mpiusempi_ext_h <<EOF
!
!     Enabled Extension: $component
!     No "use mpi" bindings available
!

EOF
    fi

    ###############
    # "use mpi_f08" bindings
    ###############
    #
    # Test if this extension has "use mpi_f08" bindings
    # If not, skip this step.
    #
    test_header="${srcdir}/ompi/mpiext/$component/use-mpi-f08/mpiext_${component}_usempif08.h"

    AC_MSG_CHECKING([if MPI Extension $component has "use mpi_f08" bindings])

    if test -e "$test_header" ; then
        AC_MSG_RESULT([yes])

        EXT_USEMPIF08_HEADERS="$EXT_USEMPIF08_HEADERS mpiext/$component/use-mpi-f08/mpiext_${component}_usempif08.h"
        $6="$$6 $component"

        component_header="mpiext_${component}_usempif08.h"

        cat >> $mpiusempif08_ext_h <<EOF
!
!     Enabled Extension: $component
!
EOF
        #
        # Include the mpif.h header if it is available.  Cannot do
        # this from inside the usempif08.h since, for VPATH builds,
        # the srcdir is needed to find the header.
        #
        if test "$enabled_mpifh" = 1; then
            mpifh_component_header="mpiext_${component}_mpifh.h"
            cat >> $mpiusempif08_ext_h <<EOF
#include "${srcdir}/ompi/mpiext/$component/mpif-h/$mpifh_component_header"
EOF
        fi

        cat >> $mpiusempif08_ext_h <<EOF
#include "${srcdir}/ompi/mpiext/$component/use-mpi-f08/$component_header"

EOF
    else
        AC_MSG_RESULT([no])

        cat >> $mpiusempif08_ext_h <<EOF
!
!     Enabled Extension: $component
!     No "use mpi_f08" bindings available
!

EOF
    fi

    m4_ifdef([OMPI_MPIEXT_]$1[_NEED_INIT],
             [echo "extern const ompi_mpiext_component_t ompi_mpiext_${component};" >> $outfile.extern
              echo "  &ompi_mpiext_${component}, " >> $outfile.struct])

    # now add the flags that were set in the environment variables
    # framework_component_FOO (for example, the flags set by
    # m4_configure components)
    m4_foreach(flags, [LDFLAGS, LIBS],
        [AS_IF([test "$mpiext_$1_WRAPPER_EXTRA_]flags[" = ""],
                [OPAL_FLAGS_APPEND_UNIQ([ompi_mca_wrapper_extra_]m4_tolower(flags), [$mpiext_$1_]flags)],
                [OPAL_FLAGS_APPEND_UNIQ([ompi_mca_wrapper_extra_]m4_tolower(flags), [$mpiext_$1_WRAPPER_EXTRA_]flags)])
        ])

    AS_IF([test "$mpiext_$1_WRAPPER_EXTRA_CPPFLAGS" != ""],
        [OPAL_FLAGS_APPEND_UNIQ([ompi_mca_wrapper_extra_cppflags], [$mpiext_$1_WRAPPER_EXTRA_CPPFLAGS])])
])


######################################################################
#
# EXT_PROCESS_DEAD_COMPONENT
#
# process a component that can not be built.  Do the last minute checks
# to make sure the user isn't doing something stupid.
#
# USAGE:
#   EXT_PROCESS_DEAD_COMPONENT((1) component_name,
#                              (2) all_components_variable)
#
# NOTE: component_name will not be determined until run time.
#
######################################################################
AC_DEFUN([EXT_PROCESS_DEAD_COMPONENT],[
    AC_MSG_CHECKING([if MPI Extension $1 can compile])

    # Need to add this component to the "all" list so that it is
    # included in DIST SUBDIRS
    $2="$$2 $1"

    AC_MSG_RESULT([no])
])



######################################################################
#
# EXT_COMPONENT_BUILD_CHECK
#
# checks the standard rules of component building to see if the
# given component should be built.
#
# USAGE:
#    EXT_COMPONENT_BUILD_CHECK(component,
#                              action-if-build, action-if-not-build)
#
######################################################################
AC_DEFUN([EXT_COMPONENT_BUILD_CHECK],[
    AC_REQUIRE([AC_PROG_GREP])

    component=$1
    component_path="$srcdir/ompi/mpiext/$component"
    want_component=0

    # build if:
    # - the component type is direct and we are that component
    # - there is no ompi_ignore file
    # - there is an ompi_ignore, but there is an empty ompi_unignore
    # - there is an ompi_ignore, but username is in ompi_unignore
    if test -d $component_path ; then
        # decide if we want the component to be built or not.  This
        # is spread out because some of the logic is a little complex
        # and test's syntax isn't exactly the greatest.  We want to
        # build the component by default.
        want_component=1
        if test -f $component_path/.ompi_ignore ; then
            # If there is an ompi_ignore file, don't build
            # the component.  Note that this decision can be
            # overridden by the unignore logic below.
            want_component=0
        fi
        if test -f $component_path/.ompi_unignore ; then
            # if there is an empty ompi_unignore, that is
            # equivalent to having your userid in the unignore file.
            # If userid is in the file, unignore the ignore file.
            if test ! -s $component_path/.ompi_unignore ; then
                want_component=1
            elif test ! -z "`$GREP $OPAL_CONFIGURE_USER $component_path/.ompi_unignore`" ; then
                want_component=1
            fi
        fi
    fi

    # if we asked for everything, then allow it to build if able
    str="ENABLED_COMPONENT_CHECK=\$ENABLE_EXT_ALL"
    eval $str
    if test ! "$ENABLED_COMPONENT_CHECK" = "1" ; then
        # if we were explicitly disabled, don't build :)
        str="ENABLED_COMPONENT_CHECK=\$ENABLE_${component}"
        eval $str
        if test ! "$ENABLED_COMPONENT_CHECK" = "1" ; then
            want_component=0
        fi
    fi

    AS_IF([test "$want_component" = "1"], [$2], [$3])
])


# OMPI_EXT_MAKE_DIR_LIST(subst'ed variable, shell list)
#
# Prefix every extension name with "mpiext/" and AC subst it.
# -------------------------------------------------------------------------
AC_DEFUN([OMPI_EXT_MAKE_DIR_LIST],[
    $1=
    for item in $2 ; do
       $1="$$1 mpiext/$item"
    done
    AC_SUBST($1)
])

# OMPI_EXT_MAKE_LISTS((1) subst'ed variable prefix,
#                     (2) shell list,
#                     (3) bindings dir name,
#                     (4) bindings suffix)
#
# Prefix every extension name with "mpiext/".
# -------------------------------------------------------------------------
AC_DEFUN([OMPI_EXT_MAKE_LISTS],[
    # Make the directory list
    tmp=
    for item in $2 ; do
       tmp="$tmp mpiext/$item/$3"
    done
    $1_DIRS=$tmp
    AC_SUBST($1_DIRS)

    # Make the list of libraries
    tmp=
    for item in $2 ; do
       tmp="$tmp "'$(top_builddir)'"/ompi/mpiext/$item/$3/libmpiext_${item}_$4.la"
    done
    $1_LIBS=$tmp
    AC_SUBST($1_LIBS)
])
