# -*- shell-script -*-
#
# Copyright (c) 2004-2006 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2008-2015 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2015-2017 Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AC_DEFUN([MCA_ompi_io_romio321_POST_CONFIG], [
    AM_CONDITIONAL([MCA_io_romio321_SHOULD_BUILD], [test $1 -eq 1])
])


# MCA_io_romio321_CONFIG([action-if-found], [action-if-not-found])
# -----------------------------------------------------------
AC_DEFUN([MCA_ompi_io_romio321_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/io/romio321/Makefile])

    OPAL_VAR_SCOPE_PUSH([io_romio321_flags io_romio321_flags_define io_romio321_happy io_romio321_save_LIBS])
    AC_ARG_ENABLE([io-romio],
                  [AC_HELP_STRING([--disable-io-romio],
                                  [Disable the ROMIO MPI-IO component])])
    AC_ARG_WITH([io-romio-flags],
                [AC_HELP_STRING([--with-io-romio-flags=FLAGS],
                                [Pass FLAGS to the ROMIO distribution configuration script])])
    AC_DEFINE_UNQUOTED([MCA_io_romio321_USER_CONFIGURE_FLAGS], ["$with_io_romio_flags"], [Set of user-defined configure flags given to ROMIOs configure script via --with-io-romio-flags])
    AC_MSG_CHECKING([if want ROMIO component])
    AS_IF([test "$enable_io_romio" = "no"],
           [AC_MSG_RESULT([no])
            $2],
           [AC_MSG_RESULT([yes])
            AC_MSG_CHECKING([if MPI profiling is enabled])
            AS_IF([test "$enable_mpi_profile" = "no"],
                  [AC_MSG_RESULT([no])
                   AC_MSG_WARN([*** The ROMIO io component requires the MPI profiling layer])
                   AS_IF([test "$enable_io_romio" = "yes"],
                         [AC_MSG_ERROR([*** ROMIO requested but not available.  Aborting])])
                   $2],
                  [AC_MSG_RESULT([yes])

                   AS_IF([test -n "$with_io_romio_flags" && test "$with_io_romio_flags" != "no"],
                         [io_romio321_flags="$with_io_romio_flags $io_romio321_flags"],
                         [io_romio321_flags=])
                   # If ROMIO is going to end up in a DSO, all we need is
                   # shared library-ized objects, as we're only building a
                   # DSO (which is always shared).  Otherwise, build with
                   # same flags as OMPI, as we might need any combination of
                   # shared and static-ized objects...
                   AS_IF([test "$compile_mode" = "dso"],
                         [io_romio321_shared=enable
                          io_romio321_static=disable],
                         [AS_IF([test "$enable_shared" = "yes"],
                                [io_romio321_shared=enable],
                                [io_romio321_shared=disable])
                          AS_IF([test "$enable_static" = "yes"],
                                [io_romio321_static=enable],
                                [io_romio321_static=disable])])
                   AS_IF([test -n "$prefix" && test "$prefix" != "NONE"],
                         [io_romio321_prefix_arg="--prefix=$prefix"],
                         [io_romio321_prefix_arg=])

                   AS_IF([test "$cross_compiling" = "yes"],
                       [AS_IF([test ! -z $build], [io_romio321_flags="$io_romio321_flags --build=$build"])
                        AS_IF([test ! -z $host], [io_romio321_flags="$io_romio321_flags --host=$host"])
                        AS_IF([test ! -z $target], [io_romio321_flags="$io_romio321_flags --target=$target"])])
                   AS_IF([test "$enable_grequest_extensions" = "yes"],
                         [io_romio321_flags="$io_romio321_flags --enable-grequest-extensions"])
                   io_romio321_flags_define="$io_romio321_flags FROM_OMPI=yes CC='$CC' CFLAGS='$CFLAGS -D__EXTENSIONS__' CPPFLAGS='$CPPFLAGS' FFLAGS='$FFLAGS' LDFLAGS='$LDFLAGS' --$io_romio321_shared-shared --$io_romio321_static-static $io_romio321_flags $io_romio321_prefix_arg --disable-aio --disable-weak-symbols --enable-strict --disable-f77 --disable-f90"
                   AC_DEFINE_UNQUOTED([MCA_io_romio321_COMPLETE_CONFIGURE_FLAGS], ["$io_romio321_flags_define"], [Complete set of command line arguments given to ROMIOs configure script])

                   io_romio321_flags="$io_romio321_flags FROM_OMPI=yes CC="'"'"$CC"'"'" CFLAGS="'"'"$CFLAGS -D__EXTENSIONS__"'"'" CPPFLAGS="'"'"$CPPFLAGS"'"'" FFLAGS="'"'"$FFLAGS"'"'" LDFLAGS="'"'"$LDFLAGS"'"'" --$io_romio321_shared-shared --$io_romio321_static-static $io_romio321_flags $io_romio321_prefix_arg --disable-aio --disable-weak-symbols --enable-strict --disable-f77 --disable-f90"

                   opal_show_subtitle "Configuring ROMIO distribution"
                   OPAL_CONFIG_SUBDIR([ompi/mca/io/romio321/romio],
                                      [$io_romio321_flags],
                                      [io_romio321_happy=1], [io_romio321_happy=0])

                   AS_IF([test "$io_romio321_happy" = "1"],
                         [ # grab the libraries list from ROMIO.  We don't
                           # need this for building the component, as libtool
                           # will figure that part out.  But we do need it for
                           # the wrapper settings
                          io_romio321_save_LIBS="$LIBS"
                          LIBS=
                          . ompi/mca/io/romio321/romio/localdefs
                          io_romio321_LIBS="$LIBS"
                          LIBS="$io_romio321_save_LIBS"

                          echo "ROMIO distribution configured successfully"
                          $1],
                         [AS_IF([test "$enable_io_romio" = "yes"],
                                [AC_MSG_ERROR([ROMIO distribution did not configure successfully])],
                                [AC_MSG_WARN([ROMIO distribution did not configure successfully])])
                          $2])])])
    OPAL_VAR_SCOPE_POP
])
