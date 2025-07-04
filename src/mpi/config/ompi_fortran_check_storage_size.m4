dnl -*- shell-script -*-
dnl
dnl Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
dnl                         University Research and Technology
dnl                         Corporation.  All rights reserved.
dnl Copyright (c) 2004-2005 The University of Tennessee and The University
dnl                         of Tennessee Research Foundation.  All rights
dnl                         reserved.
dnl Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
dnl                         University of Stuttgart.  All rights reserved.
dnl Copyright (c) 2004-2005 The Regents of the University of California.
dnl                         All rights reserved.
dnl Copyright (c) 2010-2014 Cisco Systems, Inc.  All rights reserved.
dnl Copyright (c) 2017      IBM Corporation.  All rights reserved.
dnl $COPYRIGHT$
dnl
dnl Additional copyrights may follow
dnl
dnl $HEADER$
dnl

# Check whether or not the Fortran compiler supports storage_size()
# for all relevant types or not.  E.g., gfortran 4.8 supports
# storage_size() on some types, not but all.
#
# OMPI_FORTRAN_CHECK_STORAGE_SIZE([action if found], [action if not found])
# ----------------------------------------------------
AC_DEFUN([OMPI_FORTRAN_CHECK_STORAGE_SIZE],[
    AS_VAR_PUSHDEF([fortran_storage_size_var], [ompi_cv_fortran_have_storage_size])

    # Re PR: https://github.com/open-mpi/ompi/pull/3822
    # We explored correcting the following syntax to compile with gfortran 4.8
    #   -    size = storage_size(x) / 8
    #   +    size = storage_size(x(1)) / 8
    # That allowed gfortran 4.8 to pass this configure test, but fail to
    # correctly handle mpi_sizeof due to the weak test for INTERFACE in
    # ompi_fortran_check_interface.m4. Until we can strengthen that configure
    # check we reverted the commit from PR #3822 to keep the old logic here
    # so that gfortran 4.8 will disqualify itself correctly for mpi_sizeof()
    # support.
    #
    AC_CACHE_CHECK([if Fortran compiler supports STORAGE_SIZE for relevant types],
       fortran_storage_size_var,
       [AC_LANG_PUSH([Fortran])
        AC_LINK_IFELSE([AC_LANG_SOURCE([[program check_for_storage_size
    USE, INTRINSIC :: iso_fortran_elw, ONLY: REAL32, INT32
    integer size
    complex(real32) :: c32
    complex(real32), dimension(5) :: c32_array
    real(real32) :: r32
    real(real32), dimension(5) :: r32_array
    integer(int32) :: i32
    integer(int32), dimension(5) :: i32_array

    call storage_size_complex32_scalar(c32, size)
    call storage_size_complex32_r1(c32_array, size)
    call storage_size_int32_scalar(i32, size)
    call storage_size_int32_r1(i32_array, size)
    call storage_size_real32_scalar(r32, size)
    call storage_size_real32_r1(r32_array, size)
end program

SUBROUTINE storage_size_complex32_scalar(x, size)
    USE, INTRINSIC :: iso_fortran_elw, ONLY: REAL32
    COMPLEX(REAL32) ::x
    INTEGER, INTENT(OUT) :: size

    size = storage_size(x) / 8
END SUBROUTINE storage_size_complex32_scalar

SUBROUTINE storage_size_complex32_r1(x, size)
    USE, INTRINSIC :: iso_fortran_elw, ONLY: REAL32
    COMPLEX(REAL32), DIMENSION(*)::x
    INTEGER, INTENT(OUT) :: size

    size = storage_size(x) / 8
END SUBROUTINE storage_size_complex32_r1

SUBROUTINE storage_size_int32_scalar(x, size)
    USE, INTRINSIC :: iso_fortran_elw, ONLY: INT32
    INTEGER(INT32) ::x
    INTEGER, INTENT(OUT) :: size

    size = storage_size(x) / 8
END SUBROUTINE storage_size_int32_scalar

SUBROUTINE storage_size_int32_r1(x, size)
    USE, INTRINSIC :: iso_fortran_elw, ONLY: INT32
    INTEGER(INT32), DIMENSION(*)::x
    INTEGER, INTENT(OUT) :: size

    size = storage_size(x) / 8
END SUBROUTINE storage_size_int32_r1

SUBROUTINE storage_size_real32_scalar(x, size)
    USE, INTRINSIC :: iso_fortran_elw, ONLY: REAL32
    REAL(REAL32) ::x
    INTEGER, INTENT(OUT) :: size

    size = storage_size(x) / 8
END SUBROUTINE storage_size_real32_scalar

SUBROUTINE storage_size_real32_r1(x, size)
    USE, INTRINSIC :: iso_fortran_elw, ONLY: REAL32
    REAL(REAL32), DIMENSION(*)::x
    INTEGER, INTENT(OUT) :: size

    size = storage_size(x) / 8
END SUBROUTINE storage_size_real32_r1
]])],
             [AS_VAR_SET(fortran_storage_size_var, yes)],
             [AS_VAR_SET(fortran_storage_size_var, no)])
        AC_LANG_POP([Fortran])
       ])

    AS_VAR_IF(fortran_storage_size_var, [yes], [$1], [$2])
    AS_VAR_POPDEF([fortran_storage_size_var])
])
