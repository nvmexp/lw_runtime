/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/fortran/mpif-h/bindings.h"
#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mpi/fortran/base/fortran_base_strings.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_INFO_GET_NTHKEY = ompi_info_get_nthkey_f
#pragma weak pmpi_info_get_nthkey = ompi_info_get_nthkey_f
#pragma weak pmpi_info_get_nthkey_ = ompi_info_get_nthkey_f
#pragma weak pmpi_info_get_nthkey__ = ompi_info_get_nthkey_f

#pragma weak PMPI_Info_get_nthkey_f = ompi_info_get_nthkey_f
#pragma weak PMPI_Info_get_nthkey_f08 = ompi_info_get_nthkey_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_INFO_GET_NTHKEY,
                            pmpi_info_get_nthkey,
                            pmpi_info_get_nthkey_,
                            pmpi_info_get_nthkey__,
                            pompi_info_get_nthkey_f,
                            (MPI_Fint *info, MPI_Fint *n, char *key, MPI_Fint *ierr, int key_len),
                            (info, n, key, ierr, key_len) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_INFO_GET_NTHKEY = ompi_info_get_nthkey_f
#pragma weak mpi_info_get_nthkey = ompi_info_get_nthkey_f
#pragma weak mpi_info_get_nthkey_ = ompi_info_get_nthkey_f
#pragma weak mpi_info_get_nthkey__ = ompi_info_get_nthkey_f

#pragma weak MPI_Info_get_nthkey_f = ompi_info_get_nthkey_f
#pragma weak MPI_Info_get_nthkey_f08 = ompi_info_get_nthkey_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_INFO_GET_NTHKEY,
                            mpi_info_get_nthkey,
                            mpi_info_get_nthkey_,
                            mpi_info_get_nthkey__,
                            ompi_info_get_nthkey_f,
                            (MPI_Fint *info, MPI_Fint *n, char *key, MPI_Fint *ierr, int key_len),
                            (info, n, key, ierr, key_len) )
#else
#define ompi_info_get_nthkey_f pompi_info_get_nthkey_f
#endif
#endif


static const char FUNC_NAME[] = "MPI_INFO_GET_NTHKEY";

/* Note that the key_len parameter is silently added by the Fortran
   compiler, and will be filled in with the actual length of the
   character array from the caller.  Hence, it's the max length of the
   string that we can use. */

void ompi_info_get_nthkey_f(MPI_Fint *info, MPI_Fint *n, char *key,
			   MPI_Fint *ierr, int key_len)
{
    int c_ierr, ret;
    MPI_Info c_info;
    char c_key[MPI_MAX_INFO_KEY + 1];

    c_info = PMPI_Info_f2c(*info);

    c_ierr = PMPI_Info_get_nthkey(c_info,
                                 OMPI_FINT_2_INT(*n),
                                 c_key);
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);

    if (OMPI_SUCCESS != (ret = ompi_fortran_string_c2f(c_key, key, key_len))) {
        c_ierr = OMPI_ERRHANDLER_ILWOKE(MPI_COMM_WORLD, ret, FUNC_NAME);
        if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
        return;
    }
}
