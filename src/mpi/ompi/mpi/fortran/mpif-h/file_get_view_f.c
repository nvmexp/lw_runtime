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
 * Copyright (c) 2011-2012 Cisco Systems, Inc.  All rights reserved.
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
#include "ompi/mpi/fortran/base/fortran_base_strings.h"
#include "ompi/file/file.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_FILE_GET_VIEW = ompi_file_get_view_f
#pragma weak pmpi_file_get_view = ompi_file_get_view_f
#pragma weak pmpi_file_get_view_ = ompi_file_get_view_f
#pragma weak pmpi_file_get_view__ = ompi_file_get_view_f

#pragma weak PMPI_File_get_view_f = ompi_file_get_view_f
#pragma weak PMPI_File_get_view_f08 = ompi_file_get_view_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_FILE_GET_VIEW,
                           pmpi_file_get_view,
                           pmpi_file_get_view_,
                           pmpi_file_get_view__,
                           pompi_file_get_view_f,
                           (MPI_Fint *fh, MPI_Offset *disp, MPI_Fint *etype, MPI_Fint *filetype, char *datarep, MPI_Fint *ierr, int datarep_len),
                           (fh, disp, etype, filetype, datarep, ierr, datarep_len) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_FILE_GET_VIEW = ompi_file_get_view_f
#pragma weak mpi_file_get_view = ompi_file_get_view_f
#pragma weak mpi_file_get_view_ = ompi_file_get_view_f
#pragma weak mpi_file_get_view__ = ompi_file_get_view_f

#pragma weak MPI_File_get_view_f = ompi_file_get_view_f
#pragma weak MPI_File_get_view_f08 = ompi_file_get_view_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_FILE_GET_VIEW,
                           mpi_file_get_view,
                           mpi_file_get_view_,
                           mpi_file_get_view__,
                           ompi_file_get_view_f,
                           (MPI_Fint *fh, MPI_Offset *disp, MPI_Fint *etype, MPI_Fint *filetype, char *datarep, MPI_Fint *ierr, int datarep_len),
                           (fh, disp, etype, filetype, datarep, ierr, datarep_len) )
#else
#define ompi_file_get_view_f pompi_file_get_view_f
#endif
#endif


void ompi_file_get_view_f(MPI_Fint *fh, MPI_Offset *disp,
			 MPI_Fint *etype, MPI_Fint *filetype,
			 char *datarep, MPI_Fint *ierr, int datarep_len)
{
    int c_ierr;
    MPI_File c_fh = PMPI_File_f2c(*fh);
    MPI_Datatype c_etype, c_filetype;
    MPI_Offset c_disp;
    char c_datarep[MPI_MAX_DATAREP_STRING];

    c_ierr = PMPI_File_get_view(c_fh, &c_disp, &c_etype,
                               &c_filetype, c_datarep);
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);

    if (MPI_SUCCESS == c_ierr) {
        *disp = (MPI_Offset) c_disp;
        *etype = PMPI_Type_c2f(c_etype);
        *filetype = PMPI_Type_c2f(c_filetype);
        ompi_fortran_string_c2f(c_datarep, datarep, datarep_len);
    }
}
