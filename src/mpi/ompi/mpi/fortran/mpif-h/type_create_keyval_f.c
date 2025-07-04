/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007-2012 Cisco Systems, Inc.  All rights reserved.
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
#include "ompi/communicator/communicator.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_TYPE_CREATE_KEYVAL = ompi_type_create_keyval_f
#pragma weak pmpi_type_create_keyval = ompi_type_create_keyval_f
#pragma weak pmpi_type_create_keyval_ = ompi_type_create_keyval_f
#pragma weak pmpi_type_create_keyval__ = ompi_type_create_keyval_f

#pragma weak PMPI_Type_create_keyval_f = ompi_type_create_keyval_f
#pragma weak PMPI_Type_create_keyval_f08 = ompi_type_create_keyval_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_TYPE_CREATE_KEYVAL,
                           pmpi_type_create_keyval,
                           pmpi_type_create_keyval_,
                           pmpi_type_create_keyval__,
                           pompi_type_create_keyval_f,
                           (ompi_aint_copy_attr_function* type_copy_attr_fn, ompi_aint_delete_attr_function* type_delete_attr_fn, MPI_Fint *type_keyval, MPI_Aint *extra_state, MPI_Fint *ierr),
                           (type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_TYPE_CREATE_KEYVAL = ompi_type_create_keyval_f
#pragma weak mpi_type_create_keyval = ompi_type_create_keyval_f
#pragma weak mpi_type_create_keyval_ = ompi_type_create_keyval_f
#pragma weak mpi_type_create_keyval__ = ompi_type_create_keyval_f

#pragma weak MPI_Type_create_keyval_f = ompi_type_create_keyval_f
#pragma weak MPI_Type_create_keyval_f08 = ompi_type_create_keyval_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_TYPE_CREATE_KEYVAL,
                           mpi_type_create_keyval,
                           mpi_type_create_keyval_,
                           mpi_type_create_keyval__,
                           ompi_type_create_keyval_f,
                           (ompi_aint_copy_attr_function* type_copy_attr_fn, ompi_aint_delete_attr_function* type_delete_attr_fn, MPI_Fint *type_keyval, MPI_Aint *extra_state, MPI_Fint *ierr),
                           (type_copy_attr_fn, type_delete_attr_fn, type_keyval, extra_state, ierr) )
#else
#define ompi_type_create_keyval_f pompi_type_create_keyval_f
#endif
#endif

static char FUNC_NAME[] = "MPI_Type_create_keyval_f";

void ompi_type_create_keyval_f(ompi_aint_copy_attr_function* type_copy_attr_fn,
                               ompi_aint_delete_attr_function* type_delete_attr_fn,
                               MPI_Fint *type_keyval, MPI_Aint *extra_state, MPI_Fint *ierr)
{
    int ret, c_ierr;
    OMPI_SINGLE_NAME_DECL(type_keyval);
    ompi_attribute_fn_ptr_union_t copy_fn;
    ompi_attribute_fn_ptr_union_t del_fn;

    copy_fn.attr_aint_copy_fn = type_copy_attr_fn;
    del_fn.attr_aint_delete_fn = type_delete_attr_fn;

    /* Note that we only set the "F77" bit and exclude the "F77_OLD"
       bit, indicating that the callbacks should use the new MPI-2
       INTEGER(KIND=MPI_ADDRESS_KIND)-parameter functions (as opposed
       to the old MPI-1 INTEGER-parameter functions). */

    ret = ompi_attr_create_keyval_aint(TYPE_ATTR, copy_fn, del_fn,
                                       OMPI_SINGLE_NAME_COLWERT(type_keyval), *extra_state, OMPI_KEYVAL_F77,
                                       NULL);

    if (MPI_SUCCESS != ret) {
        c_ierr = OMPI_ERRHANDLER_ILWOKE(MPI_COMM_WORLD,
                                        MPI_ERR_OTHER,
                                        FUNC_NAME);
        if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
    } else {
        if (NULL != ierr) *ierr = OMPI_INT_2_FINT(MPI_SUCCESS);
        OMPI_SINGLE_INT_2_FINT(type_keyval);
    }
}
