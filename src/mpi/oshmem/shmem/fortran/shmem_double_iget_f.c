/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2013 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "oshmem_config.h"
#include "oshmem/shmem/fortran/bindings.h"
#include "oshmem/include/shmem.h"
#include "oshmem/shmem/shmem_api_logger.h"
#include "oshmem/runtime/runtime.h"
#include "oshmem/mca/spml/spml.h"
#include "ompi/datatype/ompi_datatype.h"
#include "stdio.h"

#if OSHMEM_PROFILING
#include "oshmem/shmem/fortran/profile/pbindings.h"
SHMEM_GENERATE_WEAK_BINDINGS(SHMEM_DOUBLE_IGET, shmem_double_iget)
#include "oshmem/shmem/fortran/profile/defines.h"
#endif

SHMEM_GENERATE_FORTRAN_BINDINGS_SUB (void,
        SHMEM_DOUBLE_IGET,
        shmem_double_iget_,
        shmem_double_iget__,
        shmem_double_iget_f,
        (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe),
        (target,source,tst,sst,len,pe) )

void shmem_double_iget_f(FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *tst, MPI_Fint *sst, MPI_Fint *len, MPI_Fint *pe)
{
    int i;
    int length = OMPI_FINT_2_INT(*len);
    int tst_c = OMPI_FINT_2_INT(*tst);
    int sst_c = OMPI_FINT_2_INT(*sst);

    size_t double_type_size = 0;
    ompi_datatype_type_size(&ompi_mpi_cplex.dt, &double_type_size);

    for (i=0; i<length; i++)
    {
        MCA_SPML_CALL(get(oshmem_ctx_default, (uint8_t *)FPTR_2_VOID_PTR(source) + i * sst_c * double_type_size,
            double_type_size,
            (uint8_t *)FPTR_2_VOID_PTR(target) + i * tst_c * double_type_size,
            OMPI_FINT_2_INT(*pe)));
    }
}

