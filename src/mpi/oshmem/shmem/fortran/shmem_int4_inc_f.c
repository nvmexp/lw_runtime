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
#include "oshmem/mca/atomic/atomic.h"
#include "ompi/datatype/ompi_datatype.h"
#include "oshmem/op/op.h"
#include "stdio.h"

#if OSHMEM_PROFILING
#include "oshmem/shmem/fortran/profile/pbindings.h"
SHMEM_GENERATE_WEAK_BINDINGS(SHMEM_INT4_INC, shmem_int4_inc)
#include "oshmem/shmem/fortran/profile/defines.h"
#endif

SHMEM_GENERATE_FORTRAN_BINDINGS_SUB (void,
        SHMEM_INT4_INC,
        shmem_int4_inc_,
        shmem_int4_inc__,
        shmem_int4_inc_f,
        (FORTRAN_POINTER_T target, MPI_Fint *pe),
        (target,pe) )

void shmem_int4_inc_f(FORTRAN_POINTER_T target, MPI_Fint *pe)
{
    ompi_fortran_integer4_t value = 1;

    MCA_ATOMIC_CALL(add(oshmem_ctx_default, FPTR_2_VOID_PTR(target),
        value,
        sizeof(value),
        OMPI_FINT_2_INT(*pe)));
}
