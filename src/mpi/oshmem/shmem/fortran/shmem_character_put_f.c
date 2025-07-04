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
SHMEM_GENERATE_WEAK_BINDINGS(SHMEM_CHARACTER_PUT, shmem_character_put)
#include "oshmem/shmem/fortran/profile/defines.h"
#endif

SHMEM_GENERATE_FORTRAN_BINDINGS_SUB (void,
        SHMEM_CHARACTER_PUT,
        shmem_character_put_,
        shmem_character_put__,
        shmem_character_put_f,
        (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe),
        (target,source,length,pe) )

void shmem_character_put_f(FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *length, MPI_Fint *pe)
{
    size_t character_type_size = 0;
    ompi_datatype_type_size(&ompi_mpi_character.dt, &character_type_size);

    MCA_SPML_CALL(put(oshmem_ctx_default, FPTR_2_VOID_PTR(target),
        OMPI_FINT_2_INT(*length) * character_type_size,
        FPTR_2_VOID_PTR(source),
        OMPI_FINT_2_INT(*pe)));
}


