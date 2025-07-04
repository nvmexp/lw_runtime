/*
 * Copyright (c) 2013-2018 Mellanox Technologies, Inc.
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
#include "oshmem/constants.h"
#include "oshmem/mca/scoll/scoll.h"
#include "oshmem/proc/proc.h"
#include "oshmem/op/op.h"

#if OSHMEM_PROFILING
#include "oshmem/shmem/fortran/profile/pbindings.h"
SHMEM_GENERATE_WEAK_BINDINGS(SHMEM_BROADCAST4, shmem_broadcast4)
SHMEM_GENERATE_WEAK_BINDINGS(SHMEM_BROADCAST8, shmem_broadcast8)
SHMEM_GENERATE_WEAK_BINDINGS(SHMEM_BROADCAST32, shmem_broadcast32)
SHMEM_GENERATE_WEAK_BINDINGS(SHMEM_BROADCAST64, shmem_broadcast64)
#include "oshmem/shmem/fortran/profile/defines.h"
#endif

SHMEM_GENERATE_FORTRAN_BINDINGS_SUB (void,
        SHMEM_BROADCAST4,
        shmem_broadcast4_,
        shmem_broadcast4__,
        shmem_broadcast4_f,
        (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_root, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync),
        (target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync))

SHMEM_GENERATE_FORTRAN_BINDINGS_SUB (void,
        SHMEM_BROADCAST8,
        shmem_broadcast8_,
        shmem_broadcast8__,
        shmem_broadcast8_f,
        (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_root, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync),
        (target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync))

SHMEM_GENERATE_FORTRAN_BINDINGS_SUB (void,
        SHMEM_BROADCAST32,
        shmem_broadcast32_,
        shmem_broadcast32__,
        shmem_broadcast32_f,
        (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_root, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync),
        (target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync))

SHMEM_GENERATE_FORTRAN_BINDINGS_SUB (void,
        SHMEM_BROADCAST64,
        shmem_broadcast64_,
        shmem_broadcast64__,
        shmem_broadcast64_f,
        (FORTRAN_POINTER_T target, FORTRAN_POINTER_T source, MPI_Fint *nlong, MPI_Fint *PE_root, MPI_Fint *PE_start, MPI_Fint * logPE_stride, MPI_Fint *PE_size, FORTRAN_POINTER_T pSync),
        (target, source, nlong, PE_root, PE_start, logPE_stride, PE_size, pSync))

#define SHMEM_BROADCAST(F_NAME, T_NAME) void F_NAME(FORTRAN_POINTER_T target, \
    FORTRAN_POINTER_T source, \
    MPI_Fint *nlong,\
    MPI_Fint *PE_root, \
    MPI_Fint *PE_start, \
    MPI_Fint *logPE_stride, \
    MPI_Fint *PE_size, \
    FORTRAN_POINTER_T pSync)\
{\
    int rc;\
    oshmem_group_t *group;\
    int rel_PE_root = 0;\
    oshmem_op_t* op = T_NAME;\
\
    if ((0 <= OMPI_FINT_2_INT(*PE_root)) && \
            (OMPI_FINT_2_INT(*PE_root) < OMPI_FINT_2_INT(*PE_size)))\
    {\
        group = oshmem_proc_group_create_nofail(OMPI_FINT_2_INT(*PE_start), \
                (1 << OMPI_FINT_2_INT(*logPE_stride)), \
                OMPI_FINT_2_INT(*PE_size));\
        if (OMPI_FINT_2_INT(*PE_root) >= group->proc_count)\
        {\
            rc = OSHMEM_ERROR;\
            goto out;\
        }\
        \
        /* Define actual PE using relative in active set */\
        rel_PE_root = oshmem_proc_pe(group->proc_array[OMPI_FINT_2_INT(*PE_root)]);\
        \
        /* Call collective broadcast operation */\
        rc = group->g_scoll.scoll_broadcast( group, \
                rel_PE_root, \
                FPTR_2_VOID_PTR(target), \
                FPTR_2_VOID_PTR(source), \
                OMPI_FINT_2_INT(*nlong) * op->dt_size, \
                FPTR_2_VOID_PTR(pSync), \
                true, \
                SCOLL_DEFAULT_ALG );\
    out: \
        oshmem_proc_group_destroy(group);\
        RUNTIME_CHECK_RC(rc); \
  }\
}

SHMEM_BROADCAST(shmem_broadcast4_f, oshmem_op_prod_fint4)
SHMEM_BROADCAST(shmem_broadcast8_f, oshmem_op_prod_fint8)
SHMEM_BROADCAST(shmem_broadcast32_f, oshmem_op_prod_fint4)
SHMEM_BROADCAST(shmem_broadcast64_f, oshmem_op_prod_fint8)
