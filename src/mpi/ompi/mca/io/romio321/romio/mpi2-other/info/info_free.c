/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/* 
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "mpioimpl.h"

#ifdef HAVE_WEAK_SYMBOLS

#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Info_free = PMPI_Info_free
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Info_free MPI_Info_free
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Info_free as PMPI_Info_free
/* end of weak pragmas */
#endif

/* Include mapping from MPI->PMPI */
#define MPIO_BUILD_PROFILING
#include "mpioprof.h"
#endif

/*@
    MPI_Info_free - Frees an info object

Input Parameters:
. info - info object (handle)

.N fortran
@*/
int MPI_Info_free(MPI_Info *info)
{
    MPI_Info lwrr, next;

    if ((*info <= (MPI_Info) 0) || ((*info)->cookie != MPIR_INFO_COOKIE)) {
        FPRINTF(stderr, "MPI_Info_free: Invalid info object\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    lwrr = (*info)->next;
    ADIOI_Free(*info);
    *info = MPI_INFO_NULL;

    while (lwrr) {
	next = lwrr->next;
	ADIOI_Free(lwrr->key);
	ADIOI_Free(lwrr->value);
	ADIOI_Free(lwrr);
	lwrr = next;
    }

    return MPI_SUCCESS;
}
