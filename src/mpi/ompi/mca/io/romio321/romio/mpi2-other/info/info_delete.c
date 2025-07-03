/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/* 
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "mpioimpl.h"

#ifdef HAVE_WEAK_SYMBOLS

#if defined(HAVE_PRAGMA_WEAK)
#pragma weak MPI_Info_delete = PMPI_Info_delete
#elif defined(HAVE_PRAGMA_HP_SEC_DEF)
#pragma _HP_SECONDARY_DEF PMPI_Info_delete MPI_Info_delete
#elif defined(HAVE_PRAGMA_CRI_DUP)
#pragma _CRI duplicate MPI_Info_delete as PMPI_Info_delete
/* end of weak pragmas */
#endif

/* Include mapping from MPI->PMPI */
#define MPIO_BUILD_PROFILING
#include "mpioprof.h"
#endif

/*@
    MPI_Info_delete - Deletes a (key,value) pair from info

Input Parameters:
. info - info object (handle)
. key - key (string)

.N fortran
@*/
int MPI_Info_delete(MPI_Info info, char *key)
{
    MPI_Info prev, lwrr;
    int done;

    if ((info <= (MPI_Info) 0) || (info->cookie != MPIR_INFO_COOKIE)) {
        FPRINTF(stderr, "MPI_Info_delete: Invalid info object\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (key <= (char *) 0) {
	FPRINTF(stderr, "MPI_Info_delete: key is an invalid address\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (strlen(key) > MPI_MAX_INFO_KEY) {
	FPRINTF(stderr, "MPI_Info_delete: key is longer than MPI_MAX_INFO_KEY\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!strlen(key)) {
	FPRINTF(stderr, "MPI_Info_delete: key is a null string\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    prev = info;
    lwrr = info->next;
    done = 0;

    while (lwrr) {
	if (!strcmp(lwrr->key, key)) {
	    ADIOI_Free(lwrr->key); 
	    ADIOI_Free(lwrr->value);
	    prev->next = lwrr->next;
	    ADIOI_Free(lwrr);
	    done = 1;
	    break;
	}
	prev = lwrr;
	lwrr = lwrr->next;
    }

    if (!done) {
	FPRINTF(stderr, "MPI_Info_delete: key not defined in info\n");
        MPI_Abort(MPI_COMM_WORLD, 1);	
    }

    return MPI_SUCCESS;
}
