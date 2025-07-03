/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/* 
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "adio.h"
#include "adio_extern.h"

void ADIO_End(int *error_code)
{
    ADIOI_Flatlist_node *lwrr, *next;
    ADIOI_Datarep *datarep, *datarep_next;
    
/*    FPRINTF(stderr, "reached end\n"); */

    /* if a default errhandler was set on MPI_FILE_NULL then we need to ensure
     * that our reference to that errhandler is released */
/* Open MPI: The call to PMPI_File_set_errhandler has to be done in romio/src/io_romio_file_open.c
   in routine mca_io_romio_file_close()
*/
#if 0
    PMPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_RETURN);
#endif

/* delete the flattened datatype list */
    lwrr = ADIOI_Flatlist;
    while (lwrr) {
	if (lwrr->blocklens) ADIOI_Free(lwrr->blocklens);
	if (lwrr->indices) ADIOI_Free(lwrr->indices);
	next = lwrr->next;
	ADIOI_Free(lwrr);
	lwrr = next;
    }
    ADIOI_Flatlist = NULL;

/* free file and info tables used for Fortran interface */
    if (ADIOI_Ftable) ADIOI_Free(ADIOI_Ftable);
#ifndef HAVE_MPI_INFO
    if (MPIR_Infotable) ADIOI_Free(MPIR_Infotable);
#endif


/* free the memory allocated for a new data representation, if any */
    datarep = ADIOI_Datarep_head;
    while (datarep) {
        datarep_next = datarep->next;
        ADIOI_Free(datarep->name);
        ADIOI_Free(datarep);
        datarep = datarep_next;
    }

    if( ADIOI_syshints != MPI_INFO_NULL)
	    MPI_Info_free(&ADIOI_syshints);

    MPI_Op_free(&ADIO_same_amode);

    *error_code = MPI_SUCCESS;
}



/* This is the delete callback function associated with
   ADIO_Init_keyval when MPI_COMM_SELF is freed */

int ADIOI_End_call(MPI_Comm comm, int keyval, void *attribute_val, void
		  *extra_state)
{
    int error_code;

    ADIOI_UNREFERENCED_ARG(comm);
    ADIOI_UNREFERENCED_ARG(attribute_val);
    ADIOI_UNREFERENCED_ARG(extra_state);

    MPI_Comm_free_keyval (&keyval);

    /* The end call will be called after all possible uses of this keyval, even
     * if a file was opened with MPI_COMM_SELF.  Note, this assumes LIFO
     * MPI_COMM_SELF attribute destruction behavior mandated by MPI-2.2. */
    if (ADIOI_cb_config_list_keyval != MPI_KEYVAL_ILWALID)
        MPI_Comm_free_keyval (&ADIOI_cb_config_list_keyval);

    ADIO_End(&error_code);
    return error_code;
}
