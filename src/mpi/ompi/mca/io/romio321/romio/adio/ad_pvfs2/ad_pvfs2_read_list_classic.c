/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- 
 * vim: ts=8 sts=4 sw=4 noexpandtab 
 * 
 *   Copyright (C) 2008 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "adio.h"
#include "adio_extern.h"
#include "ad_pvfs2.h"

#include "ad_pvfs2_common.h"

void ADIOI_PVFS2_OldReadStrided(ADIO_File fd, void *buf, int count,
			     MPI_Datatype datatype, int file_ptr_type,
			     ADIO_Offset offset, ADIO_Status *status, int
			     *error_code)
{
    /* offset is in units of etype relative to the filetype. */
    ADIOI_Flatlist_node *flat_buf, *flat_file;
    int i, j, k,  brd_size, frd_size=0, st_index=0;
    int sum, n_etypes_in_filetype, size_in_filetype;
    MPI_Count bufsize;
    int n_filetypes, etype_in_filetype;
    ADIO_Offset abs_off_in_filetype=0;
    MPI_Count filetype_size, etype_size, buftype_size;
    MPI_Aint filetype_extent, buftype_extent, filetype_lb, buftype_lb; 
    int buf_count, buftype_is_contig, filetype_is_contig;
    ADIO_Offset off, disp, start_off, initial_off;
    int flag, st_frd_size, st_n_filetypes;

    int mem_list_count, file_list_count;
    PVFS_size *mem_offsets;
    int64_t *file_offsets;
    int *mem_lengths;
    int32_t *file_lengths;
    int total_blks_to_read;

    int max_mem_list, max_file_list;

    int b_blks_read;
    int f_data_read;
    int size_read=0, n_read_lists, extra_blks;

    int end_brd_size, end_frd_size;
    int start_k, start_j, new_file_read, new_buffer_read;
    int start_mem_offset;
    PVFS_Request mem_req, file_req;
    ADIOI_PVFS2_fs * pvfs_fs;
    PVFS_sysresp_io resp_io;
    int err_flag=0;
    MPI_Offset total_bytes_read = 0;
    static char myname[] = "ADIOI_PVFS2_ReadStrided";

#define MAX_ARRAY_SIZE 64

    *error_code = MPI_SUCCESS;  /* changed below if error */

    ADIOI_Datatype_iscontig(datatype, &buftype_is_contig);
    ADIOI_Datatype_iscontig(fd->filetype, &filetype_is_contig);

    /* the HDF5 tests showed a bug in this list processing code (see many many
     * lines down below).  We added a workaround, but common HDF5 file types
     * are actually contiguous and do not need the expensive workarond */
    if (!filetype_is_contig) {
	flat_file = ADIOI_Flatlist;
	while (flat_file->type != fd->filetype) flat_file = flat_file->next;
	if (flat_file->count == 1 && !buftype_is_contig)
	    filetype_is_contig = 1;
    }

    MPI_Type_size_x(fd->filetype, &filetype_size);
    if ( ! filetype_size ) {
#ifdef HAVE_STATUS_SET_BYTES
	MPIR_Status_set_bytes(status, datatype, 0);
#endif
	*error_code = MPI_SUCCESS; 
	return;
    }

    MPI_Type_get_extent(fd->filetype, &filetype_lb, &filetype_extent);
    MPI_Type_size_x(datatype, &buftype_size);
    MPI_Type_get_extent(datatype, &buftype_lb, &buftype_extent);
    etype_size = fd->etype_size;

    bufsize = buftype_size * count;
    
    pvfs_fs = (ADIOI_PVFS2_fs*)fd->fs_ptr;

    if (!buftype_is_contig && filetype_is_contig) {

/* noncontiguous in memory, contiguous in file. */
       int64_t file_offset;
	int32_t file_length;

	flat_buf = ADIOI_Flatten_and_find(datatype);

	off = (file_ptr_type == ADIO_INDIVIDUAL) ? fd->fp_ind : 
	    fd->disp + etype_size * offset;

	file_list_count = 1;
	file_offset = off;
	file_length = 0;
	total_blks_to_read = count*flat_buf->count;
	b_blks_read = 0;

	/* allocate arrays according to max usage */
	if (total_blks_to_read > MAX_ARRAY_SIZE)
	    mem_list_count = MAX_ARRAY_SIZE;
	else mem_list_count = total_blks_to_read;
	mem_offsets = (PVFS_size*)ADIOI_Malloc(mem_list_count*sizeof(PVFS_size));
	mem_lengths = (int*)ADIOI_Malloc(mem_list_count*sizeof(int));

	/* TODO: CHECK RESULTS OF MEMORY ALLOCATION */

	j = 0;
	/* step through each block in memory, filling memory arrays */
	while (b_blks_read < total_blks_to_read) {
	    for (i=0; i<flat_buf->count; i++) {
		mem_offsets[b_blks_read % MAX_ARRAY_SIZE] = 
		    /* TODO: fix this compiler warning */
		    ((PVFS_size)buf + j*buftype_extent + flat_buf->indices[i]);
		mem_lengths[b_blks_read % MAX_ARRAY_SIZE] = 
		    flat_buf->blocklens[i];
		file_length += flat_buf->blocklens[i];
		b_blks_read++;
		if (!(b_blks_read % MAX_ARRAY_SIZE) ||
		    (b_blks_read == total_blks_to_read)) {

		    /* in the case of the last read list call,
		       adjust mem_list_count */
		    if (b_blks_read == total_blks_to_read) {
		        mem_list_count = total_blks_to_read % MAX_ARRAY_SIZE;
			/* in case last read list call fills max arrays */
			if (!mem_list_count) mem_list_count = MAX_ARRAY_SIZE;
		    }
		    err_flag = PVFS_Request_hindexed(mem_list_count, 
			    mem_lengths, mem_offsets, PVFS_BYTE, &mem_req);
		    if (err_flag < 0) break;
		    err_flag = PVFS_Request_contiguous(file_length,
			    PVFS_BYTE, &file_req);
		    if (err_flag < 0) break;
#ifdef ADIOI_MPE_LOGGING
                    MPE_Log_event( ADIOI_MPE_read_a, 0, NULL );
#endif
		    err_flag = PVFS_sys_read(pvfs_fs->object_ref, file_req, 
			    file_offset, PVFS_BOTTOM, mem_req,
			    &(pvfs_fs->credentials), &resp_io);
#ifdef ADIOI_MPE_LOGGING
                    MPE_Log_event( ADIOI_MPE_read_b, 0, NULL );
#endif
		    /* --BEGIN ERROR HANDLING-- */
		    if (err_flag != 0) {
			*error_code = MPIO_Err_create_code(MPI_SUCCESS,
							   MPIR_ERR_RECOVERABLE,
							   myname, __LINE__,
							   ADIOI_PVFS2_error_colwert(err_flag),
							   "Error in PVFS_sys_read", 0);
			goto error_state;
		    }
		    PVFS_Request_free(&mem_req);
		    PVFS_Request_free(&file_req);
		    total_bytes_read += resp_io.total_completed;
		    /* --END ERROR HANDLING-- */
		  
		    /* in the case of error or the last read list call, 
		     * leave here */
		    if (err_flag || b_blks_read == total_blks_to_read) break;

		    file_offset += file_length;
		    file_length = 0;
		} 
	    } /* for (i=0; i<flat_buf->count; i++) */
	    j++;
	} /* while (b_blks_read < total_blks_to_read) */
	ADIOI_Free(mem_offsets);
	ADIOI_Free(mem_lengths);

        if (file_ptr_type == ADIO_INDIVIDUAL) 
	    fd->fp_ind += total_bytes_read;

	fd->fp_sys_posn = -1;  /* set it to null. */

#ifdef HAVE_STATUS_SET_BYTES
	MPIR_Status_set_bytes(status, datatype, bufsize);
	/* This isa temporary way of filling in status.  The right way is to
	   keep tracke of how much data was actually read adn placed in buf
	   by ADIOI_BUFFERED_READ. */
#endif
	ADIOI_Delete_flattened(datatype);

	return;
    } /* if (!buftype_is_contig && filetype_is_contig) */

    /* know file is noncontiguous from above */
    /* noncontiguous in file */

    /* filetype already flattened in ADIO_Open */
    flat_file = ADIOI_Flatlist;
    while (flat_file->type != fd->filetype) flat_file = flat_file->next;

    disp = fd->disp;
    initial_off = offset;


    /* for each case - ADIO_Individual pointer or explicit, find the file
       offset in bytes (offset), n_filetypes (how many filetypes into
       file to start), frd_size (remaining amount of data in present
       file block), and st_index (start point in terms of blocks in
       starting filetype) */
    if (file_ptr_type == ADIO_INDIVIDUAL) {
        offset = fd->fp_ind; /* in bytes */
	n_filetypes = -1;
	flag = 0;
	while (!flag) {
	    n_filetypes++;
	    for (i=0; i<flat_file->count; i++) {
	        if (disp + flat_file->indices[i] + 
		    ((ADIO_Offset) n_filetypes)*filetype_extent +
		    flat_file->blocklens[i]  >= offset) {
		    st_index = i;
		    frd_size = (int) (disp + flat_file->indices[i] + 
				    ((ADIO_Offset) n_filetypes)*filetype_extent
				      + flat_file->blocklens[i] - offset);
		    flag = 1;
		    break;
		}
	    }
	} /* while (!flag) */
    } /* if (file_ptr_type == ADIO_INDIVIDUAL) */
    else {
        n_etypes_in_filetype = filetype_size/etype_size;
	n_filetypes = (int) (offset / n_etypes_in_filetype);
	etype_in_filetype = (int) (offset % n_etypes_in_filetype);
	size_in_filetype = etype_in_filetype * etype_size;
	
	sum = 0;
	for (i=0; i<flat_file->count; i++) {
	    sum += flat_file->blocklens[i];
	    if (sum > size_in_filetype) {
	        st_index = i;
		frd_size = sum - size_in_filetype;
		abs_off_in_filetype = flat_file->indices[i] +
		    size_in_filetype - (sum - flat_file->blocklens[i]);
		break;
	    }
	}
	
	/* abs. offset in bytes in the file */
	offset = disp + ((ADIO_Offset) n_filetypes)*filetype_extent + 
	    abs_off_in_filetype;
    } /* else [file_ptr_type != ADIO_INDIVIDUAL] */

    start_off = offset;
    st_frd_size = frd_size;
    st_n_filetypes = n_filetypes;
    
    if (buftype_is_contig && !filetype_is_contig) {

/* contiguous in memory, noncontiguous in file. should be the most
   common case. */

       int mem_length=0;
	intptr_t mem_offset;
	
	i = 0;
	j = st_index;
	n_filetypes = st_n_filetypes;
	
	mem_list_count = 1;
	
	/* determine how many blocks in file to read */
	f_data_read = ADIOI_MIN(st_frd_size, bufsize);
	total_blks_to_read = 1;
	if (j < (flat_file->count-1)) j++;
	else {
	    j = 0;
	    n_filetypes++;
	}
	while (f_data_read < bufsize) {
	    f_data_read += flat_file->blocklens[j];
	    total_blks_to_read++;
	    if (j<(flat_file->count-1)) j++;
	    else j = 0;	
	}
      
	j = st_index;
	n_filetypes = st_n_filetypes;
	n_read_lists = total_blks_to_read/MAX_ARRAY_SIZE;
	extra_blks = total_blks_to_read%MAX_ARRAY_SIZE;
	
	mem_offset = (intptr_t)buf;
	mem_lengths = 0;
	
	/* if at least one full readlist, allocate file arrays
	   at max array size and don't free until very end */
	if (n_read_lists) {
	    file_offsets = (int64_t*)ADIOI_Malloc(MAX_ARRAY_SIZE*
						  sizeof(int64_t));
	    file_lengths = (int32_t*)ADIOI_Malloc(MAX_ARRAY_SIZE*
						  sizeof(int32_t));
	}
	/* if there's no full readlist allocate file arrays according
	   to needed size (extra_blks) */
	else {
	    file_offsets = (int64_t*)ADIOI_Malloc(extra_blks*
						  sizeof(int64_t));
	    file_lengths = (int32_t*)ADIOI_Malloc(extra_blks*
						  sizeof(int32_t));
	}
	
	/* for file arrays that are of MAX_ARRAY_SIZE, build arrays */
	for (i=0; i<n_read_lists; i++) {
	    file_list_count = MAX_ARRAY_SIZE;
	    if(!i) {
	        file_offsets[0] = offset;
		file_lengths[0] = st_frd_size;
		mem_length = st_frd_size;
	    }
	    for (k=0; k<MAX_ARRAY_SIZE; k++) {
	        if (i || k) {
		    file_offsets[k] = disp + 
			((ADIO_Offset)n_filetypes)*filetype_extent
		      + flat_file->indices[j];
		    file_lengths[k] = flat_file->blocklens[j];
		    mem_lengths += file_lengths[k];
		}
		if (j<(flat_file->count - 1)) j++;
		else {
		    j = 0;
		    n_filetypes++;
		}
	    } /* for (k=0; k<MAX_ARRAY_SIZE; k++) */
	    err_flag = PVFS_Request_contiguous(mem_length,
					       PVFS_BYTE, &mem_req);
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_Request_contiguous (memory)", 0);
		goto error_state;
	    }
	    /* --END ERROR HANDLING-- */

	    err_flag = PVFS_Request_hindexed(file_list_count, file_lengths, 
					     file_offsets, PVFS_BYTE,
					     &file_req);
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_Request_hindexed (file)", 0);
		goto error_state;
	    }
	    /* --END ERROR HANDLING-- */

	    /* PVFS_Request_hindexed already expresses the offsets into the
	     * file, so we should not pass in an offset if we are using
	     * hindexed for the file type */
#ifdef ADIOI_MPE_LOGGING
            MPE_Log_event( ADIOI_MPE_read_a, 0, NULL );
#endif
	    err_flag = PVFS_sys_read(pvfs_fs->object_ref, file_req, 0, 
				     (void *)mem_offset, mem_req,
				     &(pvfs_fs->credentials), &resp_io);
#ifdef ADIOI_MPE_LOGGING
            MPE_Log_event( ADIOI_MPE_read_b, 0, NULL );
#endif
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_sys_read", 0);
		goto error_state;
	    }
	    /* --END ERROR HANDING-- */
	    PVFS_Request_free(&mem_req);
	    PVFS_Request_free(&file_req);

	    total_bytes_read += resp_io.total_completed;

	    mem_offset += mem_length;
	    mem_lengths = 0;
	} /* for (i=0; i<n_read_lists; i++) */

	/* for file arrays smaller than MAX_ARRAY_SIZE (last read_list call) */
	if (extra_blks) {
	    file_list_count = extra_blks;
	    if(!i) {
	        file_offsets[0] = offset;
		file_lengths[0] = ADIOI_MIN(st_frd_size, bufsize);
	    }
	    for (k=0; k<extra_blks; k++) {
	        if(i || k) {
		    file_offsets[k] = disp + 
			((ADIO_Offset)n_filetypes)*filetype_extent +
			flat_file->indices[j];
		    if (k == (extra_blks - 1)) {
		        file_lengths[k] = bufsize - (int32_t) mem_lengths
			  - mem_offset + (int32_t)  buf;
		    }
		    else file_lengths[k] = flat_file->blocklens[j];
		} /* if(i || k) */
		mem_lengths += file_lengths[k];
		if (j<(flat_file->count - 1)) j++;
		else {
		    j = 0;
		    n_filetypes++;
		}
	    } /* for (k=0; k<extra_blks; k++) */
	    err_flag = PVFS_Request_contiguous(mem_length,
					       PVFS_BYTE, &mem_req);
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_Request_contiguous (memory)", 0);
		goto error_state;
	    }
	    /* --END ERROR HANDLING-- */

	    err_flag = PVFS_Request_hindexed(file_list_count, file_lengths, 
		    file_offsets, PVFS_BYTE, &file_req);
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_Request_hindexed (file)", 0);
		goto error_state;
	    }
	    /* --END ERROR HANDLING-- */

	    /* as above, use 0 for 'offset' when using hindexed file type */
#ifdef ADIOI_MPE_LOGGING
            MPE_Log_event( ADIOI_MPE_read_a, 0, NULL );
#endif
	    err_flag = PVFS_sys_read(pvfs_fs->object_ref, file_req, 0, 
		    (void *)mem_offset, mem_req, &(pvfs_fs->credentials), &resp_io);
#ifdef ADIOI_MPE_LOGGING
            MPE_Log_event( ADIOI_MPE_read_b, 0, NULL );
#endif
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_sys_read", 0);		
		goto error_state;
	    }
	    /* --END ERROR HANDLING-- */
	    PVFS_Request_free(&mem_req);
	    PVFS_Request_free(&file_req);
	    total_bytes_read += resp_io.total_completed;
	}
    }
    else {
/* noncontiguous in memory as well as in file */
      
	flat_buf = ADIOI_Flatten_and_find(datatype);

	size_read = 0;
	n_filetypes = st_n_filetypes;
	frd_size = st_frd_size;
	brd_size = flat_buf->blocklens[0];
	buf_count = 0;
	start_mem_offset = 0;
	start_k = k = 0;
	start_j = st_index;
	max_mem_list = 0;
	max_file_list = 0;

	/* run through and file max_file_list and max_mem_list so that you 
	   can allocate the file and memory arrays less than MAX_ARRAY_SIZE
	   if possible */

	while (size_read < bufsize) {
	    k = start_k;
	    new_buffer_read = 0;
	    mem_list_count = 0;
	    while ((mem_list_count < MAX_ARRAY_SIZE) && 
		   (new_buffer_read < bufsize-size_read)) {
	        /* find mem_list_count and file_list_count such that both are
		   less than MAX_ARRAY_SIZE, the sum of their lengths are
		   equal, and the sum of all the data read and data to be
		   read in the next immediate read list is less than
		   bufsize */
	        if(mem_list_count) {
		    if((new_buffer_read + flat_buf->blocklens[k] + 
			size_read) > bufsize) {
		        end_brd_size = new_buffer_read + 
			    flat_buf->blocklens[k] - (bufsize - size_read);
			new_buffer_read = bufsize - size_read;
		    }
		    else {
		        new_buffer_read += flat_buf->blocklens[k];
			end_brd_size = flat_buf->blocklens[k];
		    }
		}
		else {
		    if (brd_size > (bufsize - size_read)) {
		        new_buffer_read = bufsize - size_read;
			brd_size = new_buffer_read;
		    }
		    else new_buffer_read = brd_size;
		}
		mem_list_count++;
		k = (k + 1)%flat_buf->count;
	     } /* while ((mem_list_count < MAX_ARRAY_SIZE) && 
	       (new_buffer_read < bufsize-size_read)) */
	    j = start_j;
	    new_file_read = 0;
	    file_list_count = 0;
	    while ((file_list_count < MAX_ARRAY_SIZE) && 
		   (new_file_read < new_buffer_read)) {
	        if(file_list_count) {
		    if((new_file_read + flat_file->blocklens[j]) > 
		       new_buffer_read) {
		        end_frd_size = new_buffer_read - new_file_read;
			new_file_read = new_buffer_read;
			j--;
		    }
		    else {
		        new_file_read += flat_file->blocklens[j];
			end_frd_size = flat_file->blocklens[j];
		    }
		}
		else {
		    if (frd_size > new_buffer_read) {
		        new_file_read = new_buffer_read;
			frd_size = new_file_read;
		    }
		    else new_file_read = frd_size;
		}
		file_list_count++;
		if (j < (flat_file->count - 1)) j++;
		else j = 0;
		
		k = start_k;
		if ((new_file_read < new_buffer_read) && 
		    (file_list_count == MAX_ARRAY_SIZE)) {
		    new_buffer_read = 0;
		    mem_list_count = 0;
		    while (new_buffer_read < new_file_read) {
		        if(mem_list_count) {
			    if((new_buffer_read + flat_buf->blocklens[k]) >
			       new_file_read) {
			        end_brd_size = new_file_read - new_buffer_read;
				new_buffer_read = new_file_read;
				k--;
			    }
			    else {
			        new_buffer_read += flat_buf->blocklens[k];
				end_brd_size = flat_buf->blocklens[k];
			    }
			}
			else {
			    new_buffer_read = brd_size;
			    if (brd_size > (bufsize - size_read)) {
			        new_buffer_read = bufsize - size_read;
				brd_size = new_buffer_read;
			    }
			}
			mem_list_count++;
			k = (k + 1)%flat_buf->count;
		    } /* while (new_buffer_read < new_file_read) */
		} /* if ((new_file_read < new_buffer_read) && (file_list_count
		     == MAX_ARRAY_SIZE)) */
	    } /* while ((mem_list_count < MAX_ARRAY_SIZE) && 
		 (new_buffer_read < bufsize-size_read)) */

	    /*  fakes filling the readlist arrays of lengths found above  */
	    k = start_k;
	    j = start_j;
	    for (i=0; i<mem_list_count; i++) {	     
		if(i) {
		    if (i == (mem_list_count - 1)) {
			if (flat_buf->blocklens[k] == end_brd_size)
			    brd_size = flat_buf->blocklens[(k+1)%
							  flat_buf->count];
			else {
			    brd_size = flat_buf->blocklens[k] - end_brd_size;
			    k--;
			    buf_count--;
			}
		    }
		}
		buf_count++;
		k = (k + 1)%flat_buf->count;
	    } /* for (i=0; i<mem_list_count; i++) */
	    for (i=0; i<file_list_count; i++) {
		if (i) {
		    if (i == (file_list_count - 1)) {
			if (flat_file->blocklens[j] == end_frd_size)
			    frd_size = flat_file->blocklens[(j+1)%
							  flat_file->count];   
			else {
			    frd_size = flat_file->blocklens[j] - end_frd_size;
			    j--;
			}
		    }
		}
		if (j < flat_file->count - 1) j++;
		else {
		    j = 0;
		    n_filetypes++;
		}
	    } /* for (i=0; i<file_list_count; i++) */
	    size_read += new_buffer_read;
	    start_k = k;
	    start_j = j;
	    if (max_mem_list < mem_list_count)
	        max_mem_list = mem_list_count;
	    if (max_file_list < file_list_count)
	        max_file_list = file_list_count;
	} /* while (size_read < bufsize) */

	/* one last check before we actually carry out the operation:
	 * this code has hard-to-fix bugs when a noncontiguous file type has
	 * such large pieces that the sum of the lengths of the memory type is
	 * not larger than one of those pieces (and vice versa for large memory
	 * types and many pices of file types.  In these cases, give up and
	 * fall back to naive reads and writes.  The testphdf5 test created a
	 * type with two very large memory regions and 600 very small file
	 * regions.  The same test also created a type with one very large file
	 * region and many (700) very small memory regions.  both cases caused
	 * problems for this code */

	if ( ( (file_list_count == 1) && 
		    (new_file_read < flat_file->blocklens[0] ) ) ||
		((mem_list_count == 1) && 
		    (new_buffer_read < flat_buf->blocklens[0]) ) ||
		((file_list_count == MAX_ARRAY_SIZE) && 
		    (new_file_read < flat_buf->blocklens[0]) ) ||
		( (mem_list_count == MAX_ARRAY_SIZE) &&
		    (new_buffer_read < flat_file->blocklens[0])) )
	{

	    ADIOI_Delete_flattened(datatype);
	    ADIOI_GEN_ReadStrided_naive(fd, buf, count, datatype,
		    file_ptr_type, initial_off, status, error_code);
	    return;
	}

	mem_offsets = (PVFS_size*)ADIOI_Malloc(max_mem_list*sizeof(PVFS_size));
	mem_lengths = (int *)ADIOI_Malloc(max_mem_list*sizeof(int));
	file_offsets = (int64_t *)ADIOI_Malloc(max_file_list*sizeof(int64_t));
	file_lengths = (int32_t *)ADIOI_Malloc(max_file_list*sizeof(int32_t));
	    
	size_read = 0;
	n_filetypes = st_n_filetypes;
	frd_size = st_frd_size;
	brd_size = flat_buf->blocklens[0];
	buf_count = 0;
	start_mem_offset = 0;
	start_k = k = 0;
	start_j = st_index;

	/*  this section callwlates mem_list_count and file_list_count
	    and also finds the possibly odd sized last array elements
	    in new_frd_size and new_brd_size  */
	
	while (size_read < bufsize) {
	    k = start_k;
	    new_buffer_read = 0;
	    mem_list_count = 0;
	    while ((mem_list_count < MAX_ARRAY_SIZE) && 
		   (new_buffer_read < bufsize-size_read)) {
	        /* find mem_list_count and file_list_count such that both are
		   less than MAX_ARRAY_SIZE, the sum of their lengths are
		   equal, and the sum of all the data read and data to be
		   read in the next immediate read list is less than
		   bufsize */
	        if(mem_list_count) {
		    if((new_buffer_read + flat_buf->blocklens[k] + 
			size_read) > bufsize) {
		        end_brd_size = new_buffer_read + 
			    flat_buf->blocklens[k] - (bufsize - size_read);
			new_buffer_read = bufsize - size_read;
		    }
		    else {
		        new_buffer_read += flat_buf->blocklens[k];
			end_brd_size = flat_buf->blocklens[k];
		    }
		}
		else {
		    if (brd_size > (bufsize - size_read)) {
		        new_buffer_read = bufsize - size_read;
			brd_size = new_buffer_read;
		    }
		    else new_buffer_read = brd_size;
		}
		mem_list_count++;
		k = (k + 1)%flat_buf->count;
	     } /* while ((mem_list_count < MAX_ARRAY_SIZE) && 
	       (new_buffer_read < bufsize-size_read)) */
	    j = start_j;
	    new_file_read = 0;
	    file_list_count = 0;
	    while ((file_list_count < MAX_ARRAY_SIZE) && 
		   (new_file_read < new_buffer_read)) {
	        if(file_list_count) {
		    if((new_file_read + flat_file->blocklens[j]) > 
		       new_buffer_read) {
		        end_frd_size = new_buffer_read - new_file_read;
			new_file_read = new_buffer_read;
			j--;
		    }
		    else {
		        new_file_read += flat_file->blocklens[j];
			end_frd_size = flat_file->blocklens[j];
		    }
		}
		else {
		    if (frd_size > new_buffer_read) {
		        new_file_read = new_buffer_read;
			frd_size = new_file_read;
		    }
		    else new_file_read = frd_size;
		}
		file_list_count++;
		if (j < (flat_file->count - 1)) j++;
		else j = 0;
		
		k = start_k;
		if ((new_file_read < new_buffer_read) && 
		    (file_list_count == MAX_ARRAY_SIZE)) {
		    new_buffer_read = 0;
		    mem_list_count = 0;
		    while (new_buffer_read < new_file_read) {
		        if(mem_list_count) {
			    if((new_buffer_read + flat_buf->blocklens[k]) >
			       new_file_read) {
			        end_brd_size = new_file_read - new_buffer_read;
				new_buffer_read = new_file_read;
				k--;
			    }
			    else {
			        new_buffer_read += flat_buf->blocklens[k];
				end_brd_size = flat_buf->blocklens[k];
			    }
			}
			else {
			    new_buffer_read = brd_size;
			    if (brd_size > (bufsize - size_read)) {
			        new_buffer_read = bufsize - size_read;
				brd_size = new_buffer_read;
			    }
			}
			mem_list_count++;
			k = (k + 1)%flat_buf->count;
		    } /* while (new_buffer_read < new_file_read) */
		} /* if ((new_file_read < new_buffer_read) && (file_list_count
		     == MAX_ARRAY_SIZE)) */
	    } /* while ((mem_list_count < MAX_ARRAY_SIZE) && 
		 (new_buffer_read < bufsize-size_read)) */

	    /*  fills the allocated readlist arrays  */
	    k = start_k;
	    j = start_j;
	    for (i=0; i<mem_list_count; i++) {	     
	        mem_offsets[i] = ((PVFS_size)buf + buftype_extent*
					 (buf_count/flat_buf->count) +
					 (int)flat_buf->indices[k]);
		if(!i) {
		    mem_lengths[0] = brd_size;
		    mem_offsets[0] += flat_buf->blocklens[k] - brd_size;
		}
		else {
		    if (i == (mem_list_count - 1)) {
		        mem_lengths[i] = end_brd_size;
			if (flat_buf->blocklens[k] == end_brd_size)
			    brd_size = flat_buf->blocklens[(k+1)%
							  flat_buf->count];
			else {
			    brd_size = flat_buf->blocklens[k] - end_brd_size;
			    k--;
			    buf_count--;
			}
		    }
		    else {
		        mem_lengths[i] = flat_buf->blocklens[k];
		    }
		}
		buf_count++;
		k = (k + 1)%flat_buf->count;
	    } /* for (i=0; i<mem_list_count; i++) */
	    for (i=0; i<file_list_count; i++) {
	        file_offsets[i] = disp + flat_file->indices[j] + 
		    ((ADIO_Offset)n_filetypes) * filetype_extent;
	        if (!i) {
		    file_lengths[0] = frd_size;
		    file_offsets[0] += flat_file->blocklens[j] - frd_size;
		}
		else {
		    if (i == (file_list_count - 1)) {
		        file_lengths[i] = end_frd_size;
			if (flat_file->blocklens[j] == end_frd_size)
			    frd_size = flat_file->blocklens[(j+1)%
							  flat_file->count];   
			else {
			    frd_size = flat_file->blocklens[j] - end_frd_size;
			    j--;
			}
		    }
		    else file_lengths[i] = flat_file->blocklens[j];
		}
		if (j < flat_file->count - 1) j++;
		else {
		    j = 0;
		    n_filetypes++;
		}
	    } /* for (i=0; i<file_list_count; i++) */
	    err_flag = PVFS_Request_hindexed(mem_list_count, mem_lengths, 
		    mem_offsets, PVFS_BYTE, &mem_req);
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0 ) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_Request_hindexed (memory)", 0);
		goto error_state;
	    }
	    /* -- END ERROR HANDLING-- */
	    err_flag = PVFS_Request_hindexed(file_list_count, file_lengths, 
		    file_offsets, PVFS_BYTE, &file_req);
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_Request_hindexed (file)", 0);
		goto error_state;
	    }
	    /* --END ERROR HANDLING-- */

	    /* offset will be expressed in memory and file datatypes */
#ifdef ADIOI_MPE_LOGGING
            MPE_Log_event( ADIOI_MPE_read_a, 0, NULL );
#endif
	    err_flag = PVFS_sys_read(pvfs_fs->object_ref, file_req, 0, 
		    PVFS_BOTTOM, mem_req, &(pvfs_fs->credentials), &resp_io);
#ifdef ADIOI_MPE_LOGGING
            MPE_Log_event( ADIOI_MPE_read_b, 0, NULL );
#endif
	    /* --BEGIN ERROR HANDLING-- */
	    if (err_flag != 0) {
		*error_code = MPIO_Err_create_code(MPI_SUCCESS,
						   MPIR_ERR_RECOVERABLE,
						   myname, __LINE__,
						   ADIOI_PVFS2_error_colwert(err_flag),
						   "Error in PVFS_sys_read", 0);
	    }
	    /* --END ERROR HANDLING-- */
	    PVFS_Request_free(&mem_req);
	    PVFS_Request_free(&file_req);
	    total_bytes_read += resp_io.total_completed;
	    size_read += new_buffer_read;
	    start_k = k;
	    start_j = j;
	} /* while (size_read < bufsize) */
	ADIOI_Free(mem_offsets);
	ADIOI_Free(mem_lengths);
    }
    /* Other ADIO routines will colwert absolute bytes into counts of datatypes */
    /* when incrementing fp_ind, need to also take into account the file type:
     * consider an N-element 1-d subarray with a lb and ub: ( |---xxxxx-----|
     * if we wrote N elements, offset needs to point at beginning of type, not
     * at empty region at offset N+1) 
     *
     * As we dislwssed on mpich-discuss in may/june 2009, the code below might
     * look wierd, but by putting fp_ind at the last byte written, the next
     * time we run through the strided code we'll update the fp_ind to the
     * right location. */
    if (file_ptr_type == ADIO_INDIVIDUAL) {
	fd->fp_ind = file_offsets[file_list_count-1]+
	    file_lengths[file_list_count-1];
    }
    
    ADIOI_Free(file_offsets);
    ADIOI_Free(file_lengths);
    
    if (err_flag == 0) *error_code = MPI_SUCCESS;

error_state:
    fd->fp_sys_posn = -1;   /* set it to null. */

#ifdef HAVE_STATUS_SET_BYTES
    MPIR_Status_set_bytes(status, datatype, bufsize);
    /* This is a temporary way of filling in status. The right way is to 
       keep track of how much data was actually read and placed in buf 
       by ADIOI_BUFFERED_READ. */
#endif
    
    if (!buftype_is_contig) ADIOI_Delete_flattened(datatype);
}

