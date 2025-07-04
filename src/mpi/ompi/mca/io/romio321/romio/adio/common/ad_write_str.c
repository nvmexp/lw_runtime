/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/* 
 *
 *   Copyright (C) 1997 University of Chicago. 
 *   See COPYRIGHT notice in top-level directory.
 */

#include "adio.h"
#include "adio_extern.h"

#define ADIOI_BUFFERED_WRITE \
{ \
    if (req_off >= writebuf_off + writebuf_len) { \
        if (writebuf_len) { \
           ADIO_WriteContig(fd, writebuf, writebuf_len, MPI_BYTE, \
                  ADIO_EXPLICIT_OFFSET, writebuf_off, &status1, error_code); \
           if (!(fd->atomicity)) ADIOI_UNLOCK(fd, writebuf_off, SEEK_SET, writebuf_len); \
           if (*error_code != MPI_SUCCESS) { \
               *error_code = MPIO_Err_create_code(*error_code, \
                                                  MPIR_ERR_RECOVERABLE, myname, \
                                                  __LINE__, MPI_ERR_IO, \
                                                  "**iowswc", 0); \
               goto fn_exit; \
           } \
        } \
	writebuf_off = req_off; \
        writebuf_len = (unsigned) (ADIOI_MIN(max_bufsize,end_offset-writebuf_off+1));\
	if (!(fd->atomicity)) ADIOI_WRITE_LOCK(fd, writebuf_off, SEEK_SET, writebuf_len); \
	ADIO_ReadContig(fd, writebuf, writebuf_len, MPI_BYTE, \
                 ADIO_EXPLICIT_OFFSET, writebuf_off, &status1, error_code); \
	if (*error_code != MPI_SUCCESS) { \
	    *error_code = MPIO_Err_create_code(*error_code, \
					       MPIR_ERR_RECOVERABLE, myname, \
					       __LINE__, MPI_ERR_IO, \
					       "**iowsrc", 0); \
	    goto fn_exit; \
	} \
    } \
    write_sz = (unsigned) (ADIOI_MIN(req_len, writebuf_off + writebuf_len - req_off)); \
    ADIOI_Assert((ADIO_Offset)write_sz == ADIOI_MIN(req_len, writebuf_off + writebuf_len - req_off));\
    memcpy(writebuf+req_off-writebuf_off, (char *)buf +userbuf_off, write_sz);\
    while (write_sz != req_len) { \
        ADIO_WriteContig(fd, writebuf, writebuf_len, MPI_BYTE, \
                  ADIO_EXPLICIT_OFFSET, writebuf_off, &status1, error_code); \
        if (!(fd->atomicity)) ADIOI_UNLOCK(fd, writebuf_off, SEEK_SET, writebuf_len); \
        if (*error_code != MPI_SUCCESS) { \
            *error_code = MPIO_Err_create_code(*error_code, \
                                               MPIR_ERR_RECOVERABLE, myname, \
                                               __LINE__, MPI_ERR_IO, \
                                               "**iowswc", 0); \
            goto fn_exit; \
        } \
        req_len -= write_sz; \
        userbuf_off += write_sz; \
        writebuf_off += writebuf_len; \
        writebuf_len = (unsigned) (ADIOI_MIN(max_bufsize,end_offset-writebuf_off+1));\
	if (!(fd->atomicity)) ADIOI_WRITE_LOCK(fd, writebuf_off, SEEK_SET, writebuf_len); \
        ADIO_ReadContig(fd, writebuf, writebuf_len, MPI_BYTE, \
                  ADIO_EXPLICIT_OFFSET, writebuf_off, &status1, error_code); \
	if (*error_code != MPI_SUCCESS) { \
	    *error_code = MPIO_Err_create_code(*error_code, \
					       MPIR_ERR_RECOVERABLE, myname, \
					       __LINE__, MPI_ERR_IO, \
					       "**iowsrc", 0); \
	    goto fn_exit; \
	} \
        write_sz = ADIOI_MIN(req_len, writebuf_len); \
        memcpy(writebuf, (char *)buf + userbuf_off, write_sz);\
    } \
}


/* this macro is used when filetype is contig and buftype is not contig.
   it does not do a read-modify-write and does not lock*/
#define ADIOI_BUFFERED_WRITE_WITHOUT_READ \
{ \
    if (req_off >= writebuf_off + writebuf_len) { \
        ADIO_WriteContig(fd, writebuf, writebuf_len, MPI_BYTE, \
                 ADIO_EXPLICIT_OFFSET, writebuf_off, &status1, error_code); \
        if (*error_code != MPI_SUCCESS) { \
            *error_code = MPIO_Err_create_code(*error_code, \
                                               MPIR_ERR_RECOVERABLE, myname, \
                                               __LINE__, MPI_ERR_IO, \
                                               "**iowswc", 0); \
            goto fn_exit; \
        } \
	writebuf_off = req_off; \
        writebuf_len = (unsigned) (ADIOI_MIN(max_bufsize,end_offset-writebuf_off+1));\
    } \
    write_sz = (unsigned) (ADIOI_MIN(req_len, writebuf_off + writebuf_len - req_off)); \
    ADIOI_Assert((ADIO_Offset)write_sz == ADIOI_MIN(req_len, writebuf_off + writebuf_len - req_off));\
    memcpy(writebuf+req_off-writebuf_off, (char *)buf +userbuf_off, write_sz);\
    while (write_sz != req_len) { \
        ADIO_WriteContig(fd, writebuf, writebuf_len, MPI_BYTE, \
                ADIO_EXPLICIT_OFFSET, writebuf_off, &status1, error_code); \
        if (*error_code != MPI_SUCCESS) { \
            *error_code = MPIO_Err_create_code(*error_code, \
                                               MPIR_ERR_RECOVERABLE, myname, \
                                               __LINE__, MPI_ERR_IO, \
                                               "**iowswc", 0); \
            goto fn_exit; \
        } \
        req_len -= write_sz; \
        userbuf_off += write_sz; \
        writebuf_off += writebuf_len; \
        writebuf_len = (unsigned) (ADIOI_MIN(max_bufsize,end_offset-writebuf_off+1));\
        write_sz = ADIOI_MIN(req_len, writebuf_len); \
        memcpy(writebuf, (char *)buf + userbuf_off, write_sz);\
    } \
}
void ADIOI_GEN_WriteStrided(ADIO_File fd, const void *buf, int count,
                       MPI_Datatype datatype, int file_ptr_type,
                       ADIO_Offset offset, ADIO_Status *status, int
                       *error_code)
{

/* offset is in units of etype relative to the filetype. */

    ADIOI_Flatlist_node *flat_buf, *flat_file;
    ADIO_Offset i_offset, sum, size_in_filetype;
    int i, j, k, st_index=0;
    ADIO_Offset num, size, n_filetypes, etype_in_filetype, st_n_filetypes;
    ADIO_Offset n_etypes_in_filetype, abs_off_in_filetype=0;
    MPI_Count filetype_size, etype_size, buftype_size;
    MPI_Aint filetype_extent, buftype_extent, lb; 
    int buf_count, buftype_is_contig, filetype_is_contig;
    ADIO_Offset userbuf_off;
    ADIO_Offset off, req_off, disp, end_offset=0, writebuf_off, start_off;
    char *writebuf=NULL;
    unsigned writebuf_len, max_bufsize, write_sz;
    MPI_Aint bufsize;
    ADIO_Status status1;
    ADIO_Offset new_bwr_size, new_fwr_size, st_fwr_size, fwr_size=0, bwr_size, req_len;
    static char myname[] = "ADIOI_GEN_WriteStrided";

    if (fd->hints->ds_write == ADIOI_HINT_DISABLE) {
    	/* if user has disabled data sieving on reads, use naive
	 * approach instead.
	 */

	ADIOI_GEN_WriteStrided_naive(fd, 
				    buf,
				    count,
				    datatype,
				    file_ptr_type,
				    offset,
				    status,
				    error_code);
    	return;
    }


    *error_code = MPI_SUCCESS;  /* changed below if error */

    ADIOI_Datatype_iscontig(datatype, &buftype_is_contig);
    ADIOI_Datatype_iscontig(fd->filetype, &filetype_is_contig);

    MPI_Type_size_x(fd->filetype, &filetype_size);
    if ( ! filetype_size ) {
#ifdef HAVE_STATUS_SET_BYTES
	MPIR_Status_set_bytes(status, datatype, 0);
#endif
	*error_code = MPI_SUCCESS; 
	return;
    }

    MPI_Type_get_extent(fd->filetype, &lb, &filetype_extent);
    MPI_Type_size_x(datatype, &buftype_size);
    MPI_Type_get_extent(datatype, &lb, &buftype_extent);
    etype_size = fd->etype_size;

    ADIOI_Assert((buftype_size * count) == ((MPI_Count)buftype_size * (ADIO_Offset)count));
    bufsize = buftype_size * count;

/* get max_bufsize from the info object. */

    max_bufsize = fd->hints->ind_wr_buffer_size;

    if (!buftype_is_contig && filetype_is_contig) {

/* noncontiguous in memory, contiguous in file. */

	flat_buf = ADIOI_Flatten_and_find(datatype);

        off = (file_ptr_type == ADIO_INDIVIDUAL) ? fd->fp_ind : 
                 fd->disp + (ADIO_Offset)etype_size * offset;

	start_off = off;
	end_offset = off + bufsize - 1;
        writebuf_off = off;
        writebuf = (char *) ADIOI_Malloc(max_bufsize);
        writebuf_len = (unsigned) (ADIOI_MIN(max_bufsize, end_offset-writebuf_off+1));

/* if atomicity is true, lock the region to be accessed */
	if (fd->atomicity) 
	    ADIOI_WRITE_LOCK(fd, start_off, SEEK_SET, end_offset-start_off+1);

        for (j=0; j<count; j++) 
        {
              for (i=0; i<flat_buf->count; i++) {
                  userbuf_off = (ADIO_Offset)j*(ADIO_Offset)buftype_extent + flat_buf->indices[i];
      req_off = off;
      req_len = flat_buf->blocklens[i];
      ADIOI_BUFFERED_WRITE_WITHOUT_READ
                  off += flat_buf->blocklens[i];
              }
        }

        /* write the buffer out finally */
        ADIO_WriteContig(fd, writebuf, writebuf_len, MPI_BYTE, ADIO_EXPLICIT_OFFSET,
			writebuf_off, &status1, error_code);

	if (fd->atomicity) 
	    ADIOI_UNLOCK(fd, start_off, SEEK_SET, end_offset-start_off+1);

	if (*error_code != MPI_SUCCESS) goto fn_exit;

        if (file_ptr_type == ADIO_INDIVIDUAL) fd->fp_ind = off;
    }

    else {  /* noncontiguous in file */

/* filetype already flattened in ADIO_Open */
	flat_file = ADIOI_Flatlist;
	while (flat_file->type != fd->filetype) flat_file = flat_file->next;
	disp = fd->disp;

	if (file_ptr_type == ADIO_INDIVIDUAL) {
	/* Wei-keng reworked type processing to be a bit more efficient */
            offset       = fd->fp_ind - disp;
            n_filetypes  = (offset - flat_file->indices[0]) / filetype_extent;
            offset      -= (ADIO_Offset)n_filetypes * filetype_extent;
            /* now offset is local to this extent */

            /* find the block where offset is located, skip blocklens[i]==0 */
            for (i=0; i<flat_file->count; i++) {
                ADIO_Offset dist;
                if (flat_file->blocklens[i] == 0) continue;
                dist = flat_file->indices[i] + flat_file->blocklens[i] - offset;
                /* fwr_size is from offset to the end of block i */
                if (dist == 0) {
                    i++;
                    offset   = flat_file->indices[i];
                    fwr_size = flat_file->blocklens[i];
                    break;
                }
                if (dist > 0) {
                    fwr_size = dist;
                    break;
                }
            }
            st_index = i;  /* starting index in flat_file->indices[] */
            offset += disp + (ADIO_Offset)n_filetypes*filetype_extent;
        }
	else {
	    n_etypes_in_filetype = filetype_size/etype_size;
	    n_filetypes = offset / n_etypes_in_filetype;
	    etype_in_filetype = offset % n_etypes_in_filetype;
	    size_in_filetype = etype_in_filetype * etype_size;
 
	    sum = 0;
	    for (i=0; i<flat_file->count; i++) {
		sum += flat_file->blocklens[i];
		if (sum > size_in_filetype) {
		    st_index = i;
		    fwr_size = sum - size_in_filetype;
		    abs_off_in_filetype = flat_file->indices[i] +
			size_in_filetype - (sum - flat_file->blocklens[i]);
		    break;
		}
	    }

	    /* abs. offset in bytes in the file */
	    offset = disp + (ADIO_Offset) n_filetypes*filetype_extent + 
		    abs_off_in_filetype;
	}

	start_off = offset;

        /* Wei-keng Liao:write request is within single flat_file contig block*/
	/* this could happen, for example, with subarray types that are
	 * actually fairly contiguous */
        if (buftype_is_contig && bufsize <= fwr_size) {
	    /* though MPI api has an integer 'count' parameter, derived
	     * datatypes might describe more bytes than can fit into an integer.
	     * if we've made it this far, we can pass a count of original
	     * datatypes, instead of a count of bytes (which might overflow)
	     * Other WriteContig calls in this path are operating on data
	     * sieving buffer */
            ADIO_WriteContig(fd, buf, count, datatype, ADIO_EXPLICIT_OFFSET,
                             offset, status, error_code);

	    if (file_ptr_type == ADIO_INDIVIDUAL) {
                /* update MPI-IO file pointer to point to the first byte 
		 * that can be accessed in the fileview. */
                fd->fp_ind = offset + bufsize;
                if (bufsize == fwr_size) {
                    do {
                        st_index++;
                        if (st_index == flat_file->count) {
                            st_index = 0;
                            n_filetypes++;
                        }
                    } while (flat_file->blocklens[st_index] == 0);
                    fd->fp_ind = disp + flat_file->indices[st_index]
                               + (ADIO_Offset)n_filetypes*filetype_extent;
                }
            }
	    fd->fp_sys_posn = -1;   /* set it to null. */ 
#ifdef HAVE_STATUS_SET_BYTES
	    MPIR_Status_set_bytes(status, datatype, bufsize);
#endif 
            goto fn_exit;
        }

       /* Callwlate end_offset, the last byte-offset that will be accessed.
         e.g., if start_offset=0 and 100 bytes to be write, end_offset=99*/

	st_fwr_size = fwr_size;
	st_n_filetypes = n_filetypes;
	i_offset = 0;
	j = st_index;
	off = offset;
	fwr_size = ADIOI_MIN(st_fwr_size, bufsize);
	while (i_offset < bufsize) {
	    i_offset += fwr_size;
	    end_offset = off + fwr_size - 1;

            j = (j+1) % flat_file->count;
            n_filetypes += (j == 0) ? 1 : 0;
            while (flat_file->blocklens[j]==0) {
                j = (j+1) % flat_file->count;
                n_filetypes += (j == 0) ? 1 : 0;
            }

	    off = disp + flat_file->indices[j] + 
		    n_filetypes*(ADIO_Offset)filetype_extent;
	    fwr_size = ADIOI_MIN(flat_file->blocklens[j], bufsize-i_offset);
	}

/* if atomicity is true, lock the region to be accessed */
	if (fd->atomicity) 
	    ADIOI_WRITE_LOCK(fd, start_off, SEEK_SET, end_offset-start_off+1);

        writebuf_off = 0;
        writebuf_len = 0;
        writebuf = (char *) ADIOI_Malloc(max_bufsize);
	memset(writebuf, -1, max_bufsize);

	if (buftype_is_contig && !filetype_is_contig) {

/* contiguous in memory, noncontiguous in file. should be the most
   common case. */

	    i_offset = 0;
	    j = st_index;
	    off = offset;
	    n_filetypes = st_n_filetypes;
	    fwr_size = ADIOI_MIN(st_fwr_size, bufsize);
	    while (i_offset < bufsize) {
                if (fwr_size) { 
                    /* TYPE_UB and TYPE_LB can result in 
                       fwr_size = 0. save system call in such cases */ 
		    /* lseek(fd->fd_sys, off, SEEK_SET);
		    err = write(fd->fd_sys, ((char *) buf) + i_offset, fwr_size);*/

		    req_off = off;
		    req_len = fwr_size;
		    userbuf_off = i_offset;
		    ADIOI_BUFFERED_WRITE
		}
		i_offset += fwr_size;

		if (off + fwr_size < disp + flat_file->indices[j] +
	           flat_file->blocklens[j] + n_filetypes*(ADIO_Offset)filetype_extent)
		       off += fwr_size;
		/* did not reach end of contiguous block in filetype.
                   no more I/O needed. off is incremented by fwr_size. */
		else {
                    j = (j+1) % flat_file->count;
                    n_filetypes += (j == 0) ? 1 : 0;
                    while (flat_file->blocklens[j]==0) {
                        j = (j+1) % flat_file->count;
                        n_filetypes += (j == 0) ? 1 : 0;
                    }
		    off = disp + flat_file->indices[j] + 
                                    n_filetypes*(ADIO_Offset)filetype_extent;
		    fwr_size = ADIOI_MIN(flat_file->blocklens[j], 
				    bufsize-i_offset);
		}
	    }
	}
	else {
/* noncontiguous in memory as well as in file */

	    flat_buf = ADIOI_Flatten_and_find(datatype);

	    k = num = buf_count = 0;
	    i_offset = flat_buf->indices[0];
	    j = st_index;
	    off = offset;
	    n_filetypes = st_n_filetypes;
	    fwr_size = st_fwr_size;
	    bwr_size = flat_buf->blocklens[0];

	    while (num < bufsize) {
		size = ADIOI_MIN(fwr_size, bwr_size);
		if (size) {
		    /* lseek(fd->fd_sys, off, SEEK_SET);
		    err = write(fd->fd_sys, ((char *) buf) + i_offset, size); */

		    req_off = off;
		    req_len = size;
		    userbuf_off = i_offset;
		    ADIOI_BUFFERED_WRITE
		}

		new_fwr_size = fwr_size;
		new_bwr_size = bwr_size;

		if (size == fwr_size) {
/* reached end of contiguous block in file */
 		    j = (j+1) % flat_file->count;
 		    n_filetypes += (j == 0) ? 1 : 0;
 		    while (flat_file->blocklens[j]==0) {
 			j = (j+1) % flat_file->count;
 			n_filetypes += (j == 0) ? 1 : 0;
		    }

		    off = disp + flat_file->indices[j] + 
                                      n_filetypes*(ADIO_Offset)filetype_extent;

		    new_fwr_size = flat_file->blocklens[j];
		    if (size != bwr_size) {
			i_offset += size;
			new_bwr_size -= size;
		    }
		}

		if (size == bwr_size) {
/* reached end of contiguous block in memory */

		    k = (k + 1)%flat_buf->count;
		    buf_count++;
		    i_offset = (ADIO_Offset)buftype_extent*(ADIO_Offset)(buf_count/flat_buf->count) +
			flat_buf->indices[k]; 
		    new_bwr_size = flat_buf->blocklens[k];
		    if (size != fwr_size) {
			off += size;
			new_fwr_size -= size;
		    }
		}
		num += size;
		fwr_size = new_fwr_size;
                bwr_size = new_bwr_size;
	    }
	}

        /* write the buffer out finally */	
	if (writebuf_len) {
	    ADIO_WriteContig(fd, writebuf, writebuf_len, MPI_BYTE, ADIO_EXPLICIT_OFFSET,
			writebuf_off, &status1, error_code);
	    if (!(fd->atomicity)) 
		ADIOI_UNLOCK(fd, writebuf_off, SEEK_SET, writebuf_len);
	    if (*error_code != MPI_SUCCESS) goto fn_exit;
	}
	if (fd->atomicity) 
	    ADIOI_UNLOCK(fd, start_off, SEEK_SET, end_offset-start_off+1);

	if (file_ptr_type == ADIO_INDIVIDUAL) fd->fp_ind = off;
    }

    fd->fp_sys_posn = -1;   /* set it to null. */

#ifdef HAVE_STATUS_SET_BYTES
    /* datatypes returning negagive values, probably related to tt 1893 */
    MPIR_Status_set_bytes(status, datatype, bufsize);
/* This is a temporary way of filling in status. The right way is to 
   keep track of how much data was actually written by ADIOI_BUFFERED_WRITE. */
#endif

    if (!buftype_is_contig) ADIOI_Delete_flattened(datatype);
fn_exit:
    if (writebuf != NULL) ADIOI_Free(writebuf);
}

