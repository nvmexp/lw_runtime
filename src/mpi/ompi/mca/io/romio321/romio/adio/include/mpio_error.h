/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */
/* MPI_ERR_FILE */
#define MPIR_ERR_FILE_NULL 3
#define MPIR_ERR_FILE_CORRUPT 5

/* MPI_ERR_AMODE */
/* kind values set directly to 3,5,7 in mpi-io/open.c */

/* MPI_ERR_ARG */
#define MPIR_ERR_OFFSET_ARG 65
#define MPIR_ERR_DATAREP_ARG 67
#define MPIR_ERR_COUNT_ARG 69
#define MPIR_ERR_SIZE_ARG 71
#define MPIR_ERR_WHENCE_ARG 73
#define MPIR_ERR_FLAG_ARG 75
#define MPIR_ERR_DISP_ARG 77
#define MPIR_ERR_ETYPE_ARG 79
#define MPIR_ERR_FILETYPE_ARG 81
#define MPIR_ERR_SIZE_ARG_NOT_SAME 83
#define MPIR_ERR_OFFSET_ARG_NEG 85
#define MPIR_ERR_WHENCE_ARG_NOT_SAME 87
#define MPIR_ERR_OFFSET_ARG_NOT_SAME 89

/* MPI_ERR_TYPE */
#ifndef MPIR_ERR_TYPE_NULL
#define MPIR_ERR_TYPE_NULL 5
#endif

/* MPI_ERR_UNSUPPORTED_OPERATION */
#define MPIR_ERR_NO_SHARED_FP 3
#define MPIR_ERR_AMODE_SEQ 5
#define MPIR_ERR_MODE_WRONLY 7
#define MPIR_ERR_NO_MODE_SEQ 9

/* MPI_ERR_REQUEST */
#ifndef MPIR_ERR_REQUEST_NULL
#define MPIR_ERR_REQUEST_NULL 3
#endif

/* MPI_ERR_IO */
#define MPIR_ADIO_ERROR 1  /* used for strerror(errno) */
#define MPIR_ERR_ETYPE_FRACTIONAL 3
#define MPIR_ERR_NO_FSTYPE 5
#define MPIR_ERR_NO_PFS 7
#define MPIR_ERR_NO_PIOFS 9
#define MPIR_ERR_NO_UFS 11
#define MPIR_ERR_NO_NFS 13
#define MPIR_ERR_NO_HFS 15
#define MPIR_ERR_NO_XFS 17
#define MPIR_ERR_NO_SFS 19
#define MPIR_ERR_NO_PVFS 21
#define MPIR_ERR_NO_PANFS 22
#define MPIR_ERR_MULTIPLE_SPLIT_COLL 23
#define MPIR_ERR_NO_SPLIT_COLL 25
#define MPIR_ERR_ASYNC_OUTSTANDING 27
#define MPIR_READ_PERM 29
#define MPIR_PREALLOC_PERM 31
#define MPIR_ERR_FILETYPE 33 
#define MPIR_ERR_NO_NTFS 35
#define MPIR_ERR_NO_TESTFS 36
#define MPIR_ERR_NO_LUSTRE 37
#define MPIR_ERR_NO_BGL 38

/* MPI_ERR_COMM */
#ifndef MPIR_ERR_COMM_NULL
#define MPIR_ERR_COMM_NULL 3
#define MPIR_ERR_COMM_INTER 5 
#endif

/* MPI_ERR_UNSUPPORTED_DATAREP */
#define MPIR_ERR_NOT_NATIVE_DATAREP 3

