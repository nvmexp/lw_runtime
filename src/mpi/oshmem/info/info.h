/*
 * Copyright (c) 2015      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef OSHMEM_INFO_H
#define OSHMEM_INFO_H

#include "oshmem_config.h"
#include "oshmem/types.h"
#include "oshmem/constants.h"

/*
 * Environment variables
 */
#define OSHMEM_ELW_SYMMETRIC_SIZE      "SMA_SYMMETRIC_SIZE"
#define OSHMEM_ELW_DEBUG               "SMA_DEBUG"
#define OSHMEM_ELW_INFO                "SMA_INFO"
#define OSHMEM_ELW_VERSION             "SMA_VERSION"

/**
 * \internal
 * oshmem_info_t structure.
 */
struct oshmem_info_t {
    /**< print the library version at start-up */
    bool print_version;
    /**< print helpful text about all these environment variables */
    bool print_info;
    /**< enable debugging messages */
    bool debug;
    /**< number of bytes to allocate for symmetric heap */
    size_t symmetric_heap_size;
};

/**
 * \internal
 * Colwenience typedef
 */
typedef struct oshmem_info_t oshmem_info_t;


BEGIN_C_DECLS

/**
 * Global instance for oshmem_info_elw
 */
OSHMEM_DECLSPEC extern oshmem_info_t oshmem_shmem_info_elw;

/**
 * This function is ilwoked during oshmem_shmem_init() and sets up
 * oshmem_shmem_info_elw handling.
 */
int oshmem_info_init(void);

/**
 * This functions is called during oshmem_shmem_finalize() and shuts
 * down oshmem_shmem_info_elw handling.
 */
int oshmem_info_finalize(void);

END_C_DECLS

#endif /* OSHMEM_INFO_H */
