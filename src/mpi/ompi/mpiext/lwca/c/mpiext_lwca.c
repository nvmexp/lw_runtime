/*
 * Copyright (c) 2004-2009 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2010-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2010      Oracle and/or its affiliates.  All rights reserved.
 * Copyright (c) 2012      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015      LWPU, Inc. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "ompi_config.h"

#include <stdio.h>
#include <string.h>

#include "opal/constants.h"
#include "ompi/mpiext/lwca/c/mpiext_lwda_c.h"

/* If LWCA-aware support is configured in, return 1. Otherwise, return 0.
 * This API may be extended to return more features in the future. */
int MPIX_Query_lwda_support(void)
{
    return OPAL_LWDA_SUPPORT;
}
