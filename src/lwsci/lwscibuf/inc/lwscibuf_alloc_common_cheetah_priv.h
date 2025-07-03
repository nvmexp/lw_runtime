/*
 * Copyright (c) 2019-2020, LWPU CORPORATION. All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

/**
 * \file
 * \brief <b>LwSciBuf CheetAh Common Allocate Structures</b>
 *
 * @b Description: This file contains LwSciBuf CheetAh common private structures.
 */

#ifndef INCLUDED_LWSCIBUF_ALLOC_COMMON_TEGRA_PRIV_H
#define INCLUDED_LWSCIBUF_ALLOC_COMMON_TEGRA_PRIV_H

#include "lwscibuf_alloc_common_tegra.h"

/**
 * @brief CheetAh common access permissions map for LwSciBuf allocation.
 */
typedef struct {
    /** represents memory access permissions */
    uint32_t lwRmAccessPerm;
} LwSciBufAllocCommonTegraAccPermMap;

#endif /* INCLUDED_LWSCIBUF_ALLOC_COMMON_TEGRA_PRIV_H */
