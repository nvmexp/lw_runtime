/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef _CRAY_PMI_H
#define _CRAY_PMI_H

#define PMI_SUCCESS                   0
#define PMI_FAIL                      1
#define PMI_ERR_INIT                  2
#define PMI_ERR_NOMEM                 3
#define PMI_ERR_ILWALID_ARG           4
#define PMI_ERR_ILWALID_KEY           5
#define PMI_ERR_ILWALID_KEY_LENGTH    6
#define PMI_ERR_ILWALID_VAL           7
#define PMI_ERR_ILWALID_VAL_LENGTH    8
#define PMI_ERR_ILWALID_LENGTH        9
#define PMI_ERR_ILWALID_NUM_ARGS      10
#define PMI_ERR_ILWALID_ARGS          11
#define PMI_ERR_ILWALID_NUM_PARSED    12
#define PMI_ERR_ILWALID_KEYVALP       13
#define PMI_ERR_ILWALID_SIZE          14
#define PMI_Init(_spawned)            ({*(_spawned)=0; 0;})
#define PMI_Get_size(...)             0
#define PMI_Get_rank(...)             0

#endif

