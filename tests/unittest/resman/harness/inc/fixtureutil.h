/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2009 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file   fixtureutil.h
 * @brief  declarations for FIXTUREUTIL functions for allocated object's
 *         linked list management
 */

#ifndef _FIXTUREUTIL_H_
#define _FIXTUREUTIL_H_
#include <stdlib.h>
#include "odbinfra.h"

//
// enum to distinguish between different classes of
// objects allocated for the infra
//
typedef enum
{
    UNIT_CLASS_RM_OBJECT,
    UNIT_CLASS_INFOBLK,
    UNIT_CLASS_MISC
} UNIT_CLASS;

// Adds the object viz,OBJXXX to the Object list
void addToRmOjectList(void *pData);

// Adds the allocated infoblock to the list
void addToRmInfoBlockList(void *pData);

// Adds all other allocated structures to this list
void addToRmMiscNodeList(void *pData);

// destroy all lists
void destroyAllList();

// unit test infra specific malloc
void * unitMalloc(LwU32 size, UNIT_CLASS uClass);

// Infra specific function for DBG_PRINTF
void utDbg_Printf(const char* file, int line, const char *function, int debuglevel, const char* s, ...);

// API to stub/enable DBG_PRINTF
void utEnableDbgPrintf(LwBool flag);

#endif // _FIXTUREUTIL_H_
