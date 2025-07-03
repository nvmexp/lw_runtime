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
 * @file   unitodb.h
 * @brief  declarations to functions to init odbcommon structure and
 *         set a list of properties
 8         implemented in unitodb.c
 */

#ifndef _UNITODB_H_
#define _UNITODB_H_

#include <memory.h>
#include "lwrm.h"
#include "odb.h"

//Initialize a common object data base structure.
void unitOdbInitCommon(PODBCOMMON thisOdbCommon);

//
//Set all the properties passed in to TRUE. This function is
//used by the rmconfig generated code to set a list of properties
//
void
odbSetProperties(PODBCOMMON thisCommon, PDB_PROP_BASE_TYPE* thisPropertyList, LwU32 thisNum);

#endif // _UNITODB_H_
