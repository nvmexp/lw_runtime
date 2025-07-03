/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

/*!
 * @file  flcngdbTypes.h
 * @brief flcngdb defined datatypes.
 *
 *  */
#ifndef _FLCNGDBTYPES_H_
#define _FLCNGDBTYPES_H_

#ifdef __LOCAL_MACHINE
    typedef unsigned int LwU32;
    typedef int LwS32;
    typedef int LwBool;
#else
    #include "lwtypes.h"
#endif

#endif /* _FLCNGDBTYPES_H_ */

