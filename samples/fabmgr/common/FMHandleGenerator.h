
/*
 *  Copyright 2021-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include "fm_log.h"
#include "FMCommonTypes.h"
#include "FmThread.h"
#include "FMAutoLock.h"

#include <set>

/*
    This class basically exists to provide handles for import/export RM calls
    Since handles cannot be shared between the importer/exporter, we need a 
    handle generator that has global scope and ensures that handles are not
    reused, unless freed. This can also be called from multiple
    thread contexts so it has be thread-safe. 
*/

class FMHandleGenerator
{
    static std::set< uint32 > mUsedHandles;
    static uint32 mNextUnusedHandle;
    static LWOSCriticalSection mLock;

public:

    // this static class is created to basically help in initializing the lock
    // and deleting the lock. 
    static class _initLock
    {
    public:
        _initLock() { lwosInitializeCriticalSection(&mLock); }
        ~_initLock() { lwosDeleteCriticalSection(&mLock); }
    } _initializer;

    static bool allocHandle(uint32 &handle);
    static void freeHandle( uint32 handle );
};


