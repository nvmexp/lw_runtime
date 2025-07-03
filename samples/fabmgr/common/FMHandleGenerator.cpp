
/*
 *  Copyright 2021-2022 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#include "fm_log.h"
#include "FMCommonTypes.h"
#include "FmThread.h"
#include "FMAutoLock.h"
#include "FMHandleGenerator.h"

#include <set>

uint32 FMHandleGenerator::mNextUnusedHandle;
std::set<uint32> FMHandleGenerator::mUsedHandles;
LWOSCriticalSection FMHandleGenerator::mLock;
FMHandleGenerator::_initLock FMHandleGenerator::_initializer;

bool FMHandleGenerator::allocHandle( uint32 &handle ) {
    FMAutoLock lock(FMHandleGenerator::mLock);
    uint32 handleIndexLwrrent = mNextUnusedHandle;
    do {
        if (handleIndexLwrrent == 0) {
            handleIndexLwrrent = 1;
        }
        if( mUsedHandles.find( handleIndexLwrrent ) == mUsedHandles.end() ) {
            handle = handleIndexLwrrent++;
            mNextUnusedHandle = handleIndexLwrrent;
            FM_LOG_DEBUG( "allocated handle %d", handle );
            return true;
        } else {
            handleIndexLwrrent++;
        }

    } while( handleIndexLwrrent != mNextUnusedHandle );
    return false;
}

void FMHandleGenerator::freeHandle( uint32 handle )
{
    FMAutoLock lock(mLock);
    mUsedHandles.erase( handle );
}
