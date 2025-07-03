/*
 *  Copyright 2018-2019 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */
#pragma once

#include "lwos.h"

/*****************************************************************************/
/*  Fabric Manager : Implements Scoped Lock                                  */
/*****************************************************************************/

/*
 * NonCopyable:
 *  Make copy constructor and copy assignment as private to ensure classes
 *  derived rom class NonCopyable cannot be copied.
 * FMAutoLock:
 *  Manages the Critical Section automatically. It'll be locked when 
 *  FMAutoLock is constructed and released when FMAutoLock
 *  goes out of scope.
 */

class NonCopyable 
{
protected:
    NonCopyable () {}
    ~NonCopyable () {} 
private:
    // make copy constructor and assignment operator as private
    NonCopyable (const NonCopyable &);
    NonCopyable & operator = (const NonCopyable &);
};

class FMAutoLock : NonCopyable
{
public:

    FMAutoLock(LWOSCriticalSection &lock)
    : mLock(lock)
    {
        // lock will be taken when the class object is created
        lwosEnterCriticalSection( &mLock );
    }

   ~FMAutoLock()
    {
        // lock will be released when the class object is deleted.    
        lwosLeaveCriticalSection( &mLock );
    }

private:
    LWOSCriticalSection &mLock;
};

