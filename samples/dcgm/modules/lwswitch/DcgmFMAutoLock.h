#pragma once

#include "lwos.h"

/*****************************************************************************/
/*  Fabric Manager : Implements Scoped Lock                                  */
/*****************************************************************************/

/*
 * NonCopyable:
 *  Make copy constructor and copy assignment as private to ensure classes
 *  derived rom class NonCopyable cannot be copied.
 * DcgmFMAutoLock:
 *  Manages the Critical Section automatically. It'll be locked when 
 *  DcgmFMAutoLock is constructed and released when DcgmFMAutoLock
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

class DcgmFMAutoLock : NonCopyable
{
public:

    DcgmFMAutoLock(LWOSCriticalSection &lock)
    : mLock(lock)
    {
        // lock will be taken when the class object is created
        lwosEnterCriticalSection( &mLock );
    }

   ~DcgmFMAutoLock()
    {
        // lock will be released when the class object is deleted.    
        lwosLeaveCriticalSection( &mLock );
    }

private:
    LWOSCriticalSection &mLock;
};
