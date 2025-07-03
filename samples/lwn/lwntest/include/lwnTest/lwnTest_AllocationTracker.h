/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_AllocationTracker_h__
#define __lwnTest_AllocationTracker_h__

#include <algorithm>
#include <vector>

#include "lwn/lwn.h"

namespace lwnTest {

//
//                  LWN TRACKED ALLOCATION SUPPORT
//
// Unlike OpenGL, LWN doesn't have a notion of a "namespace" in which all
// objects live and doesn't automatically clean up allocated API objects when
// a context or namespace is deleted.  Allocations that aren't cleaned up will
// appear to the driver to be memory leaks, and will result in teardown
// assertions in the driver.
//
// To deal with this, we create an Allocation tracker template class that
// keeps track of outstanding allocations of all types.  This can be used to
// clean up all objects allocated by a specific test by calling a cleanup
// function in ExitGraphics().
//
template <typename T>
class AllocationTracker
{
    // Each Entry structure keeps an object (the API C object classes, which
    // are simply pointers to opaque structures) and a reference count.  It
    // also defines a "==" operator used to find an object in the list of
    // entries.
    struct Entry {
        T               m_object;
        Entry(T data) : m_object(data) {}
        bool operator == (const Entry &other) const
        {
            return m_object == other.m_object;
        }
    };

    // Array of tracked objects.
    std::vector<Entry> allAllocations;

public:
    // Tracks the creation of a new object <object>, which starts with a
    // reference count of 1.
    void TrackCreate(T &object)
    {
        if ((void *)object == NULL) {
            return;     // don't track NULL objects from failed creates
        }
        Entry entry(object);
        allAllocations.push_back(entry);
    }

    // Tracks a free operation on <object>.
    void TrackFree(T &object)
    {
        if ((void *)object == NULL) {
            return;     // don't track NULL objects from failed creates
        }
        Entry entry(object);
        // Order is irrelevant, so we can just move the last element to the open location
        // and avoid moving other elements.
        auto entryIt = std::find(allAllocations.begin(), allAllocations.end(), entry);
        if (entryIt < allAllocations.end()) {
            *entryIt = allAllocations.back();
            allAllocations.pop_back();
        }
    }

    // Performs a free operation on the API object <object>.
    static void CleanupAPIObject(T &object);

    // Cleans up all tracked allocations still on the tracking list at the end
    // of a test run.  Also clears/frees the tracking list.
    void TrackedObjectCleanup()
    {
        for (size_t i = 0; i < allAllocations.size(); i++) {
            T object = allAllocations[i].m_object;
            CleanupAPIObject(object);
        }
        allAllocations.clear();
    }

};

// allocationCleanup:  Built-in function to clean up LWN object allocations of
// all types at the end of a test.
extern void allocationCleanup();

// Utility code to enable or disable object allocation tracking support for
// our wrapped object types.
extern void EnableLWNObjectTracking();
extern void DisableLWNObjectTracking();
extern bool IsLWNObjectTrackingEnabled();

// Utility code to push/pop object tracking state for a temporary change.
// Only supports a one-deep stack.
extern void PushLWNObjectTracking();
extern void PopLWNObjectTracking();

} // namespace lwnTest

#endif // #ifndef __lwnTest_AllocationTracker_h__
