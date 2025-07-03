/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#pragma once

#include <cstdlib>
#include <type_traits>

#include <nn/fs.h>
#include <nn/fs/fs_Debug.h>

namespace AftermathTest {
namespace Utils {

template<typename T, typename A>
T AlignUp(T val, A alignment)
{
    static_assert(std::is_unsigned<T>::value, "Requires unsigned type");
    if (alignment == 0) {
        return val;
    }
    return ((val + (T)alignment - 1) / (T)alignment) * (T)alignment;
}

template<typename T, typename A>
bool IsMultipleOf(T val, A alignment)
{
    static_assert(std::is_unsigned<T>::value, "Requires unsigned type");
    if (alignment == 0) {
        return false;
    }
    return val % (T)alignment == 0;
}

// Simple wrapper for scoped mounting of the host root
class ScopedHostRootMount
{
public:
    ScopedHostRootMount()
        : ready(false)
        , mounted(false)
    {
        nn::Result result = nn::fs::MountHostRoot();
        mounted = result.IsSuccess();
        if (mounted || nn::fs::ResultMountNameAlreadyExists::Includes(result)) {
            ready = true;
        }
    }

    ~ScopedHostRootMount()
    {
        if (mounted) {
            nn::fs::UnmountHostRoot();
        }
    }

    bool Ready() const
    {
        return ready;
    }

private:

    ScopedHostRootMount(const ScopedHostRootMount&) = delete;
    ScopedHostRootMount& operator= (const ScopedHostRootMount&) = delete;

    bool ready;
    bool mounted;
};

// Simple wrapper for scoped mounting of the SD card
class ScopedSdCardMount
{
public:
    ScopedSdCardMount()
        : ready(false)
        , mounted(false)
        , noCard(true)
    {
        nn::Result result = nn::fs::MountSdCardForDebug("AmTestSD");
        noCard = nn::fs::ResultSdCardAccessFailed::Includes(result);
        if (!noCard) {
            mounted = result.IsSuccess();
            if (mounted || nn::fs::ResultMountNameAlreadyExists::Includes(result)) {
                ready = true;
            }
        }
    }

    ~ScopedSdCardMount()
    {
        if (mounted) {
            nn::fs::Unmount("AmTestSD");
        }
    }

    bool Ready() const
    {
        return ready;
    }

    bool NoCard() const
    {
        return noCard;
    }

private:

    ScopedSdCardMount(const ScopedSdCardMount&) = delete;
    ScopedSdCardMount& operator= (const ScopedSdCardMount&) = delete;

    bool ready;
    bool mounted;
    bool noCard;
};

} // namespace Utils
} // namespace AftermathTest
