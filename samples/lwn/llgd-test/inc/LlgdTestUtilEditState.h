/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#pragma once

#include <lwn/lwn.h>
#include <lwn/lwn_Cpp.h>
#include <lwn/lwn_CppMethods.h>
#include <lwn_DeviceConstantsNX.h>

#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>
#include <LlgdGpuState.h>
#include <LlgdLwnState.h>

#include <algorithm>
#include <sstream>
#include <functional>
#include <cstdio>

#define __LLGD_TEST 1 // unlocks llgdCommandSetGetMethods
#include <liblwn-llgd.h>

using namespace lwn;

namespace llgd_lwn
{

template <typename T>
/* c++14 constexpr */ inline uint32_t to_u32(const T& data) noexcept
{
    //https://stackoverflow.com/a/36413432
    uint32_t res = 0;
    static_assert(sizeof(data) == sizeof(res), "size mismatch");
    memcpy(&res, &data, sizeof(res));
    return res;
}

template <typename T>
/* c++14 constexpr */ inline T from_u32(const uint32_t data) noexcept
{
    //https://stackoverflow.com/a/36413432
    T res;
    static_assert(sizeof(data) == sizeof(res), "size mismatch");
    memcpy(&res, &data, sizeof(res));
    return res;
}

inline void EmptyMethodUpdatedFn(uint32_t eventIndex, uint32_t method, uint32_t count, const uint32_t* values, void* callback)
{
}

class CommandHandleEditingHelper
{
public:
    using BuildCommandFn = std::function<void(CommandBuffer*)>;

    CommandHandleEditingHelper(llgd_lwn::QueueHolder& queueHolder);
    bool Initialize();

    // Create command handle
    LWNcommandHandle MakeHandle(const BuildCommandFn& buildFn);

    // Run command handle on queue
    void Run(LWNcommandHandle handle);

    // Extract GpuState after Run the command handle
    GpuState RunAndExtractGpuState(LWNcommandHandle handle);

    // Patch command handle to be able to run on GPU
    LWNcommandHandle MakeCommandHandleRunnable(LWNcommandHandle baseCommandHandle);

    // Extract LlgdLwnStates from current gpu state (if targetGpuState is null, current gpuState will be used)
    RasterState ExtractRasterState(const GpuState* targetGpuState = nullptr) const;
    PixelState ExtractPixelState(const GpuState* targetGpuState = nullptr) const;
    VertexSpecificationState ExtractVertexSpecificationState(const GpuState* targetGpuState = nullptr) const;
    TransformState ExtractTransformState(const GpuState* targetGpuState = nullptr) const;

    // Write control / command memory for editing. Used by state editor
    static void* WriteControlMemoryForEditing(const void* data, size_t size, void* userData);
    static void* WriteCommandMemoryForEditing(const void* data, size_t size, void* userData);

    // Reset ctrl&cmd pointers used by state editor
    void ResetPointersForEditingCB();

private:
    llgd_lwn::QueueHolder& qh;
    llgd_lwn::MemoryPoolHolder mph;

    // 1 for original CB, 1 for patched
    static const size_t ONE_CTRL_SIZE = 4096;
    static const size_t CTRL_COUNT = 2;
    static const size_t CTRL_OFFSET_ORIGINAL_CB = ONE_CTRL_SIZE * 0;
    static const size_t CTRL_OFFSET_EDITING_CB = ONE_CTRL_SIZE * 1;
    static const size_t CTRL_SIZE = ONE_CTRL_SIZE * CTRL_COUNT;
    uint8_t ctrl_space[CTRL_SIZE] __attribute__((aligned(4096)));
    uint8_t* m_ctrl;

    // 1 PAGE for original CB, 1 for decoded, 1 for patched
    static const size_t ONE_POOL_SIZE = 2 * LWN_DEVICE_INFO_CONSTANT_NX_MEMORY_POOL_PAGE_SIZE;
    static const size_t POOL_COUNT = 3;
    static const size_t POOL_OFFSET_ORIGINAL_CB = ONE_POOL_SIZE * 0;
    static const size_t POOL_OFFSET_DECODED_CB = ONE_POOL_SIZE * 1;
    static const size_t POOL_OFFSET_EDITING_CB = ONE_POOL_SIZE * 2;
    static const size_t TOTAL_POOL_SIZE = POOL_COUNT * ONE_POOL_SIZE;
    uint8_t pool[TOTAL_POOL_SIZE] __attribute__((aligned(4096)));
    uint8_t* m_cmd;
};

inline auto GetMethods(LWNcommandHandle handle)
{
    std::vector<LlgdCommandSetMethodTrackerMethodInfo> res;
    llgdCommandSetGetMethods(handle, res);
    return res;
}

template<class A, class B>
inline bool CompareMethods(const A& a, const B& b)
{
    TEST_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        TEST_EQ(a[i].method, b[i].method);
        TEST_EQ(a[i].eventIndex, b[i].eventIndex);
    }
    return true;
}

inline bool FindMethod(const std::vector<LlgdCommandSetMethodTrackerMethodInfo>& infos, uint32_t target)
{
    return infos.end() != std::find_if(infos.begin(), infos.end(), [&] (const auto& info) { return info.method == target; });
}

} // llgd_lwn
