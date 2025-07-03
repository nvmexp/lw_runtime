/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.  Any
 * use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation
 * is strictly prohibited.
 */

#include <LlgdTestUtilEditState.h>
#include <LlgdTestUtil.h>

#include <liblwn-llgd.h>

#include <cstring>

namespace llgd_lwn
{

//-------------------------------
// Ctor
//-------------------------------
CommandHandleEditingHelper::CommandHandleEditingHelper(llgd_lwn::QueueHolder& queueHolder)
    : qh(queueHolder)
{
}

//-------------------------------
// Initialize
//-------------------------------
bool CommandHandleEditingHelper::Initialize()
{
    // To detect memory overrun, set dirty data in advance. When dirty data is read, a parser will emit assert.
    memset(pool, 0xcd, sizeof(pool));
    memset(ctrl_space, 0xef, sizeof(ctrl_space));

    ResetPointersForEditingCB();

    // Create memory pool for command buffer's command memory
    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(g_device).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED)
        .SetStorage(pool, TOTAL_POOL_SIZE);
    return mph.Initialize(&pool_builder);
}

//-------------------------------
// MakeHandle
//-------------------------------
LWNcommandHandle CommandHandleEditingHelper::MakeHandle(const BuildCommandFn& buildFn)
{
    auto handle = LWNcommandHandle(0);

    // Create base command handle
    llgd_lwn::CommandBufferHolder cbh;
    if (!cbh.Initialize((Device*)g_device)) {
        return handle;
    }
    cbh->AddCommandMemory(mph, POOL_OFFSET_ORIGINAL_CB, ONE_POOL_SIZE);
    cbh->AddControlMemory(ctrl_space, ONE_CTRL_SIZE);

    cbh->BeginRecording();
    buildFn(cbh);
    handle = cbh->EndRecording();

    return handle;
}

//-------------------------------
// Run
//-------------------------------
void CommandHandleEditingHelper::Run(LWNcommandHandle handle)
{
    qh->SubmitCommands(1, &handle);
    qh->Finish();
}

//-------------------------------
// RunAndExtractGpuState
//-------------------------------
GpuState CommandHandleEditingHelper::RunAndExtractGpuState(LWNcommandHandle handle)
{
    Run(handle);
    return llgd_lwn::ExtractGpuState(qh);
}

//-------------------------------
// MakeCommandHandleRunnable
//-------------------------------
LWNcommandHandle CommandHandleEditingHelper::MakeCommandHandleRunnable(LWNcommandHandle baseCommandHandle)
{
    // Re-create the "commandHandle" by patching gpuAddress (like llgd-host does)
    // Encode
    struct EncodeMemData {
        uint8_t* ptr;
        size_t wroteSize;
    };
    auto startPtr = reinterpret_cast<uint8_t*>(pool) + POOL_OFFSET_DECODED_CB;
    EncodeMemData data{ startPtr, 0 };

    auto ret = llgdEncodeCommandSet(baseCommandHandle, [](const void* data, size_t size, size_t alignedSize, void* userData) {
        EncodeMemData& storageData = *reinterpret_cast<EncodeMemData*>(userData);
        CHECK(storageData.wroteSize + size <= ONE_POOL_SIZE);
        std::memcpy(storageData.ptr + storageData.wroteSize, data, size);
        storageData.wroteSize += alignedSize;
    }, &data);

    // Overwrite source with garbage
    {
        const auto original = ctrl_space + CTRL_OFFSET_ORIGINAL_CB;
        memset(original, 0xef, ONE_CTRL_SIZE);
    }
    {
        const auto original = pool + POOL_OFFSET_ORIGINAL_CB;
        memset(original, 0xcd, ONE_POOL_SIZE);
    }

    if (!ret) {
        // Failed to encode
        return LWNcommandHandle(0);
    }

    // Decode
    return llgdDecodeCommandSet(startPtr, mph->GetBufferAddress() + POOL_OFFSET_DECODED_CB, false /* stripDebugTokens */);
}

template <typename StateType, void (*ExtractFn)(const GpuState&, StateType&)>
static StateType ExtractLlgdLwnState(const GpuState& targetGpuState)
{
    StateType st;
    ExtractFn(targetGpuState, st);
    return st;
}

//-------------------------------
// ExtractRasterState
//-------------------------------
RasterState CommandHandleEditingHelper::ExtractRasterState(const GpuState* targetGpuState) const
{
    return ExtractLlgdLwnState<RasterState, llgdLwnExtractRasterState>(targetGpuState ? *targetGpuState : llgd_lwn::ExtractGpuState(qh));
}

//-------------------------------
// ExtractPixelState
//-------------------------------
PixelState CommandHandleEditingHelper::ExtractPixelState(const GpuState* targetGpuState) const
{
    return ExtractLlgdLwnState<PixelState, llgdLwnExtractPixelState>(targetGpuState ? *targetGpuState : llgd_lwn::ExtractGpuState(qh));
}

//-------------------------------
// ExtractVertexSpecificationState
//-------------------------------
VertexSpecificationState CommandHandleEditingHelper::ExtractVertexSpecificationState(const GpuState* targetGpuState) const
{
    return ExtractLlgdLwnState<VertexSpecificationState, llgdLwnExtractVertexSpecificationState>(targetGpuState ? *targetGpuState : llgd_lwn::ExtractGpuState(qh));
}

//-------------------------------
// ExtractTransformState
//-------------------------------
TransformState CommandHandleEditingHelper::ExtractTransformState(const GpuState* targetGpuState) const
{
    return ExtractLlgdLwnState<TransformState, llgdLwnExtractTransformState>(targetGpuState ? *targetGpuState : llgd_lwn::ExtractGpuState(qh));
}

void* CommandHandleEditingHelper::WriteControlMemoryForEditing(const void* data, size_t size, void* userData)
{
    auto helper = reinterpret_cast<CommandHandleEditingHelper*>(userData);
    CHECK(helper->m_ctrl < &helper->ctrl_space[CTRL_OFFSET_EDITING_CB] + ONE_CTRL_SIZE);
    // Make sure you have enough space to store passed as 'data'
    CHECK(helper->m_ctrl + size < &helper->ctrl_space[CTRL_OFFSET_EDITING_CB] + ONE_CTRL_SIZE);

    std::memcpy(helper->m_ctrl, data, size);
    auto ret = helper->m_ctrl;
    helper->m_ctrl += size;

    return ret;
}
void* CommandHandleEditingHelper::WriteCommandMemoryForEditing(const void* data, size_t size, void* userData)
{
    auto helper = reinterpret_cast<CommandHandleEditingHelper*>(userData);
    CHECK(helper->m_cmd < &helper->pool[POOL_OFFSET_EDITING_CB] + ONE_POOL_SIZE);
    // Make sure you have enough space to store passed as 'data'
    CHECK(helper->m_cmd + size < &helper->pool[POOL_OFFSET_EDITING_CB] + ONE_POOL_SIZE);

    std::memcpy(helper->m_cmd, data, size);
    auto ret = helper->m_cmd;
    helper->m_cmd += size;

    return ret;
}


// Reset ctrl&cmd pointers used by state editor
void CommandHandleEditingHelper::ResetPointersForEditingCB()
{
    m_ctrl = ctrl_space + CTRL_OFFSET_EDITING_CB;
    m_cmd = pool + POOL_OFFSET_EDITING_CB;
}

} // llgd_lwn

