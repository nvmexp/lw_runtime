/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>
#include <LlgdTestUtilEditState.h>

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>

#include <string>
#include <functional>

#include <nn/os.h>

namespace {
static const float BLUE[4]{ 0, 0, 1, 1 };
static const uint32_t IDX_OFFSET = 10;
static const auto indexMapFn = [](uint32_t idx, void*) { return idx + IDX_OFFSET; };

class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool TestRenderEnable();
    bool TestConditionalRenderEnable();
    bool TestAttributeEnable();
    bool TestAttributeFormat();
    bool TestAttributeRelativeOffset();
    bool TestAttributeStreamIndex();
    bool TestStreamEnable();
    bool TestStreamStride();
    bool TestStreamDivisor();

    // Test helper methods
    void AttributeTestSetup(CommandHandle& createdBaseCommand, GpuState& createdGpuState, bool isAttributesEnabled = true);
    void StreamTestSetup(CommandHandle& createdBaseCommand, GpuState& createdGpuState, uint32_t numOfEnableStreams = VertexSpecificationState::NUM_STREAMS);

private:
    llgd_lwn::QueueHolder qh;
    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;

    static const lwn::Format::Enum ATTRIB_TEST_DEFAULT_FORMAT = lwn::Format::RGBA8;
    static const uint32_t ATTRIB_TEST_DEFAULT_RELATIVE_OFFSET = 1;
    static const uint32_t ATTRIB_TEST_DEFAULT_STREAM_INDEX = 3;

    static const uint32_t STREAM_TEST_DEFAULT_STRIDE = 5;
    static const uint32_t STREAM_TEST_DEFAULT_DIVISOR = 4;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    return true;
}

// ----------------------------------------------------------------------------
// Test Helpers
// ----------------------------------------------------------------------------

void Validator::AttributeTestSetup(CommandHandle& createdBaseCommand, GpuState& createdGpuState, bool isAttributesEnabled)
{
    // Set values for all attributes
    VertexAttribState vertexAttributes[VertexSpecificationState::NUM_ATTRIBUTES];
    for (uint32_t i = 0; i < VertexSpecificationState::NUM_ATTRIBUTES; ++i) {
        vertexAttributes[i].SetDefaults();
        if (isAttributesEnabled) {
            vertexAttributes[i].SetFormat(ATTRIB_TEST_DEFAULT_FORMAT, ATTRIB_TEST_DEFAULT_RELATIVE_OFFSET);
            vertexAttributes[i].SetStreamIndex(ATTRIB_TEST_DEFAULT_STREAM_INDEX);
        }
    }

    createdBaseCommand = m_spCommandHelper->MakeHandle([=](CommandBuffer* cbh) {
        cbh->BindVertexAttribState(VertexSpecificationState::NUM_ATTRIBUTES, vertexAttributes);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
    });

    // Extract gpuState for start/end
    createdGpuState = m_spCommandHelper->RunAndExtractGpuState(createdBaseCommand);
}

void Validator::StreamTestSetup(CommandHandle& createdBaseCommand, GpuState& createdGpuState, uint32_t numOfEnableStreams)
{
    VertexStreamState vertexStreams[VertexSpecificationState::NUM_STREAMS];
    for (uint32_t i = 0; i < VertexSpecificationState::NUM_STREAMS; ++i) {
        vertexStreams[i].SetDefaults();
        vertexStreams[i].SetStride(STREAM_TEST_DEFAULT_STRIDE);
        vertexStreams[i].SetDivisor(STREAM_TEST_DEFAULT_DIVISOR);
    }

    createdBaseCommand = m_spCommandHelper->MakeHandle([=](CommandBuffer* cbh) {
        cbh->BindVertexStreamState(numOfEnableStreams, vertexStreams);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
    });

    // Extract gpuState for start/end
    createdGpuState = m_spCommandHelper->RunAndExtractGpuState(createdBaseCommand);
}

// ----------------------------------------------------------------------------
// Test Global States
// ----------------------------------------------------------------------------

bool Validator::TestRenderEnable()
{
    static const int TEST_CASE = 2;
    static const bool defaultStates[TEST_CASE] = { false, true };

    auto colwToExelwtedStateValue = [](bool enable) -> uint32_t {
        return uint32_t(enable ? VertexSpecificationState::RENDER_ENABLE_MODE_HW_EVALUATED_TRUE : VertexSpecificationState::RENDER_ENABLE_MODE_HW_EVALUATED_FALSE);
    };

    for (int testI = 0; testI < TEST_CASE; ++testI) {
        // Create base command
        auto baseCommand = m_spCommandHelper->MakeHandle([=](CommandBuffer* cbh) {
            cbh->SetRenderEnable(defaultStates[testI]);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        const auto defaultGpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);
        // Evaluate default state
        const uint32_t expectedDefaultValue = colwToExelwtedStateValue(defaultStates[testI]);
        TEST_EQ_FMT(defaultGpuState.mmeShadowRegisters.GetBanks().bank6[2].value, expectedDefaultValue, "Failed on test case %d", testI);

        // Edit
        const auto editedState = !defaultStates[testI];
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditRenderEnable(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            defaultGpuState, defaultGpuState,
            editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        // Execute
        const auto editedGpuState = m_spCommandHelper->RunAndExtractGpuState(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test edited state
        const uint32_t expectedEditedValue = colwToExelwtedStateValue(editedState);
        TEST_EQ_FMT(editedGpuState.mmeShadowRegisters.GetBanks().bank6[2].value, expectedEditedValue, "Failed on test case %d", testI);
    }
    return true;
}

bool Validator::TestConditionalRenderEnable()
{
    static const int TEST_CASE = 2;
    static const LWNconditionalRenderMode defaultStates[TEST_CASE] = { LWN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL, LWN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL };
    static const LWNconditionalRenderMode editedStates[TEST_CASE] = { LWN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL, LWN_CONDITIONAL_RENDER_MODE_RENDER_IF_EQUAL };

    auto colwToExelwtedStateValue = [](LWNconditionalRenderMode mode) -> uint32_t {
        // We prepare a random address which won't be used at rendering. Thus "IF_NOT_EQUAL" will be "TRUE"
        return uint32_t(mode == LWN_CONDITIONAL_RENDER_MODE_RENDER_IF_NOT_EQUAL ? VertexSpecificationState::RENDER_ENABLE_MODE_HW_EVALUATED_TRUE : VertexSpecificationState::RENDER_ENABLE_MODE_HW_EVALUATED_FALSE);
    };

    // Create a random GPU address to use conditionalRenderEnable state
    static const size_t POOL_SIZE = 65536;
    llgd_lwn::MemoryPoolHolder pool;
    auto storage = LlgdAlignedAllocPodType<uint8_t>(POOL_SIZE, 4096);
    MemoryPoolBuilder pool_builder;
    pool_builder.SetDevice(g_device).SetDefaults()
        .SetFlags(MemoryPoolFlags::CPU_CACHED | MemoryPoolFlags::GPU_CACHED | MemoryPoolFlags::COMPRESSIBLE)
        .SetStorage(storage.get(), POOL_SIZE);
    TEST(pool.Initialize(&pool_builder));

    const auto conditionalRenderAddress = pool->GetBufferAddress();

    for (int testI = 0; testI < TEST_CASE; ++testI) {
        // Create base command
        auto baseCommand = m_spCommandHelper->MakeHandle([=](CommandBuffer* cbh) {
            cbh->SetRenderEnableConditional(lwn::ConditionalRenderMode::Enum(defaultStates[testI]), conditionalRenderAddress);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        const auto defaultGpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);
        // Evaluate default state
        const uint32_t expectedDefaultValue = colwToExelwtedStateValue(defaultStates[testI]);
        TEST_EQ_FMT(defaultGpuState.mmeShadowRegisters.GetBanks().bank6[2].value, expectedDefaultValue, "Failed on test case %d", testI);

        // Edit
        const auto editedState = editedStates[testI];
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditConditionalRenderEnable(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            defaultGpuState, defaultGpuState,
            editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        // Execute
        const auto editedGpuState = m_spCommandHelper->RunAndExtractGpuState(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test edited state
        const uint32_t expectedEditedValue = colwToExelwtedStateValue(editedState);
        TEST_EQ_FMT(editedGpuState.mmeShadowRegisters.GetBanks().bank6[2].value, expectedEditedValue, "Failed on test case %d", testI);
    }
    return true;
}

// Note: llgdCommandSetEditConditionalRenderDisable is alias of llgdCommandSetEditRenderEnable, thus we don't test for it.

// ----------------------------------------------------------------------------
// Test Attribute States
// ----------------------------------------------------------------------------

bool Validator::TestAttributeEnable()
{
    static const uint32_t TEST_CASES = 2; // true => false, false => true
    static const bool defaultStates[TEST_CASES] = { true, false };
    static const bool editedStates[TEST_CASES] = { false, true };

    for (uint32_t testI = 0; testI < TEST_CASES; ++testI) {
        for (uint32_t attI = 0; attI < VertexSpecificationState::NUM_ATTRIBUTES; ++attI) {
            // Setup state & create baseCommand / gpuState
            CommandHandle baseCommand;
            GpuState gpuState;
            AttributeTestSetup(baseCommand, gpuState, defaultStates[testI]);
            // Verify default state
            auto st = m_spCommandHelper->ExtractVertexSpecificationState(&gpuState);
            TEST_EQ(st.attributes[attI].enable.value, defaultStates[testI]);

            // Edit
            m_spCommandHelper->ResetPointersForEditingCB();
            auto editedHandle = llgdCommandSetEditVertexSpecificationStateAttributeEnable(
                baseCommand,
                indexMapFn,
                IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
                gpuState, gpuState,
                attI, editedStates[testI],
                m_spCommandHelper->WriteControlMemoryForEditing,
                m_spCommandHelper->WriteCommandMemoryForEditing,
                llgd_lwn::EmptyMethodUpdatedFn,
                m_spCommandHelper.get()
            );
            // Execute
            m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

            // Test state
            st = m_spCommandHelper->ExtractVertexSpecificationState();
            for (uint32_t compareAttI = 0; compareAttI < VertexSpecificationState::NUM_ATTRIBUTES; ++compareAttI) {
                const auto expected = compareAttI == attI ? editedStates[testI] : defaultStates[testI];
                const auto actual = st.attributes[compareAttI].enable.value;
                TEST_EQ_FMT(actual, expected, "failed at editing attribute index = %d", attI);
            }
        }
    }

    return true;
}

bool Validator::TestAttributeFormat()
{
    static const lwn::Format::Enum editedState = lwn::Format::RG32F;

    for (uint32_t attI = 0; attI < VertexSpecificationState::NUM_ATTRIBUTES; ++attI) {
        // Setup state & create baseCommand / gpuState
        CommandHandle baseCommand;
        GpuState gpuState;
        AttributeTestSetup(baseCommand, gpuState);
        // Verify default state
        auto st = m_spCommandHelper->ExtractVertexSpecificationState(&gpuState);
        TEST_EQ(st.attributes[attI].format.value, LWNformat(ATTRIB_TEST_DEFAULT_FORMAT));

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditVertexSpecificationStateAttributeFormat(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            attI, LWNformat(editedState),
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = m_spCommandHelper->ExtractVertexSpecificationState();
        for (uint32_t compareAttI = 0; compareAttI < VertexSpecificationState::NUM_ATTRIBUTES; ++compareAttI) {
            const auto expected = compareAttI == attI ? editedState : ATTRIB_TEST_DEFAULT_FORMAT;
            const auto actual = st.attributes[compareAttI].format.value;
            TEST_EQ_FMT(actual, LWNformat(expected), "failed at editing attribute index = %d", attI);
        }
    }

    return true;
}

bool Validator::TestAttributeRelativeOffset()
{
    static const uint32_t editedState = ATTRIB_TEST_DEFAULT_RELATIVE_OFFSET + 2;

    for (uint32_t attI = 0; attI < VertexSpecificationState::NUM_ATTRIBUTES; ++attI) {
        // Setup state & create baseCommand / gpuState
        CommandHandle baseCommand;
        GpuState gpuState;
        AttributeTestSetup(baseCommand, gpuState);
        // Verify default state
        auto st = m_spCommandHelper->ExtractVertexSpecificationState(&gpuState);
        TEST_EQ(st.attributes[attI].relativeOffset.value, ATTRIB_TEST_DEFAULT_RELATIVE_OFFSET);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditVertexSpecificationStateAttributeRelativeOffset(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            attI, editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = m_spCommandHelper->ExtractVertexSpecificationState();
        for (uint32_t compareAttI = 0; compareAttI < VertexSpecificationState::NUM_ATTRIBUTES; ++compareAttI) {
            const auto expected = compareAttI == attI ? editedState : ATTRIB_TEST_DEFAULT_RELATIVE_OFFSET;
            const auto actual = st.attributes[compareAttI].relativeOffset.value;
            TEST_EQ_FMT(actual, expected, "failed at editing attribute index = %d", attI);
        }
    }

    return true;
}

bool Validator::TestAttributeStreamIndex()
{
    static const uint32_t editedState = ATTRIB_TEST_DEFAULT_STREAM_INDEX + 2;

    for (uint32_t attI = 0; attI < VertexSpecificationState::NUM_ATTRIBUTES; ++attI) {
        // Setup state & create baseCommand / gpuState
        CommandHandle baseCommand;
        GpuState gpuState;
        AttributeTestSetup(baseCommand, gpuState);
        // Verify default state
        auto st = m_spCommandHelper->ExtractVertexSpecificationState(&gpuState);
        TEST_EQ(st.attributes[attI].streamIndex.value, ATTRIB_TEST_DEFAULT_STREAM_INDEX);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditVertexSpecificationStateAttributeStreamIndex(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            attI, editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = m_spCommandHelper->ExtractVertexSpecificationState();
        for (uint32_t compareAttI = 0; compareAttI < VertexSpecificationState::NUM_ATTRIBUTES; ++compareAttI) {
            const auto expected = compareAttI == attI ? editedState : ATTRIB_TEST_DEFAULT_STREAM_INDEX;
            const auto actual = st.attributes[compareAttI].streamIndex.value;
            TEST_EQ_FMT(actual, expected, "failed at editing attribute index = %d", attI);
        }
    }

    return true;
}

// ----------------------------------------------------------------------------
// Test Stream States
// ----------------------------------------------------------------------------

bool Validator::TestStreamEnable()
{
    for (uint32_t streamI = 0; streamI < VertexSpecificationState::NUM_STREAMS; ++streamI) {
        // Setup state & create baseCommand / GpuState
        CommandHandle baseCommand;
        GpuState gpuState;
        StreamTestSetup(baseCommand, gpuState, 0 /*numOfEnableStreams*/);

        // Verify default state
        auto st = m_spCommandHelper->ExtractVertexSpecificationState(&gpuState);
        TEST(!st.streams[streamI].enable.value);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditVertexSpecificationStateStreamEnable(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            streamI, true,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = m_spCommandHelper->ExtractVertexSpecificationState();
        for (uint32_t compareStreamI = 0; compareStreamI < VertexSpecificationState::NUM_STREAMS; ++compareStreamI) {
            const auto expected = compareStreamI == streamI ? true : false;
            const auto actual = st.streams[compareStreamI].enable.value;
            TEST_EQ_FMT(actual, expected, "failed at editing stream index = %d", streamI);
        }
    }

    return true;
}

bool Validator::TestStreamStride()
{
    static const uint32_t editedState = STREAM_TEST_DEFAULT_STRIDE + 2;

    for (uint32_t streamI = 0; streamI < VertexSpecificationState::NUM_STREAMS; ++streamI) {
        // Setup state & create baseCommand / GpuState
        CommandHandle baseCommand;
        GpuState gpuState;
        StreamTestSetup(baseCommand, gpuState);

        // Verify default state
        auto st = m_spCommandHelper->ExtractVertexSpecificationState(&gpuState);
        TEST_EQ(st.streams[streamI].stride.value, STREAM_TEST_DEFAULT_STRIDE);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditVertexSpecificationStateStreamStride(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            streamI, editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = m_spCommandHelper->ExtractVertexSpecificationState();
        for (uint32_t compareStreamI = 0; compareStreamI < VertexSpecificationState::NUM_STREAMS; ++compareStreamI) {
            const auto expected = compareStreamI == streamI ? editedState : STREAM_TEST_DEFAULT_STRIDE;
            const auto actual = st.streams[compareStreamI].stride.value;
            TEST_EQ_FMT(actual, expected, "failed at editing stream index = %d", streamI);
        }
    }

    return true;
}

bool Validator::TestStreamDivisor()
{
    static const uint32_t editedState = STREAM_TEST_DEFAULT_DIVISOR + 2;

    for (uint32_t streamI = 0; streamI < VertexSpecificationState::NUM_STREAMS; ++streamI) {
        // Setup state & create baseCommand / GpuState
        CommandHandle baseCommand;
        GpuState gpuState;
        StreamTestSetup(baseCommand, gpuState);

        // Verify default state
        auto st = m_spCommandHelper->ExtractVertexSpecificationState(&gpuState);
        TEST_EQ(st.streams[streamI].divisor.value, STREAM_TEST_DEFAULT_DIVISOR);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditVertexSpecificationStateStreamDivisor(
            baseCommand,
            indexMapFn,
            IDX_OFFSET - 1, IDX_OFFSET + 2,  // Edit everything in command handle
            gpuState, gpuState,
            streamI, editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );

        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = m_spCommandHelper->ExtractVertexSpecificationState();
        for (uint32_t compareStreamI = 0; compareStreamI < VertexSpecificationState::NUM_STREAMS; ++compareStreamI) {
            const auto expected = compareStreamI == streamI ? editedState : STREAM_TEST_DEFAULT_DIVISOR;
            const auto actual = st.streams[compareStreamI].divisor.value;
            TEST_EQ_FMT(actual, expected, "failed at editing stream index = %d", streamI);
        }
    }

    return true;
}


bool Validator::Test()
{
    TEST(TestRenderEnable());
    TEST(TestConditionalRenderEnable());
    TEST(TestAttributeEnable());
    TEST(TestAttributeFormat());
    TEST(TestAttributeRelativeOffset());
    TEST(TestAttributeStreamIndex());
    TEST(TestStreamEnable());
    TEST(TestStreamStride());
    TEST(TestStreamDivisor());

    return true;
}

LLGD_DEFINE_TEST(EditVertexSpecificationState, UNIT,
LwError Execute()
{
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
}
);
}
