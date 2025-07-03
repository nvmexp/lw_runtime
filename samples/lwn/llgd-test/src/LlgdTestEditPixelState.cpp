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
    bool TestColorLogicOp();
    bool TestColorAlphaTest();
    bool TestColorAlphaRef();
    bool TestColorBlendColor();

    bool TestDepthTest();
    bool TestDepthWrite();
    bool TestDepthFunc();
    bool TestDepthBoundsEnable();
    bool TestDepthBoundsNear();
    bool TestDepthBoundsFar();

    bool TestStencilTest();
    bool TestStencilValueMask();
    bool TestStencilMask();
    bool TestStencilRef();
    bool TestStencilFunc();
    bool TestStencilOpFail();
    bool TestStencilOpZFail();
    bool TestStencilOpZPass();

    bool TestColorTargetBlendEnable();
    bool TestColorTargetBlendColorEquation();
    bool TestColorTargetBlendAlphaEquation();
    bool TestColorTargetBlendColorSrcFunc();
    bool TestColorTargetBlendColorDstFunc();
    bool TestColorTargetBlendAlphaSrcFunc();
    bool TestColorTargetBlendAlphaDstFunc();
    bool TestColorTargetChannelMask();

private:
    //--------------------------------------------------------------
    // Test Helper methods
    //
    // Note: These methods are similar and we can refactor to one method
    // if we would like to do, but we don't do because of readability
    //--------------------------------------------------------------

    template <typename EditFn, typename StateGetterFn, typename StateType>
    bool TestStartEdgePatchCommon(
        const EditFn& editFn,               // Editor function: e.g. llgdCommandSetEditPixelStateColorAlphaTest
        const StateGetterFn& stateGetter,   // Extract state value function from PixelState
        const StateType& startState         // start state come from start edge patch
    );

    // Color test helper method
    template <typename EditFn, typename InitStateSetter, typename StateGetterFn, typename StateType>
    bool TestColorStateCommon(
        const EditFn& editFn,               // Editor function: e.g. llgdCommandSetEditPixelStateColorAlphaTest
        const InitStateSetter& initStateFn, // Setup command buffer function with given initValue
        const StateGetterFn& stateGetter,   // Extract state value function from PixelState::Color
        const StateType& defaultState,      // default state value
        const StateType& editedState,       // edited state value
        bool insertStartEdgePatch = false
    );

    // Depth test helper method
    template <typename EditFn, typename InitStateSetter, typename StateGetterFn, typename StateType>
    bool TestDepthStateCommon(
        const EditFn& editFn,               // Editor function: e.g. llgdCommandSetEditPixelStateDepthTest
        const InitStateSetter& initStateFn, // Setup command buffer function with given initValue
        const StateGetterFn& stateGetter,   // Extract state value function from PixelState::Depth
        const StateType& defaultState,      // default state value
        const StateType& editedState        // edited state value
    );

    // Stencil test with faces helper method
    template <typename EditFn, typename InitStateSetter, typename StateGetterFn, typename StateType>
    bool TestStencilFaceStateCommon(
        const EditFn& editFn,               // Editor function: e.g. llgdCommandSetEditPixelStateStencilValueMask
        const InitStateSetter& initStateFn, // Setup command buffer function with given faces & initValue
        const StateGetterFn& stateGetter,   // Extract state value function from PixelState::Stencil::StencilFace
        const StateType& defaultState,      // default state value
        const StateType& editedState        // edited state value
    );

    // ColorTarget BlendState test helper method
    template <typename EditFn, typename InitStateSetter, typename StateGetterFn, typename StateType>
    bool TestColorTargetBlendStateCommon(
        const EditFn& editFn,               // Editor function: e.g. llgdCommandSetEditPixelStateColorTargetBlendColorEquation
        const InitStateSetter& initStateFn, // Setup command buffer function with a given value
        const StateGetterFn& stateGetter,   // Extract state value function from PixelState::Blend
        const StateType& defaultState,      // default state value
        const StateType& editedState        // edited state value
    );

private:
    llgd_lwn::QueueHolder qh;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    return true;
}

template <typename EditFn, typename StateGetterFn, typename StateType>
bool Validator::TestStartEdgePatchCommon(
    const EditFn& editFn,               // Editor function: e.g. llgdCommandSetEditPixelStateColorAlphaTest
    const StateGetterFn& stateGetter,   // Extract state value function from PixelState
    const StateType& startState         // start state come from start edge patch
)
{
    enum class TestEvent : uint32_t {
        CLEAR_COLOR = 0,
        CLEAR_DEPTH_STENCIL,
        BARRIER,
        DEBUG_MARKER,

        EVENT_MAX_NUM
    };
    static const char* testDebugMarker = "test_debug_marker";
    static const auto debugDomainId = g_device->GenerateDebugDomainId(testDebugMarker);
    for (uint32_t testEvent = 0; testEvent < uint32_t(TestEvent::EVENT_MAX_NUM); testEvent++) {
        // Create basecommand
        auto baseCommand = m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
            switch (TestEvent(testEvent)) {
            case TestEvent::CLEAR_COLOR:
                cbh->ClearColor(0, BLUE, ClearColorMask::RGBA);
                break;
            case TestEvent::CLEAR_DEPTH_STENCIL:
                cbh->ClearDepthStencil(0.f, true, 0, 0x1D);
                break;
            case TestEvent::BARRIER:
                cbh->Barrier(~0);
                break;
            case TestEvent::DEBUG_MARKER:
                cbh->InsertDebugMarkerStatic(debugDomainId, testDebugMarker);
                break;
            default:
                CHECK(false);
                return;
            }
        });
        const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = editFn(
            baseCommand,
            indexMapFn,
            IDX_OFFSET + 1,  // Insert start edge patch before event method
            IDX_OFFSET + 5,  // Don't insert end edge patch
            gpuState, gpuState,
            startState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        TEST_NEQ(editedHandle, 0);

        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Check state is changed by start edge patch
        const auto st = m_spCommandHelper->ExtractPixelState();
        const auto actual = stateGetter(st);
        TEST_EQ_FMT(actual, startState, "Failed on testEvent=%d", testEvent);
    }

    return true;
}

//---------------------------------
// Test Color methods
//---------------------------------

// Color test helper method
template <typename EditFn, typename InitStateSetter, typename StateGetterFn, typename StateType>
bool Validator::TestColorStateCommon(
    const EditFn& editFn,               // Editor function: e.g. llgdCommandSetEditPixelStateColorAlphaTest
    const InitStateSetter& initStateFn, // Setup command buffer function with given initValue
    const StateGetterFn& stateGetter,   // Extract state value function from PixelState::Color
    const StateType& defaultState,      // default state value
    const StateType& editedState,       // edited state value
    bool insertStartEdgePatch
)
{
    // Create basecommand
    const auto baseCommand = m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
        initStateFn(cbh, defaultState);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        // Test with inserting start edge patch
        // => Start edge patch is inserted just before ClearColor.
        //    Thus we need to setup default state after ClearColor in order to test "insert edge patch" and "editing"
        if (insertStartEdgePatch) {
            initStateFn(cbh, defaultState);
        }
    });

    // Extract the baseCommand gpuState (use as start/end state)
    const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);

    // Verify default state
    {
        const auto st = m_spCommandHelper->ExtractPixelState(&gpuState);
        TEST_EQ(stateGetter(st.color), defaultState);
    }

    // Edit
    m_spCommandHelper->ResetPointersForEditingCB();
    const auto editedHandle = editFn(
        baseCommand,
        indexMapFn,
        insertStartEdgePatch ? IDX_OFFSET + 1 : IDX_OFFSET - 1,
        IDX_OFFSET + 1000,  // Don't insert end edge patch
        gpuState, gpuState,
        editedState,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        llgd_lwn::EmptyMethodUpdatedFn,
        m_spCommandHelper.get()
    );
    TEST_NEQ(editedHandle, 0);

    // Execute
    {
        const auto runHandle = m_spCommandHelper->MakeCommandHandleRunnable(editedHandle);
        m_spCommandHelper->Run(runHandle);
    }

    // Test state
    {
        const auto st = m_spCommandHelper->ExtractPixelState();
        const auto actual = stateGetter(st.color);
        TEST_EQ(actual, editedState);
    }

    return true;
}

bool Validator::TestColorLogicOp()
{
    // LWN_LOGIC_OP_COPY requires special handling in the driver.
    // It is good to test edit to/from LWN_LOGIC_OP_COPY.
    // Note: editing from LogicOpCopy to another, inserting start edge patch is required.
    static const int TEST_NUM = 3;
    static const LWNlogicOp defaultStates[TEST_NUM] = { LWN_LOGIC_OP_NOR, LWN_LOGIC_OP_COPY, LWN_LOGIC_OP_EQUIV };
    static const LWNlogicOp editedStates[TEST_NUM]  = { LWN_LOGIC_OP_AND, LWN_LOGIC_OP_OR, LWN_LOGIC_OP_COPY  };

    for (int testI = 0; testI < TEST_NUM; ++testI)
    {
        const auto defaultState = defaultStates[testI];
        const auto editedState = editedStates[testI];

        // Test1. Insert start edge patch test. LogicOp editing relies on
        // start edge patch insertion. This test should be done just in case.
        TEST(TestStartEdgePatchCommon(
            // EditFunction
            llgdCommandSetEditPixelStateColorLogicOp,
            // State getter
            [](const PixelState& ps) {
                return ps.color.logicOp.value;
            },
            // Start state
            editedState
        ));

        // Test2. Normal editing
        TEST_FMT(TestColorStateCommon(
            // EditFunction
            llgdCommandSetEditPixelStateColorLogicOp,
            // Setup initialize state
            [](CommandBuffer* cbh, LWNlogicOp defaultValue) {
                ColorState cs;
                cs.SetDefaults().SetLogicOp(static_cast<LogicOp::Enum>(defaultValue));
                cbh->BindColorState(&cs);
            },
            // State value getter
            [](const PixelState::Color& color) {
                return color.logicOp.value;
            },
            // Default / Edit value
            defaultState, editedState,
            // Insert start edge patch
            true
        ), "Failed on test case %d", testI);
    }
    return true;
}

bool Validator::TestColorAlphaTest()
{
    static const LWNalphaFunc defaultState = LWN_ALPHA_FUNC_LESS;
    static const LWNalphaFunc editedState = LWN_ALPHA_FUNC_GREATER;

    return TestColorStateCommon(
        // EditFunction
        llgdCommandSetEditPixelStateColorAlphaTest,
        // Setup initialize state
        [](CommandBuffer* cbh, LWNalphaFunc defaultValue) {
            ColorState cs;
            cs.SetDefaults();
            cs.SetAlphaTest(static_cast<AlphaFunc::Enum>(defaultValue));
            cbh->BindColorState(&cs);
        },
        // State value getter
        [](const PixelState::Color& color) {
            return color.alphaTest.value;
        },
        // Default / Edit value
        defaultState, editedState
    );
}

bool Validator::TestColorAlphaRef()
{
    static const float defaultState = 0.54f;
    static const float editedState = 0.325f;

    return TestColorStateCommon(
        // EditFunction
        llgdCommandSetEditPixelStateColorAlphaRef,
        // Setup initialize state
        [](CommandBuffer* cbh, float defaultValue) {
            cbh->SetAlphaRef(defaultValue);
        },
        // State value getter
        [](const PixelState::Color& color) {
            return color.alphaRef.value;
        },
        // Default / Edit value
        defaultState, editedState
    );
}

bool Validator::TestColorBlendColor()
{
    static const int COLOR_NUM = 4;
    static const float defaultState[COLOR_NUM]{ 0.1f, 0.2f, 0.3f, 0.4f };
    static const float editedState[COLOR_NUM]{ 0.45f, 0.55f, 0.65f, 0.75f };

    // Create basecommand
    auto baseCommand = m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
        cbh->SetBlendColor(defaultState);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
    });
    // Extract the baseCommand gpuState (use as start/end state)
    const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);
    // Verify default state
    auto st = m_spCommandHelper->ExtractPixelState(&gpuState);
    TEST_EQ(st.color.blendColor.value[0], defaultState[0]);
    TEST_EQ(st.color.blendColor.value[1], defaultState[1]);
    TEST_EQ(st.color.blendColor.value[2], defaultState[2]);
    TEST_EQ(st.color.blendColor.value[3], defaultState[3]);

    // Edit
    m_spCommandHelper->ResetPointersForEditingCB();
    auto editedHandle = llgdCommandSetEditPixelStateColorBlendColor(
        baseCommand,
        indexMapFn,
        0, IDX_OFFSET + 100,  // Edit everything in command handle
        gpuState, gpuState,
        editedState[0], editedState[1], editedState[2], editedState[3],
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        llgd_lwn::EmptyMethodUpdatedFn,
        m_spCommandHelper.get()
    );
    TEST_NEQ(editedHandle, 0);

    // Execute
    m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));
    // Test state
    st = m_spCommandHelper->ExtractPixelState();
    TEST_EQ(st.color.blendColor.value[0], editedState[0]);
    TEST_EQ(st.color.blendColor.value[1], editedState[1]);
    TEST_EQ(st.color.blendColor.value[2], editedState[2]);
    TEST_EQ(st.color.blendColor.value[3], editedState[3]);

    return true;
}

//---------------------------------
// Test Depth methods
//---------------------------------

template <typename EditFn, typename InitStateSetter, typename StateGetterFn, typename StateType>
bool Validator::TestDepthStateCommon(const EditFn& editFn, const InitStateSetter& initStateFn, const StateGetterFn& stateGetter, const StateType& defaultState, const StateType& editedState)
{
    // Create basecommand
    auto baseCommand = m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
        initStateFn(cbh, defaultState);
        cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
    });
    // Extract the baseCommand gpuState (use as start/end state)
    const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);
    // Verify default state
    auto st = m_spCommandHelper->ExtractPixelState(&gpuState);
    TEST_EQ(stateGetter(st.depth), defaultState);

    // Edit
    m_spCommandHelper->ResetPointersForEditingCB();
    auto editedHandle = editFn(
        baseCommand,
        indexMapFn,
        0, IDX_OFFSET + 100,  // Edit everything in command handle
        gpuState, gpuState,
        editedState,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        llgd_lwn::EmptyMethodUpdatedFn,
        m_spCommandHelper.get()
    );
    TEST_NEQ(editedHandle, 0);

    // Execute
    m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

    // Test state
    st = m_spCommandHelper->ExtractPixelState();
    TEST_EQ(stateGetter(st.depth), editedState);
    return true;
}


bool Validator::TestDepthTest()
{
    static const bool defaultState = true;
    static const bool editedState = false;

    return TestDepthStateCommon(
        // EditFunction
        llgdCommandSetEditPixelStateDepthTest,
        // Initialize default state function
        [](CommandBuffer* cbh, bool defaultValue) {
            DepthStencilState dss;
            dss.SetDefaults().SetDepthTestEnable(defaultValue);
            cbh->BindDepthStencilState(&dss);
        },
        // State value getter
        [](const PixelState::Depth& depth) {
            return depth.test.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestDepthWrite()
{
    static const bool defaultState = false;
    static const bool editedState = true;

    return TestDepthStateCommon(
        // EditFunction
        llgdCommandSetEditPixelStateDepthWrite,
        // Initialize default state function
        [](CommandBuffer* cbh, bool defaultValue) {
            DepthStencilState dss;
            dss.SetDefaults().SetDepthTestEnable(true).SetDepthWriteEnable(defaultValue);
            cbh->BindDepthStencilState(&dss);
        },
        // State value getter
        [](const PixelState::Depth& depth) {
            return depth.write.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestDepthFunc()
{
    static const LWNdepthFunc defaultState = LWN_DEPTH_FUNC_LESS;
    static const LWNdepthFunc editedState = LWN_DEPTH_FUNC_LEQUAL;

    return TestDepthStateCommon(
        // EditFunction
        llgdCommandSetEditPixelStateDepthFunc,
        // Initialize default state function
        [](CommandBuffer* cbh, LWNdepthFunc defaultValue) {
            DepthStencilState dss;
            dss.SetDefaults().SetDepthTestEnable(true).SetDepthFunc(lwn::DepthFunc::Enum(defaultValue));
            cbh->BindDepthStencilState(&dss);
        },
        // State value getter
        [](const PixelState::Depth& depth) {
            return depth.func.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestDepthBoundsEnable()
{
    static const bool defaultState = false;
    static const bool editedState = true;

    return TestDepthStateCommon(
        // EditFunction
        llgdCommandSetEditPixelStateDepthBoundsEnable,
        // Initialize default state function
        [](CommandBuffer* cbh, bool defaultValue) {
            cbh->SetDepthBounds(defaultValue, 1.f, 200.f);
        },
        // State value getter
        [](const PixelState::Depth& depth) {
            return depth.depthBoundsEnable.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestDepthBoundsNear()
{
    static const float defaultState = 1.2f;
    static const float editedState = 0.6f;

    return TestDepthStateCommon(
        // EditFunction
        llgdCommandSetEditPixelStateDepthBoundsNear,
        // Initialize default state function
        [](CommandBuffer* cbh, float defaultValue) {
            cbh->SetDepthBounds(true, defaultValue, 200.f);
        },
        // State value getter
        [](const PixelState::Depth& depth) {
            return depth.depthBoundsNear.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestDepthBoundsFar()
{
    static const float defaultState = 200.f;
    static const float editedState = 234.2f;

    return TestDepthStateCommon(
        // EditFunction
        llgdCommandSetEditPixelStateDepthBoundsFar,
        // Initialize default state function
        [](CommandBuffer* cbh, float defaultValue) {
            cbh->SetDepthBounds(true, 1.0f, defaultValue);
        },
        // State value getter
        [](const PixelState::Depth& depth) {
            return depth.depthBoundsFar.value;
        },
        defaultState, editedState
    );
}

//---------------------------------
// Test Stencil methods
//---------------------------------

bool Validator::TestStencilTest()
{
    static const int TEST_NUM = 2;
    static const bool defaultStates[TEST_NUM] = { false , true };
    static const bool editedStates[TEST_NUM] = { true , false };

    for (int i = 0; i < TEST_NUM; ++i) {
        const bool defaultState = defaultStates[i];
        const bool editedState = editedStates[i];

        // Test1. Insert start edge patch test. StencilEnable editing relies on
        // start edge patch insertion. This test should be done just in case.
        TEST(TestStartEdgePatchCommon(
            // EditFunction
            llgdCommandSetEditPixelStateStencilTest,
            // State getter
            [](const PixelState& ps) {
                return bool(ps.stencil.test.value);
            },
            // Start state
            editedState
        ));

        // Test2. Normal editing

        // Create basecommand
        auto baseCommand = m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
            DepthStencilState dss;
            dss.SetDefaults().SetStencilTestEnable(defaultState);
            cbh->BindDepthStencilState(&dss);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        // Extract the baseCommand gpuState (use as start/end state)
        const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);
        // Verify default state
        auto st = m_spCommandHelper->ExtractPixelState(&gpuState);
        TEST_EQ(bool(st.stencil.test.value), defaultState);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditPixelStateStencilTest(
            baseCommand,
            indexMapFn,
            IDX_OFFSET + 1, IDX_OFFSET + 5,  // Insert start edge patch, don't insert end edge patch
            gpuState, gpuState,
            editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        TEST_NEQ(editedHandle, 0);

        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = m_spCommandHelper->ExtractPixelState();
        TEST_EQ_FMT(bool(st.stencil.test.value), editedState, "Failed on test case %d", i);
    }

    return true;
}

template <typename EditFn, typename InitStateSetter, typename StateGetterFn, typename StateType>
bool Validator::TestStencilFaceStateCommon(const EditFn& editFn, const InitStateSetter& initStateFn, const StateGetterFn& stateGetter, const StateType& defaultState, const StateType& editedState)
{
    static const int FACE_COUNT = 3;
    static const LWNface faces[FACE_COUNT] = { LWN_FACE_FRONT, LWN_FACE_BACK, LWN_FACE_FRONT_AND_BACK };

    for (int faceI = 0; faceI < FACE_COUNT; ++faceI) {
        const auto face = faces[faceI];

        // Create basecommand
        auto baseCommand = m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
            initStateFn(cbh, lwn::Face::FRONT_AND_BACK, defaultState);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        // Extract the baseCommand gpuState (use as start/end state)
        const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);
        // Verify default state
        auto st = m_spCommandHelper->ExtractPixelState(&gpuState);
        TEST_EQ(stateGetter(st.stencil.front), defaultState);
        TEST_EQ(stateGetter(st.stencil.back) , defaultState);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = editFn(
            baseCommand,
            indexMapFn,
            0, IDX_OFFSET + 100,  // Edit everything in command handle
            gpuState, gpuState,
            face,
            editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        TEST_NEQ(editedHandle, 0);

        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));

        // Test state
        st = m_spCommandHelper->ExtractPixelState();
        switch (face)
        {
        case LWN_FACE_FRONT:
            TEST_EQ(stateGetter(st.stencil.front), editedState);
            TEST_EQ(stateGetter(st.stencil.back) , defaultState);
            break;
        case LWN_FACE_BACK:
            TEST_EQ(stateGetter(st.stencil.front), defaultState);
            TEST_EQ(stateGetter(st.stencil.back) , editedState);
            break;
        case LWN_FACE_FRONT_AND_BACK:
            TEST_EQ(stateGetter(st.stencil.front), editedState);
            TEST_EQ(stateGetter(st.stencil.back) , editedState);
            break;
        default:
            TEST(!"Unknown face");
        }
    }

    return true;
}

bool Validator::TestStencilValueMask()
{
    static const uint8_t defaultState = 0x43;
    static const uint8_t editedState  = 0x1F;
    return TestStencilFaceStateCommon(
        // EditorFunction
        llgdCommandSetEditPixelStateStencilValueMask,
        // State initialize function
        [](CommandBuffer* cbh, lwn::Face::Enum faces, uint8_t defaultValue) {
            cbh->SetStencilValueMask(faces, defaultValue);
        },
        // State extractor function
        [](const PixelState::Stencil::StencilFace& stencilFace) {
            return stencilFace.valueMask.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestStencilMask()
{
    static const int TEST_CASE = 3;
    static const uint8_t defaultStates[TEST_CASE] = { 0x13, 0x3E, 0x00 };
    static const uint8_t editedStates[TEST_CASE] = { 0xAF, 0x00, 0x02 };

    for (int i=0; i<TEST_CASE; ++i) {
        // Test1. Insert start edge patch test.
        // Set stencil mask is used in ClearDepthStencil (shadow ram control passthrough mode).
        // We need to check the editor correctly works with it.
        TEST_FMT(TestStartEdgePatchCommon(
            // EditFunction
            [](LWNcommandHandle baseCommand,
               IndexMapFn setIndexToGpuIndex,
               uint32_t startIndex,
               uint32_t endIndex,
               const GpuState& startState,
               const GpuState& endState,
               uint8_t mask,
               WriteMemoryFn writeControlMemoryFn,
               WriteMemoryFn writeCommandMemoryFn,
               MethodUpdatedFn methodUpdated,
               void* callbacksData) {
                return llgdCommandSetEditPixelStateStencilMask(baseCommand, setIndexToGpuIndex, startIndex, endIndex,
                    startState, endState, LWN_FACE_FRONT, mask,
                    writeControlMemoryFn, writeCommandMemoryFn, methodUpdated, callbacksData);
            },
            // State getter
            [](const PixelState& ps) {
                return ps.stencil.front.mask.value;
            },
            // Start state
            editedStates[i]
        ), "Failed on test case %d", i);

        // 2. Normal edit
        TEST_FMT(TestStencilFaceStateCommon(
            // EditorFunction
            llgdCommandSetEditPixelStateStencilMask,
            // State initialize function
            [](CommandBuffer* cbh, lwn::Face::Enum faces, uint8_t defaultValue) {
                cbh->SetStencilMask(faces, defaultValue);
            },
            // State extractor function
            [](const PixelState::Stencil::StencilFace& stencilFace) {
                return stencilFace.mask.value;
            },
            defaultStates[i], editedStates[i]
        ), "Failed on test case %d", i);
    }

    return true;
}

bool Validator::TestStencilRef()
{
    static const uint8_t defaultState = 0x2D;
    static const uint8_t editedState = 0x89;
    return TestStencilFaceStateCommon(
        // EditorFunction
        llgdCommandSetEditPixelStateStencilRef,
        // State initialize function
        [](CommandBuffer* cbh, lwn::Face::Enum faces, uint8_t defaultValue) {
            cbh->SetStencilRef(faces, defaultValue);
        },
        // State extractor function
        [](const PixelState::Stencil::StencilFace& stencilFace) {
            return stencilFace.ref.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestStencilFunc()
{
    static const LWNstencilFunc defaultState = LWN_STENCIL_FUNC_EQUAL;
    static const LWNstencilFunc editedState = LWN_STENCIL_FUNC_GREATER;

    return TestStencilFaceStateCommon(
        // EditorFunction
        llgdCommandSetEditPixelStateStencilFunc,
        // State initialize function
        [](CommandBuffer* cbh, lwn::Face::Enum faces, LWNstencilFunc defaultValue) {
            DepthStencilState dss;
            dss.SetDefaults()
                .SetStencilTestEnable(true)
                .SetStencilFunc(faces, lwn::StencilFunc::Enum(defaultValue));
            cbh->BindDepthStencilState(&dss);
        },
        // State extractor function
        [](const PixelState::Stencil::StencilFace& stencilFace) {
            return stencilFace.func.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestStencilOpFail()
{
    static const LWNstencilOp defaultState = LWN_STENCIL_OP_REPLACE;
    static const LWNstencilOp editedState = LWN_STENCIL_OP_DECR;

    return TestStencilFaceStateCommon(
        // EditorFunction
        llgdCommandSetEditPixelStateStencilOpFail,
        // State initialize function
        [](CommandBuffer* cbh, lwn::Face::Enum face, LWNstencilOp defaultValue) {
            const auto defSt = lwn::StencilOp::Enum(defaultValue);
            DepthStencilState dss;
            dss.SetDefaults()
                .SetStencilTestEnable(true)
                .SetStencilOp(face, defSt, defSt, defSt);
            cbh->BindDepthStencilState(&dss);
        },
        // State extractor function
        [](const PixelState::Stencil::StencilFace& stencilFace) {
            return stencilFace.opFail.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestStencilOpZFail()
{
    static const LWNstencilOp defaultState = LWN_STENCIL_OP_INCR;
    static const LWNstencilOp editedState = LWN_STENCIL_OP_ZERO;

    return TestStencilFaceStateCommon(
        // EditorFunction
        llgdCommandSetEditPixelStateStencilOpZFail,
        // State initialize function
        [](CommandBuffer* cbh, lwn::Face::Enum face, LWNstencilOp defaultValue) {
            const auto defSt = lwn::StencilOp::Enum(defaultValue);
            DepthStencilState dss;
            dss.SetDefaults()
                .SetStencilTestEnable(true)
                .SetStencilOp(face, defSt, defSt, defSt);
            cbh->BindDepthStencilState(&dss);
        },
        // State extractor function
        [](const PixelState::Stencil::StencilFace& stencilFace) {
            return stencilFace.opZFail.value;
        },
        defaultState, editedState
    );
}

bool Validator::TestStencilOpZPass()
{
    static const LWNstencilOp defaultState = LWN_STENCIL_OP_DECR_WRAP;
    static const LWNstencilOp editedState = LWN_STENCIL_OP_INCR_WRAP;

    return TestStencilFaceStateCommon(
        // EditorFunction
        llgdCommandSetEditPixelStateStencilOpZPass,
        // State initialize function
        [](CommandBuffer* cbh, lwn::Face::Enum face, LWNstencilOp defaultValue) {
            const auto defSt = lwn::StencilOp::Enum(defaultValue);
            DepthStencilState dss;
            dss.SetDefaults()
                .SetStencilTestEnable(true)
                .SetStencilOp(face, defSt, defSt, defSt);
            cbh->BindDepthStencilState(&dss);
        },
        // State extractor function
        [](const PixelState::Stencil::StencilFace& stencilFace) {
            return stencilFace.opZPass.value;
        },
        defaultState, editedState
    );
}

//---------------------------------
// Test ColorTarget methods
//---------------------------------

template <typename EditFn, typename InitStateSetter, typename StateGetterFn, typename StateType>
bool Validator::TestColorTargetBlendStateCommon(
    const EditFn& editFn,               // Editor function: e.g. llgdCommandSetEditPixelStateColorTargetBlendColorEquation
    const InitStateSetter& initStateFn, // Setup command buffer function with a given value
    const StateGetterFn& stateGetter,   // Extract state value function from PixelState::Blend
    const StateType& defaultState,      // default state value
    const StateType& editedState        // edited state value
)
{
    // Test over color targets
    for (uint32_t targetIdx = 0; targetIdx < PixelState::NUM_COLOR_TARGETS; ++targetIdx) {
        // Create basecommand
        auto baseCommand = m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
            initStateFn(cbh, defaultState);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        // Extract the baseCommand gpuState (use as start/end state)
        const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);
        // Verify default state
        auto st = m_spCommandHelper->ExtractPixelState(&gpuState);
        TEST_EQ(stateGetter(st.colorTargets[targetIdx].blend), defaultState);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = editFn(
            baseCommand,
            indexMapFn,
            0, IDX_OFFSET + 100,  // Edit everything in command handle
            gpuState, gpuState,
            targetIdx, editedState,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        TEST_NEQ(editedHandle, 0);

        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));
        // Test state
        st = m_spCommandHelper->ExtractPixelState();
        for (uint32_t idx = 0; idx < PixelState::NUM_COLOR_TARGETS; ++idx) {
            const auto& compareTarget = idx == targetIdx ? editedState : defaultState;
            const auto actual = stateGetter(st.colorTargets[idx].blend);
            TEST_EQ_FMT(actual, compareTarget, "Failed to test blend state: editing color index=%d failed checking index=%d", targetIdx, idx);
        }
    }

    return true;
}

bool Validator::TestColorTargetBlendEnable()
{
    static const uint32_t TEST_CASE = 2;
    static const bool defaultStates[TEST_CASE] = { true, false };
    static const bool editedStates[TEST_CASE] = { false, true };

    // Test over color targets
    for (uint32_t testI = 0; testI < TEST_CASE; ++testI) {
        const auto defaultState = defaultStates[testI];
        const auto editedState  = editedStates[testI];

        TEST_FMT(TestColorTargetBlendStateCommon(
            // Editor function
            llgdCommandSetEditPixelStateColorTargetBlendEnable,
            // Setup initial state
            [](CommandBuffer* cbh, bool defautValue) {
                ColorState cs;
                cs.SetDefaults();
                for (uint32_t i = 0; i < PixelState::NUM_COLOR_TARGETS; ++i) {
                    cs.SetBlendEnable(i, defautValue);
                }
                cbh->BindColorState(&cs);
            },
            // State getter from BlendState
            [](const PixelState::Blend& blendState) {
                return blendState.enable.value;
            },
            // Default / Edit values
            defaultState, editedState
        ), "Failed on Test case %d", testI);
    }

    return true;
}

bool Validator::TestColorTargetBlendColorEquation()
{
    static const LWNblendEquation defaultState = LWN_BLEND_EQUATION_SUB;
    static const LWNblendEquation editedState = LWN_BLEND_EQUATION_MIN;

    return TestColorTargetBlendStateCommon(
        // Editor function
        llgdCommandSetEditPixelStateColorTargetBlendColorEquation,
        // Setup initial state
        [](CommandBuffer* cbh, LWNblendEquation defautValue) {
            BlendState bs;
            bs.SetDefaults();
            for (uint32_t i = 0; i < PixelState::NUM_COLOR_TARGETS; ++i) {
                bs.SetBlendTarget(i).SetBlendEquation(BlendEquation::Enum(defautValue), lwn::BlendEquation::ADD);
                cbh->BindBlendState(&bs);
            }
        },
        // State getter from BlendState
        [](const PixelState::Blend& blendState) {
            return blendState.colorEquation.value;
        },
        // Default / Edit values
        defaultState, editedState
    );
}

bool Validator::TestColorTargetBlendAlphaEquation()
{
    static const LWNblendEquation defaultState = LWN_BLEND_EQUATION_SUB;
    static const LWNblendEquation editedState = LWN_BLEND_EQUATION_MIN;

    return TestColorTargetBlendStateCommon(
        // Editor function
        llgdCommandSetEditPixelStateColorTargetBlendAlphaEquation,
        // Setup initial state
        [](CommandBuffer* cbh, LWNblendEquation defautValue) {
            BlendState bs;
            bs.SetDefaults();
            for (uint32_t i = 0; i < PixelState::NUM_COLOR_TARGETS; ++i) {
                bs.SetBlendTarget(i).SetBlendEquation(lwn::BlendEquation::ADD, BlendEquation::Enum(defautValue));
                cbh->BindBlendState(&bs);
            }
        },
        // State getter from BlendState
        [](const PixelState::Blend& blendState) {
            return blendState.alphaEquation.value;
        },
        // Default / Edit values
        defaultState, editedState
    );
}

bool Validator::TestColorTargetBlendColorSrcFunc()
{
    static const LWNblendFunc defaultState = LWN_BLEND_FUNC_ONE_MINUS_SRC_COLOR;
    static const LWNblendFunc editedState = LWN_BLEND_FUNC_DST_ALPHA;

    return TestColorTargetBlendStateCommon(
        // Editor function
        llgdCommandSetEditPixelStateColorTargetBlendColorSrcFunc,
        // Setup initial state
        [](CommandBuffer* cbh, LWNblendFunc defautValue) {
            BlendState bs;
            bs.SetDefaults();
            for (uint32_t i = 0; i < PixelState::NUM_COLOR_TARGETS; ++i) {
                bs.SetBlendTarget(i).SetBlendFunc(BlendFunc::Enum(defautValue), lwn::BlendFunc::ONE, lwn::BlendFunc::ONE, lwn::BlendFunc::ONE);
                cbh->BindBlendState(&bs);
            }
        },
        // State getter from BlendState
        [](const PixelState::Blend& blendState) {
            return blendState.srcFunc.value;
        },
        // Default / Edit values
        defaultState, editedState
    );
}

bool Validator::TestColorTargetBlendColorDstFunc()
{
    static const LWNblendFunc defaultState = LWN_BLEND_FUNC_ONE_MINUS_SRC_COLOR;
    static const LWNblendFunc editedState = LWN_BLEND_FUNC_DST_ALPHA;

    return TestColorTargetBlendStateCommon(
        // Editor function
        llgdCommandSetEditPixelStateColorTargetBlendColorDstFunc,
        // Setup initial state
        [](CommandBuffer* cbh, LWNblendFunc defautValue) {
            BlendState bs;
            bs.SetDefaults();
            for (uint32_t i = 0; i < PixelState::NUM_COLOR_TARGETS; ++i) {
                bs.SetBlendTarget(i).SetBlendFunc(lwn::BlendFunc::ONE, BlendFunc::Enum(defautValue), lwn::BlendFunc::ONE, lwn::BlendFunc::ONE);
                cbh->BindBlendState(&bs);
            }
        },
        // State getter from BlendState
        [](const PixelState::Blend& blendState) {
            return blendState.dstFunc.value;
        },
        // Default / Edit values
        defaultState, editedState
    );
}

bool Validator::TestColorTargetBlendAlphaSrcFunc()
{
    static const LWNblendFunc defaultState = LWN_BLEND_FUNC_ONE_MINUS_SRC_COLOR;
    static const LWNblendFunc editedState = LWN_BLEND_FUNC_DST_ALPHA;

    return TestColorTargetBlendStateCommon(
        // Editor function
        llgdCommandSetEditPixelStateColorTargetBlendAlphaSrcFunc,
        // Setup initial state
        [](CommandBuffer* cbh, LWNblendFunc defautValue) {
            BlendState bs;
            bs.SetDefaults();
            for (uint32_t i = 0; i < PixelState::NUM_COLOR_TARGETS; ++i) {
                bs.SetBlendTarget(i).SetBlendFunc(lwn::BlendFunc::ONE, lwn::BlendFunc::ONE, BlendFunc::Enum(defautValue), lwn::BlendFunc::ONE);
                cbh->BindBlendState(&bs);
            }
        },
        // State getter from BlendState
        [](const PixelState::Blend& blendState) {
            return blendState.srcFuncAlpha.value;
        },
        // Default / Edit values
        defaultState, editedState
    );
}

bool Validator::TestColorTargetBlendAlphaDstFunc()
{
    static const LWNblendFunc defaultState = LWN_BLEND_FUNC_ONE_MINUS_SRC_COLOR;
    static const LWNblendFunc editedState = LWN_BLEND_FUNC_DST_ALPHA;

    return TestColorTargetBlendStateCommon(
        // Editor function
        llgdCommandSetEditPixelStateColorTargetBlendAlphaDstFunc,
        // Setup initial state
        [](CommandBuffer* cbh, LWNblendFunc defautValue) {
            BlendState bs;
            bs.SetDefaults();
            for (uint32_t i = 0; i < PixelState::NUM_COLOR_TARGETS; ++i) {
                bs.SetBlendTarget(i).SetBlendFunc(lwn::BlendFunc::ONE, lwn::BlendFunc::ONE, lwn::BlendFunc::ONE, BlendFunc::Enum(defautValue));
                cbh->BindBlendState(&bs);
            }
        },
        // State getter from BlendState
        [](const PixelState::Blend& blendState) {
            return blendState.dstFuncAlpha.value;
        },
        // Default / Edit values
        defaultState, editedState
    );
}

bool Validator::TestColorTargetChannelMask()
{
    static const int TARGET_NUM = PixelState::NUM_COLOR_TARGETS;
    static const PixelState::ChannelMask defaultState{ true, false, true, false };
    static const PixelState::ChannelMask editedState{ false, true, false, true };

    // Setup channel mask state function
    auto setupState = [&](ChannelMaskState& cms) {
        cms.SetDefaults();
        for (int i = 0; i < TARGET_NUM; ++i) {
            cms.SetChannelMask(i, defaultState.r, defaultState.g, defaultState.b, defaultState.a);
        }
    };

    // Test over color targets
    for (int targetIdx = 0; targetIdx < TARGET_NUM; ++targetIdx) {
        // Create basecommand
        auto baseCommand = m_spCommandHelper->MakeHandle([&](CommandBuffer* cbh) {
            ChannelMaskState cms;
            setupState(cms);
            cbh->BindChannelMaskState(&cms);
            cbh->ClearColor(0, BLUE, ClearColorMask::RGBA); // Event1 method
        });
        // Extract the baseCommand gpuState (use as start/end state)
        const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(baseCommand);
        // Verify default state
        auto st = m_spCommandHelper->ExtractPixelState(&gpuState);
        TEST_EQ_FMT(st.colorTargets[targetIdx].channelMask.value.r, defaultState.r, "color index failed to edit=%d", targetIdx);
        TEST_EQ_FMT(st.colorTargets[targetIdx].channelMask.value.g, defaultState.g, "color index failed to edit=%d", targetIdx);
        TEST_EQ_FMT(st.colorTargets[targetIdx].channelMask.value.b, defaultState.b, "color index failed to edit=%d", targetIdx);
        TEST_EQ_FMT(st.colorTargets[targetIdx].channelMask.value.a, defaultState.a, "color index failed to edit=%d", targetIdx);

        // Edit
        m_spCommandHelper->ResetPointersForEditingCB();
        auto editedHandle = llgdCommandSetEditPixelStateColorTargetChannelMask(
            baseCommand,
            indexMapFn,
            0, IDX_OFFSET + 100,  // Edit everything in command handle
            gpuState, gpuState,
            targetIdx, editedState.r, editedState.g, editedState.b, editedState.a,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        TEST_NEQ(editedHandle, 0);

        // Execute
        m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedHandle));
        // Test state
        st = m_spCommandHelper->ExtractPixelState();
        for (int idx = 0; idx < TARGET_NUM; ++idx) {
            const auto& compareTarget = idx == targetIdx ? editedState : defaultState;
            TEST_EQ_FMT(st.colorTargets[idx].channelMask.value.r, compareTarget.r, "edit color index=%d, check failed index=%d", targetIdx, idx);
            TEST_EQ_FMT(st.colorTargets[idx].channelMask.value.g, compareTarget.g, "edit color index=%d, check failed index=%d", targetIdx, idx);
            TEST_EQ_FMT(st.colorTargets[idx].channelMask.value.b, compareTarget.b, "edit color index=%d, check failed index=%d", targetIdx, idx);
            TEST_EQ_FMT(st.colorTargets[idx].channelMask.value.a, compareTarget.a, "edit color index=%d, check failed index=%d", targetIdx, idx);
        }
    }

    return true;
}

bool Validator::Test()
{
    TEST(TestColorLogicOp());
    TEST(TestColorAlphaTest());
    TEST(TestColorAlphaRef());
    TEST(TestColorBlendColor());

    TEST(TestDepthTest());
    TEST(TestDepthWrite());
    TEST(TestDepthFunc());
    TEST(TestDepthBoundsEnable());
    TEST(TestDepthBoundsNear());
    TEST(TestDepthBoundsFar());

    TEST(TestStencilTest());
    TEST(TestStencilValueMask());
    TEST(TestStencilMask());
    TEST(TestStencilRef());
    TEST(TestStencilFunc());
    TEST(TestStencilOpFail());
    TEST(TestStencilOpZFail());
    TEST(TestStencilOpZPass());

    TEST(TestColorTargetBlendEnable());
    TEST(TestColorTargetBlendColorEquation());
    TEST(TestColorTargetBlendAlphaEquation());
    TEST(TestColorTargetBlendColorSrcFunc());
    TEST(TestColorTargetBlendColorDstFunc());
    TEST(TestColorTargetBlendAlphaSrcFunc());
    TEST(TestColorTargetBlendAlphaDstFunc());
    TEST(TestColorTargetChannelMask());

    return true;
}

LLGD_DEFINE_TEST(EditPixelState, UNIT,
LwError Execute()
{
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
}
);
}
