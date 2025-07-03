/*
 * Copyright (c) 2017-2019, LWPU CORPORATION.  All rights reserved.
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

#include <nn/os.h>
#include <iostream>
#include <vector>
#include <sstream>

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>

class EditRasterStateValidator {
public:
    bool Initialize();
    bool Test();

private:
    // Test editing without start/end insert pattern
    template <typename EditFunction, typename StateGetterFunc, typename StateType, typename InitContructFunction>
    bool TestEditingWithoutInsertion(
        EditFunction&& editFunction,                // StateEditor function e.g.: llgdCommandSetEditRasterStateRasterizationRasterizerDiscard
        InitContructFunction&& initConstructFunc,   // Function to construct an initial command buffer
        StateGetterFunc&& stateGetterFunc,          // Target state extractor from RasterState
        const StateType &initState,                 // Init state
        const StateType &editState                  // Edit state. initState should be edited to this value
    );

    //---------------------------------
    // Test rasterization states
    //---------------------------------
    bool TestEditDiscard();
    bool TestEditPolygonOffset();
    bool TestEditPointSize();
    bool TestEditLineWidth();
    bool TestEditSampleMask();
    bool TestEditConservativeRasterEnable();
    bool TestEditConservativeRasterDilate();
    bool TestEditSubpixelPrecisionBias();

    //---------------------------------
    // Test polygon states
    //---------------------------------
    bool TestPolygonLwllFace();
    bool TestPolygonFrontFace();
    bool TestPolygonPolygonMode();
    bool TestPolygonPolygonOffsetEnables();

    //---------------------------------
    // Test multisample states
    //---------------------------------
    bool TestEditMultisampleAntialiasEnable();
    bool TestEditMultisampleAlphaToCoverageEnable();
    bool TestEditMultisampleAlphaToCoverageDitheringEnable();
    bool TestEditMultisampleaCoverageToColorEnable();
    bool TestEditMultisampleCoverageToColorOutput();
    bool TestEditMultisampleCoverageModulationMode();
    bool TestEditMultisampleRasterSamples();

private:
    const LWNdevtoolsBootstrapFunctions* devtools;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;

    llgd_lwn::QueueHolder qh;
};

static const float blue[4]{ 0, 0, 1, 1 };
static const float green[4]{ 0, 1, 0, 1 };
static const uint32_t INDEX_OFFSET = 10;
static const auto indexMapFn = [](uint32_t gpfifoOrderIdx, void*) { return gpfifoOrderIdx + INDEX_OFFSET; };

//---------------------------------
// Initializers
//---------------------------------
bool EditRasterStateValidator::Initialize()
{
    devtools = lwnDevtoolsBootstrap();

    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());
    return true;
}

//---------------------------------
// Test helper methods
//---------------------------------
std::ostream & operator<< (std::ostream& out, const RasterState::Rasterization::SubpixelPrecisionBias &spb) {
    out << "(x,y)=(" << spb.x << "," << spb.y << ")";
    return out;
}

static bool operator==(const RasterState::Rasterization::SubpixelPrecisionBias &lhs, const RasterState::Rasterization::SubpixelPrecisionBias &rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

// Common template function to test editing without inserting start/end state
template <typename EditFunction, typename StateGetterFunc, typename StateType, typename InitContructFunction>
bool EditRasterStateValidator::TestEditingWithoutInsertion(
    EditFunction&& editFunction,
    InitContructFunction&& initConstructFunc,
    StateGetterFunc&& stateGetterFunc,
    const StateType &initState,
    const StateType &editState)
{
    // Create base command handle and test init state
    LWNcommandHandle handle = m_spCommandHelper->MakeHandle(initConstructFunc);
    TEST(handle != 0);
    const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(handle);

    // Extract init state
    auto rasterState = m_spCommandHelper->ExtractRasterState(&gpuState);
    const auto initActual = stateGetterFunc(rasterState);
    TEST_EQ(initActual, initState);

    // Edit all. Don't insert start/end state
    m_spCommandHelper->ResetPointersForEditingCB();
    auto editedCommandHandle = editFunction(
        handle,
        indexMapFn,
        0, 1000,
        gpuState, gpuState,
        editState,
        m_spCommandHelper->WriteControlMemoryForEditing,
        m_spCommandHelper->WriteCommandMemoryForEditing,
        llgd_lwn::EmptyMethodUpdatedFn,
        m_spCommandHelper.get());
    TEST(uint64_t(editedCommandHandle) != 0);

    // Test by exelwting
    m_spCommandHelper->Run(m_spCommandHelper->MakeCommandHandleRunnable(editedCommandHandle));
    rasterState = m_spCommandHelper->ExtractRasterState();
    const auto editedActual = stateGetterFunc(rasterState);
    TEST_EQ(editedActual, editState);

    return true;
}

//---------------------------------
// Test Rasterization methods
//---------------------------------

bool EditRasterStateValidator::TestEditDiscard()
{
    static const bool initState = false;
    static const auto editedState = !initState;

    auto bufferConstructor = [](CommandBuffer *cbh) {
        cbh->SetRasterizerDiscard(initState);
        cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
    };

    return TestEditingWithoutInsertion(
        // Editor function
        llgdCommandSetEditRasterStateRasterizationRasterizerDiscard,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState &st) {
            return st.rasterization.rasterizerDiscard.value;
        },
        // Init/Edit values
        initState, editedState
    );
}

bool EditRasterStateValidator::TestEditPolygonOffset()
{
    // Create base command handle
    static const float polygonOffsetFactor = 0.23f;
    static const float polygonOffsetUnits = 0.448f;
    static const float polygonOffsetClamp = 0.9667;

    static const float editPolygonOffsetFactor = 0.75f;
    static const float editPolygonOffsetUnits = 0.119f;
    static const float editPolygonOffsetClamp = 0.887f;

    auto bufferConstructor = [&](CommandBuffer *cbh){
        cbh->SetPolygonOffsetClamp(polygonOffsetFactor, polygonOffsetUnits, polygonOffsetClamp);
        cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
    };

    TEST(TestEditingWithoutInsertion(
        // Editor function
        llgdCommandSetEditRasterStateRasterizationPolygonOffsetFactor,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState& rasterState) {
            return rasterState.rasterization.polygonOffsetFactor.value;
        },
        // Init/Edit values
        polygonOffsetFactor, editPolygonOffsetFactor
    ));

    TEST(TestEditingWithoutInsertion(
        // Editor function
        llgdCommandSetEditRasterStateRasterizationPolygonOffsetUnits,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState& rasterState) {
            return rasterState.rasterization.polygonOffsetUnits.value;
        },
        polygonOffsetUnits, editPolygonOffsetUnits
    ));

    TEST(TestEditingWithoutInsertion(
        // Editor function
        llgdCommandSetEditRasterStateRasterizationPolygonOffsetClamp,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState& rasterState) {
            return rasterState.rasterization.polygonOffsetClamp.value;
        },
        // Init/Edit values
        polygonOffsetClamp, editPolygonOffsetClamp
    ));

    return true;
}

bool EditRasterStateValidator::TestEditPointSize()
{
    static const float initPointSize = 1.4f;
    static const float editedPointSize = 4.336f;

    auto bufferConstructor = [](CommandBuffer *cbh) {
        cbh->SetPointSize(initPointSize);
        cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
    };

    return TestEditingWithoutInsertion(
        // Editor function
        llgdCommandSetEditRasterStateRasterizationPointSize,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState &st) {
            return st.rasterization.pointSize.value;
        },
        // Init/Edit values
        initPointSize, editedPointSize
    );
}

bool EditRasterStateValidator::TestEditLineWidth()
{
    static const float initLineWidth = 1.259;
    static const float editedState = 0.5527f;

    auto bufferConstructor = [](CommandBuffer *cbh) {
        // clear are events
        cbh->SetLineWidth(initLineWidth);
        cbh->ClearColor(0, blue, ClearColorMask::RGBA);
    };

    return TestEditingWithoutInsertion(
        // Editor function
        llgdCommandSetEditRasterStateRasterizationLineWidth,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState &st) {
            return st.rasterization.lineWidth.value;
        },
        // Init/Edit values
        initLineWidth, editedState
    );
}

bool EditRasterStateValidator::TestEditSampleMask()
{
    static const uint32_t initSampleMask = 0xF3;
    static const uint32_t editSampleMask = 0xEF73;

    auto bufferConstructor = [](CommandBuffer *cbh) {
        cbh->SetSampleMask(initSampleMask);
        cbh->ClearColor(0, green, ClearColorMask::RGBA); // Event 1
    };

    return TestEditingWithoutInsertion(
        // Editor function
        llgdCommandSetEditRasterStateRasterizationSampleMask,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState &st) {
            return st.rasterization.sampleMask.value;
        },
        // Init/Edit values
        initSampleMask, editSampleMask
    );
}

bool EditRasterStateValidator::TestEditConservativeRasterEnable()
{
    static const int TEST_CASE = 2;
    static const bool initStates[TEST_CASE]   = { false, true };
    static const bool editedStates[TEST_CASE] = { true, false };

    for (int i = 0; i < TEST_CASE; ++i) {
        const auto initState   = initStates[i];
        const auto editedState = editedStates[i];

        TEST_FMT(TestEditingWithoutInsertion(
            // Editor function
            llgdCommandSetEditRasterStateRasterizationConservativeRasterEnable,
            // Init command buffer creator
            [initState](CommandBuffer *cbh) {
                // clear are events
                cbh->SetConservativeRasterEnable(initState);
                cbh->ClearColor(0, blue, ClearColorMask::RGBA);
            },
            // State value extractor from RasterState
            [](const RasterState &st) {
                return st.rasterization.conservativeRasterEnable.value;
            },
            // Init/Edit values
            initState, editedState
        ), "Failed on test case %d", i);
    }

    return true;
}

bool EditRasterStateValidator::TestEditConservativeRasterDilate()
{
    static const float initState = 0.25f;
    static const float editedState = 0.75f;

    auto bufferConstructor = [](CommandBuffer *cbh) {
        cbh->SetConservativeRasterEnable(true);
        cbh->SetSubpixelPrecisionBias(1, 2);
        cbh->SetConservativeRasterDilate(initState);
        cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
    };

    return TestEditingWithoutInsertion(
        // Editor function
        llgdCommandSetEditRasterStateRasterizationConservativeRasterDilate,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState &st) {
            return st.rasterization.conservativeRasterDilate.value;
        },
        // Init/Edit values
        initState, editedState
    );
}

bool EditRasterStateValidator::TestEditSubpixelPrecisionBias()
{
    static const RasterState::Rasterization::SubpixelPrecisionBias initState{ 0, 1 };
    static const RasterState::Rasterization::SubpixelPrecisionBias editedState{ 2, 3 };

    auto bufferConstructor = [](CommandBuffer *cbh) {
        cbh->SetConservativeRasterEnable(true);
        cbh->SetSubpixelPrecisionBias(initState.x, initState.y);
        cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
    };

    for (uint32_t viewportIndex = 0; viewportIndex < RasterState::Rasterization::NUM_VIEWPORTS; viewportIndex++) {
        auto stateGetter = [viewportIndex](const RasterState &st) {
            return st.rasterization.subpixelPrecisionBias.value[viewportIndex];
        };

        // Test editing
        TEST_FMT(TestEditingWithoutInsertion(
                // Edit function (a wrapper function. This state editor requires extra arguments)
                [viewportIndex](LWNcommandHandle handle,
                                IndexMapFn indexMapFn,
                                uint32_t startIndex,
                                uint32_t endIndex,
                                const GpuState& startState,
                                const GpuState& endState,
                                const RasterState::Rasterization::SubpixelPrecisionBias& editValue,
                                WriteMemoryFn writeControlMemoryFn,
                                WriteMemoryFn writeCommandMemoryFn,
                                MethodUpdatedFn methodUpdated,
                                void* callbacksData) {
                    return llgdCommandSetEditRasterStateRasterizationSubpixelPrecisionBias(
                        handle, indexMapFn, startIndex, endIndex, startState, endState,
                        viewportIndex, editValue.x, editValue.y,
                        writeControlMemoryFn, writeCommandMemoryFn, methodUpdated, callbacksData
                    );
                },
                // Init command buffer creator
                bufferConstructor,
                // State value extractor from RasterState
                stateGetter,
                // Init/Edit value
                initState, editedState
            ), "Failed on viewportIndex=%d", viewportIndex
        );
    }

    return true;
}

//---------------------------------
// Test Polygon methods
//---------------------------------
bool EditRasterStateValidator::TestPolygonLwllFace()
{
    static const int testPatternCount = 3;
    static const LWNface initStates[testPatternCount]   = { LWN_FACE_FRONT, LWN_FACE_FRONT_AND_BACK, LWN_FACE_NONE };
    static const LWNface editedStates[testPatternCount] = { LWN_FACE_FRONT_AND_BACK, LWN_FACE_NONE, LWN_FACE_FRONT };
    static const char* lwnFaceNames[] = { "NONE", "FRONT", "BACK", "FRONT_AND_BACK" };

    for (int testIdx = 0; testIdx < testPatternCount; ++testIdx) {

        auto initState = initStates[testIdx];
        auto editedState = editedStates[testIdx];

        auto bufferConstructor = [initState](CommandBuffer *cbh) {
            PolygonState polygonState;
            polygonState.SetDefaults();
            // Bind FACE_FRONT first in order to ensure SET_LWLL_FACE method will be pushed in the buffer
            polygonState.SetLwllFace(lwn::Face::FRONT);
            cbh->BindPolygonState(&polygonState);
            // Then set to initState
            polygonState.SetLwllFace(static_cast<lwn::Face::Enum>(initState));
            cbh->BindPolygonState(&polygonState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
        };

        TEST_FMT(TestEditingWithoutInsertion(
            // Edit function
            llgdCommandSetEditRasterStatePolygonLwllFace,
            // Init command buffer creator
            bufferConstructor,
            // State value extractor from RasterState
            [](const RasterState &st) {
                return st.polygon.lwllFace.value;
            },
            // Init/Edit value
            initState, editedState
        ), "Failed to test PolygonLwllFace editing from %s to %s", lwnFaceNames[initState], lwnFaceNames[editedState]);
    }
    return true;
}

bool EditRasterStateValidator::TestPolygonFrontFace()
{
    static const auto initState = LWN_FRONT_FACE_CCW;
    static const auto editedState = LWN_FRONT_FACE_CW;

    auto bufferConstructor = [](CommandBuffer *cbh) {
        PolygonState polygonState;
        polygonState.SetDefaults()
                    .SetFrontFace(lwn::FrontFace::Enum(initState));
        cbh->BindPolygonState(&polygonState);
        cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
    };

    return TestEditingWithoutInsertion(
        // Edit function
        llgdCommandSetEditRasterStatePolygonFrontFace,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState &st) {
            return st.polygon.frontFace.value;
        },
        // Init/Edit value
        initState, editedState
    );
}

bool EditRasterStateValidator::TestPolygonPolygonMode()
{
    static const auto initState = LWN_POLYGON_MODE_LINE;
    static const auto editedState = LWN_POLYGON_MODE_FILL;

    auto bufferConstructor = [](CommandBuffer *cbh) {
        PolygonState polygonState;
        polygonState.SetDefaults()
                    .SetPolygonMode(PolygonMode::Enum(initState));
        cbh->BindPolygonState(&polygonState);
        cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
    };

    return TestEditingWithoutInsertion(
        // Edit function
        llgdCommandSetEditRasterStatePolygonPolygonMode,
        // Init command buffer creator
        bufferConstructor,
        // State value extractor from RasterState
        [](const RasterState &st) {
            return st.polygon.polygonMode.value;
        },
        // Init/Edit value
        initState, editedState
    );
}

bool EditRasterStateValidator::TestPolygonPolygonOffsetEnables()
{
    static const LWNpolygonMode polygonMode = LWN_POLYGON_MODE_FILL;
    static const LWNpolygonOffsetEnable targetModeBit = LWN_POLYGON_OFFSET_ENABLE_FILL_BIT;
    static const int testPatternCount = 2;
    static const std::pair<bool, bool> testPatterns[testPatternCount] = { std::make_pair(true, false), std::make_pair(false, true) };

    // Loop for test cases
    for (int testIdx = 0; testIdx < testPatternCount; testIdx++) {
        auto initState = testPatterns[testIdx].first;
        auto editedState = testPatterns[testIdx].second;

        auto bufferConstructor = [initState](CommandBuffer *cbh) {
            PolygonState polygonState;
            polygonState.SetDefaults()
                        .SetPolygonMode(PolygonMode::Enum(polygonMode))
                        .SetPolygonOffsetEnables(lwn::PolygonOffsetEnable::Enum(initState ? targetModeBit : 0));
            cbh->BindPolygonState(&polygonState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
        };

        TEST_FMT(TestEditingWithoutInsertion(
            // Edit function (a wrapper function. This state editor requires extra arguments)
            [](LWNcommandHandle handle,
               IndexMapFn indexMapFn,
               uint32_t startIndex,
               uint32_t endIndex,
               const GpuState& startState,
               const GpuState& endState,
               const bool& editValue,
               WriteMemoryFn writeControlMemoryFn,
               WriteMemoryFn writeCommandMemoryFn,
               MethodUpdatedFn methodUpdated,
               void* callbacksData) {
                return llgdCommandSetEditRasterStatePolygonPolygonOffsetEnables(
                    handle, indexMapFn, startIndex, endIndex, startState, endState,
                    polygonMode, editValue,
                    writeControlMemoryFn, writeCommandMemoryFn, methodUpdated, callbacksData
                );
            },
            // Init command buffer creator
            bufferConstructor,
            // State value extractor from RasterState
            [](const RasterState &st) {
                const auto offsetEnables = st.polygon.polygonOffsetEnables.value;
                return static_cast<bool>(!!(offsetEnables & targetModeBit));
            },
            // Init/Edit value
            initState, editedState
        ), "Failed on test case=%d", testIdx);
    }

    return true;
}

bool EditRasterStateValidator::TestEditMultisampleAntialiasEnable()
{
    static const int testPatterns = 2;
    static const bool initStates[testPatterns] = { false, true };
    static const bool editedStates[testPatterns] = { true, false };

    for (int testIdx = 0; testIdx < testPatterns; ++testIdx) {

        auto initState = initStates[testIdx];
        auto editedState = editedStates[testIdx];

        auto bufferConstructor = [initState](CommandBuffer *cbh) {
            MultisampleState msState;
            msState.SetDefaults().SetMultisampleEnable(initState);
            // AA never be enabled if samples <= 1
            msState.SetSamples(2);
            cbh->BindMultisampleState(&msState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
        };

        TEST_FMT(TestEditingWithoutInsertion(
            // Edit function
            llgdCommandSetEditRasterStateMultisampleAntiAliasEnable,
            // Init command buffer creator
            bufferConstructor,
            // State value extractor from RasterState
            [](const RasterState &st) {
                return bool(st.multisample.antiAliasEnable.value);
            },
            // Init/Edit value
            initState, editedState
        ), "Failed to edit MultisampleAntialiasEnable, testIdx=%d, state from %d to %d", testIdx, initState, editedState);
    }

    return true;
}

bool EditRasterStateValidator::TestEditMultisampleAlphaToCoverageEnable()
{
    static const int testPatterns = 2;
    static const bool initStates[testPatterns] = { false, true };
    static const bool editedStates[testPatterns] = { true, false };

    for (int testIdx = 0; testIdx < testPatterns; ++testIdx) {

        auto initState = initStates[testIdx];
        auto editedState = editedStates[testIdx];

        auto bufferConstructor = [initState](CommandBuffer *cbh) {
            MultisampleState msState;
            msState.SetDefaults().SetAlphaToCoverageEnable(initState);
            cbh->BindMultisampleState(&msState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
        };

        TEST_FMT(TestEditingWithoutInsertion(
            // Edit function
            llgdCommandSetEditRasterStateMultisampleAlphaToCoverageEnable,
            // Init command buffer creator
            bufferConstructor,
            // State value extractor from RasterState
            [](const RasterState &st) {
                return bool(st.multisample.alphaToCoverageEnable.value);
            },
            // Init/Edit value
            initState, editedState
        ), "Failed to edit MultisampleAlphaToCoverageEnable, testIdx=%d, state from %d to %d", testIdx, initState, editedState);
    }

    return true;
}

bool EditRasterStateValidator::TestEditMultisampleAlphaToCoverageDitheringEnable()
{
    static const int testPatterns = 2;
    static const bool initStates[testPatterns] = { false, true };
    static const bool editedStates[testPatterns] = { true, false };

    for (int testIdx = 0; testIdx < testPatterns; ++testIdx) {

        const auto initState = initStates[testIdx];
        const auto editedState = editedStates[testIdx];

        auto bufferConstructor = [initState](CommandBuffer *cbh) {
            MultisampleState msState;
            msState.SetDefaults();
            msState.SetAlphaToCoverageDither(initState);
            cbh->BindMultisampleState(&msState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
        };

        TEST_FMT(TestEditingWithoutInsertion(
            // Edit function
            llgdCommandSetEditRasterStateMultisampleAlphaToCoverageDitheringEnable,
            // Init command buffer creator
            bufferConstructor,
            // State value extractor from RasterState
            [](const RasterState &st) {
                return st.multisample.alphaToCoverageDitheringEnable.value;
            },
            // Init/Edit value
            initState, editedState
        ), "Failed to edit MultisampleSetAlphaToCoverageDitherEnable, testIdx=%d, state from %d to %d", testIdx, initState, editedState);
    }

    return true;
}

bool EditRasterStateValidator::TestEditMultisampleaCoverageToColorEnable()
{
    // Test CoverageToColorEnable editing, and also confirm CoverageToColorOutput hasn't been changed after CoverageToColorEnable was changed.
    // Because these values are stored in one 32bit value.

    static const int testPatterns = 2;
    static const bool initStates[testPatterns] = { false, true };
    static const bool editedStates[testPatterns] = { true, false };
    static const uint32_t coverageToColorOutput = 0x5;

    for (int testIdx = 0; testIdx < testPatterns; ++testIdx) {

        auto initState = initStates[testIdx];
        auto editedState = editedStates[testIdx];

        auto bufferConstructor = [initState](CommandBuffer *cbh) {
            MultisampleState msState;
            msState.SetDefaults().SetCoverageToColorEnable(initState);
            msState.SetCoverageToColorOutput(coverageToColorOutput);  // Also do this state test to confirm this won't be edited
            cbh->BindMultisampleState(&msState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
        };

        // Test ColorEnable
        TEST_FMT(TestEditingWithoutInsertion(
            // Edit function
            llgdCommandSetEditRasterStateMultisampleCoverageToColorEnable,
            // Init command buffer creator
            bufferConstructor,
            // State value extractor from RasterState
            [](const RasterState &st) {
                return bool(st.multisample.coverageToColorEnable.value);
            },
            // Init/Edit values
            initState, editedState
        ), "Failed to edit MultisampleAlphaToCoverageEnable, testIdx=%d, state from %d to %d", testIdx, initState, editedState);

        // Test ColorOutput
        auto rasterState = m_spCommandHelper->ExtractRasterState();
        TEST_EQ_FMT(rasterState.multisample.coverageToColorOutput.value, coverageToColorOutput,
            "CoverageToColorOutput must not be changed when editing CoverageToColorEnable. TestIdx=%d", testIdx);
    }

    return true;

}

bool EditRasterStateValidator::TestEditMultisampleCoverageToColorOutput()
{
    // Test CoverageToColorOutput editing, and also confirm CoverageToColorEnable hasn't been changed even if CoverageToColorOutput was changed
   
    static const int testPatterns = 3;
    static const uint32_t initStates[testPatterns]   = { 0x05, 0x03, 0x00 };
    static const uint32_t editedStates[testPatterns] = { 0x01, 0x00, 0x07 };
    static const bool coverageToColorEnable = true;

    for (int testIdx = 0; testIdx < testPatterns; ++testIdx) {

        auto initState = initStates[testIdx];
        auto editedState = editedStates[testIdx];

        auto bufferConstructor = [initState](CommandBuffer *cbh) {
            MultisampleState msState;
            msState.SetDefaults().SetCoverageToColorEnable(coverageToColorEnable); // Also test to confirm this value won't be edited
            msState.SetCoverageToColorOutput(initState);
            cbh->BindMultisampleState(&msState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
        };

        // Test ColorOutput
        TEST_FMT(TestEditingWithoutInsertion(
            // Edit function
            llgdCommandSetEditRasterStateMultisampleCoverageToColorOutput,
            // Init command buffer creator
            bufferConstructor,
            // State value extractor from RasterState
            [](const RasterState &st) {
                return uint32_t(st.multisample.coverageToColorOutput.value);
            },
            // Init/Edit values
            initState, editedState
        ), "Failed to edit MultisampleCoverageToColorOutput, testIdx=%d, state from %d to %d", testIdx, initState, editedState);

        // Test ColorEnable
        auto rasterState = m_spCommandHelper->ExtractRasterState();
        TEST_EQ_FMT(rasterState.multisample.coverageToColorEnable.value, coverageToColorEnable,
            "CoverageToColorEnable must not be changed when editing CoverageToColorOutput. TestIdx=%d", testIdx);
    }

    return true;
}

bool EditRasterStateValidator::TestEditMultisampleCoverageModulationMode()
{
    static const int testPatterns = 2;
    static const bool stateAllowTIR[testPatterns]                     = { true, false };
    static const LWNcoverageModulationMode initModes[testPatterns]   = { LWN_COVERAGE_MODULATION_MODE_RGB,  LWN_COVERAGE_MODULATION_MODE_RGBA };
    static const LWNcoverageModulationMode editedModes[testPatterns] = { LWN_COVERAGE_MODULATION_MODE_RGBA, LWN_COVERAGE_MODULATION_MODE_ALPHA };

    const auto modeGetter = [](const RasterState &st) {
        return st.multisample.coverageModulationMode.value;
    };

    for (int testIdx = 0; testIdx < testPatterns; ++testIdx) {
        const auto initMode = initModes[testIdx];
        const auto editedMode = editedModes[testIdx];
        const auto allowTIR = stateAllowTIR[testIdx];

        const auto base = m_spCommandHelper->MakeHandle([=](CommandBuffer *cbh) {
            MultisampleState msState;
            msState.SetDefaults().SetCoverageModulationMode(CoverageModulationMode::Enum(initMode));
            if (allowTIR) {
                msState.SetSamples(1).SetRasterSamples(2); // Make TIR enable
            }
            cbh->BindMultisampleState(&msState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event method 1
        });

        const auto gpuState = m_spCommandHelper->RunAndExtractGpuState(base);

        // Validate initState
        {
            const auto actual = modeGetter(m_spCommandHelper->ExtractRasterState(&gpuState));
            const auto expected = allowTIR ? initMode : LWN_COVERAGE_MODULATION_MODE_NONE;
            TEST_EQ_FMT(actual, expected, "Init state value validation failed: testCase=%d", testIdx);
        }

        // Edit & execute
        m_spCommandHelper->ResetPointersForEditingCB();
        const auto edited = llgdCommandSetEditRasterStateMultisampleCoverageModulationMode(
            base,
            indexMapFn,
            0, 1000, // Edit entire command handle
            gpuState, gpuState,
            editedMode,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            llgd_lwn::EmptyMethodUpdatedFn,
            m_spCommandHelper.get()
        );
        const auto decoded = m_spCommandHelper->MakeCommandHandleRunnable(edited);
        m_spCommandHelper->Run(decoded);

        // Validate the state is edited
        {
            const auto actual = modeGetter(m_spCommandHelper->ExtractRasterState());
            const auto expected = allowTIR ? editedMode : LWN_COVERAGE_MODULATION_MODE_NONE;
            TEST_EQ_FMT(actual, expected, "Failed to edit MultisampleCoverageModulationMode, testIdx=%d, state from %d to %d", testIdx, initMode, editedMode);
        }
    }

    return true;
}

bool EditRasterStateValidator::TestEditMultisampleRasterSamples()
{
    // RasterSamples test. (0, 1,) 2, 4, 8, 16 are allowed to use as rasterSamples.
    // Note: strictly speaking 0 or 1 is not valid. They will be treated as 2.
    static const int testPatterns = 2;
    static const uint32_t initStates[testPatterns] = { 4, 16 };
    static const uint32_t editedStates[testPatterns] = { 8, 2 };

    for (int testIdx = 0; testIdx < testPatterns; ++testIdx) {
        auto initState = initStates[testIdx];
        auto editedState = editedStates[testIdx];

        auto bufferConstructor = [initState, editedState](CommandBuffer *cbh) {
            MultisampleState msState;
            msState.SetDefaults().SetRasterSamples(editedState);
            msState.SetRasterSamples(initState);
            cbh->BindMultisampleState(&msState);
            cbh->ClearColor(0, blue, ClearColorMask::RGBA); // Event 1
        };

        TEST_FMT(TestEditingWithoutInsertion(
            // Edit function
            llgdCommandSetEditRasterStateMultisampleRasterSamples,
            // Init command buffer creator
            bufferConstructor,
            // State value extractor from RasterState
            [](const RasterState &st) {
                return st.multisample.rasterSamples.value;
            },
            // Init/Edit value
            initState, editedState
        ), "Failed to edit MultisampleRasterSamples, testIdx=%d, state from %d to %d", testIdx, initState, editedState);
    }

    return true;
}

//---------------------------------
// Entry point
//---------------------------------
bool EditRasterStateValidator::Test()
{
    // Test rasterization states
    TEST(TestEditDiscard());
    TEST(TestEditPolygonOffset());
    TEST(TestEditPointSize());
    TEST(TestEditLineWidth());
    TEST(TestEditSampleMask());
    TEST(TestEditConservativeRasterEnable());
    TEST(TestEditConservativeRasterDilate());
    TEST(TestEditSubpixelPrecisionBias());

    // Test polygon states
    TEST(TestPolygonLwllFace());
    TEST(TestPolygonFrontFace());
    TEST(TestPolygonPolygonMode());
    TEST(TestPolygonPolygonOffsetEnables());

    // Test multisample states
    TEST(TestEditMultisampleAntialiasEnable());
    TEST(TestEditMultisampleAlphaToCoverageEnable());
    TEST(TestEditMultisampleAlphaToCoverageDitheringEnable());
    TEST(TestEditMultisampleaCoverageToColorEnable());
    TEST(TestEditMultisampleCoverageToColorOutput());
    TEST(TestEditMultisampleCoverageModulationMode());
    TEST(TestEditMultisampleRasterSamples());

    return true;
}

LLGD_DEFINE_TEST(EditRasterState, UNIT,
LwError Execute()
{
    auto v = std::make_unique<EditRasterStateValidator>();
    if (!v->Initialize()) {
        return LwError_IlwalidState;
    }
    return v->Test() ? LwSuccess : LwError_IlwalidState;
}
);
