/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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

#include <array>
#include <cassert>
#include <functional>
#include <string>
#include <vector>

#include <class/cl9097.h>
#include <include/g_gf100lwnmmemethods.h>

namespace {
static const float GREEN[4]{ 0, 1, 0, 1 };

class Validator {
public:
    bool Initialize();
    bool Test();

private:
    bool TestDrawRemove(DrawEventType drawType);
    void TrackMethodInfos(LWNcommandHandle handle, std::vector<LlgdCommandSetMethodTrackerMethodInfo>& methodInfos, int indexOffset = 10);

private:
    llgd_lwn::QueueHolder qh;
    llgd_lwn::SyncHolder sync; // Used for FenceSync (create event token)

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> m_spCommandHelper;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    m_spCommandHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(m_spCommandHelper->Initialize());

    return true;
}


namespace {
int ExpectReducedGpSegments(DrawEventType drawType)
{
    switch (drawType) {
    case DrawEventType::Elements: return 0;
    case DrawEventType::ElementsIndirect: return 1;
    case DrawEventType::ElementsInstanced: return 0;
    case DrawEventType::Arrays: return 0;
    case DrawEventType::ArraysIndirect: return 1;
    case DrawEventType::ArraysInstanced: return 0;
    case DrawEventType::TransformFeedback: return 1;
    case DrawEventType::MultiElementsIndirect: return 2;
    case DrawEventType::MultiArraysIndirect: return 2;
    case DrawEventType::Texture: return 0;
    default: return 99;
    }
}

bool InsertDraw(DrawEventType drawType, CommandBuffer* cb)
{
    const LWNbufferAddress randomIndexBufferAddress = 0x4f8130fac000;
    const LWNbufferAddress randomIndirectAddress = 0xe38130dc1000;
    const LWNbufferAddress randomParameterAddress = 0xe84730dc1000;
    switch (drawType)
    {
    case DrawEventType::Elements:
        cb->DrawElements(lwn::DrawPrimitive::POINTS, lwn::IndexType::UNSIGNED_SHORT, 32, randomIndexBufferAddress);
        break;
    case DrawEventType::ElementsIndirect:
        cb->DrawElementsIndirect(lwn::DrawPrimitive::POINTS, lwn::IndexType::UNSIGNED_SHORT, randomIndexBufferAddress, randomIndirectAddress);
        break;
    case DrawEventType::ElementsInstanced: {
        const int baseVertex = (rand() % 2 == 0) ? 0 : (rand() % 10 + 1);
        cb->DrawElementsInstanced(lwn::DrawPrimitive::LINES, lwn::IndexType::UNSIGNED_SHORT, 32, randomIndexBufferAddress, baseVertex, 0 /*baseInstance*/, 12 /*instanceCount*/);
        return baseVertex == 0;
    }
    case DrawEventType::Arrays:
        cb->DrawArrays(lwn::DrawPrimitive::LINES, 4 /*first*/, 16 /*count*/);
        break;
    case DrawEventType::ArraysIndirect:
        cb->DrawArraysIndirect(lwn::DrawPrimitive::LINES, randomIndirectAddress);
        break;
    case DrawEventType::ArraysInstanced: {
        const int baseInstance = (rand() % 2 == 0) ? 0 : (rand() % 10 + 1);
        cb->DrawArraysInstanced(lwn::DrawPrimitive::LINES, 4 /*first*/, 16 /*count*/, baseInstance /*baseInstance*/, 12 /*instanceCount*/);
        return baseInstance == 0;
    }
    case DrawEventType::TransformFeedback:
        cb->DrawTransformFeedback(lwn::DrawPrimitive::POLYGON, randomIndirectAddress);
        break;
    case DrawEventType::MultiElementsIndirect:
        cb->MultiDrawElementsIndirectCount(lwn::DrawPrimitive::POLYGON, lwn::IndexType::UNSIGNED_SHORT, randomIndexBufferAddress, randomIndirectAddress, randomParameterAddress, 4, 2);
        break;
    case DrawEventType::MultiArraysIndirect:
        cb->MultiDrawArraysIndirectCount(lwn::DrawPrimitive::POLYGON, randomIndirectAddress, randomParameterAddress, 4, 1);
        break;
    case DrawEventType::Texture: {
        uint64_t randomTextureId = (312u << 20) | 0x326f;
        lwn::DrawTextureRegion srcRegion, dstRegion;
        srcRegion.x0 = 100;
        srcRegion.x1 = 200;
        srcRegion.y0 = 300;
        srcRegion.y1 = 400;
        dstRegion.x0 = 500;
        dstRegion.x1 = 600;
        dstRegion.y0 = 700;
        dstRegion.y1 = 800;
        cb->DrawTexture(LWNtextureHandle(randomTextureId), &dstRegion, &srcRegion);
        break;
    }
    default:
        assert(!"impossible");
        break;
    }
    return true;
}

uint32_t GetMercaptanMethod(DrawEventType drawType, bool alternate)
{
   switch (drawType) {
   case DrawEventType::Elements: return LW9097_LWN_MME_DRAW_ELEMENTS;
   case DrawEventType::ElementsIndirect: return LW9097_LWN_MME_DRAW_ELEMENTS_INDIRECT;
   case DrawEventType::ElementsInstanced:
        return alternate ? LW9097_LWN_MME_DRAW_ELEMENTS_INSTANCED_NO_VPC_UPDATES : LW9097_LWN_MME_DRAW_ELEMENTS_INSTANCED;
   case DrawEventType::Arrays: return LW9097_LWN_MME_DRAW_ARRAYS;
   case DrawEventType::ArraysIndirect: return LW9097_LWN_MME_DRAW_ARRAYS_INDIRECT;
   case DrawEventType::ArraysInstanced:
       return alternate ? LW9097_LWN_MME_DRAW_ARRAYS_INSTANCED_NO_VPC_UPDATES : LW9097_LWN_MME_DRAW_ARRAYS_INSTANCED;
   case DrawEventType::TransformFeedback:
       return LW9097_LWN_MME_DRAW_TRANSFORM_FEEDBACK;
   default:
       return 0u;
   }
}

uint32_t NeedFullCompare(DrawEventType drawType)
{
   switch (drawType) {
   case DrawEventType::MultiElementsIndirect:
   case DrawEventType::MultiArraysIndirect:
   case DrawEventType::Texture:
       return true;
   default:
       return false;
   }
}

}

bool Validator::TestDrawRemove(DrawEventType drawType)
{
    using CreateCbFn = std::function<void(CommandBuffer*)>;
    const std::vector<CreateCbFn> CreateEvent1Fns{
        [&](CommandBuffer* cb) { cb->Barrier(~0); /* has gp segment close */ },
        [&](CommandBuffer* cb) { cb->ClearColor(0, GREEN, ClearColorMask::RGBA); },
    };
    const std::vector<CreateCbFn> CreateEvent2Fns{
        [&](CommandBuffer* cb) { cb->FenceSync(sync, lwn::SyncCondition::ALL_GPU_COMMANDS_COMPLETE, 0); /* is a token */ },
        [&](CommandBuffer* cb) { cb->ClearColor(0, GREEN, ClearColorMask::RGBA); },
    };

    const bool fullCompare = NeedFullCompare(drawType);

    for (int command = 0; command < 2; command++) {
        // CommandBuffer without the draw method
        m_spCommandHelper->ResetPointersForEditingCB();
        const auto reference = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
            CreateEvent1Fns[command](cb);
            CreateEvent2Fns[command](cb);
            CreateEvent2Fns[1 - command](cb);
        });
        const auto referenceMethods = llgd_lwn::GetMethods(reference);

        // CommandBuffer with the draw method
        bool alternateMmeDraw = false;
        const auto handle = m_spCommandHelper->MakeHandle([&](CommandBuffer* cb) {
            CreateEvent1Fns[command](cb);
            CreateEvent2Fns[command](cb);
            alternateMmeDraw = InsertDraw(drawType, cb);
            CreateEvent2Fns[1 - command](cb);
        });
        const auto gpsegmentCountWDraw = llgdLwnGetGpfifoSegmentCount(handle);

        // Execute removing!
        m_spCommandHelper->ResetPointersForEditingCB();
        const auto patched = llgdCommandSetRemoveDraw(
            handle,
            [](uint32_t index, void*) { return 10 + index; }, 13,
            drawType,
            m_spCommandHelper->WriteControlMemoryForEditing,
            m_spCommandHelper->WriteCommandMemoryForEditing,
            m_spCommandHelper.get());
        const auto decoded = m_spCommandHelper->MakeCommandHandleRunnable(patched);
        const auto gpsegmentCountAfterRemoved = llgdLwnGetGpfifoSegmentCount(decoded);
        const auto patchedMethods = llgd_lwn::GetMethods(decoded);

        // Preparation finished, first do testing on methods
        if (fullCompare) {
            TEST(llgd_lwn::CompareMethods(patchedMethods, referenceMethods));
        } else {
            const auto targetMethod = GetMercaptanMethod(drawType, alternateMmeDraw);
            TEST(!llgd_lwn::FindMethod(patchedMethods, targetMethod));
        }

        // Testing gpsegment counts
        TEST_EQ(gpsegmentCountAfterRemoved + ExpectReducedGpSegments(drawType), gpsegmentCountWDraw);
    }

    return true;
}

bool Validator::Test()
{
    for (int i = 0; i <= int(DrawEventType::Texture); ++i) {
        TEST(TestDrawRemove(DrawEventType(i)));
    }

    return true;
}
}

LLGD_DEFINE_TEST(EditDrawRemove, UNIT, LwError Execute() {
    Validator v{};
    return (v.Initialize() && v.Test()) ? LwSuccess : LwError_IlwalidState;
});
