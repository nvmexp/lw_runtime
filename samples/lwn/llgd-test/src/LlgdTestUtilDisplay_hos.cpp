/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTestUtil.h>
#include <LlgdTestUtilDisplay.h>

#include <nn/vi.h>

namespace llgd_lwn
{
void DisplayUtil::Initialize()
{
    nn::vi::Initialize();

    nn::Result viResult = nn::vi::OpenDefaultDisplay(&_display);
    CHECK(viResult.IsSuccess());

    viResult = nn::vi::CreateLayer(&_displayLayer, _display);
    CHECK(viResult.IsSuccess());

    viResult = nn::vi::GetNativeWindow(&_nativeWindow, _displayLayer);
    CHECK(viResult.IsSuccess());
}

void DisplayUtil::Finalize()
{
    nn::vi::DestroyLayer(_displayLayer);
    nn::vi::CloseDisplay(_display);
    nn::vi::Finalize();
}
} // llgd_lwn
