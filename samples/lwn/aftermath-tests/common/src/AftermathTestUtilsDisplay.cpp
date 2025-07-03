/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <AftermathTestUtilsDisplay.h>

#include <nn/vi.h>
#include <lwassert.h>

namespace AftermathTest {
namespace LWN {

void DisplayUtil::Initialize()
{
    nn::vi::Initialize();

    nn::Result viResult = nn::vi::OpenDefaultDisplay(&_display);
    LW_ASSERT(viResult.IsSuccess());

    viResult = nn::vi::CreateLayer(&_displayLayer, _display);
    LW_ASSERT(viResult.IsSuccess());

    viResult = nn::vi::GetNativeWindow(&_nativeWindow, _displayLayer);
    LW_ASSERT(viResult.IsSuccess());
}

void DisplayUtil::Finalize()
{
    nn::vi::DestroyLayer(_displayLayer);
    nn::vi::CloseDisplay(_display);
    nn::vi::Finalize();
}

} // namespace LWN
} // namespace AftermathTest
