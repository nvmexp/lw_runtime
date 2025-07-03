#if _WIN32
#include "stdafx.h"

 /*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "SettingsDialog.h"

using namespace ms;

void RunSettings()
{
    SettingsDialog ^form = gcnew SettingsDialog();
    Application::Run(form);
}
#endif
