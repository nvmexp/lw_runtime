#pragma once

/*
** Copyright (c) 2020 LWPU CORPORATION.  All rights reserved.
**
** LWPU CORPORATION and its licensors retain all intellectual property
** and proprietary rights in and to this software, related documentation
** and any modifications thereto.  Any use, reproduction, disclosure or
** distribution of this software and related documentation without an express
** license agreement from LWPU CORPORATION is strictly prohibited.
*/

#include "wayland_input.h"

void InitGraphics(void *window, int width, int height);
void RenderFrame(EventInfo *eventInfo);
void TerminateGraphics();
