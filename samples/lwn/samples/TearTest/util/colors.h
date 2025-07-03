#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

enum PMColor {
#define COLOR(name, r, g, b, a) name,
#include "colors.inc"
#undef  COLOR
    num_colors,
};

static const int num_pm_colors = black;

static const float colors[num_colors][4] = {
#define COLOR(name, r, g, b, a) {r, g, b, a},
#include "colors.inc"
#undef  COLOR
};
