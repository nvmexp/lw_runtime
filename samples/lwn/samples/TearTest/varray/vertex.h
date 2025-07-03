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

struct Vertex3 {
    float x;
    float y;
    float z;
};

struct Color4 {
    float r;
    float g;
    float b;
    float a;
};

struct Tex2 {
    float u;
    float v;
};

struct VertexVCT {
    Vertex3 vertex;
    Color4  color;
    Tex2    tex;
};
