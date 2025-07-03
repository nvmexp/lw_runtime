/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include "trirate.hpp"
#include "glprogram.hpp"

class BenchmarkTrirateOGL : public BenchmarkCaseOGL
{
private:
    struct SegAttrs
    {
        LwnUtil::Vec4f offset;
        LwnUtil::Vec4f color;
    };

    LwnUtil::OGLMesh* m_mesh;
    GlProgram*        m_program;
    uint64_t m_numTrisRendered;

public:
    BenchmarkTrirateOGL(int w, int h);
    ~BenchmarkTrirateOGL();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
