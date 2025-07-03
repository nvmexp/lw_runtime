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

#include "bench.hpp"
#include "glprogram.hpp"

class BenchmarkCpuOverheadLWN : public BenchmarkCaseLWN
{
public:
    enum UniformMode
    {
        UNIFORM_INLINE,
        UNIFORM_UBO
    };

    struct TestDescr
    {
        // Geometric complexity of what we render
        int numSegments;

        // Should we precompile rendering commands into a command buffer
        // and reuse that each frame or re-create the command buffer each
        // frame?
        bool precompiled;

        UniformMode uniformMode;
    };

    struct SegAttrs
    {
        LwnUtil::Vec4f offset;
        LwnUtil::Vec4f color;
        uint8_t        padding[256-32]; // sizeof must by 256 byte aligned
    };

private:
    LWNprogram*                m_pgm;
    LwnUtil::VertexState*      m_vertex;
    LwnUtil::CmdBuf*           m_cmdBuf;
    LWNcommandHandle           m_prebuiltCmdHandle;
    LwnUtil::Mesh*             m_mesh;
    LwnUtil::UboArr<SegAttrs>* m_cb;

    int                        m_subtestIdx;
    uint64_t m_numInstancesRendered;


    LWNcommandHandle renderCommands(LWNcommandBuffer* cmdBuf);

public:
    BenchmarkCpuOverheadLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkCpuOverheadLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};

static_assert(sizeof(BenchmarkCpuOverheadLWN::SegAttrs) == 256, "SegAttrs sizeof must match HW CB alignment");

class BenchmarkCpuOverheadOGL : public BenchmarkCaseOGL
{
private:
    LwnUtil::OGLMesh* m_mesh;
    GlProgram*        m_program;
    GLuint            m_ubo;

    uint64_t m_numInstancesRendered;

    const BenchmarkCpuOverheadLWN::TestDescr* m_testDescr;

    BenchmarkCpuOverheadLWN::SegAttrs* m_objectAttrs;
public:
    BenchmarkCpuOverheadOGL(int w, int h);
    ~BenchmarkCpuOverheadOGL();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
