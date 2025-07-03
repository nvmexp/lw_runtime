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

class BenchmarkFillrateLWN : public BenchmarkCaseLWN
{
private:
    struct SegAttrs
    {
        LwnUtil::Vec4f offset;
        LwnUtil::Vec4f color;
    };

    LWNprogram*            m_pgm;
    LwnUtil::VertexState*  m_vertex;
    LWNcommandBuffer*      m_cmd;
    LwnUtil::CmdBuf*       m_cmdBuf;
    LwnUtil::Mesh*         m_mesh;

    LwnUtil::RenderTarget* m_rtNormal;
    LwnUtil::RenderTarget* m_rtSubregions;

    uint64_t m_numPixRendered;
public:
    enum DepthTest
    {
        DEPTHTEST_OFF,
        DEPTHTEST_PASS,
        DEPTHTEST_FAIL,
    };

    enum SubregionMode
    {
        // No ADAPTIVE_ZLWLL
        NO_SUBREGIONS,
        // Use ADAPTIVE_ZLWLL
        SUBREGIONS,
        // Use ADAPTIVE_ZLWLL and Save/RestoreZLwllData between RT changes
        SUBREGIONS_SAVE_RESTORE
    };

    enum RtChange
    {
        // No middle-of-drawing SetRenderTargets calls.  ZLwll should
        // always work.
        NO_RT_CHANGES,
        // Call SetRenderTargets with always the same color & depth in
        // the middle of drawing.  This should not ilwalidate zlwll.
        REDUNDANT_RT_CHANGE,
        // Set a new color buffer with SetRenderTargest and a null
        // depth, then switch back to original.  This should not
        // ilwalidate zlwll.
        COLOR_ONLY_RT_CHANGE
    };

    struct TestDescr
    {
        bool colorWrite;
        bool depthWrite;
        DepthTest depthTest;
        bool tiledCache;
        RtChange rtMode;
        SubregionMode subregions;
        bool stencil;
        bool transitionToFromSubregions;
    };

    static int getNumSubTests();
    static const TestDescr subtests[];

    enum { NUM_OVERDRAWS = 16 };

    BenchmarkFillrateLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkFillrateLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);

private:
    const TestDescr* m_testDescr;

    void clearBuffers(LWNcommandBuffer *cmd);

};
