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

class BenchmarkKickoffLWN : public BenchmarkCaseLWN
{
public:
    enum CmdBufMode
    {
        CMDBUF_SINGLE,
        CMDBUF_MULTI
    };

    struct TestDescr
    {
        CmdBufMode cmdBufMode;
        bool       useCachedCmdBufMemory;
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
    LwnUtil::Mesh*             m_mesh;
    LwnUtil::UboArr<SegAttrs>* m_cb;

    int                        m_numCommandBuffers;
    LwnUtil::CompiledCmdBuf**  m_commandBuffers;
    LWNcommandHandle*          m_commandHandles;

    int                        m_subtestIdx;
    uint64_t                   m_numCmdbufsSubmitted;

    void                       renderCommands(LWNcommandBuffer* cmdBuf, int drawIdx);

public:
    BenchmarkKickoffLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkKickoffLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};

static_assert(sizeof(BenchmarkKickoffLWN::SegAttrs) == 256, "SegAttrs sizeof must match HW CB alignment");
