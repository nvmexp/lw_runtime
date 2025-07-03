/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include "bench.hpp"

class BenchmarkDrawTestLWN : public BenchmarkCaseLWN
{
public :
    enum DrawFunc {
	    DrawElements = 0,
	    DrawArrays = 1,
	    DrawElementsBaseVertex = 2,
	    DrawElementsInstanced = 3,
	    DrawArraysInstanced = 4,
	    DrawElementsIndirect = 5,
	    DrawArraysIndirect = 6
	};

    struct TestDescr
    {
        int  numSegments;
        DrawFunc drawFunc;
    };

    static const TestDescr subtests[];
    static const char * DrawFuncToString(BenchmarkDrawTestLWN::DrawFunc drawFunc);
private:
    struct SegAttrs
    {
        LwnUtil::Vec4f offset;
        LwnUtil::Vec4f color;
    };

    LWNprogram*                m_pgm;
    LwnUtil::VertexState*      m_vertex;
    LwnUtil::CmdBuf*           m_cmdBuf;
    LWNcommandHandle           m_prebuiltCmdHandle;
    LwnUtil::Mesh*             m_mesh;
    LwnUtil::UboArr<SegAttrs>* m_cb;
    LwnUtil::IndirectArr<LWNdrawElementsIndirectData>* m_dei;
    LwnUtil::IndirectArr<LWNdrawArraysIndirectData>* m_dai;

    uint64_t m_numInstancesRendered;

    LWNcommandHandle renderCommands(LWNcommandBuffer* cmdBuf,
                                    BenchmarkDrawTestLWN::DrawFunc drawFunc);

public:
    BenchmarkDrawTestLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkDrawTestLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
