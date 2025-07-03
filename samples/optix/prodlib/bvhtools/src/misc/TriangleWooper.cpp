// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "TriangleWooper.hpp"
#include "TriangleWooperKernels.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <math.h>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void TriangleWooper::configure(const Config& cfg)
{
    RT_ASSERT(cfg.inNumRemaps.getNumElems() == 1);
    RT_ASSERT(cfg.inModel.isValid() && !cfg.inModel.isAABBs());

    m_cfg = cfg;
    m_cfg.outWoop.setNumElems(m_cfg.ioRemap.getNumElems());
}

//------------------------------------------------------------------------

void TriangleWooper::execute(void)
{
    if (m_cfg.lwca)
    {
        m_cfg.lwca->beginTimer(getName());
        execDevice();
        m_cfg.lwca->endTimer();
    }
    else
    {
        execHost();
    }
}

//------------------------------------------------------------------------

void TriangleWooper::execDevice(void)
{
    // Launch WooperExec.
    {
        WooperExecParams p      = {};
        p.outWoop               = m_cfg.outWoop.writeDiscardLWDA();
        p.ioRemap               = m_cfg.ioRemap.readWriteLWDA();
        p.inRemapSize           = m_cfg.inNumRemaps.readLWDA();
        p.inModel               = ModelPointers(m_cfg.inModel, MemorySpace_LWDA);
        p.maxRemapSize          = (int)m_cfg.ioRemap.getNumElems();
        p.uncomplementRemaps    = m_cfg.uncomplementRemaps;

        LAUNCH(*m_cfg.lwca, WooperExec, WOOPER_EXEC_WARPS_PER_BLOCK,
            p.maxRemapSize, p);
    }
}

//------------------------------------------------------------------------

void TriangleWooper::execHost(void)
{
    m_cfg.outWoop           .writeDiscardHost();
    m_cfg.ioRemap           .readWriteHost();
    m_cfg.inNumRemaps       .readHost();
    ModelPointers inModel   (m_cfg.inModel, MemorySpace_Host);

    // Fetch the size of the remapping table.

    int numRemaps = (m_cfg.inNumRemaps.getNumElems()) ? *m_cfg.inNumRemaps : (int)m_cfg.ioRemap.getNumElems();
    RT_ASSERT(numRemaps >= 0 && numRemaps <= (int)m_cfg.ioRemap.getNumElems());

    // Process each output triangle.

    for (int outputIdx = 0; outputIdx < numRemaps; outputIdx++)
    {
        // Un-complement remap entry.

        int inputIdx = m_cfg.ioRemap[outputIdx];
        int isLastInLeaf = inputIdx >> 31;
        inputIdx ^= isLastInLeaf;

        if (m_cfg.uncomplementRemaps)
            m_cfg.ioRemap[outputIdx] = inputIdx;

        // Fetch vertices of the input triangle.

        float3 v0, v1, v2;
        inModel.loadVertexPositions(v0, v1, v2, inputIdx);

        // Callwlate edge direction vectors and triangle normal.
        // d1 = v1 - v0
        // d2 = v2 - v0
        // t = cross(d1, d2)

        float3 d1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 d2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        float3 t = make_float3(d1.y * d2.z - d1.z * d2.y, d1.z * d2.x - d1.x * d2.z, d1.x * d2.y - d1.y * d2.x);

        // Form T plane equation.
        // |t| ~=~ 1
        // dot(t, v0) - tw = 0

        float3 a = make_float3(fabsf(t.x), fabsf(t.y), fabsf(t.z));
        float am = fmaxf(fmaxf(a.x, a.y), a.z);
        float tc = 1.0f / am;

        WoopTriangle out;
        out.t.x = t.x * tc, out.t.y = t.y * tc, out.t.z = t.z * tc;
        out.t.w = out.t.x * v0.x + out.t.y * v0.y + out.t.z * v0.z;

        // Callwlate edge normals.
        // r = round(t) // nearest axis vector to avoid loss of precision
        // u = cross(d2, r)
        // v = cross(r, d1)

        float3 r = make_float3(
            (a.x < am) ? 0.0f : (t.x >= 0.0f) ? 1.0f : -1.0f,
            (a.y < am) ? 0.0f : (t.y >= 0.0f) ? 1.0f : -1.0f,
            (a.z < am) ? 0.0f : (t.z >= 0.0f) ? 1.0f : -1.0f);

        float3 u = make_float3(d2.y * r.z - d2.z * r.y, d2.z * r.x - d2.x * r.z, d2.x * r.y - d2.y * r.x);
        float3 v = make_float3(r.y * d1.z - r.z * d1.y, r.z * d1.x - r.x * d1.z, r.x * d1.y - r.y * d1.x);

        // Expand the triangle slightly to avoid leaks due to limited precision.

        float epsA = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fabsf(v0.z));
        float epsB = epsA * epsA * 1.0e-9f;
        float epsC = epsB * 0.4142135f; // sqrt(2) - 1

        // Form U and V plane equations.
        // dot(u, d1) = 1
        // dot(v, d2) = 1
        // dot(u, v0) + uw = 0
        // dot(v, v0) + vw = 0

        float uc = 1.0f / (u.x * d1.x + u.y * d1.y + u.z * d1.z + epsB);
        out.u.x = u.x * uc, out.u.y = u.y * uc, out.u.z = u.z * uc;
        out.v.x = v.x * uc, out.v.y = v.y * uc, out.v.z = v.z * uc;
        out.u.w = epsC * uc - out.u.x * v0.x - out.u.y * v0.y - out.u.z * v0.z;
        out.v.w = epsC * uc - out.v.x * v0.x - out.v.y * v0.y - out.v.z * v0.z;

        // Degenerate triangle => ensure that it cannot be hit.

        if ((v0.x == v1.x && v0.y == v1.y && v0.z == v1.z) ||
            (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z) ||
            (v2.x == v0.x && v2.y == v0.y && v2.z == v0.z))
        {
            out.t = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
        }

        // Set the LSB of woopT.w to indicate the last triangle in a leaf.

        out.t.w = clearLSB(out.t.w);
        if (isLastInLeaf)
            out.t.w = setLSB(out.t.w); 

        // Output.

        m_cfg.outWoop[outputIdx] = out;
    }

    // Empty remapping table => output a dummy triangle that cannot be hit, potentially referenced by the root node.

    if (!numRemaps)
    {
        m_cfg.outWoop[0].u = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        m_cfg.outWoop[0].v = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
        m_cfg.outWoop[0].t = make_float4( 0.0f, 0.0f, 0.0f, setLSB(1.0f) );
    }
}

//------------------------------------------------------------------------
