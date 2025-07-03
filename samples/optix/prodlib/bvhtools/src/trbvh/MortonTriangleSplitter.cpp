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

#include "MortonTriangleSplitter.hpp"
#include <prodlib/bvhtools/src/common/Utils.hpp>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/Assert.h>
#include <math.h>
#include <vector>

using namespace prodlib::bvhtools;
using namespace prodlib;

//------------------------------------------------------------------------

void MortonTriangleSplitter::configure(const Config& cfg)
{
    // Check for errors.

    RT_ASSERT(cfg.maxInputTris >= 0);
    RT_ASSERT(cfg.maxOutputPrims >= cfg.maxInputTris);
    RT_ASSERT(cfg.inTriOrder.getNumElems() >= static_cast<size_t>( cfg.maxInputTris ) ||
              cfg.inTriOrder.getNumElems() == 0);
    RT_ASSERT(cfg.inModel.isValid() && !cfg.inModel.isAABBs());
    RT_ASSERT(cfg.inApexPointMap.getNumBytes() >= sizeof(ApexPointMap));

    if (!(cfg.splitBeta >= 0.0f))
        throw IlwalidValue( RT_EXCEPTION_INFO, "splitBeta must be non-negative!", cfg.splitBeta );

    if (cfg.splitTuningRounds < 1)
        throw IlwalidValue( RT_EXCEPTION_INFO, "splitTuningRounds must be at least one!", cfg.splitTuningRounds );

    if (!(cfg.splitPriorityX > 0.0f))
        throw IlwalidValue( RT_EXCEPTION_INFO, "splitPriorityX must be positive!", cfg.splitPriorityX );

    if (!(cfg.splitPriorityY > 0.0f))
        throw IlwalidValue( RT_EXCEPTION_INFO, "splitPriorityY must be positive!", cfg.splitPriorityY );

    if (cfg.splitMaxAABBsPerTriangle < 1)
        throw IlwalidValue( RT_EXCEPTION_INFO, "splitMaxAABBsPerTriangle must be at least one!", cfg.splitMaxAABBsPerTriangle );

    if (!(cfg.splitEpsilon >= 0.0f && cfg.splitEpsilon <= 0.5f))
        throw IlwalidValue( RT_EXCEPTION_INFO, "splitEpsilon must be between 0.0 and 0.5!", cfg.splitEpsilon );

    // Set config.

    m_cfg = cfg;
    m_cfg.splitMaxAABBsPerTriangle = min(m_cfg.splitMaxAABBsPerTriangle, 1 << MORTONSPLIT_EXEC_MAX_PIECES_LOG2);

    // Resize output buffers.

    m_cfg.outSplitAABBs .setNumElems(min(m_cfg.maxOutputPrims, (m_cfg.maxOutputPrims - m_cfg.maxInputTris) * 2));
    m_cfg.outPrimIndices.setNumElems(m_cfg.maxOutputPrims);
    m_cfg.outPrimRange  .setNumElems(1);

    // Layout temp buffers.

    m_globals       .assignNew(1);
    m_priorities    .assignNew(m_cfg.maxInputTris);

    m_cfg.tempBuffer
        .aggregate(m_globals)
        .aggregate(m_priorities);
}

//------------------------------------------------------------------------

void MortonTriangleSplitter::execute(BufferRef<const Range> inTriRange)
{
    RT_ASSERT(inTriRange.getNumElems() <= 1);
    m_inTriRange = inTriRange;

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

    m_cfg.tempBuffer.markAsUninitialized();
}

//------------------------------------------------------------------------

void MortonTriangleSplitter::execDevice(void)
{
    // Clear globals.

    m_globals.clearLWDA(0);

    // Launch MortonSplitInit.
    {
        MortonSplitInitParams p = {};
        p.outPrimIndices    = m_cfg.outPrimIndices.writeDiscardLWDA();
        p.outPrimRange      = m_cfg.outPrimRange.writeDiscardLWDA();
        p.outPriorities     = m_priorities.writeDiscardLWDA();
        p.globals           = m_globals.readWriteLWDA();
        p.inTriOrder        = m_cfg.inTriOrder.readLWDA();
        p.triRange          = m_inTriRange.readLWDA();
        p.inModel           = ModelPointers(m_cfg.inModel, MemorySpace_LWDA);
        p.inApexPointMap    = m_cfg.inApexPointMap.readLWDA();
        p.maxInputTris      = m_cfg.maxInputTris;
        p.splitBeta         = m_cfg.splitBeta;
        p.priorityLog2X     = logf(m_cfg.splitPriorityX) / logf(2.0f);
        p.priorityY         = m_cfg.splitPriorityY;
        p.epsilon           = m_cfg.splitEpsilon;

        LAUNCH(*m_cfg.lwca, MortonSplitInit, MORTONSPLIT_INIT_WARPS_PER_BLOCK, max(m_cfg.maxInputTris, 1), p);
    }

    // No splits allowed => done.

    if (m_cfg.splitBeta == 0.0f)
        return;

    // Launch MortonSplitTune.
    {
        MortonSplitTuneParams p = {};
        p.globals               = m_globals.readWriteLWDA();
        p.inPriorities          = m_priorities.readLWDA();
        p.triRange              = m_inTriRange.readLWDA();
        p.maxInputTris          = m_cfg.maxInputTris;
        p.maxOutputPrims        = m_cfg.maxOutputPrims;
        p.maxSplitsPerTriangle  = (float)m_cfg.splitMaxAABBsPerTriangle - 1.0f;
        p.splitBeta             = m_cfg.splitBeta;

        LAUNCH(*m_cfg.lwca, MortonSplitTuneA, 1, 1, p);

        for (int round = 0; round < m_cfg.splitTuningRounds; round++)
        {
            LAUNCH(*m_cfg.lwca, MortonSplitTuneB, MORTONSPLIT_TUNEB_WARPS_PER_BLOCK,
                (m_cfg.maxInputTris - 1) / MORTONSPLIT_TUNEB_TRIS_PER_THREAD + 1, p);

            LAUNCH(*m_cfg.lwca, MortonSplitTuneA, 1, 1, p);
        }
    }

    // Launch MortonSplitExec.
    {
        MortonSplitExecParams p = {};
        p.outSplitAABBs         = m_cfg.outSplitAABBs.writeDiscardLWDA();
        p.outPrimIndices        = m_cfg.outPrimIndices.readWriteLWDA();
        p.outPrimRange          = m_cfg.outPrimRange.readWriteLWDA();
        p.globals               = m_globals.readWriteLWDA();
        p.triRange              = m_inTriRange.readLWDA();
        p.inPriorities          = m_priorities.readLWDA();
        p.inModel               = ModelPointers(m_cfg.inModel, MemorySpace_LWDA);
        p.inApexPointMap        = m_cfg.inApexPointMap.readLWDA();
        p.maxInputTris          = m_cfg.maxInputTris;
        p.maxOutputPrims        = m_cfg.maxOutputPrims;
        p.maxSplitAABBs         = (int)m_cfg.outSplitAABBs.getNumElems();
        p.maxAABBsPerTriangle   = m_cfg.splitMaxAABBsPerTriangle;
        p.epsilon               = m_cfg.splitEpsilon;

        LAUNCH(*m_cfg.lwca, MortonSplitExec, MORTONSPLIT_EXEC_WARPS_PER_BLOCK,
            min(m_cfg.lwca->getMaxThreads(), m_cfg.maxInputTris), p);
    }
}

//------------------------------------------------------------------------

void MortonTriangleSplitter::execHost(void)
{
    m_cfg.outSplitAABBs     .writeDiscardHost();
    m_cfg.outPrimIndices    .writeDiscardHost();
    m_cfg.outPrimRange      .writeDiscardHost();
    m_cfg.inTriOrder        .readHost();
    ModelPointers inModel   (m_cfg.inModel, MemorySpace_Host);
    m_cfg.inApexPointMap    .readHost();
    m_inTriRange            .readHost();
    m_priorities            .writeDiscardHost();

    // Fetch model AABB, and setup transform from world space to 20-bit fixed point.

    AABB modelAABB = m_cfg.inApexPointMap->getAABB();
    float ilwScale = fmaxf(fmaxf(modelAABB.hi.x - modelAABB.lo.x, modelAABB.hi.y - modelAABB.lo.y), modelAABB.hi.z - modelAABB.lo.z) * (1.0f / (float)(1 << 20));
    float scale = 1.0f / ilwScale;
    float originX = (float)(1 << 19) - (modelAABB.lo.x + modelAABB.hi.x) * (scale * 0.5f);
    float originY = (float)(1 << 19) - (modelAABB.lo.y + modelAABB.hi.y) * (scale * 0.5f);
    float originZ = (float)(1 << 19) - (modelAABB.lo.z + modelAABB.hi.z) * (scale * 0.5f);

    // Init: Output initial AABB and evaluate heuristic priority for each triangle.

    double totalPriority = 0.0;
    float epsilonA = 1.0f - m_cfg.splitEpsilon;
    float epsilonB = m_cfg.splitEpsilon;

    Range triRange = (m_inTriRange.getNumElems()) ? *m_inTriRange : Range(0, m_cfg.maxInputTris);
    RT_ASSERT(triRange.span() >= 0 && triRange.span() <= m_cfg.maxInputTris);
    *m_cfg.outPrimRange = Range(0, triRange.span());

    for (int outputIdx = 0; outputIdx < triRange.span(); outputIdx++)
    {
        // Initialize outPrimIndices.

        int inputIdx = outputIdx + triRange.start;
        if (m_cfg.inTriOrder.getNumElems())
            inputIdx = m_cfg.inTriOrder[inputIdx];
        m_cfg.outPrimIndices[outputIdx] = inputIdx;

        // Fetch vertices and callwlate AABB.

        float3 v0, v1, v2;
        inModel.loadVertexPositions(v0, v1, v2, inputIdx);
        float3 lo = min(min(v0, v1), v2);
        float3 hi = max(max(v0, v1), v2);

        // No splits allowed => skip priority callwlation.

        if (m_cfg.splitBeta == 0.0f)
            continue;

        // Find the most important spatial median plane intersected by the AABB.

        int ilox = (int)((lo.x * epsilonA + hi.x * epsilonB) * scale + originX);
        int iloy = (int)((lo.y * epsilonA + hi.y * epsilonB) * scale + originY);
        int iloz = (int)((lo.z * epsilonA + hi.z * epsilonB) * scale + originZ);
        int ihix = max((int)((hi.x * epsilonA + lo.x * epsilonB) * scale + originX) - 1, ilox);
        int ihiy = max((int)((hi.y * epsilonA + lo.y * epsilonB) * scale + originY) - 1, iloy);
        int ihiz = max((int)((hi.z * epsilonA + lo.z * epsilonB) * scale + originZ) - 1, iloz);
        int importance = findLeadingOne((ilox ^ ihix) | (iloy ^ ihiy) | (iloz ^ ihiz));

        // Callwlate ideal surface area assuming infinite number of splits.

        float3 d1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 d2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        float aabbArea = aabbHalfArea(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z) * 2.0f;

        float idealArea =
            fabsf(d1.y * d2.z - d1.z * d2.y) +
            fabsf(d1.z * d2.x - d1.x * d2.z) +
            fabsf(d1.x * d2.y - d1.y * d2.x);

        // Evaluate split priority (Equation 5),
        // and ensure that it is finite an non-negative.

        float priority = powf(powf(m_cfg.splitPriorityX, (float)importance) * (aabbArea - idealArea), m_cfg.splitPriorityY);
        if (!(priority == priority && priority >= 0.0f))
            priority = 0.0f;

        // Output priority.

        m_priorities[outputIdx] = priority;
        totalPriority += priority;
    }

    // No splits allowed => done.

    if (m_cfg.splitBeta == 0.0f)
        return;

    // Tune: Find scaling factor for the priorities.

    int maxPrimsAfterSplit = triRange.span() + (int)((float)triRange.span() * m_cfg.splitBeta);
    maxPrimsAfterSplit = min(maxPrimsAfterSplit, m_cfg.maxOutputPrims);
    int maxSplits = maxPrimsAfterSplit - triRange.span();

    int maxSplitsPerTriangle = m_cfg.splitMaxAABBsPerTriangle - 1;
    float lwrScaleFactor = 0.0f;
    float lwrScaleInterval = (float)maxSplits / (float)totalPriority; // initial value = lower bound

    for (int round = 0; round < m_cfg.splitTuningRounds; round++)
    {
        // Callwlate how many splits we would end up performing using 3 different scale factors.

        float scale1 = lwrScaleFactor + lwrScaleInterval * 1.0f;
        float scale2 = lwrScaleFactor + lwrScaleInterval * 2.0f;
        float scale3 = lwrScaleFactor + lwrScaleInterval * 3.0f;
        int totalSplits1 = 0;
        int totalSplits2 = 0;
        int totalSplits3 = 0;

        for (int i = 0; i < triRange.span(); i++)
        {
            totalSplits1 += min((int)(m_priorities[i] * scale1), maxSplitsPerTriangle);
            totalSplits2 += min((int)(m_priorities[i] * scale2), maxSplitsPerTriangle);
            totalSplits3 += min((int)(m_priorities[i] * scale3), maxSplitsPerTriangle);
        }

        // First round => determine tight lower and upper bound based on the results.

        if (round == 0)
        {
            // Lower bound.

            float lowerBound = scale1;
            int lowerSplits = totalSplits1;
            if (totalSplits3 <= maxSplits)
                lowerBound = scale3, lowerSplits = totalSplits3;
            else if (totalSplits2 <= maxSplits)
                lowerBound = scale2, lowerSplits = totalSplits2;

            // Upper bound.

            float upperBound = lowerBound * (float)maxSplits / (float)lowerSplits;
            if (totalSplits2 > maxSplits)
                upperBound = fminf(upperBound, scale2);
            else if (totalSplits3 > maxSplits)
                upperBound = fminf(upperBound, scale3);

            // Bracket into 4 sub-intervals.

            lwrScaleFactor = lowerBound;
            lwrScaleInterval = (upperBound - lowerBound) * 0.25f;
        }

        // Remaining rounds => choose the right sub-interval, and keep bracketing.

        else
        {
            if (totalSplits3 <= maxSplits)
                lwrScaleFactor = scale3;
            else if (totalSplits2 <= maxSplits)
                lwrScaleFactor = scale2;
            else if (totalSplits1 <= maxSplits)
                lwrScaleFactor = scale1;
            lwrScaleInterval *= 0.25f;
        }
    }

    // Exec: Split each triangle relwrsively.

    std::vector<int2> stack;
    stack.reserve(m_cfg.splitMaxAABBsPerTriangle);
    int freshAABBCounter = (int)m_cfg.outSplitAABBs.getNumElems();

    for (int refIdx = 0; refIdx < triRange.span(); refIdx++)
    {
        // Callwlate desired number of pieces after splitting.
        // No need to split => skip.

        int numPieces = min((int)(m_priorities[refIdx] * lwrScaleFactor) + 1, m_cfg.splitMaxAABBsPerTriangle);
        if (numPieces < 2)
            continue;

        // Allocate and initialize splitAABB.

        int inputIdx = m_cfg.outPrimIndices[refIdx];
        int aabbIdx = --freshAABBCounter; // allocate from the end of the buffer
        m_cfg.outPrimIndices[refIdx] = aabbIdx + (1 << 30);
        m_cfg.outSplitAABBs[aabbIdx] = inModel.loadPrimitiveAABB(inputIdx);

        // Fetch vertex positions.

        float3 v0, v1, v2;
        inModel.loadVertexPositions(v0, v1, v2, inputIdx);

        // Process split tasks relwrsively.

        stack.push_back(make_int2(aabbIdx, numPieces));
        while (stack.size())
        {
            // Pop task.
            // No more splits => skip.

            int idxA = stack.back().x;
            int lwrPieces = stack.back().y;
            stack.pop_back();

            if (lwrPieces < 2)
                continue;

            // Fetch AABB.

            PrimitiveAABB& outA = m_cfg.outSplitAABBs[idxA];
            float3 lo = make_float3(outA.lox, outA.loy, outA.loz);
            float3 hi = make_float3(outA.hix, outA.hiy, outA.hiz);

            // Find the most important spatial median plane intersected by the AABB.

            int ilox = (int)((lo.x * epsilonA + hi.x * epsilonB) * scale + originX);
            int iloy = (int)((lo.y * epsilonA + hi.y * epsilonB) * scale + originY);
            int iloz = (int)((lo.z * epsilonA + hi.z * epsilonB) * scale + originZ);
            int ihix = max((int)((hi.x * epsilonA + lo.x * epsilonB) * scale + originX) - 1, ilox);
            int ihiy = max((int)((hi.y * epsilonA + lo.y * epsilonB) * scale + originY) - 1, iloy);
            int ihiz = max((int)((hi.z * epsilonA + lo.z * epsilonB) * scale + originZ) - 1, iloz);
            int importanceX = findLeadingOne(ilox ^ ihix);
            int importanceY = findLeadingOne(iloy ^ ihiy);
            int importanceZ = findLeadingOne(iloz ^ ihiz);

            int planeAxis = (importanceZ >= importanceX && importanceZ >= importanceY) ? 2 : (importanceY >= importanceX) ? 1 : 0;
            int iplanePos = chooseComponent(ihix, ihiy, ihiz, planeAxis) & (-1 << max(max(importanceX, importanceY), importanceZ));
            float planePos = ((float)iplanePos - chooseComponent(originX, originY, originZ, planeAxis)) * ilwScale;

            // The plane does not actually intersect the AABB (due to numerical inaclwracies) => skip.

            if (planePos <= chooseComponent(lo, planeAxis) || planePos >= chooseComponent(hi, planeAxis))
                continue;

            // Classify vertices on either side of the split plane.

            float w0 = chooseComponent(v0.x, v0.y, v0.z, planeAxis) - planePos;
            float w1 = chooseComponent(v1.x, v1.y, v1.z, planeAxis) - planePos;
            float w2 = chooseComponent(v2.x, v2.y, v2.z, planeAxis) - planePos;

            // Swap vertices so that v1 and v2 are on the same side.

            if (w1 * w2 < 0.0f)
            {
                float3 tv0 = v0;
                float tw0 = w0;
                if (w0 * w1 < 0.0f)
                    v0 = v1, w0 = w1, v1 = tv0, w1 = tw0;
                else
                    v0 = v2, w0 = w2, v2 = tv0, w2 = tw0;
            }

            // Determine clip vertices.

            float t01 = fminf(fmaxf(w0 / (w0 - w1), 0.0f), 1.0f);
            float t02 = fminf(fmaxf(w0 / (w0 - w2), 0.0f), 1.0f);
            float3 c01 = make_float3(v0.x + (v1.x - v0.x) * t01, v0.y + (v1.y - v0.y) * t01, v0.z + (v1.z - v0.z) * t01);
            float3 c02 = make_float3(v0.x + (v2.x - v0.x) * t02, v0.y + (v2.y - v0.y) * t02, v0.z + (v2.z - v0.z) * t02);
            if (planeAxis == 0) c01.x = c02.x = planePos;
            if (planeAxis == 1) c01.y = c02.y = planePos;
            if (planeAxis == 2) c01.z = c02.z = planePos;

            // Callwlate the clipped AABBs.

            float3 clo = min(c01, c02);
            float3 chi = max(c01, c02);
            float3 alo = max(lo, min(clo, v0));
            float3 ahi = min(hi, max(chi, v0));
            float3 blo = max(lo, min(clo, min(v1, v2)));
            float3 bhi = min(hi, max(chi, max(v1, v2)));

            // Partition pieces between the resulting tasks.

            float weightA = fmaxf(fmaxf(ahi.x - alo.x, ahi.y - alo.y), ahi.z - alo.z);
            float weightB = fmaxf(fmaxf(bhi.x - blo.x, bhi.y - blo.y), bhi.z - blo.z);
            int piecesA = min(max((int)((float)lwrPieces * weightA / (weightA + weightB) + 0.5f), 1), lwrPieces - 1);
            int piecesB = lwrPieces - piecesA;

            // Output task A.

            outA.lox = alo.x, outA.loy = alo.y, outA.loz = alo.z;
            outA.hix = ahi.x, outA.hiy = ahi.y, outA.hiz = ahi.z;
            outA.primitiveIdx = inputIdx;
            stack.push_back(make_int2(idxA, piecesA));

            // Output task B.

            int refB = m_cfg.outPrimRange->end++;
            int idxB = refB - triRange.span(); // allocate from the beginning of the buffer
            m_cfg.outPrimIndices[refB] = idxB + (1 << 30);

            PrimitiveAABB& outB = m_cfg.outSplitAABBs[idxB];
            outB.lox = blo.x, outB.loy = blo.y, outB.loz = blo.z;
            outB.hix = bhi.x, outB.hiy = bhi.y, outB.hiz = bhi.z;
            outB.primitiveIdx = inputIdx;
            stack.push_back(make_int2(idxB, piecesB));
        }
    }

    // Check that splitAABB allocation did not overflow.

    RT_ASSERT(m_cfg.outPrimRange->end - triRange.span() <= freshAABBCounter);
}

//------------------------------------------------------------------------
