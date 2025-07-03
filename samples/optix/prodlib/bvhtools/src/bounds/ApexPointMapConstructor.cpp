// Copyright LWPU Corporation 2015
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "ApexPointMapConstructor.hpp"
#include "ApexPointMapConstructorKernels.hpp"
#include "ApexPointMapDirections.hpp"
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/Assert.h>
#include <vector>
#include <map>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

void ApexPointMapConstructor::configure(const Config& cfg)
{
    // Check for errors.

    RT_ASSERT(cfg.inModel.isValid());

    if (cfg.apmResolution < 1 || cfg.apmResolution > APM_MAX_RESOLUTION)
        throw IlwalidValue(RT_EXCEPTION_INFO, "apmResolution must be between 1 and APM_MAX_RESOLUTION!", cfg.apmResolution);

    // Set config and resize outputs.

    m_cfg = cfg;
    m_cfg.outApexPointMap.setNumBytes(ApexPointMap::getNumBytes(cfg.apmResolution));

    // Layout temp buffers.

    m_workCounter.assignNew(1);
    m_cfg.tempBuffer.aggregate(m_workCounter);
}

//------------------------------------------------------------------------

void ApexPointMapConstructor::execute(void)
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


    m_cfg.tempBuffer.markAsUninitialized();
}

//------------------------------------------------------------------------

void ApexPointMapConstructor::execDevice(void)
{
    APMConstructParams p = {};
    p.outApexPointMap   = m_cfg.outApexPointMap.writeDiscardLWDA();
    p.workCounter       = m_workCounter.writeDiscardLWDA();
    p.inModel           = ModelPointers(m_cfg.inModel, MemorySpace_LWDA);
    p.apmResolution     = m_cfg.apmResolution;
    p.numDirections     = ApexPointMap::getNumDirections(m_cfg.apmResolution);

    // Launch init kernel.

    LAUNCH(*m_cfg.lwca, APMInit, 1, p.numDirections, p);

    // Empty model -> done.

    if (m_cfg.inModel.numPrimitives == 0)
        return;

    // Launch construct kernel.

    int maxBatches = (m_cfg.inModel.numPrimitives - 1) / APM_CONSTRUCT_BATCH_SIZE + 1;
    LAUNCH(*m_cfg.lwca, APMConstruct, APM_CONSTRUCT_WARPS_PER_BLOCK,
        min(m_cfg.lwca->getMaxThreads(), maxBatches * 32), p);
}

//------------------------------------------------------------------------

void ApexPointMapConstructor::execHost(void)
{
    m_cfg.outApexPointMap.writeDiscardHost();
    ModelPointers inModel(m_cfg.inModel, MemorySpace_Host);

    // Write the resolution.

    m_cfg.outApexPointMap->m_resolution = m_cfg.apmResolution;

    // Empty model => set everything to NaN (unhittable).

    if (inModel.numPrimitives == 0)
    {
        for (int i = 0; i < ApexPointMap::getNumDirections(m_cfg.apmResolution); i++)
        {
            m_cfg.outApexPointMap->m_dots[i].x = ApexPointMap::encodeFloat(LWDART_NAN_F);
            m_cfg.outApexPointMap->m_dots[i].y = ApexPointMap::encodeFloat(LWDART_NAN_F);
        }
        return;
    }

    // Callwlate the dot products. This is a trivial and horrendously slow implementation.

    float rcpResolution = 1.f / (float)m_cfg.apmResolution;
    for (int i = 0; i < ApexPointMap::getNumDirections(m_cfg.apmResolution); i++)
    {
        float3 dir = ApexPointMap::decodeDirection(g_apexPointMapDirections[m_cfg.apmResolution][i], rcpResolution);

        float dmin = +FLT_MAX;
        float dmax = -FLT_MAX;

        if (inModel.aabbs != NULL)
        {
            // AABBs.

            for (int j = 0; j < inModel.numPrimitives; j++)
            {
                const PrimitiveAABB& aabb = inModel.aabbs[j];
                float tc = dir.x * (aabb.lox + aabb.hix) + dir.y * (aabb.loy + aabb.hiy) + dir.z * (aabb.loz + aabb.hiz);
                float th = fabsf(dir.x) * (aabb.hix - aabb.lox) + fabsf(dir.y) * (aabb.hiy - aabb.loy) + fabsf(dir.z) * (aabb.hiz - aabb.loz);
                dmin = fminf(dmin, .5f * (tc - th));
                dmax = fmaxf(dmax, .5f * (tc + th));
            }
        }
        else if (inModel.indices != NULL)
        {
            // Indexed.

            for (int j = 0; j < inModel.numPrimitives; j++)
            {
                int3 tri = *(int3*)((unsigned char*)inModel.indices + j * inModel.indexStride);

                float3 v[3];
                v[0] = *(float3*)((unsigned char*)inModel.vertices + tri.x * inModel.vertexStride);
                v[1] = *(float3*)((unsigned char*)inModel.vertices + tri.y * inModel.vertexStride);
                v[2] = *(float3*)((unsigned char*)inModel.vertices + tri.z * inModel.vertexStride);

                for (int k = 0; k < 3; k++)
                {
                    float d = v[k].x * dir.x + v[k].y * dir.y + v[k].z * dir.z;
                    dmin = fminf(dmin, d);
                    dmax = fmaxf(dmax, d);
                }
            }
        }
        else
        {
            // Unindexed.

            for (int j = 0; j < inModel.numPrimitives * 3; j++)
            {
                float3 v = *(float3*)((unsigned char*)inModel.vertices + j * inModel.vertexStride);
                float d = v.x * dir.x + v.y * dir.y + v.z * dir.z;
                dmin = fminf(dmin, d);
                dmax = fmaxf(dmax, d);
            }
        }

        m_cfg.outApexPointMap->m_dots[i].x = ApexPointMap::encodeFloat(dmin);
        m_cfg.outApexPointMap->m_dots[i].y = ApexPointMap::encodeFloat(dmax);
    }
}

//------------------------------------------------------------------------
// LUT generation and printout.
//------------------------------------------------------------------------

static int addCoord(int u, int v, std::map<int2, int>& coordToDirOfs, std::vector<int2>& dirCoord, int resolution)
{
    if (coordToDirOfs.count(make_int2(u, v)) > 0)
        return coordToDirOfs.find(make_int2(u, v))->second;

    if (abs(u) + abs(v) == resolution && coordToDirOfs.count(make_int2(-u, -v)) > 0)
        return coordToDirOfs.find(make_int2(-u, -v))->second | (1u << 31);

    int ofs = (int)dirCoord.size() * sizeof(int2);
    coordToDirOfs.insert(std::pair<int2, int>(make_int2(u, v), ofs));
    dirCoord.push_back(make_int2(u, v));

    return ofs;
}

//------------------------------------------------------------------------

static bool operator<(const int2& a, const int2& b)
{
    if (a.x < b.x) return true;
    if (a.x > b.x) return false;
    return (a.y < b.y);
}

//------------------------------------------------------------------------

void ApexPointMapConstructor::printLUTSource(void)
{
    std::vector<uint4>        quadsArray     [APM_MAX_RESOLUTION];
    std::vector<unsigned int> directionsArray[APM_MAX_RESOLUTION];

    for (int resolution = 1; resolution <= APM_MAX_RESOLUTION; resolution++)
    {
        std::map<int2, int> coordToDirOfs;
        std::vector<int2>   dirCoord;

        // Prime with main coordinate axes so they end up in first three slots.

        dirCoord.push_back(make_int2(resolution, 0)); // Positive X.
        dirCoord.push_back(make_int2(0, resolution)); // Positive Y.
        dirCoord.push_back(make_int2(0, 0));          // Positive Z.
        for (int i = 0; i < 3; i++)
            coordToDirOfs.insert(std::pair<int2, int>(dirCoord[i], i * (int)sizeof(int2)));

        // Construct integer coordinates and quads inside the diamond.

        std::vector<uint4>&        quads      = quadsArray[resolution - 1];
        std::vector<unsigned int>& directions = directionsArray[resolution - 1];

        quads.resize(resolution * resolution * 4);

        for (int sv = 0; sv < 2; sv++)
        for (int su = 0; su < 2; su++)
        for (int v = 0; v < resolution; v++)
        for (int u = 0; u < resolution; u++)
        {
            if (u + v >= resolution)
                continue;

            bool dual = (u + v < resolution - 1);

            int u0 = su ? -u : u;
            int v0 = sv ? -v : v;
            int u1 = u0 + (su ? -1 : 1);
            int v1 = v0 + (sv ? -1 : 1);

            int ofs00 = addCoord(u0, v0, coordToDirOfs, dirCoord, resolution);
            int ofs10 = addCoord(u1, v0, coordToDirOfs, dirCoord, resolution);
            int ofs01 = addCoord(u0, v1, coordToDirOfs, dirCoord, resolution);
            int ofs11 = dual ? addCoord(u1, v1, coordToDirOfs, dirCoord, resolution) : ofs00;

            int sidx = su + 2 * sv;
            quads[u + resolution * v + sidx * resolution * resolution] = make_uint4(ofs00, ofs10, ofs01, ofs11);
        }

        // Fill in quads outside the diamond with mirror images of those inside
        // the diamond to make outside lookups (potentially caused by rounding errors)
        // behave correctly.

        for (int sidx = 0; sidx < 4; sidx++)
        for (int v = 0; v < resolution; v++)
        for (int u = 0; u < resolution; u++)
        {
            if (u + v < resolution)
                continue;

            int dstIdx = u + resolution * v;
            int srcIdx = (resolution - v - 1) + resolution * (resolution - u - 1);
            srcIdx += sidx * resolution * resolution;
            dstIdx += sidx * resolution * resolution;

            uint4 q = quads[srcIdx];
            quads[dstIdx] = make_uint4(q.w, q.y, q.z, q.x);
        }

        // Construct final packed directions.

        directions.resize(dirCoord.size());
        for (size_t i = 0; i < directions.size(); i++)
        {
            int2 uv = dirCoord[i];
            directions[i] = (uv.x & 0x0000ffffu) | ((uv.y & 0x0000ffffu) << 16);
        }
    }

    // Print the direction tables.

    for (int resolution = 1; resolution <= APM_MAX_RESOLUTION; resolution++)
    {
        const std::vector<unsigned int>& directions = directionsArray[resolution - 1];
        printf("APM_LUT_DECORATOR unsigned int g_apexPointMapDirections_%d[%d] = {", resolution, ApexPointMap::getNumDirections(resolution));
        for (size_t i = 0; i < directions.size(); i++)
            printf("%s0x%08xu, ", (i % 10 == 0) ? "\n    " : "", directions[i]);
        printf("\n};\n\n");
    }

    printf("APM_LUT_DECORATOR unsigned int* g_apexPointMapDirections[%d] = {\n", APM_MAX_RESOLUTION + 1);
    printf("    0,\n");
    for (int resolution = 1; resolution <= APM_MAX_RESOLUTION; resolution++)
        printf("    g_apexPointMapDirections_%d,\n", resolution);
    printf("};\n\n");

    // Print the quad lookup tables.

    for (int resolution = 1; resolution <= APM_MAX_RESOLUTION; resolution++)
    {
        const std::vector<uint4>& quads = quadsArray[resolution - 1];
        printf("APM_LUT_DECORATOR uint4 g_apexPointMapQuads_%d[%d] = {", resolution, (int)quads.size());
        for (size_t i = 0; i < quads.size(); i++)
        {
            uint4 quad = quads[i];
            printf("%s{0x%08xu,0x%08xu,0x%08xu,0x%08xu}, ", (i % 3 == 0) ? "\n    " : "", quad.x, quad.y, quad.z, quad.w);
        }
        printf("\n};\n\n");
    }

    printf("APM_LUT_DECORATOR uint4* g_apexPointMapQuads[%d] = {\n", APM_MAX_RESOLUTION + 1);
    printf("    0,\n");
    for (int resolution = 1; resolution <= APM_MAX_RESOLUTION; resolution++)
        printf("    g_apexPointMapQuads_%d,\n", resolution);
    printf("};\n\n");
}

//------------------------------------------------------------------------
