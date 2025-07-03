/*
 * Copyright (c) 2015-2016 LWPU Corporation.  All rights reserved.
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

class BenchmarkPoolFlushLWN : public BenchmarkCaseLWN
{
public:
    static const int POOL_SIZE = 4096;

    struct TestDescr
    {
        int numPools;
    };

private:
    struct Resources
    {
        class MemPool
        {
        private:
            LWNmemoryPool m_pool;
        public:
            MemPool() {
            }

            bool init(LWNdevice *device, void* assetPtr, size_t size) {
                // Create dummy pools.  See bug 1713805.
                LWNmemoryPoolBuilder poolBuilder;
                lwnMemoryPoolBuilderSetDevice(&poolBuilder, device);
                lwnMemoryPoolBuilderSetDefaults(&poolBuilder);
                lwnMemoryPoolBuilderSetStorage(&poolBuilder, assetPtr, size);
                return (lwnMemoryPoolInitialize(&m_pool, &poolBuilder) != LWN_FALSE);
            }

            ~MemPool() {
                lwnMemoryPoolFinalize(&m_pool);
            }
        };

        Resources(int numPools, size_t poolSize) :
            cmdBuf(nullptr),
            perFrameCmd(nullptr),
            mesh(nullptr),
            assetStorage(new uint8_t[LwnUtil::alignSize(numPools * poolSize, LWN_MEMORY_POOL_STORAGE_ALIGNMENT)]) {
        }

        ~Resources() {
        }

        std::unique_ptr<LwnUtil::CmdBuf>         cmdBuf;
        std::unique_ptr<LwnUtil::CompiledCmdBuf> perFrameCmd;
        std::unique_ptr<LwnUtil::Mesh>           mesh;

        std::unique_ptr<uint8_t[]>               assetStorage;
        std::vector<MemPool>                     dummyPools;
    };

    Resources*                 m_res;
    LWNprogram                 m_pgm;
    LwnUtil::VertexState*      m_vertex;

    int                        m_subtestIdx;
    uint64_t                   m_numFlushes;

    LWNcommandHandle renderCommands(LWNcommandBuffer* cmdBuf);
    void setupTexture(LWNcommandBuffer* cmd, int texWidth, int texHeight);

public:
    BenchmarkPoolFlushLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkPoolFlushLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);
};
