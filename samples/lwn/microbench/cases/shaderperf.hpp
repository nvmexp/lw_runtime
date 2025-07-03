/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __SHADERPERF_HPP
#define __SHADERPERF_HPP

#include "bench.hpp"
#include <string>

// A subtest of BenchmarkShaderPerfLWN
//
// A 'ShaderTest' is responsible for constructing an LWNprogram and binding
// it in its constructor, along with optional textures
class ShaderTest
{
public:
    struct Context
    {
        LWNdevice                            *device;
        LwnUtil::CoherentBufferPool          *coherentPool;
        LwnUtil::GPUBufferPool               *gpuPool;
        LwnUtil::DescriptorPool              *descriptorPool;
        std::unique_ptr<LwnUtil::CmdBuf>      cmdBuf;
        std::unique_ptr<LwnUtil::Mesh>        mesh;

        Context(LWNdevice *device_,
            LwnUtil::CoherentBufferPool *coherentPool_,
            LwnUtil::GPUBufferPool *gpuPool_,
            LwnUtil::DescriptorPool *descriptorPool_,
            LwnUtil::CmdBuf *cmdBuf_,
            LwnUtil::Mesh *mesh_) :
            device(device_),
            coherentPool(coherentPool_),
            gpuPool(gpuPool_),
            descriptorPool(descriptorPool_),
            cmdBuf(cmdBuf_),
            mesh(mesh_)
        {
        }
    };

    ShaderTest(Context *context);
    void setupSampler2DTexture(int texWidth, int texHeight, int samplerIndex);
    virtual ~ShaderTest();

protected:
    Context                           *m_context;
    // Textures are optional
    int                                m_texWidth;
    int                                m_texHeight;
    std::unique_ptr<LwnUtil::Buffer>   texPbo;
    LWNtexture                         m_texture;
    LWNsampler                         m_sampler;
    LWNtextureHandle                   m_textureHandle;

    LWNdevice*                         device() const         { return m_context->device; }
    LwnUtil::CoherentBufferPool*       coherentPool() const   { return m_context->coherentPool; }
    LwnUtil::GPUBufferPool*            gpuPool() const        { return m_context->gpuPool; }
    LwnUtil::DescriptorPool*           descriptorPool() const { return m_context->descriptorPool; }
    LwnUtil::CmdBuf*                   cmdBuf() const         { return m_context->cmdBuf.get(); }
};

// An umbrella test that runs a number of shader perf subtests.
class BenchmarkShaderPerfLWN : public BenchmarkCaseLWN
{
public:
    BenchmarkShaderPerfLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    ~BenchmarkShaderPerfLWN();

    int numSubtests() const;
    Description description(int subtest) const;
    void init(int subtest);
    void draw(const DrawParams* params);
    double measuredValue(int subtest, double elapsedTime);
    void deinit(int subtest);

private:
    uint64_t               m_numPixRendered;
    ShaderTest::Context*   m_context;
    ShaderTest*            m_subtest;
};

typedef ShaderTest * (*CreateShaderTestFunc)(ShaderTest::Context *params);

#define SHADERTEST_CREATE(ty, name, p) ShaderTest *createShaderTest_##name(ShaderTest::Context *ctx) { return new ty(ctx, p); }
#define DECLARE_SHADERPERF_TEST(name) extern ShaderTest *createShaderTest_##name(ShaderTest::Context *params);
#include "cases/shaderperf/tests.hpp"
#undef DECLARE_SHADERPERF_TEST

#endif
