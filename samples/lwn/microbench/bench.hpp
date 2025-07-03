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

#include "utils.hpp"

class SampleArray;

struct DrawParams
{
    uint32_t flags;
};

class BenchmarkCase
{
private:
protected:
    int m_width;
    int m_height;

public:
    enum
    {
        DISPLAY_PRESENT_BIT = 1<<0
    };

    enum TestPropFlags
    {
        TEST_PROP_BYPASS_FRAME_LOOP_BIT = 1 << 0,
        TEST_PROP_HOS_ONLY_BIT          = 1 << 1   /* run only on HOS by default */
    };

    struct Description
    {
        const char* name;
        const char* units;
    };

    virtual int numSubtests() const = 0;
    virtual Description description(int subtest) const = 0;

    // Perform bigger allocations & setup in init, this is called once
    // before ::draw().  You shouldn't initialize big allocations in
    // the constructor.
    //
    // Everything allocated in init() should be deallocated in
    // deinit().
    virtual void init(int subtest) = 0;
    virtual void draw(const DrawParams* params) = 0;
    virtual void deinit(int subtest) = 0;
    virtual double measuredValue(int subtest, double elapsedTime) = 0;
    virtual uint32_t testProps() const;

    BenchmarkCase(int w, int h);
    virtual ~BenchmarkCase();

    inline int width() const           { return m_width; }
    inline int height() const          { return m_height; }
};

class BenchmarkCaseLWN : public BenchmarkCase
{
private:
    LWNdevice*                   m_device;
    LWNqueue*                    m_queue;
    LwnUtil::Pools*              m_pools;
public:
    struct DrawParamsLWN : public DrawParams
    {
        DrawParamsLWN() : dstColor(nullptr), dstDepth(nullptr), window(nullptr)
        {
        }
        LWNtexture*            dstColor;
        LWNtexture*            dstDepth;
        LWNwindow*             window;
        LwnUtil::RenderTarget* renderTarget;
    };

    BenchmarkCaseLWN(LWNdevice* dev, LWNqueue* q, LwnUtil::Pools* pools, int w, int h);
    virtual ~BenchmarkCaseLWN();

    inline LWNdevice*                    device() const          { return m_device; }
    inline LWNqueue*                     queue() const           { return m_queue; }
    inline LwnUtil::Pools*               pools() const           { return m_pools; }
    inline LwnUtil::GPUBufferPool*       gpuPool() const         { return m_pools->gpu(); }
    inline LwnUtil::CoherentBufferPool*  coherentPool() const    { return m_pools->coherent(); }
    inline LwnUtil::CPUCachedBufferPool* cpuCachedPool() const   { return m_pools->cpuCached(); }
    inline LwnUtil::DescriptorPool*      descriptorPool() const  { return m_pools->descriptor(); }
};

class BenchmarkCaseOGL : public BenchmarkCase
{
private:
public:
    BenchmarkCaseOGL(int w, int h);
    ~BenchmarkCaseOGL();
};

class ResultPrinter
{
private:
public:
    ResultPrinter();
    virtual ~ResultPrinter();

    virtual void print(const char* test, double value, const char* units) = 0;
};

class ResultPrinterStdout : public ResultPrinter
{
private:
public:
    ResultPrinterStdout();
    ~ResultPrinterStdout();

    void print(const char* test, double value, const char* units);
};

class ResultCollector
{
private:
    SampleArray* m_samples;
public:
    ResultCollector();
    ~ResultCollector();

    void begin(const char* testname, const char* units);
    void end(double value);
    void print(ResultPrinter& printer);
};

// Context class for running a benchmark case.  Collection of some
// platform dependent utilities (like flipping the display) plus basic
// benchmark infrastructure functionality like timing, reporting
// results.
class BenchmarkContextLWN
{
private:

    LWNdevice*                   m_device;
    LWNwindow                    m_window;
    LWNqueue*                    m_queue;
    LWNsync                      m_textureAvailableSync;
    LwnUtil::Buffer*             m_counterBuf;
    LwnUtil::Timer*              m_timer;

    LwnUtil::Pools*              m_testcasePools;
    LwnUtil::Pools*              m_contextPools;
    LwnUtil::DescBufferPool*     m_descriptorMemPool;
    LwnUtil::DescriptorPool*     m_descriptorPool;

    LwnUtil::CompiledCmdBuf*     m_defaultState;
    LwnUtil::CompiledCmdBuf*     m_counterCmds0;
    LwnUtil::CompiledCmdBuf*     m_counterCmds1;
    int                          m_width;
    int                          m_height;

    LwnUtil::RenderTarget*       m_renderTarget;
    int                          m_rtBufferIdx;

    void setupDefaultState();
    void initDescriptorPool();

    bool testRunnable(const BenchmarkCaseLWN *bench, int subtestIdx) const;
    void run(ResultCollector* results, BenchmarkCaseLWN* test);
public:
    BenchmarkContextLWN(LWNdevice* dev, LWNnativeWindow nativeWindow, int w, int h);
    virtual ~BenchmarkContextLWN();

    // Run all BenchmarkCases
    void runAll(ResultCollector* results);

    virtual void flip() = 0;
};
