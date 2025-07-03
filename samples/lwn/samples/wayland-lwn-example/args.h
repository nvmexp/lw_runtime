#pragma once

class TestArgs
{
public:
    bool m_debug;
    bool m_fullScreen;
    bool m_tripleBuffer;
    bool m_compress;        // Display buffers should be compressed or not
    bool m_measureFrameRate;
#ifdef WAYLAND_LWN_PRESENT
    int  m_cpuLoad;
    int  m_gpuLoad;
#endif

    TestArgs() : m_debug(false), m_fullScreen(false), m_tripleBuffer(false),
                 m_compress(true), m_measureFrameRate(false)
#ifdef WAYLAND_LWN_PRESENT
                // Default to 12ms load for CPU/GPU
                , m_cpuLoad(12), m_gpuLoad(12)
#endif
    {}

    void ParseArgs(int argc, char *argv[]);
};

extern TestArgs g_args;
