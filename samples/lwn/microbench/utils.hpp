/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#define GL_GLEXT_PROTOTYPES

#include <lwn/lwn.h>
#include <lwn/lwn_FuncPtrInline.h>
#include <GL/gl.h>

// Define types that are not in gl.h but required by glext.h and portions of microbench
typedef double GLdouble;
typedef double GLclampd;

#define GL_UNSIGNED_INT    0x1405

#ifndef APIENTRYP
#define APIENTRYP APIENTRY *
#endif

// Core entry point callback types not defined in either gl.h or glext.h
typedef void (APIENTRYP PFNGLCLEARPROC) (GLbitfield mask);
typedef void (APIENTRYP PFNGLCLEARCOLORPROC) (GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha);
typedef void (APIENTRYP PFNGLCOLORMASKPROC) (GLboolean red, GLboolean green, GLboolean blue, GLboolean alpha);
typedef void (APIENTRYP PFNGLDEPTHFUNCPROC) (GLenum func);
typedef void (APIENTRYP PFNGLDEPTHMASKPROC) (GLboolean flag);
typedef void (APIENTRYP PFNGLDISABLEPROC) (GLenum cap);
typedef void (APIENTRYP PFNGLDRAWELEMENTSPROC) (GLenum mode, GLsizei count, GLenum type, const void *indices);
typedef void (APIENTRYP PFNGLENABLEPROC) (GLenum cap);
typedef void (APIENTRYP PFNGLFINISHPROC) (void);
typedef void (APIENTRYP PFNGLFLUSHPROC) (void);

#define GL_ARB_vertex_buffer_object 1
#include <GL/glext.h>

#include <cstdint>
#include "timer.hpp"
#include <memory>

#include <assert.h>

#if defined(LW_HOS)
#include <lwnExt/lwnExt_Internal.h>
#include <nn/nn_Log.h>
#define PRINTF NN_LOG
#define FLUSH_STDOUT()
#else
#define PRINTF printf
#define FLUSH_STDOUT() fflush(stdout)
#endif

#include <string.h>
#include "lwnUtil/lwnUtil_GlslcHelper.h"        // For GLSLC Specialization
#include "lwnUtil/lwnUtil_PoolAllocator.h"      // for definitions of BUFFER_ALIGN_*_BIT

using namespace std;
using namespace lwnUtil;

namespace LwnUtil
{
    struct Vec3i
    {
        int32_t x, y, z;
        Vec3i() {}
        Vec3i(int32_t x_, int32_t y_, int32_t z_) : x(x_), y(y_), z(z_) { }
    };

    struct Vec2f
    {
        float x, y;
        Vec2f() {}
        Vec2f(float x_, float y_) : x(x_), y(y_) { }
    };

    struct Vec3f
    {
        float x, y, z;
        Vec3f() {}
        Vec3f(float x_, float y_, float z_) : x(x_), y(y_), z(z_) { }
    };

    struct Vec4f
    {
        float x, y, z, w;
        Vec4f() {}
        Vec4f(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) { }
    };

    enum {
        LWN_INIT_DEBUG_LAYER_BIT = 1 << 0
    };

    template <typename T> inline bool isPow2(T v) { return (v & (v-1)) == 0; }

    // Round a pointer to 'align' alignment
    template <typename T> inline T align(T value, size_t align)
    {
        assert(isPow2(align));
        return (T)(((uintptr_t)value + align - 1) & ~(align - 1));
    }

    // Round an allocation size to 'align' alignment.  Use this like
    // so:
    //
    // size_t   ALIGNMENT  = 16;
    // size_t   allocSize  = alignSize(bufferSize, ALIGNMENT);
    // uint8_t* buf        = new uint8_t[allocSize];
    // uint8_t* alignedBuf = align(buf, ALIGNMENT);
    template <typename T> inline T alignSize(T value, size_t align)
    {
        assert(isPow2(align));
        return value + (T) align - 1;
    }

    LWNdevice* init(uint32_t flags, const char* glslcDllPath);
    void exit();

    class BufferPool
    {
    public:
        ~BufferPool();

        uintptr_t alloc(uintptr_t size, uintptr_t align);
        void freeAll();

        LWNmemoryPool*    pool()             { return &m_pool; }
        uintptr_t         size() const       { return m_poolSize; }
        uintptr_t         used() const       { return m_poolTop; }
        uintptr_t         free() const       { return m_poolSize - m_poolTop; }

    protected:
        BufferPool(LWNdevice* device, uintptr_t size, uint32_t poolFlags);

    private:
        BufferPool() {};

        LWNmemoryPool     m_pool;
        uintptr_t         m_poolTop;
        uintptr_t         m_poolSize;
        std::unique_ptr<uint8_t[]> m_memory;
    };

    class GPUBufferPool : public BufferPool
    {
    public:
        GPUBufferPool(LWNdevice* device, uintptr_t size) :
            BufferPool(device, size, (LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
                                      LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |
                                      LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT |
                                      LWN_MEMORY_POOL_FLAGS_COMPRESSIBLE_BIT))
        {
        }
    };

    class CoherentBufferPool : public BufferPool
    {
    public:
        CoherentBufferPool(LWNdevice* device, uintptr_t size) :
            BufferPool(device, size, (LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |
                                      LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |
                                      LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT))
        {
        }
    };

    class CPUCachedBufferPool : public BufferPool
    {
    public:
        CPUCachedBufferPool(LWNdevice* device, uintptr_t size) :
            BufferPool(device, size, (LWN_MEMORY_POOL_FLAGS_CPU_CACHED_BIT |
                                      LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |
                                      LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT))
        {
        }
    };

    class DescBufferPool : public BufferPool
    {
    public:
        DescBufferPool(LWNdevice* device, void* memory, uintptr_t size) :
            BufferPool(device, size, (
#if defined(_WIN32)
                                      LWN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS_BIT |
#else
                                      LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |
#endif
                                      LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |
                                      LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT))
        {
        }
    };

    class DescriptorPool
    {
    public:
        DescriptorPool(LWNdevice* device, DescBufferPool* descBufferPool, int maxIDs);
        ~DescriptorPool();

        int allocTextureID();
        int allocSamplerID();
        void freeAll();

        void registerTexture(uint32_t textureId, LWNtexture* tex);
        void registerImage(uint32_t textureId, LWNtexture* image);
        void registerSampler(uint32_t samplerId, LWNsampler* smp);
        void setPools(LWNcommandBuffer* cmd);

    private:
        DescriptorPool() {};

        LWNsamplerPool m_samplerPool;
        LWNtexturePool m_texturePool;

        int m_maxIDs;
        int m_numReservedTextures;
        int m_numReservedSamplers;
        int m_samplerIDTop;
        int m_textureIDTop;
    };

    class Pools
    {
    public:
        Pools(LWNdevice* device, LwnUtil::DescriptorPool* descrPool, size_t gpuPoolSize, size_t cohPoolSize, size_t cpuCachedPoolSize);
        ~Pools();

        void freeAll();

        LwnUtil::DescriptorPool*      descriptor() const { return m_descriptor; }
        LwnUtil::GPUBufferPool*       gpu() const        { return m_gpu.get(); }
        LwnUtil::CoherentBufferPool*  coherent() const   { return m_coherent.get(); }
        LwnUtil::CPUCachedBufferPool* cpuCached() const  { return m_cpuCached.get(); }

    private:
        LwnUtil::DescriptorPool*                      m_descriptor;
        std::unique_ptr<LwnUtil::GPUBufferPool>       m_gpu;
        std::unique_ptr<LwnUtil::CoherentBufferPool>  m_coherent;
        std::unique_ptr<LwnUtil::CPUCachedBufferPool> m_cpuCached;
    };

    class RingBuffer;

    class CmdBuf
    {
    public:
        CmdBuf(LWNdevice* device, LWNqueue* queue, LwnUtil::CoherentBufferPool* coherentPool, int numChunks, int cmdChunkSize, int ctrlChunkSize);
        ~CmdBuf();

        LWNcommandBuffer* cmd() { return &m_cmd; }

        void submit(uint32_t numCommands, const LWNcommandHandle* handles);
    private:
        CmdBuf() {}
        static void LWNAPIENTRY commandBufferMemoryCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event, size_t minSize, void *callbackData);

        void addCmdChunk();
        void addCtrlChunk();

        LWNdevice*          m_device;
        LWNqueue*           m_queue;

        int                 m_cmdChunkSize;
        LWNcommandBuffer    m_cmd;
        LWNmemoryPool*      m_cmdLWNPool;
        uintptr_t           m_cmdPoolOffset;
        RingBuffer*         m_cmdRB;

        int                 m_ctrlChunkSize;
        char*               m_ctrlPool;
        RingBuffer*         m_ctrlRB;
    };

    // Helper class for building static command buffers
    class CompiledCmdBuf
    {
    public:
        CompiledCmdBuf(LWNdevice* device, LwnUtil::BufferPool* cohPool, size_t cmdSize = 1024, size_t ctrlSize = 256);
        ~CompiledCmdBuf();

        void              submit(LWNqueue* queue);
        LWNcommandBuffer* cmd()          { return &m_cmd; }
        LWNcommandHandle  handle() const { return m_handle; };

        void              begin();
        void              end();
    private:
        static void LWNAPIENTRY commandBufferMemoryCallback(LWNcommandBuffer *cmdBuf, LWNcommandBufferMemoryEvent event, size_t minRequiredSize, void *callbackData);

        LWNcommandHandle           m_handle;
        LWNcommandBuffer           m_cmd;
        std::unique_ptr<uint8_t[]> m_ctrlBuf;
    };

    class RenderTarget
    {
    private:
        LWNtexture*     m_rtTex[2];
        LWNtexture      m_rtTexMSAA;
        LWNtexture*     m_depthTex;
        CompiledCmdBuf* m_setTargetsCmd[2];

        int             m_numSamples;

        uint64_t        m_zlwllBuffer;
        size_t          m_zlwllBufferSize;
    public:
        enum {
            DEST_WRITE_COLOR_BIT = 1 << 0,
            DEST_WRITE_DEPTH_BIT = 1 << 1
        };

        enum CreationFlags
        {
            MSAA_4X          = 1 << 0,
            ADAPTIVE_ZLWLL   = 1 << 1,
            DEPTH_FORMAT_D24 = 1 << 2 // allow overriding default d24s8
        };

        RenderTarget(LWNdevice* device, Pools* pools, int w, int h, uint32_t flags);
        ~RenderTarget();

        uint32_t getNumSamples() const { return m_numSamples; }

        void setTargets(LWNqueue* q, int cbIdx);
        void setTargets(LWNcommandBuffer* cmdBuf, int cbIdx);

        static void setColorDepthMode(LWNcommandBuffer* cmd, uint32_t destWriteMask, bool depthTest);

        void downsample(LWNcommandBuffer* cmd, LWNtexture* dst);

        LWNtexture** colorBuffers() { return m_rtTex; }
        LWNtexture* colorBufferMSAA()  { assert(m_numSamples > 1);  return &m_rtTexMSAA; }
        LWNtexture* depthBuffer() { return m_depthTex; }

        void getZLwllSaveRestoreBuffer(uint64_t *addr, size_t *size)
        {
            *addr = m_zlwllBuffer;
            *size = m_zlwllBufferSize;
        }
    };

    // Persistently mapped buffer object
    class Buffer
    {
    private:
        LWNbuffer* m_buf;
        void*      m_ptr;
        size_t     m_size;
    public:
        Buffer(LWNdevice* dev, BufferPool* pool, const void* data, size_t size, BufferAlignBits alignBlits);
        ~Buffer();

        void*            ptr() const      { return m_ptr; }
        size_t           size() const     { return m_size; }
        LWNbufferAddress address() const  { return lwnBufferGetAddress(m_buf); }
        LWNbuffer*       buffer() const   { return m_buf; }
    };

    template <size_t dataAlign, BufferAlignBits bufferAlignBits, typename eleT> class BufferArray
    {
    private:
        inline size_t elemAlign(size_t n) {
            return (n + (dataAlign-1)) & ~(dataAlign-1);
        }

        Buffer* m_buffer;
    public:
        BufferArray(LWNdevice* dev, CoherentBufferPool* coherentPool, int numElements)
            : m_buffer(new Buffer(dev, coherentPool, nullptr, elemAlign(sizeof(eleT))*numElements, bufferAlignBits)) {
        }

        ~BufferArray() {
            delete m_buffer;
        }

        LWNbufferAddress address() const { return m_buffer->address(); }

        inline size_t size() const  { return m_buffer->size(); }
        inline int offset(int ndx)  { return (int)elemAlign(sizeof(eleT))*ndx; }

        void set(int ndx, const eleT& d) {
            eleT* b = (eleT*)((uintptr_t)m_buffer->ptr() + offset(ndx));
            *b = d;
        }
    };

    template <typename eleT>
    using UboArr = BufferArray<256, BUFFER_ALIGN_UNIFORM_BIT, eleT>;

    template <typename eleT>
    using IndirectArr = BufferArray<4, BUFFER_ALIGN_INDIRECT_BIT, eleT>;

    class Mesh
    {
    private:
        Mesh() : m_vbo(nullptr),
                 m_texVbo(nullptr),
                 m_ibo(nullptr),
                 m_vboAddress(0UL),
                 m_texVboAddress(0UL),
                 m_iboAddress(0UL),
                 m_numVertices(0),
                 m_numTriangles(0)
        {}

        Buffer* m_vbo;
        Buffer* m_texVbo;
        Buffer* m_ibo;

        LWNbufferAddress m_vboAddress;
        LWNbufferAddress m_texVboAddress;
        LWNbufferAddress m_iboAddress;

        int     m_numVertices;
        int     m_numTriangles;
    public:
        static Mesh* createGrid(LWNdevice* dev, CoherentBufferPool* coherentPool, int gridX, int gridY, Vec2f offs, Vec2f scale, float z);
        static Mesh* createCircle(LWNdevice* dev, CoherentBufferPool* coherentPool, int nSegments, Vec2f scale, float z);
        static Mesh* createFullscreenTriangle(LWNdevice* dev, CoherentBufferPool* coherentPool, float z);

        ~Mesh();

        int numVertices()  const { return m_numVertices; }
        int numTriangles() const { return m_numTriangles; }

        LWNbufferAddress vboAddress() const    { return m_vboAddress; }
        LWNbufferAddress texVboAddress() const { return m_texVboAddress; }
        LWNbufferAddress iboAddress() const    { return m_iboAddress; }
    };

    class OGLMesh
    {
    private:
        OGLMesh() {}

        GLuint m_vbo;
        GLuint m_ibo;

        int m_numVertices;
        int m_numTriangles;

        void allocGLBuffers(const Vec3f* vtx, const Vec3i* tri);

    public:
        static OGLMesh* createGrid(int gridX, int gridY, Vec2f offs, Vec2f scale, float z);
        static OGLMesh* createFullscreenTriangle(float z);
        static OGLMesh* createCircle(int nSegments, Vec2f scale, float z);

        ~OGLMesh();

        int numVertices()  const { return m_numVertices; }
        int numTriangles() const { return m_numTriangles; }

        int vbo() const { return m_vbo; }
        int ibo() const { return m_ibo; }

        void bindGeometryGL(GLuint vtxAttrLoc);
    };

    class VertexState
    {
    private:
        int nAttribs;
        int nStreams;
        LWLwertexAttribState attribs[16];
        LWLwertexStreamState streams[16];
    public:
        VertexState() : nAttribs(0), nStreams(0) { }
        void setAttribute(int n, LWNformat format, ptrdiff_t offset, int stream);
        void resetAttribute(int n);
        void setStream(int n, ptrdiff_t stride, int divisor = 0);
        void resetStream(int n);
        void bind(LWNcommandBuffer *cmdBuf);
    };

    // For GLSLC Specialization
    // Copied from lwntest
    union ArrayUnion {
        int32_t  i[16];
        uint32_t u[16];
        float    f[16];
        double   d[16];
    };

    enum ArgTypeEnum {
        ARG_TYPE_INT = 0,
        ARG_TYPE_DOUBLE = 1,
        ARG_TYPE_FLOAT = 2,
        ARG_TYPE_UINT = 3
    };

    // All GL entry points must be dynamically queried. This macro lists all of the ones used by
    // microbanch and lets us iterate over all of them to manage this.
#define FOREACH_GL_PROC(OP) \
    OP(ATTACHSHADER,            AttachShader)               \
    OP(BINDBUFFER,              BindBuffer)                 \
    OP(BINDBUFFERBASE,          BindBufferBase)             \
    OP(BINDBUFFERRANGE,         BindBufferRange)            \
    OP(BUFFERDATA,              BufferData)                 \
    OP(BUFFERSUBDATA,           BufferSubData)              \
    OP(CLEARDEPTHF,             ClearDepthf)                \
    OP(COMPILESHADER,           CompileShader)              \
    OP(CREATEPROGRAM,           CreateProgram)              \
    OP(CREATESHADER,            CreateShader)               \
    OP(DEBUGMESSAGECALLBACK,    DebugMessageCallback)       \
    OP(DELETEBUFFERS,           DeleteBuffers)              \
    OP(DELETEPROGRAM,           DeleteProgram)              \
    OP(DRAWELEMENTS,            DrawElements)               \
    OP(DRAWELEMENTSINSTANCED,   DrawElementsInstanced)      \
    OP(ENABLEVERTEXATTRIBARRAY, EnableVertexAttribArray)    \
    OP(GENBUFFERS,              GenBuffers)                 \
    OP(GETACTIVEUNIFORMBLOCKIV, GetActiveUniformBlockiv)    \
    OP(GETACTIVEUNIFORMSIV,     GetActiveUniformsiv)        \
    OP(GETATTRIBLOCATION,       GetAttribLocation)          \
    OP(GETPROGRAMINFOLOG,       GetProgramInfoLog)          \
    OP(GETPROGRAMIV,            GetProgramiv)               \
    OP(GETSHADERINFOLOG,        GetShaderInfoLog)           \
    OP(GETSHADERIV,             GetShaderiv)                \
    OP(GETSHADERSOURCE,         GetShaderSource)            \
    OP(GETUNIFORMBLOCKINDEX,    GetUniformBlockIndex)       \
    OP(GETUNIFORMINDICES,       GetUniformIndices)          \
    OP(GETUNIFORMLOCATION,      GetUniformLocation)         \
    OP(LINKPROGRAM,             LinkProgram)                \
    OP(SHADERSOURCE,            ShaderSource)               \
    OP(UNIFORM4FV,              Uniform4fv)                 \
    OP(UNIFORMBLOCKBINDING,     UniformBlockBinding)        \
    OP(USEPROGRAM,              UseProgram)                 \
    OP(VERTEXATTRIBPOINTER,     VertexAttribPointer)

#define FOREACH_GL_10_PROC(OP) \
    OP(CLEAR,                   Clear)                      \
    OP(CLEARCOLOR,              ClearColor)                 \
    OP(COLORMASK,               ColorMask)                  \
    OP(DEPTHFUNC,               DepthFunc)                  \
    OP(DEPTHMASK,               DepthMask)                  \
    OP(DISABLE,                 Disable)                    \
    OP(ENABLE,                  Enable)                     \
    OP(FINISH,                  Finish)                     \
    OP(FLUSH,                   Flush)                      \

#define DECLARE_GL_CALLBACK(ucname, lcname) \
    extern PFNGL##ucname##PROC g_gl##lcname;

FOREACH_GL_PROC(DECLARE_GL_CALLBACK)
FOREACH_GL_10_PROC(DECLARE_GL_CALLBACK)

#undef DECLARE_GL_CALLBACK

    void enableWarpLwlling(bool enable);
    void enableCBF(bool enable);
    void setShaderScratchMemory(LwnUtil::GPUBufferPool* pool, size_t size, LWNcommandBuffer* cb);
    // GLSLC specialization helper functions copied from lwntest:
#define DEFAULT_SHADER_SCRATCH_MEMORY_SIZE 524288
    void setData(GLSLCspecializationUniform * uniform, const char * name, int numElements, ArgTypeEnum type, int numArgs, ...);
    void addSpecializationUniform(int index, const GLSLCspecializationUniform* uniform);
    void clearSpecializationUniformArrays(void);
    bool compileAndSetShaders(LWNprogram* pgm, const LWNshaderStage* stages, uint32_t count, const char** srcs);
}

#ifdef _WIN32
extern "C" void lwogSwapBuffers();
#endif
