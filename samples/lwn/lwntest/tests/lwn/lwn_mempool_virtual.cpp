/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

extern int lwnDebugEnabled;

static const char* vsString = 
    "layout(location = 0) in vec4 position;\n"
    "out vec2 texcoord;\n"
    "void main() {\n"
    "  gl_Position = position;\n"
    "  texcoord = (position.xy + vec2(1.0, 1.0)) * vec2(0.5, 0.5);\n"
    "}\n";

static const char* fsTexture = 
    "layout(location = 0) out vec4 color;\n"
    "layout (binding=0) uniform sampler2D tex;\n"
    "in vec2 texcoord;\n"
    "void main() {\n"
    "  color = vec4(texture(tex, texcoord));\n"
    "}\n";

using namespace lwn;

//////////////////////////////////////////////////////////////////////////

static const int offscreenWidth = 128, offscreenHeight = 128;
static const int texWidth = 8, texHeight = 8; // value 8 chosen because compression seems to kick in starting with this size???

// these values are arbitrarily chosen since we don't have 
// APIs that allow us to query constraints related to the values like
// alignment, max. sizes etc.)
static const int numPoolRanges = 2;
static const int numPages = 2;
static const int pageSize = 0x10000L; // 64 page size??? Need to query???
static const int memSizePhys = pageSize * numPages;
static const int memSizeVirt = memSizePhys * numPoolRanges;

class DrawWithTexture
{
public:
    explicit DrawWithTexture(Framebuffer* fb) :
        _device(DeviceState::GetActive()->getDevice()),
        _queueCB(DeviceState::GetActive()->getQueueCB()),
        _queue(DeviceState::GetActive()->getQueue()), _fb(fb)
    {
        // Set up the vertex format and buffer.
        struct Vertex {
            dt::vec4 position;
        };
        static const Vertex vertexData[] = {
            { dt::vec4(-1.0, -1.0, 0.0, 1.0) },
            { dt::vec4(-1.0, +1.0, 0.0, 1.0) },
            { dt::vec4(+1.0, -1.0, 0.0, 1.0) },
            { dt::vec4(+1.0, +1.0, 0.0, 1.0) },
        };
        _vertexDataSize = sizeof(vertexData);

        _vertex_allocator = new MemoryPoolAllocator(_device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
        VertexStream stream(sizeof(Vertex));
        LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
        VertexStreamSet streamSet(stream);
        _vertex = streamSet.CreateVertexArrayState();

        // vertex buffer
        _vbo = stream.AllocateVertexBuffer(_device, 4, *_vertex_allocator, vertexData);

        // sampler
        SamplerBuilder sb;
        sb.SetDevice(_device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST_MIPMAP_NEAREST, MagFilter::NEAREST);
        _sampler = sb.CreateSampler();
    }

    ~DrawWithTexture()
    {
        _sampler->Free();
        _vertex_allocator->freeBuffer(_vbo);
        delete _vertex_allocator;
    }

    void draw(Texture *tex) {
        // set render target
        // scissor and viewport
        _fb->bind(_queueCB);
        _fb->setViewportScissor();
        actuallyDraw(tex);
    }

    void draw(Texture *tex, int x, int y, int w, int h) {
        // set render target
        // scissor and viewport
        _fb->bind(_queueCB);
        _queueCB.SetViewportScissor(x, y, w, h);
        actuallyDraw(tex);
    }

    void actuallyDraw(Texture *tex) {
        // combined texture/sampler handle
        LWNuint textureID = tex->GetRegisteredTextureID();
        TextureHandle texHandle = _device->GetTextureHandle(textureID, _sampler->GetRegisteredID());

        // bind vertex state, texture and sampler
        _queueCB.BindVertexArrayState(_vertex);
        _queueCB.BindVertexBuffer(0, _vbo->GetAddress(), _vertexDataSize);
        _queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
        _queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        _queueCB.submit();
        _queue->Finish();
    }

    void clearColor(const LWNfloat* color) {
        _fb->bind(_queueCB);
        _fb->setViewportScissor();
        _queueCB.ClearColor(0, color, ClearColorMask::RGBA);
    }
    void clearColor(const LWNfloat* color, int x, int y, int w, int h) {
        _fb->bind(_queueCB);
        _queueCB.SetViewportScissor(x, y, w, h);
        _queueCB.ClearColor(0, color, ClearColorMask::RGBA);
    }
private:
    Device                  * _device;
    QueueCommandBuffer      &_queueCB;
    Queue                   * _queue;

    Framebuffer*            _fb;

    MemoryPoolAllocator*    _vertex_allocator;
    LWNuint                 _vertexDataSize;
    Sampler                 *_sampler;
    VertexArrayState        _vertex;
    Buffer                  *_vbo;
};

class Test
{
public:
    Test() :
        _device(DeviceState::GetActive()->getDevice()),
        _queueCB(DeviceState::GetActive()->getQueueCB()),
        _queue(DeviceState::GetActive()->getQueue()) {}
    virtual ~Test();

    Texture *execute();
    virtual void run() = 0;
    // clear FB to result color, can be overriden by test
    virtual bool getResult() {
        _fb->bind(_queueCB);
        _fb->setViewportScissor();
        const LWNfloat red[] = { 1, 0, 0, 1 };
        const LWNfloat green[] = { 0, 1, 0, 1 };
        _queueCB.ClearColor(0, _result ? green : red, ClearColorMask::RGBA);
        return _result;
    }

protected:
    Device              *_device;
    QueueCommandBuffer  &_queueCB;
    Queue               *_queue;
    Framebuffer*        _fb;
    bool                _result;

    void init();
    void setPass();

    void read(Texture *tex, void* pixel);
    int PreMapVerify(Texture *tex[], int numTex);
    int UnmapAndVerify(MemoryPool *vPool, Texture *tex[], int numTex);

    MemoryPool *createPhysicalPool(LWNuint initial_value, LWNuint size, void** physMemAllocated);
    MemoryPool *createVirtualPool(LWNuintptr poolSize = memSizeVirt);

    // LWN does not implement getting tile sizes in HR10, so hardcode what we need here.
    // We only lwrrently need dimensions for RGBA8 single sample format in this test.
    void GetPageDimensions(Format format, TextureTarget target, LWNint *width, LWNint *height)
    {
        assert(format == Format::RGBA8);
        assert(target == TextureTarget::TARGET_2D);

        *width = 128;
        *height = 128;

        assert((pageSize == 0x10000) || (pageSize == 0x20000));

        LWNint pageSize = 0;
        _device->GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);

        if (pageSize == 0x20000) {
            if (*width == *height) {
                *width <<= 1;
            }
            else {
                *height <<= 1;
            }
        }
    }
};

Test::~Test()
{
    _fb->destroy();
    delete _fb;
}

void Test::init()
{
    _result = false;
}

void Test::setPass() {
    _result = true;
}

MemoryPool *Test::createPhysicalPool(LWNuint initial_value, LWNuint size, void** physMemAllocated)
{
    // create a physical pool that will donate pages
    LWNuint* physMem = (LWNuint*)PoolStorageAlloc(size);
    *physMemAllocated = physMem;
    for (LWNuint i = 0; i < size / sizeof(LWNuint); i++) {
        physMem[i] = initial_value;
    }

    MemoryPool *memPool = new MemoryPool;

    MemoryPoolBuilder builder;
    builder.SetDevice(_device).SetDefaults();
    builder.SetStorage(physMem, size);
    MemoryPoolFlags poolFlags(MemoryPoolFlags::GPU_NO_ACCESS |
                              MemoryPoolFlags::CPU_CACHED |
                              MemoryPoolFlags::PHYSICAL |
                              MemoryPoolFlags::COMPRESSIBLE);
    builder.SetFlags(poolFlags);
    if (!memPool->Initialize(&builder)) {
        delete memPool;
        PoolStorageFree(physMem);
        *physMemAllocated = NULL;
        return NULL;
    }
    return memPool;
}


MemoryPool *Test::createVirtualPool(LWNuintptr poolSize) {
    // create a virtual pool where we will map pages from the physical pool
    MemoryPool *memPool = new MemoryPool;

    MemoryPoolBuilder builder;
    builder.SetDevice(_device).SetDefaults();
    builder.SetStorage(NULL, poolSize);
    MemoryPoolFlags poolFlags(MemoryPoolFlags::CPU_NO_ACCESS |
                              MemoryPoolFlags::GPU_CACHED |
                              MemoryPoolFlags::VIRTUAL |
                              MemoryPoolFlags::COMPRESSIBLE);
    builder.SetFlags(poolFlags);
    if (!memPool->Initialize(&builder)) {
        delete memPool;
        return NULL;
    }
    return memPool;
}

void Test::read(Texture *tex, void*pixel)
{
    MemoryPoolAllocator coherent_allocator(_device, NULL, 0, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    LWNuint data[texWidth][texHeight];

    // pink texture image
    for (int dy = 0; dy < texHeight; dy++) {
        for (int dx = 0; dx < texWidth; dx++) {
            data[dx][dy] = 0xFFFF00FF;
        }
    }

    Buffer *readbo = AllocAndFillBuffer(_device, _queue, _queueCB, coherent_allocator, data, sizeof(LWNuint),
                                        BUFFER_ALIGN_COPY_WRITE_BIT, false);

    CopyRegion copyRegion = { 0, 0, 0, 1, 1, 1 };
    _queueCB.CopyTextureToBuffer(tex, NULL, &copyRegion, readbo->GetAddress(), CopyFlags::NONE);

    _queueCB.submit();
    _queue->Finish();

    LWNuint* p = (LWNuint*)readbo->Map();
    *(LWNuint*)pixel = *p;
}

int Test::PreMapVerify(Texture *tex[], int numTex)
{
    // Clear the unpopulated texture.  On Maxwell2 GPUs, these clears should
    // effectively do nothing because pages shouldn't be mapped.  Reads from
    // such memory should return black, so we clear to non-black.  On older
    // GPUs, a "dummy" page will be mapped in unbound portions of the pool and
    // hardware does not prevent overwrites, so we clear to black to make sure
    // we read back the "expected" black.
    const LWNfloat red[] = { 1, 0, 0, 1 };
    const LWNfloat black[] = { 0, 0, 0, 0 };
    const LWNfloat *clearColor = g_lwnDeviceCaps.supportsZeroFromUndefinedMappedPoolPages ? red : black;
    for (int i = 0; i < numTex; i++) {
        _queueCB.SetRenderTargets(1, &tex[i], NULL, NULL, NULL);
        _queueCB.SetViewportScissor(0, 0, texWidth, texHeight);
        _queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
    }

    DrawWithTexture dwt(_fb);

    // Now draw something with the texture.  We should get back because of
    // hardware support (Maxwell2) or because we just wrote black (older).
    for (int i = 0; i < numTex; i++) {
        dwt.draw(tex[i]);

        LWNuint pixel = 0xFFFFFFFF;
        read(_fb->getColorTexture(0), (void*)&pixel);

        if (pixel != 0) {
            return 1;
        }
    }

    return 0;
}

int Test::UnmapAndVerify(MemoryPool *vPool, Texture *tex[], int numTex)
{
    MappingRequest req = { 0, };

    // unmap all pages
    req.physicalPool = NULL;
    req.virtualOffset = 0;
    req.physicalOffset = 0;
    req.size = memSizeVirt;
    req.storageClass = 0;

    vPool->MapVirtual(1, &req);

    // Clear the now-unpopulated texture.  On Maxwell2 GPUs, these clears
    // should effectively do nothing because pages shouldn't be mapped.  Reads
    // from such memory should return black, so we clear to non-black.  On
    // older GPUs, a "dummy" page will be mapped in unbound portions of the
    // pool and hardware does not prevent overwrites, so we clear to black to
    // make sure we read back the "expected" black.  overwrites.
    const LWNfloat blue[] = { 0, 0, 1, 1 };
    const LWNfloat black[] = { 0, 0, 0, 0 };
    const LWNfloat *clearColor = g_lwnDeviceCaps.supportsZeroFromUndefinedMappedPoolPages ? blue : black;
    for (int i = 0; i < numTex; i++) {
        _queueCB.SetRenderTargets(1, &tex[i], NULL, NULL, NULL);
        _queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);
    }

    // Now draw something with the texture.  We should get back because of
    // hardware support (Maxwell2) or because we just wrote black (older).
    DrawWithTexture dwt(_fb);
    for (int i = 0; i < numTex; i++) {
        dwt.draw(tex[i]);

        LWNuint pixel = 0xFFFFFFFF;
        read(_fb->getColorTexture(0), (void*)&pixel);
        if (pixel != 0) {
            return 1;
        }
    }
    return 0;
}

// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

Texture *Test::execute()
{
    init();

    VertexShader vs(440);
    vs << vsString;

    FragmentShader fs(440);
    fs << fsTexture;

    Program *pgm = _device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    if (!compiled) {
        return NULL;
    }

    // always use the same program
    _queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

    // setup framebuffer
    _fb = new Framebuffer;
    _fb->setSize(offscreenWidth, offscreenHeight);
    _fb->setColorFormat(0, Format::RGBA8);
    _fb->alloc(_device);

    // run test
    run();
    getResult();

    _queue->Finish();

    return _fb->getColorTexture(0);
}

// create a texture in a virtual pool and
// see if it has the right color after mapping in
// the pre-filled physical pool pages.
// This creates the texture *before* it maps the pages in.
class Simple : public Test
{
    void run() {
        void* poolAlloc;
        MemoryPool *memPoolPhys = createPhysicalPool(0xFF00FF00, memSizePhys, &poolAlloc); // green
        if (!memPoolPhys) {
            return;
        }

        // create a virtual pool where we will map pages from the physical pool
        MemoryPool *memPoolVirtual = createVirtualPool();
        if (!memPoolVirtual) {
            return;
        }

        TextureBuilder *textureBuilder = _device->CreateTextureBuilder();
        textureBuilder->SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetFormat(Format::RGBA8).
            SetSize2D(texWidth, texHeight).
            SetFlags(TextureFlags::COMPRESSIBLE);

        Texture *tex[numPoolRanges];
        for (int i = 0; i < numPoolRanges; i++) {
            tex[i] = textureBuilder->CreateTextureFromPool(memPoolVirtual, i * memSizePhys);
        }

        // verify all is unmapped
        if (PreMapVerify(tex, numPoolRanges)) {
            return;
        }

        MappingRequest req[numPoolRanges];

        // populate and expect color to be green because physical pool was all green
        for (int i = 0; i < numPoolRanges; i++) {
            req[i].physicalPool = memPoolPhys;
            req[i].virtualOffset = i * memSizePhys;
            req[i].physicalOffset = 0;
            req[i].size = memSizePhys;
            req[i].storageClass = tex[i]->GetStorageClass();
        }
        memPoolVirtual->MapVirtual(numPoolRanges, req);

        // now draw again with the texture, should be color of texture
        DrawWithTexture dwt(_fb);
        for (int i = 0; i < numPoolRanges; i++) {
            dwt.draw(tex[i]);

            LWNuint pixel = 0xFFFFFFFF;
            read(_fb->getColorTexture(0), (void*)&pixel);
            if (pixel != 0xFF00FF00) { // green
                return;
            }
        }

        if (UnmapAndVerify(memPoolVirtual, tex, numPoolRanges)) {
            return;
        }

        ////******************
        //// tear down

        for (int i = 0; i < numPoolRanges; i++) {
            tex[i]->Free();
        }

        // free virtual pool
        memPoolVirtual->Finalize();
        delete memPoolVirtual;

        // free physical pool
        memPoolPhys->Finalize();
        delete memPoolPhys;

        PoolStorageFree(poolAlloc);

        // free builder
        textureBuilder->Free();

        setPass();
    }
};

class Simple2 : public Test
{
    void run() {
        const int numPools = 2;
        MemoryPool *memPoolPhys[numPools];
        void* poolAlloc[numPools];
        LWNuint iValues[numPools] = { 0xFFFF0000, 0xFF00FF00 };

        for (unsigned int i = 0; i < __GL_ARRAYSIZE(memPoolPhys); i++) {
            memPoolPhys[i] = createPhysicalPool(iValues[i], memSizePhys, &poolAlloc[i]);
            if (!memPoolPhys[i]) {
                return;
            }
        }

        // create a virtual pool where we will map pages from the physical pool
        MemoryPool *memPoolVirtual = createVirtualPool();
        if (!memPoolVirtual) {
            return;
        }

        TextureBuilder *textureBuilder = _device->CreateTextureBuilder();
        textureBuilder->SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetFormat(Format::RGBA8).
            SetFlags(TextureFlags::COMPRESSIBLE).
            SetSize2D(texWidth, texHeight);

        Texture *tex[numPools];
        // create one texture per physical page, one should render as red, one as green when mapped for GPU
        for (int i = 0; i < numPools; i++) {
            tex[i] = textureBuilder->CreateTextureFromPool(memPoolVirtual, i * memSizePhys);
        }

        // verify all is unmapped
        if (PreMapVerify(tex, numPools)) {
            return;
        }

        MappingRequest req[numPools];

        // populate and expect color to be different for each page because another physical pool 
        // with a different color is used per page
        for (int i = 0; i < numPools; i++) {
            req[i].physicalPool = memPoolPhys[i];
            req[i].virtualOffset = i * memSizePhys;
            req[i].physicalOffset = 0;
            req[i].size = memSizePhys;
            req[i].storageClass = tex[i]->GetStorageClass();
        }
        memPoolVirtual->MapVirtual(numPools, req);

        DrawWithTexture dwt(_fb);

        // now draw again with the texture, should be color of texture
        for (int i = 0; i < numPools; i++) {
            dwt.draw(tex[i]);

            LWNuint pixel = 0xFFFFFFFF;
            read(_fb->getColorTexture(0), (void*)&pixel);
            if (pixel != iValues[i]) {
                return;
            }
        }

        if (UnmapAndVerify(memPoolVirtual, tex, numPools)) {
            return;
        }

        ////******************
        //// tear down

        for (int i = 0; i < numPools; i++) {
            tex[i]->Free();
            // free physical pool
            // free physical pool
            memPoolPhys[i]->Finalize();
            delete memPoolPhys[i];

            PoolStorageFree(poolAlloc[i]);
        }

        // free virtual pool
        memPoolVirtual->Finalize();
        delete memPoolVirtual;


        // free builder
        textureBuilder->Free();

        setPass();
    }
};

class Miplevels : public Test
{
    void run() {
        LWNint maxTextureLevels = 0;
        _device->GetInteger(DeviceInfo::MAX_TEXTURE_LEVELS, &maxTextureLevels);

        LWNint pageSize = 0;
        _device->GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);

        LWNint tileWidth, tileHeight;
        GetPageDimensions(Format::RGBA8, TextureTarget::TARGET_2D, &tileWidth, &tileHeight);

        // create a virtual pool where we will map pages from the physical pool
        void* poolAlloc;
        MemoryPool *memPoolPhys = createPhysicalPool(0xFF00FF00, pageSize, &poolAlloc); // green

        const int texWidth = 4096, texHeight = texWidth;
        const int max_tiles_x = texWidth / tileWidth;
        const int max_tiles_y = texHeight / tileHeight;
        const int numLevels = 1 + FloorLog2(texWidth);

        TextureBuilder *textureBuilder = _device->CreateTextureBuilder();
        textureBuilder->SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetFormat(Format::RGBA8).
            SetFlags(TextureFlags::COMPRESSIBLE).
            SetLevels(numLevels);
        textureBuilder->SetSize2D(texWidth, texHeight); // 16k^2 texture (query max size?)

        MemoryPool *memPoolVirtual = createVirtualPool((textureBuilder->GetStorageSize() + pageSize - 1) & ~(pageSize - 1));
        if (!memPoolVirtual) {
            return;
        }

        LWNuintptr baseTexOffset = 0;
        Texture *tex = textureBuilder->CreateTextureFromPool(memPoolVirtual, baseTexOffset);

        // render using every this mip level
        DrawWithTexture dwt(_fb);
        LWNfloat gray[4] = { 0.1, 0.1, 0.1, 1.0 };
        dwt.clearColor(gray);

        LWNfloat gray1[] = { 0.3, 0.3, 0.3, 1 };
        LWNfloat gray2[] = { 0.5, 0.5, 0.5, 1 };
        int cellSizeX = (offscreenWidth / ceil(sqrt((float)numLevels)));
        int cellSizeY = (offscreenHeight / ceil(sqrt((float)numLevels)));
        int xx = 0, yy = 0, ww = (offscreenWidth / numLevels), hh = ww;
        CellIterator2D cell(4, 4);

        VertexShader vs(440);
        vs << vsString;

        TextureView *textureView = new TextureView;
        for (int level = 0; level < numLevels; level++) {
            int tiles_x = max_tiles_x >> level;
            int tiles_y = max_tiles_y >> level;

            textureView->SetDefaults().SetLevels(level, 1);
            LWNuintptr tileVirtualOffset = tex->GetViewOffset(textureView);
            LWNuintptr tileVirtualOffsetAligned = tileVirtualOffset & ~(pageSize - 1);

            // now map in one physical page for the level 
            // diagonal pages
            MappingRequest req[128];

            int x = 0;
            if (tiles_y) {
                for (int y = 0; y < tiles_y; y++) {
                    req[y].physicalPool = memPoolPhys;
                    req[y].virtualOffset = tileVirtualOffsetAligned + pageSize*(y*tiles_x + x);
                    req[y].physicalOffset = 0;
                    req[y].size = pageSize;
                    req[y].storageClass = tex->GetStorageClass();
                    x++; // move over by one each row
                }
                memPoolVirtual->MapVirtual(tiles_y, req);
            } else {
                req[0].physicalPool = memPoolPhys;
                req[0].virtualOffset = tileVirtualOffsetAligned;
                req[0].physicalOffset = 0;
                req[0].size = pageSize;
                req[0].storageClass = tex->GetStorageClass();
                memPoolVirtual->MapVirtual(1, req);
            }

            FragmentShader fsLod(440);
            fsLod << "layout(location = 0) out vec4 color;\n"
                "layout (binding=0) uniform sampler2D tex;\n"
                "in vec2 texcoord;\n"
                "void main() {\n"
                "  vec4 c = vec4(textureLod(tex, texcoord, " << level << "));\n"
                "  if(c == vec4(0,0,0,0))\n"
                "    discard;\n"
                "  else\n"
                "    color = c;\n"
                "}\n";

            Program *pgm = _device->CreateProgram();
            LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fsLod);
            if (!compiled) {
                return;
            }

            _queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

            xx = cell.x() * cellSizeX + cellSizeX / 10;
            yy = cell.y() * cellSizeY + cellSizeY / 10;
            ww = cellSizeX - 2 * cellSizeX / 10;
            hh = cellSizeY - 2 * cellSizeY / 10;
            dwt.clearColor((!(cell.x() & 1) != !(cell.y() & 1)) ? gray1 : gray2, xx, yy, ww, hh);
            dwt.draw(tex, xx, yy, ww, hh);
            cell++;

            if (tiles_y) {
                x = 0;
                for (int y = 0; y < tiles_y; y++) {
                    req[y].physicalPool = NULL;
                    req[y].virtualOffset = tileVirtualOffsetAligned + pageSize*(y*tiles_x + x);
                    req[y].physicalOffset = 0;
                    req[y].size = pageSize;
                    req[y].storageClass = 0;
                    x++; // move over by one each row
                }
                memPoolVirtual->MapVirtual(tiles_y, req);
            }
            else {
                req[0].physicalPool = NULL;
                req[0].virtualOffset = tileVirtualOffsetAligned;
                req[0].physicalOffset = 0;
                req[0].size = pageSize;
                req[0].storageClass = 0;
                memPoolVirtual->MapVirtual(1, req);
            }
        }

        ////******************
        //// tear down
        tex->Free();

        delete textureView;

        // free virtual pool
        memPoolVirtual->Finalize();
        delete memPoolVirtual;

        // free physical pool
        memPoolPhys->Finalize();
        delete memPoolPhys;
        PoolStorageFree(poolAlloc);

        // free builder
        textureBuilder->Free();

        // mark test as pass
        setPass();
    }

    virtual bool getResult() {
        return _result;
    }
};

class Layers : public Test
{
    void run() {
        const int numLayers = 16;

        LWNint pageSize = 0;
        _device->GetInteger(DeviceInfo::MEMORY_POOL_PAGE_SIZE, &pageSize);

        LWNint tileWidth, tileHeight;
        GetPageDimensions(Format::RGBA8, TextureTarget::TARGET_2D, &tileWidth, &tileHeight);

        // create a physical pool with one page to map into virtual pool
        void* poolAlloc;
        MemoryPool *memPoolPhys = createPhysicalPool(0xFF00FF00, pageSize, &poolAlloc); // green

        const int texWidth = 4096, texHeight = texWidth;
        const int max_tiles_x = texWidth / tileWidth;

        TextureBuilder *textureBuilder = _device->CreateTextureBuilder();
        textureBuilder->SetDefaults().
            SetTarget(TextureTarget::TARGET_2D_ARRAY).
            SetFormat(Format::RGBA8).
            SetFlags(TextureFlags::COMPRESSIBLE);
        textureBuilder->SetSize3D(texWidth, texHeight, numLayers); // 16k^2 texture (query max size?)

        MemoryPool *memPoolVirtual = createVirtualPool(textureBuilder->GetStorageSize());
        if (!memPoolVirtual) {
            return;
        }

        LWNuintptr baseTexOffset = 0;
        Texture *tex = textureBuilder->CreateTextureFromPool(memPoolVirtual, baseTexOffset);

        // render using every this mip level
        DrawWithTexture dwt(_fb);
        LWNfloat gray[4] = { 0.1, 0.1, 0.1, 1.0 };
        dwt.clearColor(gray);

        LWNfloat gray1[] = { 0.3, 0.3, 0.3, 1 };
        LWNfloat gray2[] = { 0.5, 0.5, 0.5, 1 };
        int cellSizeX = (offscreenWidth / ceil(sqrt((float)numLayers)));
        int cellSizeY = (offscreenHeight / ceil(sqrt((float)numLayers)));
        int xx = 0, yy = 0, ww = (offscreenWidth / numLayers), hh = ww;
        CellIterator2D cell(4, 4);

        VertexShader vs(440);
        vs << vsString;

        TextureView *textureView = new TextureView;
        MappingRequest* req = new MappingRequest[max_tiles_x];
        for (int layer = 0; layer < numLayers; layer++) {
            textureView->SetDefaults().SetLayers(layer, 1);
            LWNuintptr tileVirtualOffset = tex->GetViewOffset(textureView);
            LWNuintptr tileVirtualOffsetAligned = tileVirtualOffset & ~(pageSize - 1);

            // now map in one physical page for the layer into
            // horizontal pages,  offset row by layer
            for (int x = 0; x < max_tiles_x; x++) {
                req[x].physicalPool = memPoolPhys;
                req[x].virtualOffset = tileVirtualOffsetAligned + pageSize*(layer*max_tiles_x + x);
                req[x].physicalOffset = 0;
                req[x].size = pageSize;
                req[x].storageClass = tex->GetStorageClass();
            }
            memPoolVirtual->MapVirtual(max_tiles_x, req);

            FragmentShader fsLayer(440);
            fsLayer << "layout(location = 0) out vec4 color;\n"
                "layout (binding=0) uniform sampler2DArray tex;\n"
                "in vec2 texcoord;\n"
                "void main() {\n"
                "  vec4 c = vec4(texture(tex, vec3(texcoord.x, texcoord.y, " << layer << ")));\n"
                "  if(c == vec4(0,0,0,0))\n"
                "    discard;\n"
                "  else\n"
                "    color = c;\n"
                "}\n";

            Program *pgm = _device->CreateProgram();
            LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fsLayer);
            if (!compiled) {
                return;
            }

            _queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);

            xx = cell.x() * cellSizeX;
            yy = cell.y() * cellSizeY;
            ww = cellSizeX;
            hh = cellSizeY;
            dwt.clearColor((!(cell.x() & 1) != !(cell.y() & 1)) ? gray1 : gray2, xx, yy, ww, hh);
            dwt.draw(tex, xx, yy, ww, hh);
            cell++;

            // unmap
            for (int x = 0; x < max_tiles_x; x++) {
                req[x].physicalPool = NULL;
                req[x].virtualOffset = tileVirtualOffsetAligned + pageSize*(layer*max_tiles_x + x);
                req[x].physicalOffset = 0;
                req[x].size = pageSize;
                req[x].storageClass = 0;
            }
            memPoolVirtual->MapVirtual(max_tiles_x, req);
        }
        delete[] req;

        ////******************
        //// tear down
        tex->Free();

        delete textureView;

        // free virtual pool
        memPoolVirtual->Finalize();
        delete memPoolVirtual;

        // free physical pool
        memPoolPhys->Finalize();
        delete memPoolPhys;
        PoolStorageFree(poolAlloc);

        // free builder
        textureBuilder->Free();

        // mark test as pass
        setPass();
    }

    virtual bool getResult() {
        return _result;
    }
};

// OrderOfCreation below will first map the pages
// then create a texture which will exercise a code path
// in MapVirtual that would not be taken otherwise.
// Test that creating a virtual pool and mapping
// memory into it works even when we did not create
// a texture in it before.
// Create a virtual pool, create a texture in it to
// get a valid storage class. Then create another
// virtual pool and map physical memory with above
// storage class into it. Only now create a texture...
// If MapVirtual doesn't create a new virtual mapping
// with that storage class we will likely (CRASH? SEE CORRUPTION?)
// because there will be no proper mapping of the
// page kind required for the texture.
class OrderOfCreation : public Test
{
    void run() {
#if 0
        LWNdeviceData deviceData;
        deviceData.deviceMemory = NULL;
        deviceData.deviceMemorySize = 0;
        deviceData.flags = LWN_DEVICE_FLAG_DEBUG_ENABLE_LEVEL_2_BIT | LWN_DEVICE_FLAG_DEBUG_SKIP_CALLS_ON_ERROR_BIT;
        _device = new LWNdevice;
        TexIDPool* defaultTexIDPool = g_lwnTexIDPool;
        TexIDPool* idPool = NULL;
        utils::CompletionTracker *tracker = NULL;
        CommandBufferMemoryManager *cmdMem = NULL;
        if (!lwnDeviceInitialize(_device, &deviceData)) {
            return;
        }
        ReloadLWNEntryPoints(_device, true);
        _queue = lwnDeviceCreateQueue(_device);
        idPool = new TexIDPool(_device);
        tracker = new utils::CompletionTracker(_device, 32);
        _queueCB = QueueCommandBuffer();
        _queueCB.init(_device, _queue, tracker);
        cmdMem = new CommandBufferMemoryManager();
        if (cmdMem) {
            cmdMem->init(_device, tracker);
        }
        cmdMem->populateCommandBuffer(_queueCB, CommandBufferMemoryManager::Coherent);
        g_lwnTexIDPool = idPool;
        g_lwnTexIDPool->Bind(_queueCB);
#endif
        void* poolAlloc;
        MemoryPool *memPoolPhys = createPhysicalPool(0xFF00FF00, memSizePhys, &poolAlloc); // green
        if (!memPoolPhys) {
            return;
        }

        // create a virtual pool where we will map pages from the physical pool
        MemoryPool *memPoolVirtual = createVirtualPool();
        if (!memPoolVirtual) {
            return;
        }

        TextureBuilder *textureBuilder = _device->CreateTextureBuilder();
        textureBuilder->SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetFormat(Format::RGBA8).
            SetFlags(TextureFlags::COMPRESSIBLE);

        Texture *tex;
        textureBuilder->SetSize2D(texWidth, texHeight);

        MappingRequest req;

        // populate and expect color to be green because physical pool was all green
        req.physicalPool = memPoolPhys;
        req.virtualOffset = 0;
        req.physicalOffset = 0;
        req.size = memSizePhys;
        req.storageClass = textureBuilder->GetStorageClass();
        memPoolVirtual->MapVirtual(1, &req);

        tex = textureBuilder->CreateTextureFromPool(memPoolVirtual, 0);

        DrawWithTexture dwt(_fb);

        // now draw again with the texture, should be color of texture
        dwt.draw(tex);

        LWNuint pixel = 0xFFFFFFFF;
        read(_fb->getColorTexture(0), (void*)&pixel);
        if (pixel != 0xFF00FF00) {
            return;
        }

        // unmap
        req.physicalPool = NULL;
        req.virtualOffset = 0;
        req.physicalOffset = 0;
        req.size = memSizePhys;
        req.storageClass = 0;
        memPoolVirtual->MapVirtual(1, &req);

        ////******************
        //// tear down

        tex->Free();


        memPoolVirtual->Finalize();
        delete memPoolVirtual;

        // free physical pool
        memPoolPhys->Finalize();
        delete memPoolPhys;

        PoolStorageFree(poolAlloc);

        // free builder
        textureBuilder->Free();

        setPass();

#if 0
        g_lwnTexIDPool = defaultTexIDPool;

        // Render the results to screen.
        ReloadLWNEntryPoints(g_lwnDevice, false);

        delete idPool;
#endif
    }
};

// verify that getting a storage class
// from a texture builder works
class BuilderStorageClass : public Test
{
    void run() {
        TextureBuilder *textureBuilder = _device->CreateTextureBuilder();
        textureBuilder->SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetFormat(Format::RGBA8).
            SetFlags(TextureFlags::COMPRESSIBLE);

        Texture tex;
        textureBuilder->SetSize2D(texWidth, texHeight);

        LWNuint s = textureBuilder->GetStorageClass();

        // free builder
        textureBuilder->Free();

        if (0 != s) {
            setPass();
        }
    }
};

// verify that resources can't be allocated from a physical pool
class FailResourceCreationInPhysPool : public Test
{
    void run() {
        bool pass = true;

        if (lwnDebugEnabled) {
            setPass();
            return;
        }

        void* poolAlloc;
        MemoryPool *memPoolPhys = createPhysicalPool(0xDEADD00D, memSizePhys, &poolAlloc); 
        if (!memPoolPhys) {
            return;
        }

        TextureBuilder *textureBuilder = _device->CreateTextureBuilder();
        textureBuilder->SetDefaults().
            SetTarget(TextureTarget::TARGET_2D).
            SetFormat(Format::RGBA8).
            SetFlags(TextureFlags::COMPRESSIBLE).
            SetSize2D(1, 1);

        Texture *tex;
        tex = textureBuilder->CreateTextureFromPool(memPoolPhys, 0);

        // free builder
        textureBuilder->Free();

        if (NULL != tex) {
            pass = false;
        }

        BufferBuilder *bufferBuilder = _device->CreateBufferBuilder();
        bufferBuilder->SetDefaults();

        Buffer *buf;
        buf = bufferBuilder->CreateBufferFromPool(memPoolPhys, 0, memSizePhys);

        bufferBuilder->Free();

        if (NULL != buf) {
            pass = false;
        }

        if (pass) {
            setPass();
        }

        // free physical pool and memory
        memPoolPhys->Finalize();
        delete memPoolPhys;
        PoolStorageFree(poolAlloc);
    }
};

//
//### lwntest required
//
class LWNMemoryPoolVirtualTest
{
public:
    LWNTEST_CppMethods();

private:
    ct_assert(offscreenWidth == 128);
    ct_assert(offscreenHeight == 128);
    static const int cellSize = 160;
    static const int cellMargin = 16;
    static const int cellsX = 4;
    static const int cellsY = 3;

    void drawResult(CellIterator2D& cell, Texture *tex) const;
};

int LWNMemoryPoolVirtualTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 2);
}

lwString LWNMemoryPoolVirtualTest::getDescription() const
{
    return "Simple tests for virtual memory pools.\n";
}

// show result as red/green square
void LWNMemoryPoolVirtualTest::drawResult(CellIterator2D& cell, Texture *tex) const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *_device = deviceState->getDevice();
    QueueCommandBuffer &_queueCB = deviceState->getQueueCB();
    Queue *_queue = deviceState->getQueue();

    if (!tex) {
        return;
    }

    VertexShader vs(440);
    vs << vsString;

    FragmentShader fs(440);
    fs << fsTexture;

    Program *pgm = _device->CreateProgram();
    LWNboolean compiled = g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
    if (!compiled) {
        return;
    }

    // bind default FB
    g_lwnWindowFramebuffer.bind();

    // Set up the vertex format and buffer.
    struct Vertex {
        dt::vec4 position;
    };
    static const Vertex vertexData[] = {
        { dt::vec4(-1.0, -1.0, 0.0, 1.0) },
        { dt::vec4(-1.0, +1.0, 0.0, 1.0) },
        { dt::vec4(+1.0, -1.0, 0.0, 1.0) },
        { dt::vec4(+1.0, +1.0, 0.0, 1.0) },
    };

    MemoryPoolAllocator vertex_allocator(_device, NULL, sizeof(vertexData), LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    VertexStream stream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(stream, Vertex, position);
    VertexStreamSet streamSet(stream);
    VertexArrayState vertex = streamSet.CreateVertexArrayState();

    // vertex buffer
    Buffer *_vbo = stream.AllocateVertexBuffer(_device, 4, vertex_allocator, vertexData);

    // sampler
    SamplerBuilder sb;
    sb.SetDevice(_device).SetDefaults().SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    Sampler *_sampler = sb.CreateSampler();

    // combined texture/sampler handle
    LWNuint textureID = tex->GetRegisteredTextureID();
    TextureHandle texHandle = _device->GetTextureHandle(textureID, _sampler->GetRegisteredID());

    // bind vertex state, texture and sampler
    // scissor
    _queueCB.SetViewportScissor(cell.x() * cellSize + cellMargin, cell.y() * cellSize + cellMargin,
        cellSize - 2 * cellMargin, cellSize - 2 * cellMargin);
    LWNfloat red[] = { 1, 0, 0, 1 };
    _queueCB.ClearColor(0, red, ClearColorMask::RGBA);
    _queueCB.BindVertexArrayState(vertex);
    _queueCB.BindVertexBuffer(0, _vbo->GetAddress(), sizeof(vertexData));
    _queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
    _queueCB.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
    _queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
    _queueCB.submit();
    _queue->Finish();

    // advance in grid
    cell++;
}


void LWNMemoryPoolVirtualTest::doGraphics() const
{
    QueueCommandBuffer &queueCB = *g_lwnQueueCB;
    CellIterator2D cell(cellsX, cellsY);

    // clear default FB
    LWNfloat clearColor[] = { 0.5, 0.3, 0.3, 1 };
    queueCB.ClearColor(0, clearColor, ClearColorMask::RGBA);

    Test* tests[] = {
        new Simple(),
        new Simple2(),
        new Miplevels(),
        new Layers(),
        new OrderOfCreation(),
        new BuilderStorageClass(),
        new FailResourceCreationInPhysPool()
    };

    for (unsigned int t = 0; t < __GL_ARRAYSIZE(tests); t++) {
        // draw test result to default framebuffer
        drawResult(cell, tests[t]->execute());
        delete tests[t];
    }
}

// ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

OGTEST_CppTest(LWNMemoryPoolVirtualTest, lwn_mempool_virtual, );


//////////////////////////////////////////////////////////////////////////

class LWNMemoryPoolVirtualMixTest
{
    // We render 64x64 views into our virtual textures with a 2-pixel margin.
    // The virtual textures themselves are larger, so we only display selected
    // portions.
    static const int cellSize = 68;
    static const int cellMargin = 2;
    static const int cellsX = 9;
    static const int cellsY = 7;

    // We allocate a physical pool that has enough memory to store
    // <physicalVersions> textures at once (each with space enough for 128-bit
    // texels).  We allocate a larger virtual pool that has space for up to
    // <maxVirtualMappings> separate mappings of the entire physical pool.
    // Each of these mappings will be used for a separate storage class.
    //
    // The test will repeatedly allocate new textures out of the physical
    // memory, and will choose an appropriate virtual range based on the
    // texture's storage class.  This "allocator" will insert GPU barriers to
    // ensure old uses of the memory have finished before new ones start, but
    // will not wait for completion on the CPU.  Because we create the pool
    // mappings up front and select appropriate virtual ranges, there isn't any
    // need to mess with mappings during the rendering loop.
    static const int texSize = 256;
    static const int physicalVersions = 3;
    static const int physicalVersionSize = texSize * texSize * 16;
    static const int maxVirtualMappings = 3;
    static const int physicalPoolSize = physicalVersions * physicalVersionSize;
    static const int virtualPoolSize = maxVirtualMappings * physicalPoolSize;

    bool m_compressible;
public:
    LWNMemoryPoolVirtualMixTest(bool compressible) : m_compressible(compressible) {}
    LWNTEST_CppMethods();
};

lwString LWNMemoryPoolVirtualMixTest::getDescription() const
{
    lwStringBuf sb;
    sb << 
        "Tests 'on the fly' re-allocations of textures with potentially "
        "different storage classes, where setting up new textures doesn't "
        "wait on the CPU for completion of previous textures using the "
        "same memory.  Instead, we use a separate virtual mapping of the "
        "pool memory for each storage class."
        "\n\n"
        "This test loops over the physical memory, setting up new textures "
        "of rotating formats, clearing and rendering to them, and then "
        "displaying them on-screen.  Each cell displays portions of these "
        "textures, with triangles in the corners showing blue, red, cyan, "
        "and yellow (from the corner of the textures).  The rest shows the "
        "center of the texture, showing a yellow vertical line and a black "
        "horizontal one.  All cells should be roughly identical, except for "
        "differences in color depth between the textures."
        "\n\n"
        "This test uses " << (m_compressible ? "" : "non-") << "compressible "
        "textures.";
    return sb.str();    
}

int LWNMemoryPoolVirtualMixTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(40, 2);
}

void LWNMemoryPoolVirtualMixTest::doGraphics() const
{
    CellIterator2D cell(cellsX, cellsY);
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // The color program renders based on interpolated color, but overlays
    // black and yellow "lines" near the center of the texture.
    VertexShader colorVS(440);
    colorVS <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec3 color;\n"
        "out vec3 ocolor;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  ocolor = color;\n"
        "}\n";

    FragmentShader colorFS(440);
    colorFS <<
        "in vec3 ocolor;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = vec4(ocolor, 1.0);\n"
        "  ivec2 delta = abs(ivec2(gl_FragCoord.xy) - ivec2(" << texSize / 2 << "));\n"
        "  if (delta.x < 4.0 && delta.y < 20.0) {\n"
        "    fcolor = vec4(1.0, 1.0, 0.0, 1.0);\n"
        "  }\n"
        "  if (delta.x < 12.0 && delta.y < 3.0) {\n"
        "    fcolor = vec4(0.0, 0.0, 0.0, 1.0);\n"
        "  }\n"
        "}\n";

    Program *colorProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(colorProgram, colorVS, colorFS)) {
        LWNFailTest();
        return;
    }

    // The texture program simply displays portions of the texture based on
    // interpolated texture coordinates.
    VertexShader textureVS(440);
    textureVS <<
        "layout(location=0) in vec3 position;\n"
        "layout(location=1) in vec2 texcoord;\n"
        "out vec2 tc;\n"
        "void main() {\n"
        "  gl_Position = vec4(position, 1.0);\n"
        "  tc = texcoord;\n"
        "}\n";

    FragmentShader textureFS(440);
    textureFS <<
        "layout(binding = 0) uniform sampler2D tex;\n"
        "in vec2 tc;\n"
        "out vec4 fcolor;\n"
        "void main() {\n"
        "  fcolor = texture(tex, tc);\n"
        "}\n";

    Program *textureProgram = device->CreateProgram();
    if (!g_glslcHelper->CompileAndSetShaders(textureProgram, textureVS, textureFS)) {
        LWNFailTest();
        return;
    }

    // The color phase draws a full-screen shaded quad.
    struct ColorVertex {
        dt::vec3 position;
        dt::vec3 color;
    };
    static const ColorVertex colorVertices[] = {
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec3(0.0, 0.0, 1.0) },
        { dt::vec3(-1.0, +1.0, 0.0), dt::vec3(0.0, 1.0, 1.0) },
        { dt::vec3(+1.0, -1.0, 0.0), dt::vec3(1.0, 0.0, 0.0) },
        { dt::vec3(+1.0, +1.0, 0.0), dt::vec3(1.0, 1.0, 0.0) },
    };

    // The texture phase draws portions of the texture with texture
    // coordinates set up to have a 1:1 pixel/texel relationship.
    static const float magnify = float(cellSize - 2 * cellMargin) / float(texSize);
    struct TextureVertex {
        dt::vec3 position;
        dt::vec2 texcoord;
    };
    static const TextureVertex textureVertices[] = {

        // We start with a full-cell quad with texture coordinates centered
        // around (0.5, 0.5).
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec2(0.5 - 0.5 * magnify, 0.5 - 0.5 * magnify) },
        { dt::vec3(-1.0, +1.0, 0.0), dt::vec2(0.5 - 0.5 * magnify, 0.5 + 0.5 * magnify) },
        { dt::vec3(+1.0, -1.0, 0.0), dt::vec2(0.5 + 0.5 * magnify, 0.5 - 0.5 * magnify) },
        { dt::vec3(+1.0, +1.0, 0.0), dt::vec2(0.5 + 0.5 * magnify, 0.5 + 0.5 * magnify) },

        // Then, we draw four triangles in the corners of the viewport with
        // texture coordinates aligned with corresponding corners of the
        // texture.
        { dt::vec3(-1.0, -1.0, 0.0), dt::vec2(0.0 + 0.00 * magnify, 0.0 + 0.00 * magnify) },
        { dt::vec3(-0.5, -1.0, 0.0), dt::vec2(0.0 + 0.25 * magnify, 0.0 + 0.00 * magnify) },
        { dt::vec3(-1.0, -0.5, 0.0), dt::vec2(0.0 + 0.00 * magnify, 0.0 + 0.25 * magnify) },

        { dt::vec3(+1.0, -1.0, 0.0), dt::vec2(1.0 - 0.00 * magnify, 0.0 + 0.00 * magnify) },
        { dt::vec3(+0.5, -1.0, 0.0), dt::vec2(1.0 - 0.25 * magnify, 0.0 + 0.00 * magnify) },
        { dt::vec3(+1.0, -0.5, 0.0), dt::vec2(1.0 - 0.00 * magnify, 0.0 + 0.25 * magnify) },

        { dt::vec3(-1.0, +1.0, 0.0), dt::vec2(0.0 + 0.00 * magnify, 1.0 - 0.00 * magnify) },
        { dt::vec3(-0.5, +1.0, 0.0), dt::vec2(0.0 + 0.25 * magnify, 1.0 - 0.00 * magnify) },
        { dt::vec3(-1.0, +0.5, 0.0), dt::vec2(0.0 + 0.00 * magnify, 1.0 - 0.25 * magnify) },

        { dt::vec3(+1.0, +1.0, 0.0), dt::vec2(1.0 - 0.00 * magnify, 1.0 - 0.00 * magnify) },
        { dt::vec3(+0.5, +1.0, 0.0), dt::vec2(1.0 - 0.25 * magnify, 1.0 - 0.00 * magnify) },
        { dt::vec3(+1.0, +0.5, 0.0), dt::vec2(1.0 - 0.00 * magnify, 1.0 - 0.25 * magnify) },

    };

    VertexStream colorStream(sizeof(ColorVertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(colorStream, ColorVertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(colorStream, ColorVertex, color);
    VertexArrayState colorVertexState = colorStream.CreateVertexArrayState();

    VertexStream textureStream(sizeof(TextureVertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(textureStream, TextureVertex, position);
    LWN_VERTEX_STREAM_ADD_MEMBER(textureStream, TextureVertex, texcoord);
    VertexArrayState textureVertexState = textureStream.CreateVertexArrayState();

    // Create a CPU-uncached memory pool filled with our vertex data.
    MemoryPool *vertexPool = device->CreateMemoryPoolWithFlags(NULL, sizeof(colorVertices) + sizeof(textureVertices),
                                                               MemoryPoolFlags::CPU_UNCACHED | MemoryPoolFlags::GPU_CACHED);
    char *mappedVertexPool = (char *) vertexPool->Map();
    memcpy(mappedVertexPool, colorVertices, sizeof(colorVertices));
    memcpy(mappedVertexPool + sizeof(colorVertices), textureVertices, sizeof(textureVertices));
    BufferAddress colorVertexAddress = vertexPool->GetBufferAddress();
    BufferAddress textureVertexAddress = colorVertexAddress + sizeof(colorVertices);

    // Create our physical and virtual pools for the test.
    MemoryPool *physicalPool = device->CreateMemoryPoolWithFlags(NULL, physicalPoolSize, 
                                                                 (MemoryPoolFlags::CPU_NO_ACCESS |
                                                                  MemoryPoolFlags::GPU_NO_ACCESS |
                                                                  MemoryPoolFlags::PHYSICAL |
                                                                  MemoryPoolFlags::COMPRESSIBLE));
    MemoryPool *virtualPool = device->CreateMemoryPoolWithFlags(NULL, virtualPoolSize,
                                                                (MemoryPoolFlags::CPU_NO_ACCESS |
                                                                 MemoryPoolFlags::GPU_CACHED |
                                                                 MemoryPoolFlags::VIRTUAL |
                                                                 MemoryPoolFlags::COMPRESSIBLE));

    // Set up a basic tetxure builder we'll used for subsequent queries and
    // texture creation.
    TextureBuilder texBuilder;
    texBuilder.SetDevice(device);
    texBuilder.SetDefaults();
    texBuilder.SetTarget(TextureTarget::TARGET_2D);
    texBuilder.SetSize2D(texSize, texSize);
    if (m_compressible) {
        texBuilder.SetFlags(TextureFlags::COMPRESSIBLE);
    }

    // Set up a sampler to use for rendering.
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults();
    Sampler *smp = sb.CreateSampler();

    // Our test loops over a variety of different texture formats.  Extract
    // information (including storage class) for each of the formats, and
    // prepare virtual mapping requests for each unique storage class.
    static const Format formats[] = { 
        Format::RGBA8,
        Format::RGB10A2,
        Format::RGBA16F,
        Format::RGBA32F,
    };
    static const int nFormats = int(__GL_ARRAYSIZE(formats));
    struct FormatInfo {
        LWNstorageClass storageClass;
        int virtualPoolIndex;
    };
    FormatInfo formatInfo[nFormats];
    MappingRequest mappingRequests[nFormats];
    int nMappings = 0;

    for (int i = 0; i < nFormats; i++) {
        FormatInfo *fi = formatInfo + i;
        texBuilder.SetFormat(formats[i]);
        fi->storageClass = texBuilder.GetStorageClass();
        fi->virtualPoolIndex = -1;
        for (int j = 0; j < i; j++) {
            if (fi->storageClass == formatInfo[j].storageClass) {
                fi->virtualPoolIndex = formatInfo[j].virtualPoolIndex;
                break;
            }
        }
        if (fi->virtualPoolIndex < 0) {
            // Whenever we find a new storage class, prepare a new whole-pool
            // mapping request.
            if (nMappings >= maxVirtualMappings) {
                LWNFailTest();
                return;
            }
            MappingRequest *request = mappingRequests + nMappings;
            request->physicalPool = physicalPool;
            request->physicalOffset = 0;
            request->virtualOffset = nMappings * physicalPoolSize;
            request->size = physicalPoolSize;
            request->storageClass = fi->storageClass;
            fi->virtualPoolIndex = nMappings;
            nMappings++;
        }
    }
    virtualPool->MapVirtual(nMappings, mappingRequests);

    queueCB.ClearColor(0, 0.4, 0.0, 0.0, 0.0);

    // Loop over all of our cells.  For each cell, we "advance" to the next
    // format and next offset in the physical pool.
    for (int i = 0; i < cellsX * cellsY; i++) {
        int formatIndex = i % nFormats;
        int physicalStorageIndex = i % physicalVersions;
        FormatInfo *fi = formatInfo + formatIndex;

        // Set up a new texture at the appropriate offset in the physical
        // storage, using the virtual mapping of the physical pool for the
        // texture's storage class.
        LWNuint offset = (fi->virtualPoolIndex * physicalPoolSize +
                          physicalStorageIndex * physicalVersionSize);
        texBuilder.SetFormat(formats[formatIndex]);
        Texture *tex = texBuilder.CreateTextureFromPool(virtualPool, offset);
        TextureHandle texHandle = device->GetTextureHandle(tex->GetRegisteredTextureID(), smp->GetRegisteredID());

        // Start by clearing the new texture and then using the color program
        // and state to fill in its contents.
        static const LWNfloat black[] = { 0, 0, 0, 0 };
        queueCB.SetRenderTargets(1, &tex, NULL, NULL, NULL);
        queueCB.SetViewportScissor(0, 0, texSize, texSize);
        queueCB.ClearColor(0, black, ClearColorMask::RGBA);
        queueCB.BindProgram(colorProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindVertexArrayState(colorVertexState);
        queueCB.BindVertexBuffer(0, colorVertexAddress, sizeof(colorVertices));
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);

        // Insert a render-to-texture barrier.
        queueCB.Barrier(BarrierBits::ORDER_FRAGMENTS | BarrierBits::ILWALIDATE_TEXTURE);

        // Now use the texture program and state to display the contents of
        // the new texture.
        g_lwnWindowFramebuffer.bind();
        queueCB.SetViewportScissor(cell.x() * cellSize + cellMargin,
                                   cell.y() * cellSize + cellMargin,
                                   cellSize - 2 * cellMargin, cellSize - 2* cellMargin);
        queueCB.BindProgram(textureProgram, ShaderStageBits::ALL_GRAPHICS_BITS);
        queueCB.BindVertexArrayState(textureVertexState);
        queueCB.BindVertexBuffer(0, textureVertexAddress, sizeof(textureVertices));
        queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLE_STRIP, 0, 4);
        queueCB.DrawArrays(DrawPrimitive::TRIANGLES, 4, 12);

        // We would potentially need a barrier here to ensure that we don't
        // try to allocate a new texture in the memory used by the texture
        // program above.  It isn't needed because the next loop will never
        // reuse the same memory and its render-to-texture barrier will ensure
        // that the texture program will be done then.

        cell++;
    }

    queueCB.submit();

    // we need to make sure everything is done because
    // the MemoryAllocator destructor will free its memory
    // pool immediately w/o sync.
    queue->Finish();
}

OGTEST_CppTest(LWNMemoryPoolVirtualMixTest, lwn_mempool_virtual_mix, (false));
OGTEST_CppTest(LWNMemoryPoolVirtualMixTest, lwn_mempool_virtual_mix_compr, (true));
