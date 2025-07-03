/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnUtil_PoolAllocator_h__
#define __lwnUtil_PoolAllocator_h__

#include "lwnUtil_Interface.h"

#include "lwn/lwn.h"
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
#include "lwn/lwn_Cpp.h"
#endif

#include <algorithm>
#include <list>
#include <map>

namespace lwnUtil {

//////////////////////////////////////////////////////////////////////////
//
//                      memory pool allocator
//
// minimal allocator for blocks from memory pool
class MemoryPoolSubAllocator;

// Flags indicating how a buffer resource will be used, which will affect the
// required alignment.
enum BufferAlignBits {
    BUFFER_ALIGN_UNIFORM_BIT = 0x00000001,
    BUFFER_ALIGN_SHADER_STORAGE_BIT = 0x00000002,
    BUFFER_ALIGN_TEXTURE_BIT = 0x00000004,
    BUFFER_ALIGN_COUNTER_BIT = 0x00000008,
    BUFFER_ALIGN_VERTEX_BIT = 0x00000010,
    BUFFER_ALIGN_INDEX_BIT = 0x00000020,
    BUFFER_ALIGN_INDIRECT_BIT = 0x00000040,
    BUFFER_ALIGN_TRANSFORM_FEEDBACK_BIT = 0x00000080,
    BUFFER_ALIGN_COPY_READ_BIT = 0x00000100,
    BUFFER_ALIGN_COPY_WRITE_BIT = 0x00000200,
    BUFFER_ALIGN_ZLWLL_SAVE_BIT = 0x00000400,
    BUFFER_ALIGN_IMAGE_BIT = 0x00000800,
    BUFFER_ALIGN_TRANSFORM_FEEDBACK_CONTROL_BIT = 0x00001000,
    BUFFER_ALIGN_NONE = 0x00000000,
};

class MemoryPoolAllocator {
    friend class DebugMemoryPoolAllocator;

public:
    MemoryPoolAllocator(LWNdevice* device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags)
    {
        init(device, memory, size, poolFlags);
    }
    virtual ~MemoryPoolAllocator();

    void                init(LWNdevice* device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags);

    LWNmemoryPool*      pool(LWNtexture* texture);
    LWNmemoryPool*      pool(LWNbuffer* buffer);

    LWNtexture*         allocTexture(LWNtextureBuilder* builder);
    void                freeTexture(LWNtexture* texture);
    ptrdiff_t           offset(LWNtexture* texture);
    LWNbuffer*          allocBuffer(LWNbufferBuilder* builder, BufferAlignBits alignBits, size_t size);
    void                freeBuffer(LWNbuffer* buffer);
    ptrdiff_t           offset(LWNbuffer* buffer);
    size_t              size(LWNbuffer* buffer);

    void                discard();
    size_t              numAllocs();

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    MemoryPoolAllocator(lwn::Device* device, void *memory, size_t size, lwn::MemoryPoolFlags poolFlags)
    {
        LWNdevice *cdev = reinterpret_cast<LWNdevice *>(device);
        LWNmemoryPoolFlags cflags = LWNmemoryPoolFlags(int(poolFlags));
        init(cdev, memory, size, cflags);
    }
    lwn::MemoryPool*    pool(lwn::Texture* texture)
    {
        LWNtexture *ctex = reinterpret_cast<LWNtexture *>(texture);
        LWNmemoryPool *cpool = pool(ctex);
        return reinterpret_cast<lwn::MemoryPool *>(cpool);
    }
    lwn::MemoryPool*    pool(lwn::Buffer* buffer)
    {
        LWNbuffer *cbuf = reinterpret_cast<LWNbuffer *>(buffer);
        LWNmemoryPool *cpool = pool(cbuf);
        return reinterpret_cast<lwn::MemoryPool *>(cpool);
    }
    lwn::Texture*       allocTexture(lwn::TextureBuilder* builder)
    {
        LWNtextureBuilder *cbuilder = reinterpret_cast<LWNtextureBuilder *>(builder);
        LWNtexture *ctex = allocTexture(cbuilder);
        return reinterpret_cast<lwn::Texture *>(ctex);
    }
    void                freeTexture(lwn::Texture* texture)
    {
        LWNtexture *ctex = reinterpret_cast<LWNtexture *>(texture);
        freeTexture(ctex);
    }
    ptrdiff_t           offset(lwn::Texture* texture)
    {
        LWNtexture *ctex = reinterpret_cast<LWNtexture *>(texture);
        return offset(ctex);
    }
    lwn::Buffer*        allocBuffer(lwn::BufferBuilder* builder, BufferAlignBits alignBits, size_t size)
    {
        LWNbufferBuilder *cbuilder = reinterpret_cast<LWNbufferBuilder *>(builder);
        LWNbuffer *cbuf = allocBuffer(cbuilder, alignBits, size);
        return reinterpret_cast<lwn::Buffer *>(cbuf);
    }
    void                freeBuffer(lwn::Buffer* buffer)
    {
        LWNbuffer *cbuf = reinterpret_cast<LWNbuffer *>(buffer);
        freeBuffer(cbuf);
    }
    ptrdiff_t           offset(lwn::Buffer* buffer)
    {
        LWNbuffer *cbuf = reinterpret_cast<LWNbuffer *>(buffer);
        return offset(cbuf);
    }
    size_t              size(lwn::Buffer* buffer)
    {
        LWNbuffer *cbuf = reinterpret_cast<LWNbuffer *>(buffer);
        return size(cbuf);
    }
#endif

private:
    typedef std::list<MemoryPoolSubAllocator*> MemoryPoolList;
    typedef std::map<LWNbuffer*, MemoryPoolSubAllocator*> MemoryAllocatorBufferMap;
    typedef std::map<LWNtexture*, MemoryPoolSubAllocator*> MemoryAllocatorTextureMap;

private:
    void                reset();
    void                initAllocator();
    void                doHouseKeeping(MemoryPoolSubAllocator* suballocator);
    size_t              pickPoolSize(size_t size);

private:
    MemoryPoolList            _poolList;
    MemoryAllocatorBufferMap  _allocatorBufferMap;
    MemoryAllocatorTextureMap _allocatorTextureMap;

    LWNdevice*          _device;
    void*               _memory;
    LWNmemoryPoolFlags  _poolFlags;
    size_t              _size;
    size_t              _numAllocs;
};

class MemoryPoolSubAllocator {
    friend class MemoryPoolAllocator;
    friend class DebugMemoryPoolSubAllocator;

    class MemoryPoolBlock {
    public:
        MemoryPoolBlock()
            : _offset(0),
            _size(0),
            _align(0),
            _allocator(NULL)
        {}

        MemoryPoolBlock(ptrdiff_t offset, size_t size, MemoryPoolSubAllocator* allocator = NULL, size_t align = 0);

        ptrdiff_t offset() const {
            return _offset;
        }

        ptrdiff_t aligned_offset() const {
            return _align ? (_offset + (_align - 1)) & ~(_align - 1) : _offset;
        }

        size_t size() const {
            return _size;
        }

        size_t align() const {
            return _align;
        }

        void setOffset(ptrdiff_t offset) {
            _offset = offset;
        }

        void setSize(size_t size){
            _size = size;
        }

        bool operator==(MemoryPoolBlock& lhs) {
            return (lhs.offset() == _offset && lhs.size() == _size);
        }

        operator LWNboolean () const { return valid(); }

        MemoryPoolSubAllocator* allocator() const {
            return _allocator;
        }

        void freeBlock();

        MemoryPoolBlock intersects(const MemoryPoolBlock& other) {
            ptrdiff_t minimum = (std::max)(_offset, other.offset());
            ptrdiff_t maximum = (std::min)(_offset + _size, other.offset() + other.size());

            size_t size = (maximum > minimum) ? size_t(maximum - minimum) : 0;

            return MemoryPoolBlock(minimum, size);
        }

    private:
        LWNboolean valid() const{
            return (_size > 0);
        }

        ptrdiff_t               _offset;
        size_t                  _size;
        size_t                  _align;

        MemoryPoolSubAllocator*    _allocator;
    };

    struct OffsetGreater {
        bool operator()(const MemoryPoolSubAllocator::MemoryPoolBlock &a, const MemoryPoolSubAllocator::MemoryPoolBlock& b) const {
            return a.offset() < b.offset();
        }
    };

private:
    MemoryPoolSubAllocator(LWNdevice* device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags);
    virtual ~MemoryPoolSubAllocator();

public:
    LWNmemoryPool*          pool();
    LWNtexture*             allocTexture(LWNtextureBuilder* builder);
    void                    freeTexture(LWNtexture* texture);
    ptrdiff_t               offset(LWNtexture* texture);
    LWNbuffer*              allocBuffer(LWNbufferBuilder* builder, BufferAlignBits alignBits, size_t size);
    void                    freeBuffer(LWNbuffer* buffer);
    ptrdiff_t               offset(LWNbuffer* buffer);
    size_t                  size(LWNbuffer* buffer);

    void                    discard();

private:
    typedef std::map<ptrdiff_t, MemoryPoolBlock> MemoryOffsetPoolBlockMap;
    typedef std::map<LWNtexture*, MemoryPoolBlock> MemoryPoolTextureMap;
    typedef std::map<LWNbuffer*, MemoryPoolBlock> MemoryPoolBufferMap;

    bool                    initPool(void);
    MemoryPoolBlock         allocBlock(size_t size, size_t align);
    void                    freeBlock(MemoryPoolBlock block);
    MemoryOffsetPoolBlockMap::iterator prev(MemoryOffsetPoolBlockMap::iterator it);
    void                    reset();
    size_t                  bufferAlignment(BufferAlignBits alignBits) const;

    bool                    contains(LWNtexture* texture);
    bool                    contains(LWNbuffer* buffer);

    bool                    _wasUsed; // true if anything was ever allocated out of this pool (for house keeping)

    LWNdevice*              _device;
    void*                   _memory;
    bool                    _memoryIsOwned;
    LWNmemoryPoolFlags      _poolFlags;
    LWNmemoryPool           * _pool;
    size_t                  _size;
    size_t                  _numAllocs;
    size_t                  _free;
    size_t                  _used;

    MemoryOffsetPoolBlockMap _freeOffsetMap;
    MemoryOffsetPoolBlockMap _usedMap;

    MemoryPoolTextureMap    _textureMap;
    MemoryPoolBufferMap     _bufferMap;
};

} // namespace lwnUtil

#endif // #ifndef __lwnUtil_PoolAllocator_h__
