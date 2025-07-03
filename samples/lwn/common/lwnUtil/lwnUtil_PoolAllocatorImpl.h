/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnUtil_PoolAllocatorImpl_h__
#define __lwnUtil_PoolAllocatorImpl_h__

#include <lwn/lwn.h>
#include <lwn/lwn_FuncPtr.h>

#include <assert.h>

#include "lwnUtil_AlignedStorage.h"
#include "lwnUtil_PoolAllocator.h"

// These are defined for the samples in lwnutils.h, and for lwntest in lwn_utils.h.
// This file ultimately needs to include one consolidated utils header file to use, and these
// externs can then be removed.
extern void lwnTextureFree(LWNtexture *);
extern void lwnBufferFree(LWNbuffer *object);
extern void lwnMemoryPoolFree(LWNmemoryPool *object);
LWNmemoryPool *lwnDeviceCreateMemoryPool(LWNdevice *device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags);
extern LWNtexture *lwnTextureBuilderCreateTextureFromPool(LWNtextureBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset);
extern LWNbuffer *lwnBufferBuilderCreateBufferFromPool(LWNbufferBuilder *builder, LWNmemoryPool *storage, ptrdiff_t offset, size_t size);


//////////////////////////////////////////////////////////////////////////

namespace lwnUtil {

void MemoryPoolAllocator::init(LWNdevice* device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags)
{
    _device = device;
    _memory = memory;
    _poolFlags = poolFlags;
    _size = size;
    reset();
}

MemoryPoolAllocator::~MemoryPoolAllocator()
{
    for (MemoryPoolList::iterator it = _poolList.begin(); it != _poolList.end(); it++) {
        MemoryPoolSubAllocator* allocator = (*it);
        delete allocator;
    }

    reset();
}

void MemoryPoolAllocator::initAllocator() {
    // init the pool list with one fixed sized pool (that could also
    // have a prebaked asset)
    // most tests will have an idea of how much memory they will need
    // but you could also just have an allocator with zero size
    // and it will manage the memory by itself (see pickPoolSize)
    if (_size && _poolList.empty()) {
        // pick a size exactly based on size and alignment of LWNtexture*
        MemoryPoolSubAllocator* pool = new MemoryPoolSubAllocator(_device, _memory, _size, _poolFlags);
        if (!pool) {
            // we did not get a new pool to satisfy our request. This is fatal!
            assert(0);
        }
        _poolList.push_back(pool);
        // never use _memory twice
        _memory = NULL;
    }
}

// clean up unused pools in pool list
void MemoryPoolAllocator::doHouseKeeping(MemoryPoolSubAllocator* suballocator) {

    if (suballocator->_numAllocs) {
        return;
    }
    MemoryPoolList::iterator it = _poolList.begin();
    while (it != _poolList.end()) {
        MemoryPoolSubAllocator* allocator = (*it);

        // sub allocator has no allocations and was never used,
        // we can drop it entirely
        // if it was used we can't know if there are resources
        // still in flight so we need to wait until the client
        // intentionally destroys the pool to avoid hangs
        if (!allocator->_numAllocs && !allocator->_wasUsed) {
            it = _poolList.erase(it);
            delete allocator;
        } else {
            it++;
        }
    }
}

// if we request more memory than an initial allocation specified through the
// constructor can handle, we'll create additional new pools.  We allocate new
// pools that are at least as large of the initial one (to avoid creating many
// separate pools for small allocations when we overflow).
static const size_t minPoolSize = LWN_MEMORY_POOL_STORAGE_ALIGNMENT;
size_t MemoryPoolAllocator::pickPoolSize(size_t size) {
    size_t actualSize;
    actualSize = std::max(size, _size);
    actualSize = (actualSize + minPoolSize - 1) & ~(minPoolSize - 1);
    return actualSize;
}

LWNmemoryPool* MemoryPoolAllocator::pool(LWNtexture* texture) {
    initAllocator();

    if (_poolList.empty()) {
        assert(0);
        return NULL;
    }

    MemoryPoolSubAllocator* allocator = _allocatorTextureMap[texture];
    if (allocator) {
        return allocator->pool();
    }
    assert(0);
    return NULL;
}

LWNmemoryPool* MemoryPoolAllocator::pool(LWNbuffer* buffer) {
    initAllocator();

    if (_poolList.empty()) {
        assert(0);
        return NULL;
    }

    MemoryPoolSubAllocator* allocator = _allocatorBufferMap[buffer];
    if (allocator) {
        return allocator->pool();
    }
    assert(0);
    return NULL;
}


// allocate a LWNtexture* from the memory pool
LWNtexture* MemoryPoolAllocator::allocTexture(LWNtextureBuilder* builder)
{
    initAllocator();

    int retries = 0;
    LWNtexture* tex = NULL;
    do {
        // try to find a pool
        for (MemoryPoolList::iterator it = _poolList.begin(); it != _poolList.end(); it++) {
            MemoryPoolSubAllocator* allocator = (*it);

            tex = allocator->allocTexture(builder);
            if (tex) {
                _allocatorTextureMap[tex] = allocator;
                break;
            }
        }

        assert(retries <= 1);

        // create a new pool
        if (!tex) {
            size_t tex_size = lwnTextureBuilderGetStorageSize(builder);
            size_t tex_align = lwnTextureBuilderGetStorageAlignment(builder);
            size_t size = pickPoolSize(tex_size + tex_align);

            // pick a size exactly based on size and alignment of LWNtexture*
            MemoryPoolSubAllocator* pool = new MemoryPoolSubAllocator(_device, NULL, size, _poolFlags);
            if (!pool) {
                // we did not get a new pool to satisfy our request. This is fatal!
                assert(0);
                return NULL;
            }
            // front because more likely to find free space in pools at front of list
            _poolList.push_front(pool);

            retries++;
        }
    } while (!tex);
    return tex;
}

void MemoryPoolAllocator::freeTexture(LWNtexture* texture)
{
    MemoryPoolSubAllocator* allocator = _allocatorTextureMap[texture];
    if (allocator) {
        allocator->freeTexture(texture);
        _allocatorTextureMap.erase(texture);
        doHouseKeeping(allocator);
    }
}

ptrdiff_t MemoryPoolAllocator::offset(LWNtexture* texture)
{
    MemoryPoolSubAllocator* allocator = _allocatorTextureMap[texture];
    if (allocator) {
        return allocator->offset(texture);
    }
    assert(0);
    return 0;
}

LWNbuffer* MemoryPoolAllocator::allocBuffer(LWNbufferBuilder* builder, BufferAlignBits alignBits, size_t size)
{
    if (!size) {
        return NULL;
    }
    initAllocator();

    int retries = 0;
    LWNbuffer* buffer = NULL;
    do {
        for (MemoryPoolList::iterator it = _poolList.begin(); it != _poolList.end(); it++) {
            MemoryPoolSubAllocator* allocator = (*it);

            buffer = allocator->allocBuffer(builder, alignBits, size);
            if (buffer) {
                _allocatorBufferMap[buffer] = allocator;
                break;
            }
        }

        assert(retries <= 1);

        // create a new pool
        if (!buffer) {
            // pick a size exactly based on size of LWNbuffer*
            MemoryPoolSubAllocator* pool = new MemoryPoolSubAllocator(_device, NULL, pickPoolSize(size), _poolFlags);
            if (!pool) {
                // we did not get a new pool to satisfy our request. This is fatal!
                assert(0);
                return NULL;
            }
            _poolList.push_front(pool);

            retries++;
        }
    } while (!buffer);
    return buffer;
}

void MemoryPoolAllocator::freeBuffer(LWNbuffer* buffer)
{
    MemoryPoolSubAllocator* allocator = _allocatorBufferMap[buffer];
    if (allocator) {
        allocator->freeBuffer(buffer);
        _allocatorBufferMap.erase(buffer);
        doHouseKeeping(allocator);
    }
}

ptrdiff_t MemoryPoolAllocator::offset(LWNbuffer* buffer)
{
    MemoryPoolSubAllocator* allocator = _allocatorBufferMap[buffer];
    if (allocator) {
        return allocator->offset(buffer);
    }
    assert(0);
    return 0;
}

size_t MemoryPoolAllocator::size(LWNbuffer* buffer)
{
    MemoryPoolSubAllocator* allocator = _allocatorBufferMap[buffer];
    if (allocator) {
        return allocator->size(buffer);
    }
    assert(0);
    return 0;
}

void MemoryPoolAllocator::reset() {
    _numAllocs = 0;

    // initially whole pool is free
    _poolList.clear();
    _allocatorTextureMap.clear();
    _allocatorBufferMap.clear();
}


void MemoryPoolAllocator::discard() {
    for (MemoryPoolList::iterator it = _poolList.begin(); it != _poolList.end(); it++) {
        MemoryPoolSubAllocator* allocator = (*it);
        allocator->discard();
    }
    reset();
}

size_t MemoryPoolAllocator::numAllocs() {
    size_t numAllocs = 0;
    for (MemoryPoolList::iterator it = _poolList.begin(); it != _poolList.end(); it++) {
        MemoryPoolSubAllocator* allocator = (*it);
        numAllocs += allocator->_numAllocs;
    }
    return numAllocs;
}

// backend allocator
MemoryPoolSubAllocator::MemoryPoolBlock::MemoryPoolBlock(ptrdiff_t offset, size_t size, MemoryPoolSubAllocator* allocator, size_t align)
: _offset(offset),
  _size(size),
  _align(align),
  _allocator(allocator)
{

}

void MemoryPoolSubAllocator::MemoryPoolBlock::freeBlock() {
    if (!valid()) {
        return;
    }
    assert(_allocator);
    _allocator->freeBlock(*this);
}

MemoryPoolSubAllocator::MemoryPoolSubAllocator(LWNdevice* device, void *memory, size_t size, LWNmemoryPoolFlags poolFlags)
: _device(device),
  _memory(memory),
  _memoryIsOwned(false),
  _poolFlags(poolFlags),
  _size(size)
{
    if (!memory) {
        // If no memory was provided, create our own. Alignment requirements might mean that we
        // store more memory than was requested.
        _memoryIsOwned = true;
        _size = PoolStorageSize(_size);
        _memory = PoolStorageAlloc(_size);
    }
    // initially whole pool is free
    // and nothing is used
    reset();
}

MemoryPoolSubAllocator::~MemoryPoolSubAllocator()
{
    // free all textures
    for (MemoryPoolTextureMap::iterator it = _textureMap.begin(); it != _textureMap.end(); it++) {
        LWNtexture* texture = it->first;
        MemoryPoolBlock block = it->second;
        block.freeBlock();
        lwnTextureFree(texture);
    }

    // free all buffers
    for (MemoryPoolBufferMap::iterator it = _bufferMap.begin(); it != _bufferMap.end(); it++) {
        LWNbuffer* buffer = it->first;
        MemoryPoolBlock block = it->second;
        block.freeBlock();
        lwnBufferFree(buffer);
    }

    // check that everything has been deallocated
    assert(_freeOffsetMap.size() == 1 && _freeOffsetMap[0].offset() == 0 && _freeOffsetMap[0].size() == _size && _numAllocs == 0);

    if (_pool) {
        lwnMemoryPoolFree(_pool);
    }

    if (_memoryIsOwned) {
        PoolStorageFree(_memory);
    }
}

bool MemoryPoolSubAllocator::initPool(void) {
    if (!_pool) {
        // allocate the actual pool
        _pool = lwnDeviceCreateMemoryPool(_device, _memory, _size, _poolFlags);
    }
    return NULL != _pool;
}

LWNmemoryPool* MemoryPoolSubAllocator::pool() {
    initPool();
    return _pool;
}

MemoryPoolSubAllocator::MemoryPoolBlock MemoryPoolSubAllocator::allocBlock(size_t size, size_t align)
{
    if (!initPool()) {
        return MemoryPoolBlock();
    }

    // look for a free block in _freeOffsetMap, using reverse_iterator because waste blocks get put
    // at the front of the map (due to lower value of their key) and we don't want to run through all 
    // of these when trying to find a free block.
    for (MemoryOffsetPoolBlockMap::reverse_iterator it = _freeOffsetMap.rbegin(); it != _freeOffsetMap.rend(); it++) {
        MemoryPoolBlock free_block = it->second;

        if (size && free_block.size() >= size) {
            // if free block is not already aligned, we need a bigger one to satisfy
            // aligned size
            if (free_block.offset() & (align - 1)) {
                ptrdiff_t aligned_offset = (free_block.offset() + (align - 1)) & ~(align - 1);
                size_t waste = aligned_offset - free_block.offset();
                // would it fit into current free block? then generate
                // a block from the unaligned offset and size and a tail block and add them
                // to the free list. Return the unaligned block. Caller will
                // return aligned offset. 
                if (free_block.size() >= (size + waste)) {
                    MemoryPoolBlock alloc(free_block.offset(), size + waste, this, align);
                    MemoryPoolBlock tail(aligned_offset + size, free_block.size() - (waste + size));

                    // remove available block from free map
                    ++it;
                    _freeOffsetMap.erase(it.base());

                    // if we have a tail add it to the free map
                    if (tail) {
                        _freeOffsetMap[tail.offset()] = tail;
                    }

                    _usedMap[alloc.offset()] = alloc;

                    _numAllocs++;
                    _free -= alloc.size();
                    _used += alloc.size();

                    return alloc;
                }
            } else { // alignment and size fit, use it
                MemoryPoolBlock alloc(free_block.offset(), size, this);

                // track used block
                _usedMap[alloc.offset()] = alloc;

                // remove available block from free map
                ++it;
                _freeOffsetMap.erase(it.base());

                MemoryPoolBlock newfree(free_block.offset() + size, free_block.size() - size);
                if (newfree) {
                    _freeOffsetMap[newfree.offset()] = newfree;
                }

                _numAllocs++;
                _free -= alloc.size();
                _used += alloc.size();

                return alloc;
            }
        }
    }
    return MemoryPoolBlock();
}

// allocate a LWNtexture* from the memory pool
LWNtexture* MemoryPoolSubAllocator::allocTexture(LWNtextureBuilder* builder)
{
    LWNtexture* tex = NULL;

    size_t tex_size = lwnTextureBuilderGetStorageSize(builder);
    size_t tex_align = lwnTextureBuilderGetStorageAlignment(builder);

    MemoryPoolBlock block = allocBlock(tex_size, tex_align);
    if (block) {
        tex = lwnTextureBuilderCreateTextureFromPool(builder, pool(), block.aligned_offset());
        if (tex) {
            _textureMap[tex] = block;
            _wasUsed = true;
        } else {
            block.freeBlock();
        }
    }

    return tex;
}

void MemoryPoolSubAllocator::freeTexture(LWNtexture* texture)
{
    MemoryPoolBlock block = _textureMap[texture];
    block.freeBlock();
    lwnTextureFree(texture);
    _textureMap.erase(texture);
}

ptrdiff_t MemoryPoolSubAllocator::offset(LWNtexture* texture)
{
    MemoryPoolBlock block = _textureMap[texture];
    return block.offset();
}

size_t MemoryPoolSubAllocator::bufferAlignment(BufferAlignBits alignBits) const
{
    //*** from lwndocs
    //* Vertex bufferformat-specific,                               1B to 4B
    //* Uniform buffer                                              256B
    //* Shader Storage buffer                                       32B
    //* Transform feedback data buffer                              4B
    //* Transform feedback control buffer                           4B
    //* Index buffer    index size,                                 1B to 4B
    //* Indirect draw buffer                                        4B
    //* Counter reports                                             16B
    //* Texture (TextureTarget::TEXTURE_BUFFER) format-specific,    1B to 16B
    //***
    // ifs are ordered in such a way that this return safe alignment
    // when multiple access bits are set (which is possible given
    // access is a bitfield)
    if (alignBits & BUFFER_ALIGN_UNIFORM_BIT) {
        return 256;
    } else if (alignBits & BUFFER_ALIGN_IMAGE_BIT) {
        return 256; // use worst case assumption for Kepler (requiring 256B)
    } else if (alignBits & BUFFER_ALIGN_SHADER_STORAGE_BIT) {
        return 32; // use worst case (dvec4) in absence of format information
    } else if (alignBits & BUFFER_ALIGN_ZLWLL_SAVE_BIT) {
        return 32;
    } else if (alignBits & BUFFER_ALIGN_TRANSFORM_FEEDBACK_CONTROL_BIT) {
        return 32;
    } else if(alignBits & BUFFER_ALIGN_TEXTURE_BIT) {
        return 16; // use worst case in absence of format information
    } else if(alignBits & BUFFER_ALIGN_COUNTER_BIT) {
        return 16;
    } else if(alignBits & BUFFER_ALIGN_VERTEX_BIT) {
        return 16;
    } else if (alignBits & BUFFER_ALIGN_INDEX_BIT) {
        return 4; // use worst case in absence of format information
    } else if (alignBits & BUFFER_ALIGN_INDIRECT_BIT) {
        return 4;
    } else if (alignBits & BUFFER_ALIGN_TRANSFORM_FEEDBACK_BIT) {
        return 4;
    } else if (alignBits & BUFFER_ALIGN_COPY_READ_BIT) {
        return 4;
    } else if (alignBits & BUFFER_ALIGN_COPY_WRITE_BIT) {
        return 4;
    }
    return 512; // return GOB alignment to be on the safe side
}

LWNbuffer* MemoryPoolSubAllocator::allocBuffer(LWNbufferBuilder* builder, BufferAlignBits alignBits, size_t size)
{
    LWNbuffer* buffer = NULL;

    MemoryPoolBlock block = allocBlock(size, bufferAlignment(alignBits));
    if (block) {
        buffer = lwnBufferBuilderCreateBufferFromPool(builder, pool(), block.aligned_offset(), size);
        if (buffer) {
            _bufferMap[buffer] = block;
            _wasUsed = true;
        } else {
            block.freeBlock();
        }
    }

    return buffer;
}

void MemoryPoolSubAllocator::freeBuffer(LWNbuffer* buffer)
{
    MemoryPoolBlock block = _bufferMap[buffer];

    block.freeBlock();
    _bufferMap.erase(buffer);
    lwnBufferFree(buffer);
}

ptrdiff_t MemoryPoolSubAllocator::offset(LWNbuffer* buffer)
{
    MemoryPoolBlock block = _bufferMap[buffer];
    return block.offset();
}

size_t MemoryPoolSubAllocator::size(LWNbuffer* buffer)
{
    MemoryPoolBlock block = _bufferMap[buffer];
    return block.size();
}

// helper to return previous iterator
MemoryPoolSubAllocator::MemoryOffsetPoolBlockMap::iterator MemoryPoolSubAllocator::prev(MemoryOffsetPoolBlockMap::iterator it)
{
    return --it;
}

void MemoryPoolSubAllocator::freeBlock(MemoryPoolBlock block)
{
    assert(block.allocator() == this);

    MemoryOffsetPoolBlockMap::iterator it = _usedMap.find(block.offset());
    if (it != _usedMap.end()) {
        MemoryOffsetPoolBlockMap::iterator this_it;

        // see if we can combine with a block directly following this block
        MemoryOffsetPoolBlockMap::iterator next_it = _freeOffsetMap.find(block.offset() + block.size());
        if (next_it != _freeOffsetMap.end()) {
            MemoryPoolBlock new_block(block.offset(), block.size() + next_it->second.size());
            _freeOffsetMap.erase(next_it);
            this_it = _freeOffsetMap.insert(std::make_pair(new_block.offset(), new_block)).first;
        } else {
            this_it = _freeOffsetMap.insert(std::make_pair(block.offset(), block)).first;
        }

        // see if we can combine this with a previous block
        MemoryPoolBlock this_block = this_it->second;
        MemoryOffsetPoolBlockMap::iterator prev_it = prev(this_it);
        if (prev_it != _freeOffsetMap.end()) {
            ptrdiff_t prev_end = prev_it->second.offset() + prev_it->second.size();
            if (prev_end == block.offset()) {
                MemoryPoolBlock new_block(prev_it->second.offset(), prev_it->second.size() + this_block.size());
                _freeOffsetMap.erase(block.offset());
                _freeOffsetMap[new_block.offset()] = new_block;
            }
        }

        _usedMap.erase(it);
        _numAllocs--;
        _free += block.size();
        _used -= block.size();
    }
}

void MemoryPoolSubAllocator::reset() {
    _wasUsed = false;
    _numAllocs = 0;
    _free = _size;
    _used = 0;
    _pool = 0;

    // initially whole pool is free
    _freeOffsetMap.clear();
    MemoryPoolBlock block(0, _size);
    _freeOffsetMap[0] = block;

    // and nothing is used...
    _usedMap.clear();

    // no textures allocated
    _textureMap.clear();
}

bool MemoryPoolSubAllocator::contains(LWNtexture* texture) {
    return _textureMap.find(texture) != _textureMap.end();
}

bool MemoryPoolSubAllocator::contains(LWNbuffer* buffer) {
    return _bufferMap.find(buffer) != _bufferMap.end();
}


void MemoryPoolSubAllocator::discard() {

    // free all textures
    for (MemoryPoolTextureMap::iterator it = _textureMap.begin(); it != _textureMap.end(); it++) {
        LWNtexture* texture = it->first;
        lwnTextureFree(texture);
    }

    // free all buffers
    for (MemoryPoolBufferMap::iterator it = _bufferMap.begin(); it != _bufferMap.end(); it++) {
        LWNbuffer* buffer = it->first;
        lwnBufferFree(buffer);
    }

    reset();
}

} // namespace lwnUtil

#endif // #ifndef __lwnUtil_PoolAllocatorImpl_h__
