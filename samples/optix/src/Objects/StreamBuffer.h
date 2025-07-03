// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <LWCA/Function.h>
#include <LWCA/Module.h>

#include <Memory/MBuffer.h>
#include <Objects/ManagedObject.h>

#include <Util/ReusableIDMap.h>

#include <o6/optix.h>

#include <atomic>
#include <mutex>
#include <vector>


namespace optix {

class LWDADevice;
class Buffer;

class StreamBuffer : public ManagedObject
{
  public:
    StreamBuffer( Context* context );
    ~StreamBuffer() override;

    void detachFromParents();

    // React to device changes
    void preSetActiveDevices( const DeviceSet& removedDevices );
    void postSetActiveDevices( const DeviceSet& removedDevices );

    int getId() const;

    void setFormat( RTformat fmt );
    RTformat getFormat();
    void setElementSize( RTsize sz );
    RTsize getElementSize();
    void setSize1D( RTsize w );
    void setSize2D( RTsize w, RTsize h );
    void setSize3D( RTsize w, RTsize h, RTsize d );
    RTsize getWidth();
    RTsize getHeight();
    RTsize getDepth();
    RTsize getLevelWidth( unsigned int level );
    RTsize getLevelHeight( unsigned int level );
    RTsize getLevelDepth( unsigned int level );
    void setMipLevelCount( unsigned int levels );
    unsigned int getMipLevelCount();
    void setSize( int dimensionality, const RTsize* sz, unsigned int levels );
    void getSize( int dimensionality, RTsize* sz );
    int  getDimensionality();
    void bindSource( Buffer* source );
    Buffer* getSource();
    void* map( unsigned int level );
    void unmap( unsigned int level );
    void* map_aclwm( unsigned int level );
    void unmap_aclwm( unsigned int level );
    void fillFromSource( unsigned int max_subframes );
    void resetAclwm();
    void updateReady( int* ready, unsigned int* subframe_count, unsigned int* max_subframes );
    void markNotReady();

  private:
    size_t sizeBytes();
    size_t sizeElems();
    void   alloc();
    int    getSrcNcomp();
    void aclwmulateOnDevice( const LWDADevice* device,
                             const float*      src_d,
                             float*            aclwm_d,
                             unsigned char*    output_d,
                             int               npixels,
                             int               nSrcChannels,
                             int               nOutputChannels,
                             int               pass,
                             float             gamma );

    ReusableID m_id;

    RTformat     m_format        = RTformat( 0 );
    RTsize       m_width         = 0;
    RTsize       m_height        = 0;
    RTsize       m_depth         = 0;
    unsigned int m_levels        = 1;
    RTsize       m_elemsize      = 0;
    int          m_ndims         = 0;
    Buffer*      m_source        = nullptr;
    int          m_naclwm        = 0;  // number of aclwmulated subframes
    unsigned int m_max_subframes = 0;  // just stored here so we can pass it through from the launch to the ready poll
    std::atomic<bool> m_update_ready;  // whether there's been an update to the stream that hasn't been mapped yet
    MBufferHandle     m_stream_storage_device = nullptr;  // post-tonemap storage (what's actually mapped by the app)
    bool              m_stream_is_mapped      = false;
    std::vector<unsigned char> m_stream_storage_host_cache;
    MBufferHandle              m_aclwm_storage_device = nullptr;  // aclwmulation buffer (pre-tonemap)
    void*                      m_aclwm_storage_host   = nullptr;
    std::mutex m_mutex;  // mutex for protecting stream between map/ready polls and re-filling after a launch

    lwca::Module   m_lwda_module;
    lwca::Function m_lwda_function;

  public:
    // Use constant defaults to avoid referencing VCA headers.
    int         stream_attrib_fps     = 30;
    int         stream_attrib_bitrate = 5000000;
    std::string stream_attrib_format  = "auto";
    float       stream_attrib_gamma   = 1.0f;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_STREAM_BUFFER};
};

inline bool StreamBuffer::isA( ManagedObjectType type ) const
{
    return type == m_objectType || ManagedObject::isA( type );
}

}  // namespace optix
