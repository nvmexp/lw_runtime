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

#include <Context/UpdateManager.h>
#include <Device/Device.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <Memory/DeviceSpecificTable.h>
#include <Memory/MBuffer.h>

#include <rtcore/interface/types.h>

#include <memory>
#include <vector>


namespace cort {
struct ProgramHeader;
struct TextureSamplerHost;
}  // namespace cort

namespace optix {
class BackedAllocator;
class Buffer;
class Context;
class LexicalScope;
class ManagedObject;
class MappedSet;
class Program;
class TextureSampler;
class UpdateCaller;

class TableManager : public UpdateEventListenerNop
{
  public:
    TableManager( Context* context );
    ~TableManager() override;
    void unmapFromHost();

    // Disallow copying
    TableManager( const TableManager& ) = delete;
    TableManager& operator=( const TableManager& ) = delete;

    // Table manager needs to allocate tables on new devices, and free
    // resources in removed devices.
    void preSetActiveDevices( const DeviceArray& removedDevices );
    void postSetActiveDevices();

    // Launch management.  allocateTables is exelwted before the memory manager
    // finalizes the allocations (syncAllMemoryBeforeLaunch).  Once allocations
    // are complete, syncTablesForLaunch will trigger manual synchronization.
    // No further writes to the tables are allowed until launchCompleted is
    // called.
    void allocateTables();
    void syncTablesForLaunch();
    void launchCompleted();

    // Object record management.  Only usable when NOT launching. Also, don't
    // store the returned pointers for later use -- they may get ilwalidated by
    // reallocations.
    std::shared_ptr<size_t> allocateObjectRecord( size_t bytes );
    char* getObjectRecordHostPointer();
    // DynamicVariableTable management.  Only usable when NOT launching. Also, don't
    // store the returned pointers for later use -- they may get ilwalidated by
    // reallocations. Note size in bytes - but we want two shorts actually.
    std::shared_ptr<size_t> allocateDynamicVariableTable( size_t bytes );
    char* getDynamicVariableTableHostPointer();

    // Get the variable table pointers for the specified device. Only usable when
    // launching. The pointer is valid after allocateTables and
    // MemoryManager::syncAllMemoryBeforeLaunch().
    char* getObjectRecordDevicePointer( const Device* device );
    cort::Buffer* getBufferHeaderDevicePointer( const Device* device );
    cort::TextureSamplerHost* getTextureHeaderDevicePointer( const Device* device );
    cort::ProgramHeader* getProgramHeaderDevicePointer( const Device* device );
    cort::TraversableHeader* getTraversableHeaderDevicePointer( const Device* device );
    char* getDynamicVariableTableDevicePointer( const Device* device );

    RtcTraversableHandle getTraversableHandleForTest( int id, unsigned int allDeviceIndex );

    // Since we don't want to have each object get a valid host pointer for each device, we
    // feed the data necessary to fill the header directly to the TableManager via these
    // functions.
    void writeBufferHeader( int id, size_t width, size_t height, size_t depth, unsigned int pageWidth, unsigned int pageHeight, unsigned int pageDepth );
    void clearBufferHeader( int id );
    void writeTextureHeader( int          id,
                             unsigned int width,
                             unsigned int height,
                             unsigned int depth,
                             unsigned int mipLevels,
                             unsigned int format,
                             unsigned int wrapMode0,
                             unsigned int wrapMode1,
                             unsigned int wrapMode2,
                             unsigned int normCoord,
                             unsigned int filterMode,
                             unsigned int normRet,
                             bool         isDemandLoad );
    void writeDemandTextureHeader( int          id,
                                   unsigned int mipTailFirstLevel,
                                   float        ilwAnisotropy,
                                   unsigned int tileWidth,
                                   unsigned int tileHeight,
                                   unsigned int tileGutterWidth,
                                   unsigned int isInitialized,
                                   unsigned int isSquarePowerOfTwo,
                                   unsigned int mipmapFilterMode );
    void writeDemandTextureDeviceHeader( int id, unsigned int firstVirtualPage, unsigned int numPages, unsigned int minMipLevel );
    void clearTextureHeader( int id );
    void writeProgramHeader( int id, unsigned int offset );
    void clearProgramHeader( int id );
    void writeTraversableHeader( int id, RtcTraversableHandle travHandle, unsigned int allDeviceIndex );
    void clearTraversableHeader( int id );
    void notifyCanonicalProgramAddedToProgram( const Program* program );

    // These are safe to call during launch.  Note that the size of the backed memory might
    // be larger than the size reported here, since the backed memory might be padded at the
    // end.  The size reported is the exact size required to hold all valid data.
    size_t getObjectRecordSize();
    size_t getDynamicVariableTableSize();
    size_t getBufferHeaderTableSizeInBytes();
    size_t getProgramHeaderTableSizeInBytes();
    size_t getTextureHeaderTableSizeInBytes();
    size_t getTraversableTableSizeInBytes();

    size_t getNumberOfBuffers( const Device* device );
    size_t getNumberOfPrograms( const Device* device );
    size_t getNumberOfTextures( const Device* device );
    size_t getNumberOfTraversables( const Device* device );

    // Called by objectmanager when new objects are added so that tables get
    // resized/updated appropriately.
    void notifyCreateBuffer( int bid, Buffer* buffer );
    void notifyCreateTextureSampler( int tid, TextureSampler* sampler );
    void notifyCreateProgram( int pid, Program* program );
    void notifyCreateTraversableHandle( int tid, GraphNode* node );

    // With these events we can fill in the device specific portions of the headers as
    // needed.
    void eventBufferMAccessDidChange( const Buffer* buffer, const Device* device, const MAccess& oldMA, const MAccess& newMA ) override;
    void eventTextureSamplerMAccessDidChange( const TextureSampler* sampler,
                                              const Device*         device,
                                              const MAccess&        oldMA,
                                              const MAccess&        newMA ) override;

  private:
    Context* m_context   = nullptr;
    bool     m_launching = false;

    // All updates to the object record and headers are done incrementally.  The host
    // pointer is mapped on demand.  A host pointer is nullptr if it was not mapped for
    // access.  In addition it is not copied to the device if it was not mapped.
    std::unique_ptr<BackedAllocator> m_objectRecordAllocator;
    char*                            m_objectRecordHostPtr = nullptr;
    // Similar to the object record handling, the dynamicVariableTable host pointer handling.
    // This host pointer is mapped on demand.  A host pointer is nullptr if it was not mapped for
    // access. In addition it is not copied to the device if it was not mapped.
    std::unique_ptr<BackedAllocator> m_dynamicVariableTableAllocator;
    char*                            m_dynamicVariableTableHostPtr = nullptr;

    // Header buffers, one per device.  Note that the size of the buffer can be larger than
    // the number of entries which is stored in a separate variable
    // (e.g. m_bufferHeaderCount).  This is to avoid churn on the allocation when we add
    // headers.  Lwrrently we only grow the allocation, but the count maintains the exact
    // size.
    DeviceSpecificTable<cort::Buffer>             m_bufferHeaders;
    DeviceSpecificTable<cort::TextureSamplerHost> m_textureHeaders;
    DeviceSpecificTable<cort::ProgramHeader>      m_programHeaders;
    DeviceSpecificTableBase                       m_traversableHeaders;

    void resizeBufferHeaders();
    void resizeTextureHeaders();
    void resizeProgramHeaders();
    void resizeTraversableHeaders();

    // These are called when we need to fill the entire table from scratch, such as when you
    // change the active device list.
    void fillBufferHeaders();
    void fillTextureHeaders();
    void fillProgramHeaders();
    void fillTraversableHeaders();
    void fillTextureHeaderForDevice( cort::TextureSamplerHost::DeviceDependent& th,
                                     const TextureSampler*                      sampler,
                                     const Device*                              device,
                                     const MAccess&                             memAccess );

    // Writes the canonicalProgramID for all active devices
    void writeProgramHeaderCanonicalProgram( const Program* program );

    // Map the object record tables to the host.
    void mapObjectRecordToHost();
    // Unmaps the object record tables from the host, and syncs mapped tables to device if syncOnUnmap is true.
    void unmapObjectRecordAllocator( bool syncOnUnmap );
    // Map the dynamicVariableTable to the host.
    void mapDynamicVariableTableToHost();
    // Unmaps the dynamicVariableTable from the host, and syncs mapped tables to device if syncOnUnmap is true.
    void unmapDynamicVariableTableAllocator( bool syncOnUnmap );

    // Unmaps all mapped tables from the host, and syncs mapped tables to device if syncOnUnmap is true.
    void syncTablesFromHost();

    cort::Buffer::DeviceIndependent* getBufferHeaderDiHostPointer( int id );
    cort::Buffer::DeviceDependent* getBufferHeaderDdHostPointer( int id, unsigned int allDeviceIndex );

    cort::TextureSamplerHost::DeviceIndependent* getTextureHeaderDiHostPointer( int id );
    cort::TextureSamplerHost::DeviceDependent* getTextureHeaderDdHostPointer( int id, unsigned int allDeviceIndex );

    cort::ProgramHeader::DeviceIndependent* getProgramHeaderDiHostPointer( int id );
    cort::ProgramHeader::DeviceDependent* getProgramHeaderDdHostPointer( int id, unsigned int allDeviceIndex );


    cort::TraversableHeader* getTraversableHeaderDdHostPointer( int id, unsigned int allDeviceIndex );

    // Only LayoutPrinter should need these methods
    friend class LayoutPrinter;
    const cort::Buffer::DeviceIndependent* getBufferHeaderDiHostPointerReadOnly( int id );
    const cort::Buffer::DeviceDependent* getBufferHeaderDdHostPointerReadOnly( int id, unsigned int allDeviceIndex );
    const cort::TextureSamplerHost::DeviceIndependent* getTextureHeaderDiHostPointerReadOnly( int id );
    const cort::TextureSamplerHost::DeviceDependent* getTextureHeaderDdHostPointerReadOnly( int id, unsigned int allDeviceIndex );
    const cort::ProgramHeader::DeviceIndependent* getProgramHeaderDiHostPointerReadOnly( int id );
    const cort::ProgramHeader::DeviceDependent* getProgramHeaderDdHostPointerReadOnly( int id, unsigned int allDeviceIndex );
    const cort::TraversableHeader* getTraversableHeaderDdHostPointerReadOnly( int id, unsigned int allDeviceIndex );

    // A cache of the printed object layout.  If the new one matches, we don't print it
    // out to the file.  This saves a lot of unnecessary printing.  Typically empty when
    // not doing printing.
    std::string m_printedLayoutCache;
};

}  // end namespace optix
