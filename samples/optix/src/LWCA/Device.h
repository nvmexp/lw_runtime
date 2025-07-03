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

// lwca nightly version of lwdaGL.h in lwca-11.0-nightly-27866028 includes gl.h without 
// including windows.h causing the WINGDIAPI macro to not be defined.
// We should include windows.h manually here.
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <lwca.h>
typedef unsigned int GLuint;  // BL: Are we ok with this?
typedef unsigned int GLenum;
#include <lwdaGL.h>
#include <string>


namespace optix {

class LWDADevice;

namespace lwca {
class ComputeCapability;

class Device
{
  public:
    Device();

    // Get the low-level device
    LWdevice       get();
    const LWdevice get() const;
    bool           isValid() const;

    bool operator==( const Device& other ) const;

    // Returns a handle to a compute device.
    static Device get( int ordinal, LWresult* returnResult = nullptr );

    // Returns the number of compute-capable devices.
    static int getCount( LWresult* returnResult = nullptr );

    // Returns the compute capability of the device.
    ComputeCapability computeCapability( LWresult* returnResult = nullptr ) const;

    // Returns information about the device.
    int getAttribute( LWdevice_attribute attrib, LWresult* returnResult = nullptr ) const;

    // Returns an identifier string for the device.
    std::string getName( LWresult* returnResult = nullptr ) const;

    // Returns the total amount of memory on the device.
    size_t totalMem( LWresult* returnResult = nullptr ) const;

    // Returns a handle to a compute device.
    static Device getByPCIBusId( const std::string& id, LWresult* returnResult = nullptr );

    // Returns a PCI Bus Id string for the device.
    std::string getPCIBusId( LWresult* returnResult = nullptr ) const;

    // Returns the LUID and node mask associated with the device.
    void getLuidAndNodeMask( char* luid, unsigned int* deviceNodeMask, LWresult* returnResult = nullptr ) const;

    // Queries if a device may directly access a peer device's memory.
    bool canAccessPeer( const Device& peerDev, LWresult* returnResult = nullptr ) const;


    // Colwenience functions for getAttribute
    int MAX_THREADS_PER_BLOCK() const;       /**< Maximum number of threads per block */
    int MAX_BLOCK_DIM_X() const;             /**< Maximum block dimension X */
    int MAX_BLOCK_DIM_Y() const;             /**< Maximum block dimension Y */
    int MAX_BLOCK_DIM_Z() const;             /**< Maximum block dimension Z */
    int MAX_GRID_DIM_X() const;              /**< Maximum grid dimension X */
    int MAX_GRID_DIM_Y() const;              /**< Maximum grid dimension Y */
    int MAX_GRID_DIM_Z() const;              /**< Maximum grid dimension Z */
    int MAX_SHARED_MEMORY_PER_BLOCK() const; /**< Maximum shared memory available per block in bytes */
    int SHARED_MEMORY_PER_BLOCK() const;     /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK */
    int TOTAL_CONSTANT_MEMORY() const; /**< Memory available on device for __constant__ variables in a LWCA C kernel in bytes */
    int WARP_SIZE() const;             /**< Warp size in threads */
    int MAX_PITCH() const;             /**< Maximum pitch in bytes allowed by memory copies */
    int MAX_REGISTERS_PER_BLOCK() const;          /**< Maximum number of 32-bit registers available per block */
    int REGISTERS_PER_BLOCK() const;              /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK */
    int CLOCK_RATE() const;                       /**< Peak clock frequency in kilohertz */
    int TEXTURE_ALIGNMENT() const;                /**< Alignment requirement for textures */
    int GPU_OVERLAP() const;                      /**< Device can possibly copy memory and execute a kernel conlwrrently. Deprecated. Use instead LW_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT. */
    int MULTIPROCESSOR_COUNT() const;             /**< Number of multiprocessors on device */
    int KERNEL_EXEC_TIMEOUT() const;              /**< Specifies whether there is a run time limit on kernels */
    int INTEGRATED() const;                       /**< Device is integrated with host memory */
    int CAN_MAP_HOST_MEMORY() const;              /**< Device can map host memory into LWCA address space */
    int COMPUTE_MODE() const;                     /**< Compute mode (See ::LWcomputemode for details) */
    int MAXIMUM_TEXTURE1D_WIDTH() const;          /**< Maximum 1D texture width */
    int MAXIMUM_TEXTURE2D_WIDTH() const;          /**< Maximum 2D texture width */
    int MAXIMUM_TEXTURE2D_HEIGHT() const;         /**< Maximum 2D texture height */
    int MAXIMUM_TEXTURE3D_WIDTH() const;          /**< Maximum 3D texture width */
    int MAXIMUM_TEXTURE3D_HEIGHT() const;         /**< Maximum 3D texture height */
    int MAXIMUM_TEXTURE3D_DEPTH() const;          /**< Maximum 3D texture depth */
    int MAXIMUM_TEXTURE2D_LAYERED_WIDTH() const;  /**< Maximum 2D layered texture width */
    int MAXIMUM_TEXTURE2D_LAYERED_HEIGHT() const; /**< Maximum 2D layered texture height */
    int MAXIMUM_TEXTURE2D_LAYERED_LAYERS() const; /**< Maximum layers in a 2D layered texture */
    int MAXIMUM_TEXTURE2D_ARRAY_WIDTH() const; /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH */
    int MAXIMUM_TEXTURE2D_ARRAY_HEIGHT() const; /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT */
    int MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES() const; /**< Deprecated, use LW_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS */
    int SURFACE_ALIGNMENT() const;                 /**< Alignment requirement for surfaces */
    int CONLWRRENT_KERNELS() const;                /**< Device can possibly execute multiple kernels conlwrrently */
    int ECC_ENABLED() const;                       /**< Device has ECC support enabled */
    int PCI_BUS_ID() const;                        /**< PCI bus ID of the device */
    int PCI_DEVICE_ID() const;                     /**< PCI device ID of the device */
    int TCC_DRIVER() const;                        /**< Device is using TCC driver model */
    int MEMORY_CLOCK_RATE() const;                 /**< Peak memory clock frequency in kilohertz */
    int GLOBAL_MEMORY_BUS_WIDTH() const;           /**< Global memory bus width in bits */
    int L2_CACHE_SIZE() const;                     /**< Size of L2 cache in bytes */
    int MAX_THREADS_PER_MULTIPROCESSOR() const;   /**< Maximum resident threads per multiprocessor */
    int ASYNC_ENGINE_COUNT() const;               /**< Number of asynchronous engines */
    int UNIFIED_ADDRESSING() const;               /**< Device shares a unified address space with the host */
    int MAXIMUM_TEXTURE1D_LAYERED_WIDTH() const;  /**< Maximum 1D layered texture width */
    int MAXIMUM_TEXTURE1D_LAYERED_LAYERS() const; /**< Maximum layers in a 1D layered texture */
    int CAN_TEX2D_GATHER() const;                 /**< Deprecated, do not use. */
    int MAXIMUM_TEXTURE2D_GATHER_WIDTH() const;   /**< Maximum 2D texture width if LWDA_ARRAY3D_TEXTURE_GATHER is set */
    int MAXIMUM_TEXTURE2D_GATHER_HEIGHT() const; /**< Maximum 2D texture height if LWDA_ARRAY3D_TEXTURE_GATHER is set */
    int MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE() const;     /**< Alternate maximum 3D texture width */
    int MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE() const;    /**< Alternate maximum 3D texture height */
    int MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE() const;     /**< Alternate maximum 3D texture depth */
    int PCI_DOMAIN_ID() const;                         /**< PCI domain ID of the device */
    int TEXTURE_PITCH_ALIGNMENT() const;               /**< Pitch alignment requirement for textures */
    int MAXIMUM_TEXTURELWBEMAP_WIDTH() const;          /**< Maximum lwbemap texture width/height */
    int MAXIMUM_TEXTURELWBEMAP_LAYERED_WIDTH() const;  /**< Maximum lwbemap layered texture width/height */
    int MAXIMUM_TEXTURELWBEMAP_LAYERED_LAYERS() const; /**< Maximum layers in a lwbemap layered texture */
    int MAXIMUM_SURFACE1D_WIDTH() const;               /**< Maximum 1D surface width */
    int MAXIMUM_SURFACE2D_WIDTH() const;               /**< Maximum 2D surface width */
    int MAXIMUM_SURFACE2D_HEIGHT() const;              /**< Maximum 2D surface height */
    int MAXIMUM_SURFACE3D_WIDTH() const;               /**< Maximum 3D surface width */
    int MAXIMUM_SURFACE3D_HEIGHT() const;              /**< Maximum 3D surface height */
    int MAXIMUM_SURFACE3D_DEPTH() const;               /**< Maximum 3D surface depth */
    int MAXIMUM_SURFACE1D_LAYERED_WIDTH() const;       /**< Maximum 1D layered surface width */
    int MAXIMUM_SURFACE1D_LAYERED_LAYERS() const;      /**< Maximum layers in a 1D layered surface */
    int MAXIMUM_SURFACE2D_LAYERED_WIDTH() const;       /**< Maximum 2D layered surface width */
    int MAXIMUM_SURFACE2D_LAYERED_HEIGHT() const;      /**< Maximum 2D layered surface height */
    int MAXIMUM_SURFACE2D_LAYERED_LAYERS() const;      /**< Maximum layers in a 2D layered surface */
    int MAXIMUM_SURFACELWBEMAP_WIDTH() const;          /**< Maximum lwbemap surface width */
    int MAXIMUM_SURFACELWBEMAP_LAYERED_WIDTH() const;  /**< Maximum lwbemap layered surface width */
    int MAXIMUM_SURFACELWBEMAP_LAYERED_LAYERS() const; /**< Maximum layers in a lwbemap layered surface */
    int MAXIMUM_TEXTURE1D_LINEAR_WIDTH() const;        /**< Maximum 1D linear texture width */
    int MAXIMUM_TEXTURE2D_LINEAR_WIDTH() const;        /**< Maximum 2D linear texture width */
    int MAXIMUM_TEXTURE2D_LINEAR_HEIGHT() const;       /**< Maximum 2D linear texture height */
    int MAXIMUM_TEXTURE2D_LINEAR_PITCH() const;        /**< Maximum 2D linear texture pitch in bytes */
    int MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH() const;     /**< Maximum mipmapped 2D texture width */
    int MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT() const;    /**< Maximum mipmapped 2D texture height */
    int COMPUTE_CAPABILITY_MAJOR() const;              /**< Major compute capability version number */
    int COMPUTE_CAPABILITY_MINOR() const;              /**< Minor compute capability version number */
    int MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH() const;     /**< Maximum mipmapped 1D texture width */
    int STREAM_PRIORITIES_SUPPORTED() const;           /**< Device supports stream priorities */


    // Gets the LWCA devices associated with the current OpenGL context.
    static void GLGetDevices( unsigned int*  pDeviceCount,
                              Device*        pDevices,
                              unsigned int   lwdaDeviceCount,
                              LWGLDeviceList deviceList,
                              LWresult*      returnResult = nullptr );

#ifdef _WIN32
    // Gets the LWCA device associated with hGpu.
    static Device lwWGLGetDevice( HGPULW hGpu, LWresult* returnResult = 0 );
#endif  // _WIN32

  private:
    friend class Context;
    friend class optix::LWDADevice;
    explicit Device( LWdevice device );

    LWdevice m_device;
};
}
}
