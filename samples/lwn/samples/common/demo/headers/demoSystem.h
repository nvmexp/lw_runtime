/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/
// -----------------------------------------------------------------------------
//  demoSystem.h
//
// -----------------------------------------------------------------------------

#ifndef __DEMO_SYSTEM_H__
#define __DEMO_SYSTEM_H__

#ifdef LW_HOS
#include <nn/os/os_Tick.h>

#define getTime() ((uint64_t)(nn::os::GetSystemTick().GetInt64Value()) * (1000000000 / nn::os::GetSystemTickFrequency()))
#endif

#include <stdio.h>
#include <string.h>
#include <assert.h>

#define DEMO_INLINE static inline

#define DEMOAssert(expr) assert(expr)

#define DEMORoundUp256B(x) (((u32)(x) + 256 - 1) & ~(256 - 1))
#define DEMORoundUp32B(x) (((u32)(x) + 32 - 1) & ~(32 - 1))
#define DEMORoundUp4B(x) (((u32)(x) + 4 - 1) & ~(4 - 1))
	
#define DEMO_RAND_MAX	  ((u32)0xFFFF)
#define DEMO_BUFFER_ALIGN (64)

/// @addtogroup demoSystem
/// @{

/// \brief Initialize System
///
/// The DEMO library provides a common application framework
/// that is used in many of the example demos distributed with SDK.
/// The source code for the DEMO library is also distributed with the SDK.
///
/// This function initializes various system components including:
/// - OS: Base operation system
/// - Memory: All of the main memory is then allocated into a 
///   heap that can be managed with \ref DEMOAlloc
void DEMOInit(void);

/// \brief Have \ref DEMOIsRunning() return false the next time it is called
///
/// This allows a controlled shutdown by letting the current loop finish
void DEMOStopRunning(void);

/// \brief Shutdown System
///
/// Shuts down all system components initialized by \ref DEMOInit
void DEMOShutdown(void);

/// \brief Set the random seed
///
/// Seeds the random number generator.
void DEMOSRand(u32 seed);

/// \brief Random float number generator
///
/// Returns a random number: 0 <= n <= 1.
/// \retval The next random number.
f32 DEMOFRand(void);

/// \brief Random integer number generator
///
/// Returns a random number: 0 <= n <= DEMO_RAND_MAX.
/// \retval The next random number.
u32 DEMORand(void);

/// \brief Type used for default DEMO memory allocator
typedef void* (*DEMODefaultAllocateFunc)(u32 byteCount, int alignment);

/// \brief Type used for default DEMO memory free function
typedef void (*DEMODefaultFreeFunc)(void* pMem);

/// \brief Set default functions to use for memory allocation/freeing.
///
/// These will be the functions called by DEMOAlloc/DEMOAllocEx/DEMOFree.
/// Those entry points are used by the DEMO libs when they need to allocate memory.
/// (Except for when non-regular-MEM2 arenas are needed.)
///
/// If not set by the user, these will just call MEMAllocFromDefaultHeap/MEMFreeToDefaultHeap.
///
/// \param pfnAlloc pointer to allocator function
/// \param pfnFree  pointer to free function
void DEMOSetDefaultAllocator(DEMODefaultAllocateFunc pfnAlloc, DEMODefaultFreeFunc pfnFree);

/// \brief Get default functions to use for memory allocation/freeing.
/// \param ppfnAlloc pointer to get pointer to allocator function
/// \param ppfnFree  pointer to get pointer to free function
void DEMOGetDefaultAllocator(DEMODefaultAllocateFunc *ppfnAlloc, DEMODefaultFreeFunc *ppfnFree);

/// \brief Allocate memory
///
/// \param size Size to allocate
/// \retval Pointer to the allocated buffer if allocation succeeded
void* DEMOAlloc(u32 size);

/// \brief Allocate memory with specific alignment
///
/// \param size Size to allocate
/// \param align Alignment to use for allocation
/// \retval Pointer to the allocated buffer if allocation succeeded
void* DEMOAllocEx(u32 size, u32 align);

/// \brief Free memory
///
/// \param ptr Pointer to the buffer to be deallocated
void DEMOFree(void* ptr);

/// \brief Get demo running state
///
/// \note This function also calls the DEMO Test functions, and therefore
///       it is expected that this function is only called once prior to each
///       main loop iteration.
///
/// \retval TRUE if \ref DEMOInit() has been called and DEMOStopRunning() has not been called; false otherwise.
BOOL DEMOIsRunning(void);

typedef void (*DEMOReleaseCallbackFunc)(void);


#ifndef LW_HOS
/// \brief Prints formatted message to debug output
///
/// \param msg Pointer to a null-terminated string including format specification 
/// (equivalent to C's standard output function).
/// \param ... Optional argument
void DEMOPrintf (const char* msg, ...);

#else

#include <nn/nn_Log.h>
#define DEMOPrintf(...) NN_LOG(__VA_ARGS__)

#endif

/// @}

#endif /// __DEMO_SYSTEM_H__
