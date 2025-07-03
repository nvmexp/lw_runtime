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

#include <Context/ProfileMapping.h>
#include <Device/Device.h>
#include <Memory/MBuffer.h>

#include <corelib/misc/Concepts.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>


namespace llvm {
class Module;
};

namespace optix {
class Context;
class CodeRangeTimer;

class ProfileManager : private corelib::NonCopyable
{
  public:
    ProfileManager( Context* context );
    ~ProfileManager();

    void beginKernelLaunch();
    void finalizeKernelLaunch( const ProfileMapping* profile );
    void finalizeApiLaunch();
    void preSetActiveDevices( const DeviceArray& removedDevices );
    void postSetActiveDevices();

    // Interface to exelwtion strategy compile and launch
    std::unique_ptr<ProfileMapping> makeProfileMappingAndUpdateModule( int numCounters, int numEvents, int numTimers, llvm::Module* module );
    void disableProfilingInModule( llvm::Module* module );
    void* getProfileDataDevicePointer( const Device* device );

    bool launchTimingEnabled() const;

  private:
    void dumpProfilerOutputForDevice( unsigned int deviceIndex, unsigned long long* data, const ProfileMapping* profile ) const;
    void printCounter( unsigned long long* data, int counterId, const std::string& name, bool printHeader ) const;
    void printEvent( unsigned long long* data, int eventId, const std::string& name, double scale, bool printHeader ) const;
    void printTimer( unsigned long long* data, int timerId, const std::string& name, unsigned long long totalClocks, double scale, bool printHeader ) const;

    // Setup variables for profiling in module
    void specializeModule( llvm::Module* module, int numCounters, int numEvents, int numTimers );


    Context* m_context = nullptr;

    std::unique_ptr<CodeRangeTimer> m_codeRangeTimer;

    // Device profile data
    std::vector<MBufferHandle> m_profileData;
    unsigned int               m_counterOffset = 0;
    unsigned int               m_eventOffset   = 0;
    unsigned int               m_timerOffset   = 0;
    unsigned int               m_totalSize     = 0;
};
}  // end namespace optix
