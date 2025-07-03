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

#include <Device/CPUDevice.h>

#include <Device/APIDeviceAttributes.h>

#include <corelib/misc/String.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/System.h>

#include <llvm/Support/TargetSelect.h>

#include <optixu/optixu_math.h>

using namespace optix;
using namespace prodlib;
using prodlib::IlwalidValue;


namespace {
// clang-format off
  Knob<int> k_cpu_numThreads( RT_DSTRING("cpu.numThreads"),                          -1, RT_DSTRING( "Number of threads to use for CPU. -1 means use all cores." ) );
// clang-format on
}

CPUDevice::CPUDevice( Context* context )
    : Device( context )
{
    if( k_cpu_numThreads.get() == -1 )
        m_numThreads = getNumberOfCPUCores();
    else
        m_numThreads = k_cpu_numThreads.get();
}

CPUDevice::~CPUDevice()
{
}

void CPUDevice::enable()
{
    // CPU device is activated only when CPU fallback is enabled.
    // Initialize JIT for LLVM
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    llvm::InitializeNativeTargetDisassembler();

    m_enabled = true;
}

void CPUDevice::disable()
{
    m_enabled = false;
}

std::string CPUDevice::deviceName() const
{
    std::ostringstream out;
    out << "CPUDevice " << allDeviceListIndex() << " (" << m_numThreads << " threads)";
    return out.str();
}

void CPUDevice::dump( std::ostream& out ) const
{
    Device::dump( out, "CPU" );
    out << "  Number of threads    : " << m_numThreads << std::endl;
}

void CPUDevice::setNumThreads( int numThreads )
{
    if( numThreads < 1 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Number of CPU threads must be greater than zero", numThreads );
    m_numThreads = numThreads;
}

int CPUDevice::getNumThreads() const
{
    return m_numThreads;
}

size_t CPUDevice::getAvailableMemory() const
{
    return getAvailableSystemMemoryInBytes();
}

size_t CPUDevice::getTotalMemorySize() const
{
    return getTotalSystemMemoryInBytes();
}

void CPUDevice::getAPIDeviceAttributes( APIDeviceAttributes& attributes ) const
{
    attributes.maxThreadsPerBlock      = 1;
    attributes.clockRate               = getCPUClockRateInKhz();
    attributes.multiprocessorCount     = getNumberOfCPUCores();
    attributes.exelwtionTimeoutEnabled = 0;
    attributes.maxHardwareTextureCount = INT_MAX;
    attributes.name                    = getCPUName();
    attributes.computeCapability       = make_int2( 0, 0 );
    attributes.totalMemory             = getTotalMemorySize();
    attributes.tccDriver               = 0;
    attributes.lwdaDeviceOrdinal       = -1;
    attributes.pciBusId                = "";
    attributes.compatibleDevices       = getCompatibleOrdinals();
    attributes.rtcoreVersion           = 0;
}

bool CPUDevice::isCompatibleWith( const Device* otherDevice ) const
{
    // All CPU devices are compatible, but we should not have more than one.
    return deviceCast<const CPUDevice>( otherDevice ) != nullptr;
}

bool CPUDevice::supportsHWBindlessTexture() const
{
    return false;
}

bool CPUDevice::supportsTTU() const
{
    return false;
}

bool CPUDevice::supportsMotionTTU() const
{
    return false;
}

bool CPUDevice::supportsLwdaSparseTextures() const
{
    return false;
}

bool CPUDevice::supportsTextureFootprint() const
{
    return false;
}

bool CPUDevice::canAccessPeer( const Device& peerDev ) const
{
    return false;
}
