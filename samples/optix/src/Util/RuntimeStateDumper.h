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

#include <stddef.h>  // size_t on Linux
#include <string>
#include <vector>

#include <ExelwtionStrategy/CommonRuntime.h>

namespace optix {
class LWDADevice;
class ObjectManager;


// Dumps all of the state on the device. This is helpful for doing diffs between
// a run that works and one that doesn't to isolate the differences in the state.
//
// Ideally all of the state would be accessible through the cort::Global data
// structure. Some of it is not. For example buffer element sizes are not included
// in the buffer table.
//
// TODO: Add texture sampler state and buffers.
//
class RuntimeStateDumper
{
  public:
    RuntimeStateDumper( int launchCount );

    // Ideally we wouldn't have to call this function if all the information in
    // the runtime were self-contained.
    void computeBufferSizes( ObjectManager* om );

    void dump( LWDADevice* device, cort::Global* global_d );

  private:
    int                 m_launchCount;
    int                 m_deviceId;
    std::vector<size_t> m_bufferSizes;

    // This method takes all state as parameters to identify what the pieces are. Ideally
    // this method would only take global_d.
    void dump( LWDADevice* device, cort::Global* global_d, std::vector<size_t>& bufferSizes );

    // Write data from buffer_d to file and optionally copy to out_buffer_h.
    void dumpBuffer( const std::string& name, const void* buffer_d, size_t size, std::vector<char>* out_buffer_h = nullptr );

    // Copy the data from buffer_d to buffer_h and write to a file based on name
    void dumpBuffer( const std::string& name, const void* buffer_d, size_t size, void* buffer_h );
};
}
