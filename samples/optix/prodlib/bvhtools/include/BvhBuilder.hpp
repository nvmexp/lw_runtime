// Copyright LWPU Corporation 2013
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
#include <driver_types.h> // for lwdaStream_t

#include "Types.hpp"
#include <string>

#include <rtcore/interface/rtcore.h>

namespace prodlib
{
namespace bvhtools
{

//------------------------------------------------------------------------------

class BvhBuilder
{
public:
    static BvhBuilder*              create                  (const char* builderSpec);
    virtual ~BvhBuilder                                     (void);

    virtual void                    setBuilderSpec          (const char* builderSpec);
    virtual void                    setDisableLwda          (bool disableLwda) = 0;
    virtual void                    setLwdaStream           (lwdaStream_t stream) = 0;

    virtual void                    computeMemUsage         (const char* builderSpec, bool buildGpu, int numPrims, int motionSteps, InputType type,  MemoryUsage* memUsage);

    // FIXME: Remove this in favor of the other method
    virtual void                    computeMemUsage         (const char* builderSpec, bool buildGpu, int numPrimAabbs, int numTriangles, MemoryUsage* memUsage) = 0;

    virtual void                    computeMemUsage         (const char* builderSpec, const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem, bool buildGpu, MemoryUsage* memUsage) = 0;
    virtual void                    computeMemUsage         (const char* builderSpec, bool buildGpu, int numInputs, const RtcBuildInput* buildInputs, MemoryUsage* memUsage) = 0;

    virtual void                    build                   (const std::vector<TriangleMesh>& meshes, bool meshInDeviceMem);
    virtual void                    build                   (const Group& group, bool groupInDeviceMem);
    virtual void                    build                   (int numAabbs, const PrimitiveAABB* aabbs, bool aabbsInDeviceMem) = 0;
    virtual void                    build                   (int numAabbs, int motionSteps, const float* aabbs, bool aabbsInDeviceMem);
    virtual void                    build                   (int numInstances, const InstanceDesc* instances, bool inDeviceMem);
    virtual void                    build                   (int numInputs, const RtcBuildInput* buildInputs, bool inDeviceMem) = 0;

    virtual void                    copyHeader              (void* dst, size_t dstSize); // Copy the header into the specified buffer. 

    virtual void                    freeData                (void) = 0;
    virtual void                    setDataLocation         (bool inDeviceMem) = 0;         // No duplicate buffers remain after the call.

    virtual const BvhNode*          getNodeBufferPtr        (bool inDeviceMem = false) = 0; // Duplicates the data lazily to CPU/GPU memory if needed.
    virtual const int*              getRemapBufferPtr       (bool inDeviceMem = false) = 0;
    virtual const WoopTriangle*     getWoopBufferPtr        (bool inDeviceMem = false) = 0;
    virtual const ApexPointMap*     getApexPointMapPtr      (bool inDeviceMem = false) = 0;
    virtual const unsigned char*    getOutputBufferPtr      (bool inDeviceMem = false) = 0;

    virtual size_t                  getNodeBufferSize       (void) const = 0;
    virtual size_t                  getRemapBufferSize      (void) const = 0;
    virtual size_t                  getWoopBufferSize       (void) const = 0;
    virtual size_t                  getOutputBufferSize     (void) const = 0;
    virtual size_t                  getApexPointMapSize     (void) const = 0;

    virtual void                    setExternalBuffers      (unsigned char* outputPtr, size_t outputNumBytes, 
                                                             unsigned char* tempPtr, size_t tempNumBytes, 
                                                             unsigned char* readbackPtr, size_t readbackNumBytes, bool buildGpu) = 0;

    virtual void                    setExternalOutputBuffer (unsigned char* outputPtr, size_t outputNumBytes, bool buildGpu) = 0;

protected:
    std::string   m_builderSpec;

};

} // namespace bvhtools
} // namespace prodlib
