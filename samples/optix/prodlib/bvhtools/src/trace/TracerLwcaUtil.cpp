// Copyright LWPU Corporation 2017
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES
 
#include "TracerLwda.hpp"
#include "src/common/Utils.hpp"

#include <string.h>
#include <vector>

namespace prodlib
{
namespace bvhtools
{ 
// copyMesh is in separate build unit to cut linker dependence chain
// between TTU traversal and BVH2 traversal
void copyMesh(TracerParamsMesh& dst, const TracerDataMesh& src)
{
    dst.nodes        = (float4*)src.nodes;
    dst.triangles    = (float4*)src.woopTriangles;
    dst.indices      = (int3*)src.mesh.indices;
    dst.vertices     = src.mesh.vertices;
    dst.remap        = src.remap;
    dst.indexStride  = src.mesh.indexStride;
    dst.vertexStride = src.mesh.vertexStride;
    dst.rootNode     = (src.nodes) ? 0 : EntrypointSentinel;
    dst.numNodes     = src.numNodes;
    dst.numRemaps    = src.numRemaps;
}

} // namespace bvhtools
} // namespace prodlib
