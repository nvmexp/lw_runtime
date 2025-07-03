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

#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Memory/DemandLoad/PagingMode.h>
#include <Memory/MTextureSampler.h>
#include <Memory/MapMode.h>
#include <Objects/SemanticType.h>

#include <corelib/misc/Concepts.h>

#include <o6/optix.h>
#include <vector_types.h>

#include <stddef.h>
#include <vector>


namespace optix {
class AbstractGroup;
class Acceleration;
class Buffer;
class CallSiteIdentifier;
class CanonicalProgram;
class Context;
class Device;
class Geometry;
class GeometryInstance;
class GraphNode;
class LexicalScope;
class ManagedObject;
class Material;
class Program;
class TextureSampler;
class Variable;
struct VariableReferenceBinding;

typedef std::vector<optix::Device*> DeviceArray;

class UpdateEventListener : private corelib::AbstractInterface
{
  public:
#define EVT_PURE = 0
#include "UpdateEvents.h"
#undef EVT_PURE
};

class UpdateManager : public UpdateEventListener
{
  public:
    UpdateManager( Context* context );
    ~UpdateManager() override;

    void registerUpdateListener( UpdateEventListener* listener );
    void unregisterUpdateListener( UpdateEventListener* listener );

#define EVT_PURE override
#include "UpdateEvents.h"
#undef EVT_PURE
  private:
    Context* m_context;

    typedef std::vector<UpdateEventListener*> ListenerListType;
    ListenerListType                          m_listeners;
};

class UpdateEventListenerNop : public UpdateEventListener
{
  public:
#define EVT_PURE override
#include "UpdateEvents.h"
#undef EVT_PURE
};

}  // end namespace optix
