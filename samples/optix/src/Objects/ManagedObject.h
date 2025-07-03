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

#include <Objects/ObjectClass.h>
#include <Util/IndexedVector.h>
#include <Util/LinkedPtr.h>
#include <corelib/misc/Concepts.h>

#include <set>

namespace optix {
class Buffer;
class Context;
class Geometry;
class GraphNode;
class LexicalScope;
class LinkedPtr_Link;
class Material;
class Program;
class TextureSampler;

enum ManagedObjectType
{
    // abstract types
    MO_TYPE_MANAGED_OBJECT,
    MO_TYPE_LEXICAL_SCOPE,
    MO_TYPE_ABSTRACT_GROUP,
    MO_TYPE_GRAPH_NODE,
    MO_TYPE_POSTPROC_STAGE,

    // concrete types
    MO_TYPE_BUFFER,
    MO_TYPE_STREAM_BUFFER,
    MO_TYPE_COMMAND_LIST,
    MO_TYPE_TEXTURE_SAMPLER,
    MO_TYPE_MATERIAL,
    MO_TYPE_PROGRAM,
    MO_TYPE_ACCELERATION,
    MO_TYPE_GEOMETRY_INSTANCE,
    MO_TYPE_GEOMETRY_TRIANGLES,
    MO_TYPE_GEOMETRY,
    MO_TYPE_POSTPROC_STAGE_DENOISER,
    MO_TYPE_POSTPROC_STAGE_SSIM,
    MO_TYPE_POSTPROC_STAGE_TONEMAP,
    MO_TYPE_SELECTOR,
    MO_TYPE_TRANSFORM,
    MO_TYPE_GROUP,
    MO_TYPE_GEOMETRY_GROUP
};

class ManagedObject : public corelib::NonCopyable
{
  public:
    ManagedObject();
    ManagedObject( Context* context, ObjectClass objclass );
    virtual ~ManagedObject() NOEXCEPT_FALSE;


    Context*    getContext() const;
    ObjectClass getClass() const;

    void addLink( LinkedPtr_Link* );
    void removeLink( LinkedPtr_Link* );


  protected:
    // the owning context
    Context* m_context;

    // This is a critical piece of infrastructure. It holds all of the
    // LinkedPtrs that lwrrently point to this object. The data
    // structure should have O(1) insert time, O(1) removal time,
    // consistent ordering, and inexpensive traversal. Only
    // IntrusiveList and IndexedVector qualify.
    typedef IndexedVector<LinkedPtr_Link*, LinkedPtr_Link::managedObjectIndex_fn> LinkedPointerType;
    LinkedPointerType m_linkedPointers;

  private:
    ObjectClass m_class;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    virtual bool isA( ManagedObjectType type ) const;

    static const ManagedObjectType m_objectType{MO_TYPE_MANAGED_OBJECT};
};

inline Context* ManagedObject::getContext() const
{
    return m_context;
}

inline ObjectClass ManagedObject::getClass() const
{
    return m_class;
}

//------------------------------------------------------------------------
// RTTI
//------------------------------------------------------------------------

template <typename T>
T* managedObjectCast( ManagedObject* mo )
{
    return mo && mo->isA( T::m_objectType ) ? reinterpret_cast<T*>( mo ) : nullptr;
}

template <typename T>
const T* managedObjectCast( const ManagedObject* mo )
{
    return mo && mo->isA( T::m_objectType ) ? reinterpret_cast<T*>( mo ) : nullptr;
}

inline bool ManagedObject::isA( ManagedObjectType type ) const
{
    return type == m_objectType;
}

}  // namespace optix
