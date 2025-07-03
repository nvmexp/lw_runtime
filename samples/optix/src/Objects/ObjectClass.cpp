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

#include <Objects/ManagedObject.h>
#include <Objects/ObjectClass.h>

namespace optix {

const char* const getNameForClass( ObjectClass c )
{
    switch( c )
    {
        case RT_OBJECT_UNKNOWN:
            return "Unknown";
        case RT_OBJECT_ACCELERATION:
            return "Acceleration";
        case RT_OBJECT_BUFFER:
            return "Buffer";
        case RT_OBJECT_GLOBAL_SCOPE:
            return "GlobalScope";
        case RT_OBJECT_DEVICE:
            return "Device";
        case RT_OBJECT_GEOMETRY:
            return "Geometry";
        case RT_OBJECT_GEOMETRY_INSTANCE:
            return "GeometryInstance";
        case RT_OBJECT_GROUP:
            return "Group";
        case RT_OBJECT_GEOMETRY_GROUP:
            return "GeometryGroup";
        case RT_OBJECT_TRANSFORM:
            return "Transform";
        case RT_OBJECT_MATERIAL:
            return "Material";
        case RT_OBJECT_PROGRAM:
            return "Program";
        case RT_OBJECT_SELECTOR:
            return "Selector";
        case RT_OBJECT_TEXTURE_SAMPLER:
            return "TextureSampler";
        case RT_OBJECT_VARIABLE:
            return "Variable";
        case RT_OBJECT_POSTPROCESSINGSTAGE:
            return "PostprocessingStage";
        default:
            return "UNKNOWN CLASS";
    }
}

bool isGraphNode( ObjectClass c )
{
    return c == RT_OBJECT_GROUP || c == RT_OBJECT_SELECTOR || c == RT_OBJECT_GEOMETRY_GROUP || c == RT_OBJECT_TRANSFORM;
}

bool isGraphNode( RTobject object )
{
    ManagedObject* managed = static_cast<ManagedObject*>( object );
    return object && isGraphNode( managed->getClass() );
}

bool isLegalGroupChild( ObjectClass c )
{
    return isGraphNode( c );
}

bool isLegalGroupChild( RTobject object )
{
    ManagedObject* managed = static_cast<ManagedObject*>( object );
    return object && isLegalGroupChild( managed->getClass() );
}

bool isLegalSelectorChild( ObjectClass c )
{
    return isGraphNode( c );
}

bool isLegalSelectorChild( RTobject object )
{
    ManagedObject* managed = static_cast<ManagedObject*>( object );
    return object && isLegalSelectorChild( managed->getClass() );
}

bool isLegalTransformChild( ObjectClass c )
{
    return isGraphNode( c );
}

bool isLegalTransformChild( RTobject object )
{
    ManagedObject* managed = static_cast<ManagedObject*>( object );
    return object && isLegalTransformChild( managed->getClass() );
}

bool isLegalGeometryGroupChild( ObjectClass c )
{
    return c == RT_OBJECT_GEOMETRY_INSTANCE;
}

bool isLegalGeometryGroupChild( RTobject object )
{
    ManagedObject* managed = static_cast<ManagedObject*>( object );
    return object && isLegalGeometryGroupChild( managed->getClass() );
}

}  // namespace optix
