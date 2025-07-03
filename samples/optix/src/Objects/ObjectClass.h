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

#include <o6/optix.h>

namespace optix {

enum ObjectClass
{
    // API objects.
    // Make sure the object types exposed in optix.h match the ones declared here.
    // The reason why these are two different enums in the first place is that not
    // *all* of the types declared here appear in optix.h.
    RT_OBJECT_UNKNOWN             = RT_OBJECTTYPE_UNKNOWN,
    RT_OBJECT_GROUP               = RT_OBJECTTYPE_GROUP,
    RT_OBJECT_GEOMETRY_GROUP      = RT_OBJECTTYPE_GEOMETRY_GROUP,
    RT_OBJECT_TRANSFORM           = RT_OBJECTTYPE_TRANSFORM,
    RT_OBJECT_SELECTOR            = RT_OBJECTTYPE_SELECTOR,
    RT_OBJECT_GEOMETRY_INSTANCE   = RT_OBJECTTYPE_GEOMETRY_INSTANCE,
    RT_OBJECT_BUFFER              = RT_OBJECTTYPE_BUFFER,
    RT_OBJECT_TEXTURE_SAMPLER     = RT_OBJECTTYPE_TEXTURE_SAMPLER,
    RT_OBJECT_PROGRAM             = RT_OBJECTTYPE_PROGRAM,
    RT_OBJECT_COMMANDLIST         = RT_OBJECTTYPE_COMMANDLIST,
    RT_OBJECT_POSTPROCESSINGSTAGE = RT_OBJECTTYPE_POSTPROCESSINGSTAGE,

    RT_OBJECT_ACCELERATION = 0x300,
    RT_OBJECT_GLOBAL_SCOPE,
    RT_OBJECT_DEVICE,
    RT_OBJECT_GEOMETRY,
    RT_OBJECT_MATERIAL,
    RT_OBJECT_VARIABLE,
    RT_OBJECT_STREAM_BUFFER
};

// Get the name for a class as string.
const char* const getNameForClass( ObjectClass c );

// Return if a type is a graph node
bool isGraphNode( ObjectClass c );
bool isGraphNode( RTobject object );

// Return if a type is legal as a child for groups.
bool isLegalGroupChild( ObjectClass c );
bool isLegalGroupChild( RTobject object );

// Return if a type is legal as a child for selectors.
bool isLegalSelectorChild( ObjectClass c );
bool isLegalSelectorChild( RTobject object );

// Return if a type is legal as a child for transforms.
bool isLegalTransformChild( ObjectClass c );
bool isLegalTransformChild( RTobject object );

// Return if a type is legal as a child for geometry groups.
bool isLegalGeometryGroupChild( ObjectClass c );
bool isLegalGeometryGroupChild( RTobject object );

}  // namespace optix
