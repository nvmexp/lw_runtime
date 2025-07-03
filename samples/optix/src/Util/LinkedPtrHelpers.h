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

namespace optix {
class LinkedPtr_Link;

// Return a link From
template <typename From, typename To>
From* getLinkFrom( LinkedPtr_Link* link )
{
    LinkedPtr<From, To>* ptr = dynamic_cast<LinkedPtr<From, To>*>( link );
    return ptr ? ptr->getFrom() : nullptr;
}

// Return a link from a pointer of type F1, F2, F3, ... to a pointer
// of type To.  Do not use these directly - they are intended to
// support the variants below.
template <typename F1, typename F2, typename To, typename Base>
Base* getLinkFrom( LinkedPtr_Link* link )
{
    if( Base* from = getLinkFrom<F1, To>( link ) )
        return from;
    return getLinkFrom<F2, To>( link );
}
template <typename F1, typename F2, typename F3, typename To, typename Base>
Base* getLinkFrom( LinkedPtr_Link* link )
{
    if( Base* from = getLinkFrom<F1, F2, To, Base>( link ) )
        return from;
    return getLinkFrom<F3, To>( link );
}
template <typename F1, typename F2, typename F3, typename F4, typename To, typename Base>
Base* getLinkFrom( LinkedPtr_Link* link )
{
    if( Base* from = getLinkFrom<F1, F2, F3, To, Base>( link ) )
        return from;
    return getLinkFrom<F4, To>( link );
}
template <typename F1, typename F2, typename F3, typename F4, typename F5, typename To, typename Base>
Base* getLinkFrom( LinkedPtr_Link* link )
{
    if( Base* from = getLinkFrom<F1, F2, F3, F4, To, Base>( link ) )
        return from;
    return getLinkFrom<F5, To>( link );
}

/*
   * The variants below are syntactic sugar - the avoid the need to
   * remember which argument is which in the more general versions
   * above. Prefer these functions.
   */

// Pointers from one of the template types to Acceleration
class Acceleration;
template <typename F1>
F1* getLinkToAccelerationFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, Acceleration>( link );
}

// Pointers from one of the template types to Buffer
class Buffer;
template <typename F1>
F1* getLinkToBufferFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, Buffer>( link );
}

// Pointers from one of the template types to PostprocessingStage.
class PostprocessingStage;
template <typename F1>
F1* getLinkToPostprocessingStageFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, PostprocessingStage>( link );
}

// Pointers from one of the template types to Geometry
class Geometry;
template <typename F1>
F1* getLinkToGeometryFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, Geometry>( link );
}

// Pointers from one of the template types to GraphNode
class GraphNode;
template <typename F1>
F1* getLinkToGraphNodeFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, GraphNode>( link );
}
template <typename F1, typename F2>
GraphNode* getLinkToGraphNodeFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, F2, GraphNode, GraphNode>( link );
}

// Pointers from one of the template types to LexicalScope
class LexicalScope;
template <typename F1>
F1* getLinkToLexicalScopeFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, LexicalScope>( link );
}

// Pointers from one of the template types to Material
class Material;
template <typename F1>
F1* getLinkToMaterialFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, Material>( link );
}

// Pointers from one of the template types to Programs.
class Program;
template <typename F1>
F1* getLinkToProgramFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, Program>( link );
}
template <typename F1, typename F2, typename F3>
LexicalScope* getLinkToProgramFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, F2, F3, Program, LexicalScope>( link );
}
template <typename F1, typename F2, typename F3, typename F4>
LexicalScope* getLinkToProgramFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, F2, F3, F4, Program, LexicalScope>( link );
}

// Pointers from one of the template types to TextureSampler
class TextureSampler;
template <typename F1>
F1* getLinkToTextureSamplerFrom( LinkedPtr_Link* link )
{
    return getLinkFrom<F1, TextureSampler>( link );
}

// Utility function to return the index of a vector element given
// the address of the element. Returns true if the element is
// contained in the vector. index will be valid only if the function
// returns true. T2 must be a subclass of T1 (or identical to T1).
template <typename T1, typename T2>
bool getElementIndex( const std::vector<T1>& vec, T2* elt, unsigned int& index )
{
    if( vec.empty() )
    {
        index = ~0;
        return false;
    }
    else
    {
        size_t diff = (T1*)elt - vec.data();
        index       = diff;
        return diff < vec.size();
    }
}
}
