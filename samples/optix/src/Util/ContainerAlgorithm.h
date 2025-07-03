// Copyright (c) 2018 LWPU Corporation.  All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from LWPU Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES

#pragma once

#include <algorithm>
#include <iterator>

// Variants of the standard algorithms that take the container and automatically operate
// on the entire range of elements within the container via std::begin() and std::end().
//
// As you consume more algorithms from <algorithm>, add the container variants here.
//
// A note about auto, -> (trailing return type) and decltype();
// In order for algorithms that return the container's iterator to work on plain C
// style arrays, we can't declare the return type as Container::iterator because that
// is invalid syntax for C style arrays.  Instead, declare the return type as auto
// and use trailing return type syntax to specify the return type to be the same as
// std::begin( c ).  Trailing return type is needed to access an expression referencing
// c.
//
namespace optix {
namespace algorithm {

// std::copy
template <typename Container, typename OutputIt>
OutputIt copy( const Container& c, OutputIt destFirst )
{
    return std::copy( std::begin( c ), std::end( c ), destFirst );
}

// std::equal
template <typename Container, typename InputIt>
bool equal( const Container& c, InputIt first2 )
{
    return std::equal( std::begin( c ), std::end( c ), first2 );
}

// std::fill
template <typename Container, typename T>
void fill( Container& c, const T& val )
{
    std::fill( std::begin( c ), std::end( c ), val );
}

// std::find
template <typename Container, typename T>
auto find( const Container& c, const T& value ) -> decltype( std::begin( c ) )
{
    return std::find( std::begin( c ), std::end( c ), value );
}

template <typename Container, typename T>
auto find( Container& c, const T& value ) -> decltype( std::begin( c ) )
{
    return std::find( std::begin( c ), std::end( c ), value );
}

// std::find_if
template <typename Container, typename UnaryPredicate>
auto find_if( const Container& c, UnaryPredicate p ) -> decltype( std::begin( c ) )
{
    return std::find_if( std::begin( c ), std::end( c ), p );
}

template <typename Container, typename UnaryPredicate>
auto find_if( Container& c, UnaryPredicate p ) -> decltype( std::begin( c ) )
{
    return std::find_if( std::begin( c ), std::end( c ), p );
}

// std::sort
template <typename Container>
void sort( Container& c )
{
    std::sort( std::begin( c ), std::end( c ) );
}

template <typename Container, typename Comparator>
void sort( Container& c, Comparator cmp )
{
    std::sort( std::begin( c ), std::end( c ), cmp );
}

// std::transform
template <typename Container, typename OutputIt, typename UnaryOperation>
OutputIt transform( Container& c, OutputIt destFirst, UnaryOperation unaryOp )
{
    return std::transform( std::begin( c ), std::end( c ), destFirst, unaryOp );
}

// std::all_of
template <typename Container, typename UnaryPredicate>
bool all_of( const Container& c, UnaryPredicate p )
{
    return std::all_of( std::begin( c ), std::end( c ), p );
}

// std::any_of
template <typename Container, typename UnaryPredicate>
bool any_of( const Container& c, UnaryPredicate p )
{
    return std::any_of( std::begin( c ), std::end( c ), p );
}

// std::none_of
template <typename Container, typename UnaryPredicate>
bool none_of( const Container& c, UnaryPredicate p )
{
    return std::none_of( std::begin( c ), std::end( c ), p );
}

}  // namespace algorithm
}  // namespace optix
