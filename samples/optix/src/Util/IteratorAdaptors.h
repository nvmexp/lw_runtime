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
/*
   * Instead of using iter.second everywhere, use this to iterate over the values in a map.
   */
template <typename T, typename Base>
class ValueIterator
{
  public:
    ValueIterator( const Base& base )
        : base( base )
    {
    }
    bool operator==( const ValueIterator<T, Base>& other ) { return base == other.base; }
    bool operator!=( const ValueIterator<T, Base>& other ) { return base != other.base; }

    ValueIterator& operator++()
    {
        ++base;
        return *this;
    }
    ValueIterator operator++( int )
    {
        ValueIterator tmp( base );
        base++;
        return tmp;
    }

    const T& operator*() { return base->second; }
    const T& operator*() const { return base->second; }

  private:
    Base base;
};

template <typename T, typename Base>
class KeyIterator
{
  public:
    KeyIterator( const Base& base )
        : base( base )
    {
    }
    bool operator==( const KeyIterator<T, Base>& other ) { return base == other.base; }
    bool operator!=( const KeyIterator<T, Base>& other ) { return base != other.base; }

    KeyIterator& operator++()
    {
        ++base;
        return *this;
    }
    KeyIterator operator++( int )
    {
        KeyIterator tmp( base );
        base++;
        return tmp;
    }

    const T& operator*() { return base->first; }
    const T& operator*() const { return base->first; }

  private:
    Base base;
};
}  // end namespace optix
