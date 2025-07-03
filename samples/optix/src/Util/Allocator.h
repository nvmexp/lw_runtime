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

#include <string>
#include <vector>


namespace optix {

//#define DEBUG_RANDOMIZED_ALLOCATION
void* randomizedMalloc( size_t size );
// include the following lines into a class for testing recompilation inconsistency
//#ifdef DEBUG_RANDOMIZED_ALLOCATION
//  inline void *operator new(size_t size) { return randomizedMalloc(size);}
//  inline void operator delete(void *p) {}
//#endif


struct AbstractSizeAllocatorImpl;

class AbstractSizeAllocator
{
  public:
    AbstractSizeAllocator( size_t elementSize );
    ~AbstractSizeAllocator();

    void* allocate();
    void deallocate( void* ptr );

    static const size_t MAX_THREADS = 16;

  private:
    // thread local allocators
    AbstractSizeAllocatorImpl* threads_impls[MAX_THREADS];
};

template <unsigned elementSize>
class FixedSizeAllocatorHelper
{
    static AbstractSizeAllocator allocator;

  public:
    static void* allocate() { return allocator.allocate(); }
    static void deallocate( void* ptr ) { allocator.deallocate( ptr ); }
};

template <unsigned    elementSize>
AbstractSizeAllocator FixedSizeAllocatorHelper<elementSize>::allocator( elementSize );

template <typename Ty>
class FixedSizeAllocator
{
  public:
    typedef std::size_t    size_type;
    typedef std::ptrdiff_t difference_type;
    typedef Ty*            pointer;
    typedef const Ty*      const_pointer;
    typedef Ty&            reference;
    typedef const Ty&      const_reference;
    typedef Ty             value_type;

    pointer address( reference val ) const { return &val; }
    const_pointer address( const_reference val ) const { return &val; }

    template <class Other>
    struct rebind
    {
        typedef FixedSizeAllocator<Other> other;
    };

    FixedSizeAllocator() throw() {}

    template <class Other>
    FixedSizeAllocator( const FixedSizeAllocator<Other>& ) throw()
    {
    }

    template <class Other>
    FixedSizeAllocator& operator=( const FixedSizeAllocator<Other>& )
    {
        return *this;
    }

    pointer allocate( size_type count, const void* = nullptr )
    {
        if( count == 1 )
            return static_cast<pointer>( FixedSizeAllocatorHelper<sizeof( Ty )>::allocate() );
        else
        {
            size_type count8 = count >> 3;
            switch( count8 )
            {
                case 0:
                    return static_cast<pointer>( FixedSizeAllocatorHelper<sizeof( Ty ) * 8>::allocate() );
                case 1:
                    return static_cast<pointer>( FixedSizeAllocatorHelper<sizeof( Ty ) * 16>::allocate() );
                case 2:
                    return static_cast<pointer>( FixedSizeAllocatorHelper<sizeof( Ty ) * 24>::allocate() );
                case 3:
                    return static_cast<pointer>( FixedSizeAllocatorHelper<sizeof( Ty ) * 32>::allocate() );
                default:
                    return static_cast<pointer>(::operator new( sizeof( Ty ) * count ) );
            }
        }
    }

    void deallocate( pointer ptr, size_type count )
    {
        if( count == 1 )
            return FixedSizeAllocatorHelper<sizeof( Ty )>::deallocate( ptr );
        else
        {
            size_type count8 = count >> 3;
            switch( count8 )
            {
                case 0:
                    return FixedSizeAllocatorHelper<sizeof( Ty ) * 8>::deallocate( ptr );
                case 1:
                    return FixedSizeAllocatorHelper<sizeof( Ty ) * 16>::deallocate( ptr );
                case 2:
                    return FixedSizeAllocatorHelper<sizeof( Ty ) * 24>::deallocate( ptr );
                case 3:
                    return FixedSizeAllocatorHelper<sizeof( Ty ) * 32>::deallocate( ptr );
                default:
                    return ::operator delete( ptr );
            }
        }
    }

    void construct( pointer ptr, const Ty& val ) { new( (void*)ptr ) Ty( val ); }

    void destroy( pointer ptr ) { ptr->Ty::~Ty(); }

    size_type max_size() const throw()
    {
        size_t _Count = ( size_t )( -1 ) / sizeof( Ty );
        return ( 0 < _Count ? _Count : 1 );
    }
};

template <class Ty, class Other>
inline bool operator==( const FixedSizeAllocator<Ty>&, const FixedSizeAllocator<Other>& )
{
    return sizeof( Ty ) == sizeof( Other );
}

template <class Ty, class Other>
inline bool operator!=( const FixedSizeAllocator<Ty>& left, const FixedSizeAllocator<Other>& right )
{
    return !( left == right );
}

typedef std::basic_string<char, std::char_traits<char>, FixedSizeAllocator<char>> FastString;
typedef std::vector<FastString, FixedSizeAllocator<FastString>> FastVectorString;

}  // end namespace optix
