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

/******************************************************************************
 * Copyright 1986-2008 by mental images GmbH, Fasanenstr. 81, D-10623 Berlin,
 * Germany. All rights reserved.
 ******************************************************************************
 *
 *      This module defines a small set of free functions which shall be
 *      considered an implementation detail. The application programmer's
 *      interface to atomic counters is the AtomicCounter class defined in
 *      AtomicCounter.h.
 *
 *****************************************************************************/

#pragma once

///////////////////////////////////////////////////////////////////////////////
// Implementation for x86, x86-64, GNU C Compiler, Intel C Compiler
///////////////////////////////////////////////////////////////////////////////

#if( defined( __i386__ ) || defined( __x86_64__ ) ) && ( defined( __GNUC__ ) || defined( __INTEL_COMPILER ) )

// LWCC warns that "cc" clobber is ignored.  I'm sick of that warning (which I
// can't disable), so I'm working around it by not specifying it for LWCC
// builds.
#ifdef __LWDACC__
#define CC_CLOBBER
#define CC_CLOBBER_COMMA
#else
#define CC_CLOBBER "cc"
#define CC_CLOBBER_COMMA "cc",
#endif

namespace optix {

typedef unsigned int volatile NativeAtomicCounter;

inline void create_counter( NativeAtomicCounter& counter, unsigned int i )
{
    counter = i;
}

inline void destroy_counter( NativeAtomicCounter& counter )
{
}

inline unsigned int get_counter( NativeAtomicCounter const& counter )
{
    return counter;
}

inline unsigned int atomic_add( NativeAtomicCounter& counter, unsigned int i )
{
    unsigned int retval;
    asm volatile(
        "movl %2,%0\n"
        "lock; xaddl %0,%1\n"
        "addl %2,%0\n"
        : "=&r"( retval ), "+m"( counter )
        : "r"( i )
        : CC_CLOBBER );
    return retval;
}

inline unsigned int atomic_sub( NativeAtomicCounter& counter, unsigned int i )
{
    unsigned int retval;
    asm volatile(
        "neg %2\n"
        "movl %2,%0\n"
        "lock; xaddl %0,%1\n"
        "addl %2,%0\n"
        : "=&r"( retval ), "+m"( counter )
        : "r"( i )
        : CC_CLOBBER_COMMA "%2" );
    return retval;
}

inline unsigned int atomic_inc( NativeAtomicCounter& counter )
{
    unsigned int retval;
    asm volatile(
        "movl $1,%0\n"
        "lock; xaddl %0,%1\n"
        "addl $1,%0\n"
        : "=&r"( retval ), "+m"( counter )
        :
        : CC_CLOBBER );
    return retval;
}

inline unsigned int atomic_dec( NativeAtomicCounter& counter )
{
    unsigned int retval;
    asm volatile(
        "movl $-1,%0\n"
        "lock; xaddl %0,%1\n"
        "addl $-1,%0\n"
        : "=&r"( retval ), "+m"( counter )
        :
        : CC_CLOBBER );
    return retval;
}

inline unsigned int atomic_post_inc( NativeAtomicCounter& counter )
{
    unsigned int retval;
    asm volatile(
        "movl $1,%0\n"
        "lock; xaddl %0,%1\n"
        : "=&r"( retval ), "+m"( counter )
        :
        : CC_CLOBBER );
    return retval;
}

inline unsigned int atomic_post_dec( NativeAtomicCounter& counter )
{
    unsigned int retval;
    asm volatile(
        "movl $-1,%0\n"
        "lock; xaddl %0,%1\n"
        : "=&r"( retval ), "+m"( counter )
        :
        : CC_CLOBBER );
    return retval;
}

}  // end namespace optix

///////////////////////////////////////////////////////////////////////////////
// Implementation for x86, x86-64, Microsoft Visual C++
///////////////////////////////////////////////////////////////////////////////

#elif( defined( X86 ) || defined( _M_IX86 ) || defined( _M_X64 ) ) && defined( _MSC_VER )

// In order to work around a VS9 bug, math.h needs to be included before intrin.h.
// Since this is the only place where this is done, we will pre-include math.h here.
// https://connect.microsoft.com/VisualStudio/feedback/ViewFeedback.aspx?FeedbackID=381422&wa=wsignin1.0
#include <intrin.h>
#include <math.h>

#pragma intrinsic( _InterlockedExchangeAdd )
#pragma intrinsic( _InterlockedCompareExchange )

namespace optix {

typedef long volatile NativeAtomicCounter;

inline void create_counter( NativeAtomicCounter& counter, unsigned int i )
{
    counter = i;
}

inline void destroy_counter( NativeAtomicCounter& counter )
{
}

inline unsigned int get_counter( NativeAtomicCounter const& counter )
{
    return static_cast<unsigned int>( counter );
}

inline unsigned int atomic_add( NativeAtomicCounter& counter, unsigned int i )
{
    return _InterlockedExchangeAdd( &counter, i ) + i;
}

inline unsigned int atomic_sub( NativeAtomicCounter& counter, unsigned int i )
{
    return _InterlockedExchangeAdd( &counter, -static_cast<int>( i ) ) - i;
}

inline unsigned int atomic_inc( NativeAtomicCounter& counter )
{
    return _InterlockedExchangeAdd( &counter, 1L ) + 1L;
}

inline unsigned int atomic_dec( NativeAtomicCounter& counter )
{
    return _InterlockedExchangeAdd( &counter, -1L ) - 1L;
}

inline unsigned int atomic_post_inc( NativeAtomicCounter& counter )
{
    return _InterlockedExchangeAdd( &counter, 1L );
}

inline unsigned int atomic_post_dec( NativeAtomicCounter& counter )
{
    return _InterlockedExchangeAdd( &counter, -1L );
}

}  // end namespace optix

///////////////////////////////////////////////////////////////////////////////
// Generic Implementation for the GNU C++ Compiler
///////////////////////////////////////////////////////////////////////////////

#elif( defined( __GNUC__ ) && defined( __powerpc64__ ) ) || defined( __ICC ) || defined( __aarch64__ )

#if !defined( __ia64__ ) && !defined( __APPLE__ ) && !defined( MI_CELL ) && !defined( __powerpc64__ ) && !defined( __aarch64__ )
#warning "No native atomic counting implementation available."
#warning "Using GCC library code as a fallback."
#endif

#include <ext/atomicity.h>

namespace optix {

typedef _Atomic_word volatile NativeAtomicCounter;

inline void create_counter( NativeAtomicCounter& counter, unsigned int i )
{
    counter = i;
}

inline void destroy_counter( NativeAtomicCounter& counter )
{
}

inline unsigned int get_counter( NativeAtomicCounter const& counter )
{
    return counter;
}

inline unsigned int atomic_add( NativeAtomicCounter& counter, unsigned int i )
{
    return __gnu_cxx::__exchange_and_add( &counter, static_cast<int>( i ) ) + i;
}

inline unsigned int atomic_sub( NativeAtomicCounter& counter, unsigned int i )
{
    return __gnu_cxx::__exchange_and_add( &counter, -static_cast<int>( i ) ) - i;
}

inline unsigned int atomic_inc( NativeAtomicCounter& counter )
{
    return atomic_add( counter, 1u );
}

inline unsigned int atomic_dec( NativeAtomicCounter& counter )
{
    return atomic_sub( counter, 1u );
}

inline unsigned int atomic_post_inc( NativeAtomicCounter& counter )
{
    return __gnu_cxx::__exchange_and_add( &counter, 1 );
}

inline unsigned int atomic_post_dec( NativeAtomicCounter& counter )
{
    return __gnu_cxx::__exchange_and_add( &counter, -1 );
}

}  // end namespace optix

///////////////////////////////////////////////////////////////////////////////
// Generic implementation using pthreads mutex-locking
///////////////////////////////////////////////////////////////////////////////

#elif defined( _REENTRANT ) || defined( _THREAD_SAFE ) || defined( _PTHREADS )

#if !defined( IRIX )
#warning "No lock-free atomic counting implementation available."
#warning "Using pthreads as a fallback."
#endif

#include <pthread.h>  // POSIX threads
#include <utility>    // std::pair<>

namespace optix {

typedef std::pair<pthread_mutex_t, unsigned int volatile> NativeAtomicCounter;

inline void create_counter( NativeAtomicCounter& counter, unsigned int i )
{
    pthread_mutex_init( &counter.first, 0 );
    counter.second = i;
}

inline void destroy_counter( NativeAtomicCounter& counter )
{
    pthread_mutex_destroy( &counter.first );
}

inline unsigned int get_counter( NativeAtomicCounter const& counter )
{
    return counter.second;
}

inline unsigned int atomic_add( NativeAtomicCounter& counter, unsigned int i )
{
    pthread_mutex_lock( &counter.first );
    unsigned int const retval = ( counter.second += i );
    pthread_mutex_unlock( &counter.first );
    return retval;
}

inline unsigned int atomic_sub( NativeAtomicCounter& counter, unsigned int i )
{
    pthread_mutex_lock( &counter.first );
    unsigned int const retval = ( counter.second -= i );
    pthread_mutex_unlock( &counter.first );
    return retval;
}

inline unsigned int atomic_inc( NativeAtomicCounter& counter )
{
    return atomic_add( counter, 1u );
}

inline unsigned int atomic_dec( NativeAtomicCounter& counter )
{
    return atomic_sub( counter, 1u );
}

inline unsigned int atomic_post_inc( NativeAtomicCounter& counter )
{
    pthread_mutex_lock( &counter.first );
    unsigned int const retval = counter.second++;
    pthread_mutex_unlock( &counter.first );
    return retval;
}

inline unsigned int atomic_post_dec( NativeAtomicCounter& counter )
{
    pthread_mutex_lock( &counter.first );
    unsigned int const retval = counter.second--;
    pthread_mutex_unlock( &counter.first );
    return retval;
}

}  // end namespace optix

#undef CC_CLOBBER
#undef CC_CLOBBER_COMMA

///////////////////////////////////////////////////////////////////////////////
// Report an error if no implementation was selected
///////////////////////////////////////////////////////////////////////////////

#else
#error "No atomic counter implementation available for this platform."
#endif
