#ifndef __optix_legacy_testing_rtPrintfLegacy_definition_h__
#define __optix_legacy_testing_rtPrintfLegacy_definition_h__

namespace optix_legacy_testing {

/*
   Type descriptors for printf arguments:
   0 = 32 bit integer value
   1 = 64 bit integer value
   2 = 32 bit float value
   3 = 64 bit double value
  */
template <typename T>
struct rt_print_t
{
    static const int desc = 0;
};
/*
  Template specialization for pointers.  We treat pointers like 64 bit integer
  values.  Without this specialization pointers have 0 as type descriptor,
  which stands for 32 bit values.
  */
template <typename T>
struct rt_print_t<T*>
{
    static const int desc = 1;
};
template <>
struct rt_print_t<long>
{
    static const int desc = 1;
};
template <>
struct rt_print_t<unsigned long>
{
    static const int desc = 1;
};
template <>
struct rt_print_t<long long>
{
    static const int desc = 1;
};
template <>
struct rt_print_t<unsigned long long>
{
    static const int desc = 1;
};
template <>
struct rt_print_t<float>
{
    static const int desc = 2;
};
template <>
struct rt_print_t<double>
{
    static const int desc = 3;
};

static __forceinline__ __device__ int rt_print_strlen( const char* s )
{
    const char* p = s;
    while( *p )
        ++p;
    return p - s;
}

template <typename T>
static __forceinline__ __device__ int rt_print_arg( T arg, int off )
{
    const int sz           = max( 4, (int)sizeof( arg ) );
    const int tmp_typedesc = rt_print_t<T>::desc;
    const int typedesc     = ( tmp_typedesc == 1 && sizeof( T ) == 4 ) ? 0 : tmp_typedesc;

    const unsigned int* p;

    /* Get a pointer to a (at least) 32 bit value. */
    unsigned int iarg;
    if( sizeof( arg ) < 4 )
    {
        /*        
      This redundant cast is needed to avoid a compilation warning when T is a
      pointer.  Casting a pointer to a smaller integer results in a warning, so
      we cast it first to an int of the right size and then we cast that to
      unsigned int.  Notice that this branch of the if-statement is never
      alwtally exelwted for pointers, since their size is greater than 4.
      Unfortunatelly intptr_t seems not to avaialable, so I fall back to use
      unsigned long long.
      */
        unsigned long long upcasted_arg = (unsigned long long)arg;
        iarg                            = (unsigned int)upcasted_arg;
        p                               = &iarg;
    }
    else
    {
        p = (unsigned int*)&arg;
    }

    /* Write type descriptor. */
    asm volatile( "call (), _rt_print_write32, (%0, %1);" : : "r"( typedesc ), "r"( off ) : );

    /* Write argument. */
    for( int i = 0; i < sz / 4; ++i )
    {
        asm volatile( "call (), _rt_print_write32, (%0, %1);" : : "r"( p[i] ), "r"( off + ( i + 1 ) * 4 ) : );
    }

    return sz;
}

static __forceinline__ __device__ int rt_print_active()
{
    int ret;
    asm volatile( "call (%0), _rt_print_active, ();" : "=r"( ret ) : : );
    return ret;
}

static __forceinline__ __device__ int rt_print_start( const char* fmt, int sz )
{
    int ret;
    asm volatile( "call (%0), _rt_print_start" OPTIX_BITNESS_SUFFIX ", (%1, %2);"
                  : "=r"( ret )
                  : OPTIX_ASM_PTR( fmt ), "r"( sz )
                  : );
    return ret;
}

#define _RT_PRINTF_1()                                                                                                 \
    if( !optix_legacy_testing::rt_print_active() )                                                                     \
        return;                                                                                                        \
    /* Compute length of header (=batchsize) plus format string. */                                                    \
    const int fmtlen = optix_legacy_testing::rt_print_strlen( fmt );                                                   \
    int       sz     = 4 + fmtlen + 1;                                                                                 \
    sz               = ( sz + 3 ) & ~3; /* align */

#define _RT_PRINTF_2()                                                                                                 \
    int off; /* offset where to start writing args */                                                                  \
    if( !( off = optix_legacy_testing::rt_print_start( fmt, sz ) ) )                                                   \
        return; /* print buffer is full */

#define _RT_PRINTF_ARG_1( a )                                                                                          \
    /* Sum up argument sizes. */                                                                                       \
    sz += 4; /* type descriptor */                                                                                     \
    sz += max( 4, static_cast<unsigned int>( sizeof( a ) ) );

#define _RT_PRINTF_ARG_2( a )                                                                                          \
    /* Write out argument. */                                                                                          \
    off += optix_legacy_testing::rt_print_arg( a, off );                                                               \
    off += 4; /* space for type desc */
}

static inline __device__ void rtPrintfLegacy( const char* fmt )
{
    _RT_PRINTF_1();
    optix_legacy_testing::rt_print_start( fmt, sz );
}
template <typename T1>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
}
template <typename T1, typename T2>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
}
template <typename T1, typename T2, typename T3>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
}
template <typename T1, typename T2, typename T3, typename T4>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
}
template <typename T1, typename T2, typename T3, typename T4, typename T5>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_ARG_1( arg5 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
    _RT_PRINTF_ARG_2( arg5 );
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_ARG_1( arg5 );
    _RT_PRINTF_ARG_1( arg6 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
    _RT_PRINTF_ARG_2( arg5 );
    _RT_PRINTF_ARG_2( arg6 );
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_ARG_1( arg5 );
    _RT_PRINTF_ARG_1( arg6 );
    _RT_PRINTF_ARG_1( arg7 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
    _RT_PRINTF_ARG_2( arg5 );
    _RT_PRINTF_ARG_2( arg6 );
    _RT_PRINTF_ARG_2( arg7 );
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_ARG_1( arg5 );
    _RT_PRINTF_ARG_1( arg6 );
    _RT_PRINTF_ARG_1( arg7 );
    _RT_PRINTF_ARG_1( arg8 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
    _RT_PRINTF_ARG_2( arg5 );
    _RT_PRINTF_ARG_2( arg6 );
    _RT_PRINTF_ARG_2( arg7 );
    _RT_PRINTF_ARG_2( arg8 );
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_ARG_1( arg5 );
    _RT_PRINTF_ARG_1( arg6 );
    _RT_PRINTF_ARG_1( arg7 );
    _RT_PRINTF_ARG_1( arg8 );
    _RT_PRINTF_ARG_1( arg9 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
    _RT_PRINTF_ARG_2( arg5 );
    _RT_PRINTF_ARG_2( arg6 );
    _RT_PRINTF_ARG_2( arg7 );
    _RT_PRINTF_ARG_2( arg8 );
    _RT_PRINTF_ARG_2( arg9 );
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_ARG_1( arg5 );
    _RT_PRINTF_ARG_1( arg6 );
    _RT_PRINTF_ARG_1( arg7 );
    _RT_PRINTF_ARG_1( arg8 );
    _RT_PRINTF_ARG_1( arg9 );
    _RT_PRINTF_ARG_1( arg10 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
    _RT_PRINTF_ARG_2( arg5 );
    _RT_PRINTF_ARG_2( arg6 );
    _RT_PRINTF_ARG_2( arg7 );
    _RT_PRINTF_ARG_2( arg8 );
    _RT_PRINTF_ARG_2( arg9 );
    _RT_PRINTF_ARG_2( arg10 );
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_ARG_1( arg5 );
    _RT_PRINTF_ARG_1( arg6 );
    _RT_PRINTF_ARG_1( arg7 );
    _RT_PRINTF_ARG_1( arg8 );
    _RT_PRINTF_ARG_1( arg9 );
    _RT_PRINTF_ARG_1( arg10 );
    _RT_PRINTF_ARG_1( arg11 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
    _RT_PRINTF_ARG_2( arg5 );
    _RT_PRINTF_ARG_2( arg6 );
    _RT_PRINTF_ARG_2( arg7 );
    _RT_PRINTF_ARG_2( arg8 );
    _RT_PRINTF_ARG_2( arg9 );
    _RT_PRINTF_ARG_2( arg10 );
    _RT_PRINTF_ARG_2( arg11 );
}
template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
static inline __device__ void rtPrintfLegacy( const char* fmt, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12 )
{
    _RT_PRINTF_1();
    _RT_PRINTF_ARG_1( arg1 );
    _RT_PRINTF_ARG_1( arg2 );
    _RT_PRINTF_ARG_1( arg3 );
    _RT_PRINTF_ARG_1( arg4 );
    _RT_PRINTF_ARG_1( arg5 );
    _RT_PRINTF_ARG_1( arg6 );
    _RT_PRINTF_ARG_1( arg7 );
    _RT_PRINTF_ARG_1( arg8 );
    _RT_PRINTF_ARG_1( arg9 );
    _RT_PRINTF_ARG_1( arg10 );
    _RT_PRINTF_ARG_1( arg11 );
    _RT_PRINTF_ARG_1( arg12 );
    _RT_PRINTF_2();
    _RT_PRINTF_ARG_2( arg1 );
    _RT_PRINTF_ARG_2( arg2 );
    _RT_PRINTF_ARG_2( arg3 );
    _RT_PRINTF_ARG_2( arg4 );
    _RT_PRINTF_ARG_2( arg5 );
    _RT_PRINTF_ARG_2( arg6 );
    _RT_PRINTF_ARG_2( arg7 );
    _RT_PRINTF_ARG_2( arg8 );
    _RT_PRINTF_ARG_2( arg9 );
    _RT_PRINTF_ARG_2( arg10 );
    _RT_PRINTF_ARG_2( arg11 );
    _RT_PRINTF_ARG_2( arg12 );
}
/** @} */

#undef _RT_PRINTF_1
#undef _RT_PRINTF_2
#undef _RT_PRINTF_ARG_1
#undef _RT_PRINTF_ARG_2

#endif
