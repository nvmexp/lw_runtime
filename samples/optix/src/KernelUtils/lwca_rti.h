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

////////////////////////////////////////////////////////////////////////////////
// lwda_rti.h
// Interfaces to internal functions for use in traversal code
//
////////////////////////////////////////////////////////////////////////////////

#ifndef __LWDA_RTI_H__
#define __LWDA_RTI_H__

#include <o6/optix.h>
#include <optixu/optixu_math.h>
#include <optixu/optixu_matrix.h>

#include <internal/optix_defines.h>
#include <private/optix_declarations_private.h>

#include <ExelwtionStrategy/CORTTypes.h>
#include <ExelwtionStrategy/FrameStatus.h>


namespace optix {

inline __device__ void rtiIntersectNode( unsigned int node )
{
    asm volatile( "call (), _rti_intersect_node, (%0);" : : "r"( node ) : );
}

inline __device__ void rtiIntersectNode( rtObject node )
{
    asm volatile( "call (), _rti_intersect_node, (%0);" : : "r"( *(unsigned int*)&node ) : );
}

inline __device__ void rtiIntersectPrimitive( unsigned int child, unsigned int primitive )
{
    asm volatile( "call (), _rti_intersect_primitive, (%0, %1);" : : "r"( child ), "r"( primitive ) : );
}

inline __device__ unsigned int rtiGetPrimitiveIndexOffset( unsigned int child )
{
    unsigned int tmp;
    asm volatile( "call (%0), _rti_get_primitive_index_offset, (%1);" : "=r"( tmp ) : "r"( child ) : );
    return tmp;
}

inline __device__ cort::AabbRequest rtiGetAabbRequest()
{
    cort::AabbRequest tmp;
    asm volatile( "call (), _rti_get_aabb_request, (%0);" : : OPTIX_ASM_PTR( &tmp ) : );
    return tmp;
}

inline __device__ void rtiComputeGeometryInstanceAabb( unsigned int giOffset, unsigned int primitive, unsigned int motionStep, void* aabb )
{
    asm volatile( "call (), _rti_compute_geometry_instance_aabb" OPTIX_BITNESS_SUFFIX ", (%0, %1, %2, %3);"
                  :
                  : "r"( giOffset ), "r"( primitive ), "r"( motionStep ), OPTIX_ASM_PTR( aabb )
                  : );
}


inline __device__ void rtiComputeGroupChildAabb( unsigned int groupOffset, unsigned int child, void* aabb )
{
    asm volatile( "call (), _rti_compute_group_child_aabb" OPTIX_BITNESS_SUFFIX ", (%0, %1, %2);"
                  :
                  : "r"( groupOffset ), "r"( child ), OPTIX_ASM_PTR( aabb )
                  : );
}

inline __device__ void rtiGatherMotionAabbs( unsigned int groupOffset, void* aabb )
{
    asm volatile( "call (), _rti_gather_motion_aabbs" OPTIX_BITNESS_SUFFIX ", (%0, %1);"
                  :
                  : "r"( groupOffset ), OPTIX_ASM_PTR( aabb )
                  : );
}

inline __device__ void rtiHandleTransformNode()
{
    asm volatile( "call (), _rti_handle_transform_node, ();" : : : );
}

// NOT TO BE USED IN PUBLICLY VISIBLE PTX
inline __device__ cort::FrameStatus* rtiGetStatusReturn()
{
    void* ptr;
    asm volatile( "call (%0), _rti_get_status_return, ();" : "=l"( ptr ) : : );
    return (cort::FrameStatus*)ptr;
}

inline __device__ void rtiYield()
{
    asm volatile( "call _rti_yield, ();" );
}

inline __device__ void rtiSetLwrrentAcceleration()
{
    asm volatile( "call _rti_set_lwrrent_acceleration, ();" );
}

inline __device__ void rtiResetStack()
{
    asm volatile( "call (), _rti_reset_stack, ();" );
}

inline __device__ bool rtiReportFullIntersectionFF( float t, unsigned int matlIndex, unsigned char hitkind, float f0, float f1 )
{
    int ret;
    asm volatile( "call (%0), _rti_report_full_intersection_ff, ( %1, %2, %3, %4, %5 );"
                  : "=r"( ret )
                  : "f"( t ), "r"( matlIndex ), "r"( (unsigned int)hitkind ), "f"( f0 ), "f"( f1 )
                  : );
    return ret;
}

inline __device__ void stat_add( void* ptr, int value )
{
    asm volatile( "call(), _rti_statistic_add_int" OPTIX_BITNESS_SUFFIX ", (%0, %1);"
                  :
                  : OPTIX_ASM_PTR( ptr ), "r"( value )
                  : );
}
inline __device__ void stat_add( void* ptr, unsigned int value )
{
    asm volatile( "call(), _rti_statistic_add_uint" OPTIX_BITNESS_SUFFIX ", (%0, %1);"
                  :
                  : OPTIX_ASM_PTR( ptr ), "r"( value )
                  : );
}
inline __device__ void stat_add( void* ptr, unsigned long long value )
{
    asm volatile( "call(), _rti_statistic_add_uint64" OPTIX_BITNESS_SUFFIX ", (%0, %1);"
                  :
                  : OPTIX_ASM_PTR( ptr ), "l"( value )
                  : );
}
inline __device__ void stat_add( void* ptr, float value )
{
    asm volatile( "call(), _rti_statistic_add_float" OPTIX_BITNESS_SUFFIX ", (%0, %1);"
                  :
                  : OPTIX_ASM_PTR( ptr ), "f"( value )
                  : );
}
inline __device__ void stat_vector_add( void* ptr, int value, unsigned int idx, unsigned int N )
{
    asm volatile( "call(), _rti_statistic_vector_add_int" OPTIX_BITNESS_SUFFIX ", (%0, %1, %2, %3);"
                  :
                  : OPTIX_ASM_PTR( ptr ), "r"( value ), "r"( idx ), "r"( N )
                  : );
}
inline __device__ void stat_vector_add( void* ptr, unsigned int value, unsigned int idx, unsigned int N )
{
    asm volatile( "call(), _rti_statistic_vector_add_uint" OPTIX_BITNESS_SUFFIX ", (%0, %1, %2, %3);"
                  :
                  : OPTIX_ASM_PTR( ptr ), "r"( value ), "r"( idx ), "r"( N )
                  : );
}
inline __device__ void stat_vector_add( void* ptr, unsigned long long value, unsigned int idx, unsigned int N )
{
    asm volatile( "call(), _rti_statistic_vector_add_uint64" OPTIX_BITNESS_SUFFIX ", (%0, %1, %2, %3);"
                  :
                  : OPTIX_ASM_PTR( ptr ), "l"( value ), "r"( idx ), "r"( N )
                  : );
}
inline __device__ void stat_vector_add( void* ptr, float value, unsigned int idx, unsigned int N )
{
    asm volatile( "call(), _rti_statistic_vector_add_float" OPTIX_BITNESS_SUFFIX ", (%0, %1, %2, %3);"
                  :
                  : OPTIX_ASM_PTR( ptr ), "f"( value ), "r"( idx ), "r"( N )
                  : );
}
template <typename T>
inline __device__ T stat_get( void* ptr );

template <>
inline __device__ int stat_get<int>( void* ptr )
{
    int tmp;
    asm volatile( "call (%0), _rti_statistic_get_int" OPTIX_BITNESS_SUFFIX ", (%1);"
                  : "=r"( tmp )
                  : OPTIX_ASM_PTR( ptr )
                  : );
    return tmp;
}
template <>
inline __device__ unsigned int stat_get<unsigned int>( void* ptr )
{
    unsigned int tmp;
    asm volatile( "call (%0), _rti_statistic_get_uint" OPTIX_BITNESS_SUFFIX ", (%1);"
                  : "=r"( tmp )
                  : OPTIX_ASM_PTR( ptr )
                  : );
    return tmp;
}
template <>
inline __device__ unsigned long long stat_get<unsigned long long>( void* ptr )
{
    unsigned long long tmp;
    asm volatile( "call (%0), _rti_statistic_get_uint64" OPTIX_BITNESS_SUFFIX ", (%1);"
                  : "=l"( tmp )
                  : OPTIX_ASM_PTR( ptr )
                  : );
    return tmp;
}
template <>
inline __device__ float stat_get<float>( void* ptr )
{
    float tmp;
    asm volatile( "call (%0), _rti_statistic_get_float" OPTIX_BITNESS_SUFFIX ", (%1);"
                  : "=f"( tmp )
                  : OPTIX_ASM_PTR( ptr )
                  : );
    return tmp;
}

template <typename T>
inline __device__ T stat_vector_get( void* ptr, unsigned int idx, unsigned int N );

template <>
inline __device__ int stat_vector_get<int>( void* ptr, unsigned int idx, unsigned int N )
{
    int tmp;
    asm volatile( "call (%0), _rti_statistic_vector_get_int" OPTIX_BITNESS_SUFFIX ", (%1, %2, %3);"
                  : "=r"( tmp )
                  : OPTIX_ASM_PTR( ptr ), "r"( idx ), "r"( N )
                  : );
    return tmp;
}
template <>
inline __device__ unsigned int stat_vector_get<unsigned int>( void* ptr, unsigned int idx, unsigned int N )
{
    unsigned int tmp;
    asm volatile( "call (%0), _rti_statistic_vector_get_uint" OPTIX_BITNESS_SUFFIX ", (%1, %2, %3);"
                  : "=r"( tmp )
                  : OPTIX_ASM_PTR( ptr ), "r"( idx ), "r"( N )
                  : );
    return tmp;
}
template <>
inline __device__ unsigned long long stat_vector_get<unsigned long long>( void* ptr, unsigned int idx, unsigned int N )
{
    unsigned long long tmp;
    asm volatile( "call (%0), _rti_statistic_vector_get_uint64" OPTIX_BITNESS_SUFFIX ", (%1, %2, %3);"
                  : "=l"( tmp )
                  : OPTIX_ASM_PTR( ptr ), "r"( idx ), "r"( N )
                  : );
    return tmp;
}
template <>
inline __device__ float stat_vector_get<float>( void* ptr, unsigned int idx, unsigned int N )
{
    float tmp;
    asm volatile( "call (%0), _rti_statistic_vector_get_float" OPTIX_BITNESS_SUFFIX ", (%1, %2, %3);"
                  : "=f"( tmp )
                  : OPTIX_ASM_PTR( ptr ), "r"( idx ), "r"( N )
                  : );
    return tmp;
}

template <class T>
struct statistic
{
    inline __device__ void operator++() { stat_add( this, T( 1 ) ); }
    inline __device__ void operator++( int ) { stat_add( this, T( 1 ) ); }
    inline __device__ void operator+=( T value ) { stat_add( this, value ); }
    inline __device__ void operator--() { stat_add( this, T( -1 ) ); }
    inline __device__ void operator--( int ) { stat_add( this, T( -1 ) ); }
    inline __device__ void operator-=( T value ) { stat_add( this, -value ); }

    inline __device__ T get() { return stat_get<T>( this ); }
};

template <class T, unsigned int N>
struct vector_statistic;

template <class T, unsigned int N>
struct vector_statistic_helper
{
    inline __device__ vector_statistic_helper( vector_statistic<T, N>* obj, unsigned int idx )
        : m_obj( obj )
        , m_idx( idx )
    {
    }
    inline __device__ void operator++() { stat_vector_add( m_obj, T( 1 ), m_idx, N ); }
    inline __device__ void operator++( int ) { stat_vector_add( m_obj, T( 1 ), m_idx, N ); }
    inline __device__ void operator+=( T value ) { stat_vector_add( m_obj, value, m_idx, N ); }
    inline __device__ void operator--() { stat_vector_add( m_obj, T( -1 ), m_idx, N ); }
    inline __device__ void operator--( int ) { stat_vector_add( m_obj, T( -1 ), m_idx, N ); }
    inline __device__ void operator-=( T value ) { stat_vector_add( m_obj, -value, m_idx, N ); }

  private:
    vector_statistic<T, N>* m_obj;
    unsigned int m_idx;
};

template <class T, unsigned int N>
struct vector_statistic
{
    inline __device__ vector_statistic_helper<T, N> operator[]( unsigned int idx )
    {
        return vector_statistic_helper<T, N>( this, idx );
    }
    inline __device__ T get( unsigned int idx ) { return stat_vector_get<T>( this, idx, N ); }
};

__device__ __forceinline__ void rtiProfileEvent( unsigned int idx )
{
    asm volatile( "call _rti_profile_event, (%0);"
                  :
                  /* no return value */
                  : "r"( idx )
                  : );
}

__device__ __forceinline__ void rtiCPUDebug( unsigned int val )
{
    asm volatile( "call _rti_cpu_debug, (%0);"
                  :
                  /* no return value */
                  : "r"( val )
                  : );
}

__device__ __forceinline__ void rtiCPUDebug( unsigned long long val )
{
    asm volatile( "call _rti_cpu_debug_64, (%0);"
                  :
                  /* no return value */
                  : "l"( val )
                  : );
}

__device__ __forceinline__ void rtiCPUDebug( float val )
{
    asm volatile( "call _rti_cpu_debug_float, (%0);"
                  :
                  /* no return value */
                  : "f"( val )
                  : );
}

__device__ __forceinline__ void rtiCPUDebug( char* str )
{
    asm volatile( "call _rti_cpu_debug_string" OPTIX_BITNESS_SUFFIX ", (%0);"
                  :
                  /* no return value */
                  : OPTIX_ASM_PTR( str )
                  : );
}

__device__ __forceinline__ void rtiCPUDebug( const float3& f )
{
    rtiCPUDebug( f.x );
    rtiCPUDebug( f.y );
    rtiCPUDebug( f.z );
}

__device__ __forceinline__ void rtiCPUDebug( const float4& f )
{
    rtiCPUDebug( f.x );
    rtiCPUDebug( f.y );
    rtiCPUDebug( f.z );
    rtiCPUDebug( f.w );
}
}

#define RTI_DECLARE_STATISTIC( type, name, description )                                                               \
    __device__ optix::statistic<type> name;                                                                            \
    namespace rti_description {                                                                                        \
    __device__ char name[] = description;                                                                              \
    }

#define RTI_DECLARE_VECTOR_STATISTIC( type, name, size, description )                                                  \
    __device__ optix::vector_statistic<type, size> name;                                                               \
    namespace rti_description {                                                                                        \
    __device__ char name[] = description;                                                                              \
    }

// Adds a comment to the stitched PTX.  Argument text must be a legal variable name
// (e.g. no spaces).
#define rtiComment( _a ) asm volatile( "call _rti_comment_" #_a " , ();" );

// Helper functions to bypass lwcc's inability to generate vectorized loads and stores for
// local memory.
inline __device__ void uint4_vec_load_l( void* ptr, unsigned int& a, unsigned int& b, unsigned int& c, unsigned int& d )
{
    asm volatile( "ld.local.v4.u32 {%0, %1, %2, %3}, [%4];"
                  /*output*/
                  : "=r"( a ), "=r"( b ), "=r"( c ), "=r"( d )
                  /*input*/
                  : OPTIX_ASM_PTR( ptr )
                  /*clobber*/
                  : );
}

inline __device__ void uint4_vec_store_l( void* ptr, unsigned int a, unsigned int b, unsigned int c, unsigned int d )
{
    asm volatile( "st.local.v4.u32 [%0], {%1, %2, %3, %4};"
                  /*output*/
                  :
                  /*input*/
                  : OPTIX_ASM_PTR( ptr ), "r"( a ), "r"( b ), "r"( c ), "r"( d )
                  /*clobber*/
                  : );
}

inline __device__ void dbgbreak()
{
#if defined __LWDA_ARCH__
#if __LWDA_ARCH__ >= 110
    asm volatile( "brkpt;" );
#else
#pragma message( "WARNING: dbgbreak() disabled. Requires sm_11 or greater. " )
#endif
#endif
}

static __device__ __forceinline__ void rtiMainSwitch()
{
    asm volatile( "call (), _rti_main_switch, ();" : : : );
}

static __device__ __forceinline__ void rtiObjectRecordsSet( unsigned long long ptr, char*& addr, unsigned int& size, optix::ObjectStorageType& type )
{
    unsigned long long return_addr;
    unsigned int       return_size;
    unsigned int       return_type;

    asm volatile( "call (%0,%1,%2), _rti_object_records_set, (%3);"
                  : "=l"( return_addr ), "=r"( return_size ), "=r"( return_type )
                  : "l"( ptr )
                  : );

    addr = (char*)(size_t)return_addr;
    size = return_size;
    type = (optix::ObjectStorageType)return_type;
}


#endif /* __LWDA_RTI_H__ */
