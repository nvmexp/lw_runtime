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

//
// NOTE: wherever possible, this file should just forward to the api
// object memory functions.  Do not put control logic here - it is too
// hard to find.
//

#define OPTIX_OPTIONAL_FEATURE_DEPRECATED_ATTRIBUTES
#define OPTIX_OPTIONAL_FEATURE_INTERNAL_ATTRIBUTES
#include <c-api/rtapi.h>

#include <o6/optix.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <optix_gl_interop.h>
#include <private/optix_declarations_private.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/RTCore.h>
#include <Context/SharedProgramManager.h>
#include <Context/ValidationManager.h>
#include <Control/ErrorManager.h>
#include <Device/Device.h>
#include <Device/DeviceManager.h>
#include <Device/DriverVersion.h>
#include <Exceptions/ExceptionHelpers.h>
#include <Objects/Acceleration.h>
#include <Objects/Buffer.h>
#include <Objects/CommandList.h>
#include <Objects/Geometry.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/GlobalScope.h>
#include <Objects/Group.h>
#include <Objects/Material.h>
#include <Objects/PostprocessingStageDenoiser.h>
#include <Objects/PostprocessingStageSSIMPredictor.h>
#include <Objects/PostprocessingStageTonemap.h>
#include <Objects/Program.h>
#include <Objects/Selector.h>
#include <Objects/StreamBuffer.h>
#include <Objects/TextureSampler.h>
#include <Objects/Transform.h>
#include <Objects/Variable.h>
#include <c-api/ApiCapture.h>

#include <Exceptions/AlreadyMapped.h>
#include <Exceptions/TypeMismatch.h>
#include <corelib/math/MathUtil.h>
#include <corelib/misc/Cast.h>
#include <corelib/system/Preprocessor.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/Exception.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/GLFunctions.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <optix_d3d10_interop.h>
#include <optix_d3d11_interop.h>
#include <optix_d3d9_interop.h>
#endif

#include <private/optix_version_string.h>  // for OPTIX_BUILD_VERSION

using namespace optix;
using namespace prodlib;
using namespace corelib;


// Make sure our size type is appropriate.
static_assert( sizeof( RTsize ) == sizeof( size_t ), "Bad RTsize size" );

/************************************
 **
 **    Colwenience functions for casting back and forth between the opaque types and
 **    internal types.  These functions also guarantees a certain amount of type safety as
 **    the parameter type dictates the output type avoiding accidentally casting to the
 **    wrong type.
 **
 ***********************************/
#define DECLARE_FROM_API_TYPE_CAST( implType, apiType )                                                                \
    static implType api_cast( apiType ptr ) { return reinterpret_cast<implType>( ptr ); }
#define DECLARE_TO_API_TYPE_CAST( implType, apiType )                                                                  \
    static apiType api_cast( implType ptr ) { return reinterpret_cast<apiType>( ptr ); }
#define DECLARE_API_TYPE_CAST( implType, apiType )                                                                     \
    DECLARE_FROM_API_TYPE_CAST( implType, apiType )                                                                    \
    DECLARE_TO_API_TYPE_CAST( implType, apiType )

DECLARE_API_TYPE_CAST( Acceleration*, RTacceleration );
DECLARE_API_TYPE_CAST( Buffer*, RTbuffer );
DECLARE_FROM_API_TYPE_CAST( Buffer**, RTbuffer* );
DECLARE_API_TYPE_CAST( CommandList*, RTcommandlist );
DECLARE_API_TYPE_CAST( Context*, RTcontext );
DECLARE_API_TYPE_CAST( Geometry*, RTgeometry );
DECLARE_API_TYPE_CAST( GeometryTriangles*, RTgeometrytriangles );
DECLARE_API_TYPE_CAST( GeometryInstance*, RTgeometryinstance );
DECLARE_API_TYPE_CAST( GeometryGroup*, RTgeometrygroup );
DECLARE_API_TYPE_CAST( Group*, RTgroup );
DECLARE_API_TYPE_CAST( Material*, RTmaterial );
DECLARE_API_TYPE_CAST( PostprocessingStage*, RTpostprocessingstage );
DECLARE_API_TYPE_CAST( Selector*, RTselector );
DECLARE_API_TYPE_CAST( TextureSampler*, RTtexturesampler );
DECLARE_API_TYPE_CAST( Transform*, RTtransform );
DECLARE_API_TYPE_CAST( Variable*, RTvariable );

// Program's api_cast is handled below specially
// DECLARE_API_TYPE_CAST(Program*,            RTprogram);


/************************************
 **
 **    Colwenience functions for translating null programs back and forth to the
 **    NullProgram object.
 **
 ***********************************/
static Program* translateProgramOutOfAPI( Program* p, Context* c )
{
    Program* null_program = c->getSharedProgramManager()->getNullProgram();
    return ( p == null_program ) ? nullptr : p;
}

static Program* translateProgramIntoAPI( Program* p, Context* c )
{
    return p ? p : c->getSharedProgramManager()->getNullProgram();
}

static Program* api_cast_ignore_null( RTprogram old )
{
    return reinterpret_cast<Program*>( old );
}

static RTprogram api_cast_ignore_null( Program* old )
{
    return reinterpret_cast<RTprogram>( old );
}

static Program* api_cast( RTprogram old, Context* c )
{
    return translateProgramIntoAPI( api_cast_ignore_null( old ), c );
}

static RTprogram api_cast( Program* old, Context* c )
{
    return api_cast_ignore_null( translateProgramOutOfAPI( old, c ) );
}

/************************************
 **
 **    Colwenience macros to
 **    simplify error checking
 **
 ***********************************/

#define HANDLE_EXCEPTIONS_NAME( _context, _func )                                                                      \
    catch( const Exception& e )                                                                                        \
    {                                                                                                                  \
        ( _context )->getErrorManager()->setErrorString( ( _func ), e );                                               \
        return getRTresultFromException( &e );                                                                         \
    }                                                                                                                  \
    catch( const std::exception& e )                                                                                   \
    {                                                                                                                  \
        ( _context )->getErrorManager()->setErrorString( ( _func ), e );                                               \
        return RT_ERROR_UNKNOWN;                                                                                       \
    }                                                                                                                  \
    catch( ... )                                                                                                       \
    {                                                                                                                  \
        ( _context )->getErrorManager()->setErrorString( ( _func ), "Caught unknown exception", RT_ERROR_UNKNOWN );    \
        return RT_ERROR_UNKNOWN;                                                                                       \
    }

#define HANDLE_EXCEPTIONS( _context ) HANDLE_EXCEPTIONS_NAME( _context, RTAPI_FUNC )

#define HANDLE_EXCEPTIONS_NO_CONTEXT_NAME_CODE( _func, _code )                                                         \
    catch( const Exception& e )                                                                                        \
    {                                                                                                                  \
        RTresult errorCode = getRTresultFromException( &e );                                                           \
        lerr << ( _func ) << " error: Exception caught(" << errorCode << "): " << e.getDescription() << std::endl;     \
        return errorCode;                                                                                              \
    }                                                                                                                  \
    catch( const std::exception& e )                                                                                   \
    {                                                                                                                  \
        lerr << ( _func ) << " error: std::exception caught: " << e.what() << std::endl;                               \
        return ( _code );                                                                                              \
    }                                                                                                                  \
    catch( ... )                                                                                                       \
    {                                                                                                                  \
        lerr << ( _func ) << " error: Unknown exception caught." << std::endl;                                         \
        return ( _code );                                                                                              \
    }

#define HANDLE_EXCEPTIONS_NO_CONTEXT_NAME( _func ) HANDLE_EXCEPTIONS_NO_CONTEXT_NAME_CODE( _func, RT_ERROR_UNKNOWN )

#define CHECK_OBJECT_POINTER( _o, _c )                                                                                 \
    if( !( _o ) || ( _o )->getClass() != ( _c ) )                                                                      \
    return RT_ERROR_ILWALID_VALUE

#define CHECK_CONTEXT_MATCHES( _o, _ctx )                                                                                               \
    if( static_cast<ManagedObject*>( _o )->getContext() != ( _ctx ) )                                                                   \
    {                                                                                                                                   \
        ( _ctx )->getErrorManager()->setErrorString( ( RTAPI_FUNC ), "Mismatched contexts for \"" #_o "\"", RT_ERROR_ILWALID_CONTEXT ); \
        return RT_ERROR_ILWALID_CONTEXT;                                                                                                \
    }

#define CHECK_OBJECT_POINTER_CONTEXT( _o, _c, _ctx )                                                                         \
    if( !( _o ) || ( _o )->getClass() != ( _c ) )                                                                            \
    {                                                                                                                        \
        ( _ctx )->getErrorManager()->setErrorString( ( RTAPI_FUNC ), "Invalid object \"" #_o "\"", RT_ERROR_ILWALID_VALUE ); \
        return RT_ERROR_ILWALID_VALUE;                                                                                       \
    }                                                                                                                        \
    else                                                                                                                     \
    {                                                                                                                        \
        CHECK_CONTEXT_MATCHES( _o, _ctx )                                                                                    \
    }

#define CHECK_NULL_NAME( _o, _ctx, _func )                                                                                   \
    if( !( _o ) )                                                                                                            \
    {                                                                                                                        \
        if( _ctx )                                                                                                           \
            ( _ctx )->getErrorManager()->setErrorString( ( _func ), "Pointer \"" #_o "\" is null", RT_ERROR_ILWALID_VALUE ); \
        return RT_ERROR_ILWALID_VALUE;                                                                                       \
    }

#define CHECK_NULL( _o, _ctx ) CHECK_NULL_NAME( _o, _ctx, RTAPI_FUNC )

#define CHECK_EXPRESSION( _expr, _ctx, _errstr )                                                                       \
    if( !( _expr ) )                                                                                                   \
    {                                                                                                                  \
        if( _ctx )                                                                                                     \
            ( _ctx )->getErrorManager()->setErrorString( ( RTAPI_FUNC ), _errstr, RT_ERROR_ILWALID_VALUE );            \
        return RT_ERROR_ILWALID_VALUE;                                                                                 \
    }

#define CHECK_NULL_AND_CONTEXT_MATCHES( _o, _ctx )                                                                     \
    CHECK_NULL( _o, _ctx )                                                                                             \
    CHECK_CONTEXT_MATCHES( _o, _ctx )

static StreamBuffer* getStreamBuffer( RTbuffer buf )
{
    if( !buf )
        return nullptr;
    if( ( (ManagedObject*)buf )->getClass() == RT_OBJECT_STREAM_BUFFER )
        return (StreamBuffer*)buf;
    return nullptr;
}

static Context* get_context( RTcontext obj )
{
    return obj ? (Context*)obj : nullptr;
}
static Context* get_context( RTacceleration obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTbuffer obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTcommandlist obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTgeometry obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTgeometrytriangles obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTgeometryinstance obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTgeometrygroup obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTgroup obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTmaterial obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTprogram obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTselector obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTtexturesampler obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTtransform obj )
{
    return obj ? ( (ManagedObject*)obj )->getContext() : nullptr;
}
static Context* get_context( RTvariable obj )
{
    return obj ? ( (Variable*)obj )->getScope()->getContext() : nullptr;
}
static Context* get_context( RTpostprocessingstage obj )
{
    return obj ? ( (PostprocessingStage*)obj )->getContext() : nullptr;
}

template <typename T>
static inline void finishAsyncLaunches( T obj, bool async = true, bool progressive = true )
{
    Context* ctx = get_context( obj );
    if( ctx )
    {
        if( progressive )
            ctx->stopProgressiveLaunch( false );
        if( async )
            ctx->finishAsyncLaunches();
    }
}

// Macro for defining a bunch of rtVariableSet functions for a given type.
#define DEFINE_VARIABLE_SETTERS( _t, _a )                                                                              \
    RTresult _rtVariableSet1##_a( RTvariable v, _t v0 ) { return var_set<_t, 1>( v, &v0, RTAPI_FUNC ); }               \
    RTresult _rtVariableSet2##_a( RTvariable v, _t v0, _t v1 ) { return var_set<_t>( v, v0, v1, RTAPI_FUNC ); }        \
    RTresult _rtVariableSet3##_a( RTvariable v, _t v0, _t v1, _t v2 )                                                  \
    {                                                                                                                  \
        return var_set<_t>( v, v0, v1, v2, RTAPI_FUNC );                                                               \
    }                                                                                                                  \
    RTresult _rtVariableSet4##_a( RTvariable v, _t v0, _t v1, _t v2, _t v3 )                                           \
    {                                                                                                                  \
        return var_set<_t>( v, v0, v1, v2, v3, RTAPI_FUNC );                                                           \
    }                                                                                                                  \
    RTresult _rtVariableSet1##_a##v( RTvariable v, const _t* v0 ) { return var_set<_t, 1>( v, v0, RTAPI_FUNC ); }      \
    RTresult _rtVariableSet2##_a##v( RTvariable v, const _t* v0 ) { return var_set<_t, 2>( v, v0, RTAPI_FUNC ); }      \
    RTresult _rtVariableSet3##_a##v( RTvariable v, const _t* v0 ) { return var_set<_t, 3>( v, v0, RTAPI_FUNC ); }      \
    RTresult _rtVariableSet4##_a##v( RTvariable v, const _t* v0 ) { return var_set<_t, 4>( v, v0, RTAPI_FUNC ); }

// Macro for defining a bunch of rtVariableGet functions for a given type.
#define DEFINE_VARIABLE_GETTERS( _t, _a )                                                                              \
    RTresult _rtVariableGet1##_a( RTvariable v, _t* v0 ) { return var_get<_t, 1>( v, v0, RTAPI_FUNC ); }               \
    RTresult _rtVariableGet2##_a( RTvariable v, _t* v0, _t* v1 ) { return var_get<_t>( v, v0, v1, RTAPI_FUNC ); }      \
    RTresult _rtVariableGet3##_a( RTvariable v, _t* v0, _t* v1, _t* v2 )                                               \
    {                                                                                                                  \
        return var_get<_t>( v, v0, v1, v2, RTAPI_FUNC );                                                               \
    }                                                                                                                  \
    RTresult _rtVariableGet4##_a( RTvariable v, _t* v0, _t* v1, _t* v2, _t* v3 )                                       \
    {                                                                                                                  \
        return var_get<_t>( v, v0, v1, v2, v3, RTAPI_FUNC );                                                           \
    }                                                                                                                  \
    RTresult _rtVariableGet1##_a##v( RTvariable v, _t* v0 ) { return var_get<_t, 1>( v, v0, RTAPI_FUNC ); }            \
    RTresult _rtVariableGet2##_a##v( RTvariable v, _t* v0 ) { return var_get<_t, 2>( v, v0, RTAPI_FUNC ); }            \
    RTresult _rtVariableGet3##_a##v( RTvariable v, _t* v0 ) { return var_get<_t, 3>( v, v0, RTAPI_FUNC ); }            \
    RTresult _rtVariableGet4##_a##v( RTvariable v, _t* v0 ) { return var_get<_t, 4>( v, v0, RTAPI_FUNC ); }

/* Sets of primitive types */
namespace {

/* Colwenience functions for set */
template <typename T, int N>
RTresult inline var_set( RTvariable v_api, const T* value, const char* const funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();

    try
    {
        v->set<T, N>( value );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

template <unsigned int R, unsigned int C>
RTresult inline var_set_matrix( RTvariable v_api, int transpose, const float* m, const char* const funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();

    try
    {
        v->setMatrix<R, C>( transpose != 0, m );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

template <typename T>
RTresult inline var_set( RTvariable v_api, T v0, T v1, const char* const funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();

    try
    {
        const T array[2] = {v0, v1};
        v->set<T, 2>( array );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

template <typename T>
RTresult inline var_set( RTvariable v_api, T v0, T v1, T v2, const char* const funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();

    try
    {
        const T array[3] = {v0, v1, v2};
        v->set<T, 3>( array );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

template <typename T>
RTresult inline var_set( RTvariable v_api, T v0, T v1, T v2, T v3, const char* const funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();

    try
    {
        const T array[4] = {v0, v1, v2, v3};
        v->set<T, 4>( array );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

}  // namespace

// Define the set functions.
DEFINE_VARIABLE_SETTERS( float, f );
DEFINE_VARIABLE_SETTERS( int, i );
DEFINE_VARIABLE_SETTERS( unsigned, ui );
DEFINE_VARIABLE_SETTERS( long long, ll );
DEFINE_VARIABLE_SETTERS( unsigned long long, ull );
RTresult _rtVariableSetMatrix2x2fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<2, 2>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableSetMatrix2x3fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<2, 3>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableSetMatrix2x4fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<2, 4>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableSetMatrix3x2fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<3, 2>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableSetMatrix3x3fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<3, 3>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableSetMatrix3x4fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<3, 4>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableSetMatrix4x2fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<4, 2>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableSetMatrix4x3fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<4, 3>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableSetMatrix4x4fv( RTvariable v, int transpose, const float* m )
{
    return var_set_matrix<4, 4>( v, transpose, m, RTAPI_FUNC );
}

/* Gets of primitive types */
namespace {
/* Colwenience functions for get */
template <typename T, int N>
RTresult inline var_get( RTvariable v_api, T* value, const char* const funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL_NAME( value, context, funcname );

    try
    {
        v->get<T, N>( value );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

template <unsigned int R, unsigned int C>
RTresult inline var_get_matrix( RTvariable v_api, int transpose, float* m, const char* const funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    Context* context = v->getScope()->getContext();
    finishAsyncLaunches( v_api );
    CHECK_NULL_NAME( m, context, funcname );

    try
    {
        v->getMatrix<R, C>( transpose != 0, m );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

template <typename T>
RTresult inline var_get( RTvariable v_api, T* v0, T* v1, const char* funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL_NAME( v0, context, funcname );
    CHECK_NULL_NAME( v1, context, funcname );

    try
    {
        T array[2];
        v->get<T, 2>( array );
        *v0 = array[0];
        *v1 = array[1];
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

template <typename T>
RTresult inline var_get( RTvariable v_api, T* v0, T* v1, T* v2, const char* funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL_NAME( v0, context, funcname );
    CHECK_NULL_NAME( v1, context, funcname );
    CHECK_NULL_NAME( v2, context, funcname );

    try
    {
        T array[3];
        v->get<T, 3>( array );
        *v0 = array[0];
        *v1 = array[1];
        *v2 = array[2];
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}

template <typename T>
RTresult inline var_get( RTvariable v_api, T* v0, T* v1, T* v2, T* v3, const char* funcname )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL_NAME( v0, context, funcname );
    CHECK_NULL_NAME( v1, context, funcname );
    CHECK_NULL_NAME( v2, context, funcname );
    CHECK_NULL_NAME( v3, context, funcname );

    try
    {
        T array[4];
        v->get<T, 4>( array );
        *v0 = array[0];
        *v1 = array[1];
        *v2 = array[2];
        *v3 = array[3];
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NAME( context, funcname );
}
}  // namespace

// Define the get functions.
DEFINE_VARIABLE_GETTERS( float, f );
DEFINE_VARIABLE_GETTERS( int, i );
DEFINE_VARIABLE_GETTERS( unsigned, ui );
DEFINE_VARIABLE_GETTERS( long long, ll );
DEFINE_VARIABLE_GETTERS( unsigned long long, ull );
RTresult _rtVariableGetMatrix2x2fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<2, 2>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableGetMatrix2x3fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<2, 3>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableGetMatrix2x4fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<2, 4>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableGetMatrix3x2fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<3, 2>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableGetMatrix3x3fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<3, 3>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableGetMatrix3x4fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<3, 4>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableGetMatrix4x2fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<4, 2>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableGetMatrix4x3fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<4, 3>( v, transpose, m, RTAPI_FUNC );
}
RTresult _rtVariableGetMatrix4x4fv( RTvariable v, int transpose, float* m )
{
    return var_get_matrix<4, 4>( v, transpose, m, RTAPI_FUNC );
}


RTresult _rtVariableSetObject( RTvariable v_api, RTobject object )
{
    Variable* v = api_cast( v_api );

    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL_AND_CONTEXT_MATCHES( object, context );
    ManagedObject* managed = static_cast<ManagedObject*>( object );

    try
    {

        // Check if the passed API object is of legal type to be set as a variable.
        switch( managed->getClass() )
        {
            case RT_OBJECT_BUFFER:
            {
                RT_ASSERT( managedObjectCast<Buffer>( managed ) );
                Buffer* concrete = static_cast<Buffer*>( managed );
                v->setBuffer( concrete );
                break;
            }
            case RT_OBJECT_TEXTURE_SAMPLER:
            {
                RT_ASSERT( managedObjectCast<TextureSampler>( managed ) );
                TextureSampler* concrete = static_cast<TextureSampler*>( managed );
                v->setTextureSampler( concrete );
                break;
            }
            case RT_OBJECT_GROUP:
            case RT_OBJECT_SELECTOR:
            case RT_OBJECT_GEOMETRY_GROUP:
            case RT_OBJECT_TRANSFORM:
            {
                RT_ASSERT( managedObjectCast<GraphNode>( managed ) );
                GraphNode* concrete = static_cast<GraphNode*>( managed );
                v->setGraphNode( concrete );
                break;
            }
            case RT_OBJECT_PROGRAM:
            {
                RT_ASSERT( managedObjectCast<Program>( managed ) );
                Program* concrete = static_cast<Program*>( managed );
                v->setProgram( translateProgramIntoAPI( concrete, context ) );
                break;
            }
            default:
                throw TypeMismatch( RT_EXCEPTION_INFO, "Unsupported object type in variable assignment: ",
                                    getNameForClass( managed->getClass() ) );
        }

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtVariableGetObject( RTvariable v_api, RTobject* object )
{
    if( object )
        *object = nullptr;

    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL( object, context );

    try
    {

        // Make sure the variable type is one of the allowed API objects.
        switch( v->getType().baseType() )
        {
            case VariableType::Buffer:
            case VariableType::DemandBuffer:
            {
                *object = v->getBuffer();
                break;
            }
            case VariableType::TextureSampler:
            {
                *object = v->getTextureSampler();
                break;
            }
            case VariableType::GraphNode:
            {
                *object = v->getGraphNode();
                break;
            }
            case VariableType::Program:
            {
                *object = translateProgramOutOfAPI( v->getProgram(), context );
                break;
            }
            default:
                throw TypeMismatch( RT_EXCEPTION_INFO, "Variable is not of OptiX object type" );
        }

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/* Set/get for user data */
RTresult _rtVariableSetUserData( RTvariable v_api, RTsize size, const void* ptr )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    if( size != 0 )
    {
        CHECK_NULL( ptr, context );
    }

    try
    {
        v->setUserData( size, ptr );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtVariableGetUserData( RTvariable v_api, RTsize size, void* ptr )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    if( size != 0 )
    {
        CHECK_NULL( ptr, context );
    }

    try
    {
        v->getUserData( size, ptr );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtVariableGetName( RTvariable v_api, const char** name_return )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL( name_return, context );

    try
    {
        *name_return = context->getPublicString( v->getName() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtVariableGetAnnotation( RTvariable v_api, const char** annotation_return )
{
    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL( annotation_return, context );

    try
    {
#if 0  // OP-1100
    *annotation_return = context->getPublicString(v->getAnnotation());
#endif
        *annotation_return = context->getPublicString( "" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtVariableGetType( RTvariable v_api, RTobjecttype* type_return )
{
    Variable* v = api_cast( v_api );
    if( type_return )
        *type_return = RT_OBJECTTYPE_UNKNOWN;
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();

    try
    {
        // Internal variable types are different than the API - translate
        RTobjecttype type = RT_OBJECTTYPE_UNKNOWN;
        switch( v->getType().baseType() )
        {
            case VariableType::Unknown:
                type = RT_OBJECTTYPE_UNKNOWN;
                break;
            case VariableType::Float:
                type = RT_OBJECTTYPE_FLOAT;
                break;
            case VariableType::Int:
                type = RT_OBJECTTYPE_INT;
                break;
            case VariableType::Uint:
                type = RT_OBJECTTYPE_UNSIGNED_INT;
                break;
            case VariableType::LongLong:
                type = RT_OBJECTTYPE_LONG_LONG;
                break;
            case VariableType::ULongLong:
                type = RT_OBJECTTYPE_UNSIGNED_LONG_LONG;
                break;
            case VariableType::Buffer:
            case VariableType::DemandBuffer:
                type = RT_OBJECTTYPE_BUFFER;
                break;
            case VariableType::TextureSampler:
                type = RT_OBJECTTYPE_TEXTURE_SAMPLER;
                break;

            case VariableType::UserData:
            {
                // Matrices are really user data, but we know the dimensions.
                if( v->isMatrix() )
                {
                    const uint2        dim = v->getMatrixDim();
                    const unsigned int off = ( dim.x - 2 ) * 3 + dim.y - 2;
                    type                   = static_cast<RTobjecttype>( RT_OBJECTTYPE_MATRIX_FLOAT2x2 + off );
                }
                else
                {
                    type = RT_OBJECTTYPE_USER;
                }
            }
            break;

            case VariableType::GraphNode:
            {
                // Query the object type
                GraphNode* gn = v->getGraphNode();
                if( gn )
                {
                    switch( gn->getClass() )
                    {
                        case RT_OBJECT_GEOMETRY_GROUP:
                            type = RT_OBJECTTYPE_GEOMETRY_GROUP;
                            break;
                        case RT_OBJECT_GROUP:
                            type = RT_OBJECTTYPE_GROUP;
                            break;
                        case RT_OBJECT_SELECTOR:
                            type = RT_OBJECTTYPE_SELECTOR;
                            break;
                        case RT_OBJECT_TRANSFORM:
                            type = RT_OBJECTTYPE_TRANSFORM;
                            break;
                        default:
                            type = RT_OBJECTTYPE_UNKNOWN;
                            break;
                    }
                }
                else
                {
                    // Variable used to be a graph node but was set to NULL,
                    // or is an rtObject that was never assigned anything.
                    type = RT_OBJECTTYPE_OBJECT;
                }
            }
            break;

            default:
                throw IlwalidValue( RT_EXCEPTION_INFO, "Type queried for invalid variable: ", v->getName() );
        }

        // For the vector types, get the number of elements and adjust the type.
        VariableType vt = v->getType();
        if( vt.baseType() == VariableType::Float || vt.baseType() == VariableType::Int || vt.baseType() == VariableType::Uint
            || vt.baseType() == VariableType::LongLong || vt.baseType() == VariableType::ULongLong )
        {
            const size_t n = vt.numElements();
            RT_ASSERT( n >= 1 && n <= 4 );
            type = static_cast<RTobjecttype>( type + n - 1 );
        }

        if( type_return )
            *type_return = type;
    }
    HANDLE_EXCEPTIONS( context );
    return RT_SUCCESS;
}

RTresult _rtVariableGetContext( RTvariable v_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtVariableGetSize( RTvariable v_api, RTsize* size )
{
    if( size )
        *size = 0;

    Variable* v = api_cast( v_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    finishAsyncLaunches( v_api );
    Context* context = v->getScope()->getContext();
    CHECK_NULL( size, context );

    try
    {
        switch( v->getType().baseType() )
        {
            case VariableType::Float:
            case VariableType::Int:
            case VariableType::Uint:
            case VariableType::LongLong:
            case VariableType::ULongLong:
            case VariableType::ProgramId:
            case VariableType::BufferId:
            case VariableType::UserData:
                *size = v->getType().computeSize();
                break;

            case VariableType::Buffer:
            case VariableType::DemandBuffer:
            case VariableType::GraphNode:
            case VariableType::Program:
            case VariableType::TextureSampler:
                throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot get size of objects" );

            case VariableType::Ray:
            case VariableType::Unknown:
                throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot get size of variable of unknown type" );

                // default case intentionally omitted
        }
    }
    HANDLE_EXCEPTIONS( context );

    return RT_SUCCESS;
}


/************************************
 **
 **    Context-free functions
 **
 ***********************************/

RTresult _rtGetVersion( unsigned int* version )
{
    if( !version )
    {
        lerr << "rtGetVersion error: version parameter is NULL.\n";
        return RT_ERROR_ILWALID_VALUE;
    }
    *version = static_cast<unsigned int>( OPTIX_VERSION );
    return RT_SUCCESS;
}

// ID reservation attributes are special. They must work without a context so
// that reservations can take place before context creation.
//
// Forcing whole mip-level callbacks for demand load textures also needs to take
// place before context creation.
RTresult _rtGlobalSetAttribute( RTglobalattribute attrib, RTsize size, const void* p )
{
    if( p == nullptr )
        return RT_ERROR_ILWALID_VALUE;

    switch( attrib )
    {
        case RT_GLOBAL_ATTRIBUTE_ENABLE_RTX:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            const int value = *static_cast<const int*>( p );
            // NOTE: For now directly set the value on the context. If we get more global attributes
            // we might want to change that.

            // RTX is the only supported exelwtion strategy.
            if( value == 1 )
                Context::setDefaultExelwtionStrategy( "rtx" );
            else
                return RT_ERROR_ILWALID_VALUE;
            return RT_SUCCESS;
        }
        case RT_GLOBAL_ATTRIBUTE_DEVELOPER_OPTIONS:
        {
            if( size == 0 )
                return RT_ERROR_ILWALID_VALUE;
            KnobRegistry& registry = knobRegistry();
            const char*   s        = static_cast<const char*>( p );
            registry.initializeKnobs( std::string( s, &s[size] ) );  // make sure the string is null-terminated!
            return RT_SUCCESS;
        }
        case RT_GLOBAL_INTERNAL_ATTRIBUTE_FORCE_DEMAND_LOAD_WHOLE_MIP_LEVEL:
        case RT_GLOBAL_INTERNAL_ATTRIBUTE_FORCE_DEMAND_LOAD_WHOLE_MIP_LEVEL_BACK_COMPAT:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            const int forced = *static_cast<const int*>( p );
            Context::setForceDemandLoadWholeMipLevel( forced != 0 );
            return RT_SUCCESS;
        }
        case RT_GLOBAL_ATTRIBUTE_DEMAND_LOAD_NUM_VIRTUAL_PAGES:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            const int numPages = *static_cast<const int*>( p );
            if( numPages <= 0 )
                return RT_ERROR_ILWALID_VALUE;
            Context::setDemandLoadNumVirtualPages( numPages );
            return RT_SUCCESS;
        }
        case RT_GLOBAL_INTERNAL_ATTRIBUTE_RESERVE_PROGRAM_ID:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            const int id = *static_cast<const int*>( p );
            llog( 30 ) << "Received program ID reservation request: " << id << std::endl;
            ObjectManager::reserveProgramId( id );
            return RT_SUCCESS;
        }
        case RT_GLOBAL_INTERNAL_ATTRIBUTE_RESERVE_BUFFER_ID:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            const int id = *static_cast<const int*>( p );
            llog( 30 ) << "Received buffer ID reservation request: " << id << std::endl;
            ObjectManager::reserveBufferId( id );
            return RT_SUCCESS;
        }
        case RT_GLOBAL_INTERNAL_ATTRIBUTE_RESERVE_TEXTURE_SAMPLER_ID:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            const int id = *static_cast<const int*>( p );
            llog( 30 ) << "Received texture sampler ID reservation request: " << id << std::endl;
            ObjectManager::reserveTextureSamplerId( id );
            return RT_SUCCESS;
        }
        default:
            break;
    }

    return RT_ERROR_ILWALID_GLOBAL_ATTRIBUTE;
}

RTresult _rtGlobalGetAttribute( RTglobalattribute attrib, RTsize size, void* p )
{
    if( p == nullptr )
        return RT_ERROR_ILWALID_VALUE;

    switch( attrib )
    {
        case RT_GLOBAL_ATTRIBUTE_ENABLE_RTX:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            *static_cast<int*>( p ) = Context::getDefaultExelwtionStrategy() == "rtx" ? 1 : 0;
            return RT_SUCCESS;
        }
        case RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MAJOR:
        {
            if( size != sizeof( unsigned int ) )
                return RT_ERROR_ILWALID_VALUE;
            DriverVersion driverVersion;
            if( !driverVersion.isValid() )
                return RT_ERROR_DRIVER_VERSION_FAILED;
            *static_cast<unsigned int*>( p ) = driverVersion.getMajorVersion();
            return RT_SUCCESS;
        }
        case RT_GLOBAL_ATTRIBUTE_DISPLAY_DRIVER_VERSION_MINOR:
        {
            if( size != sizeof( unsigned int ) )
                return RT_ERROR_ILWALID_VALUE;
            DriverVersion driverVersion;
            if( !driverVersion.isValid() )
                return RT_ERROR_DRIVER_VERSION_FAILED;
            *static_cast<unsigned int*>( p ) = driverVersion.getMinorVersion();
            return RT_SUCCESS;
        }
        case RT_GLOBAL_ATTRIBUTE_DEVELOPER_OPTIONS:
        {
            KnobRegistry&      registry = knobRegistry();
            std::ostringstream o;
            registry.printKnobs( o );
            std::string knobs = o.str();
            if( size <= knobs.size() )
                return RT_ERROR_ILWALID_VALUE;
            char* s = static_cast<char*>( p );
            strncpy( s, knobs.c_str(), size );
            return RT_SUCCESS;
        }
        case RT_GLOBAL_INTERNAL_ATTRIBUTE_FORCE_DEMAND_LOAD_WHOLE_MIP_LEVEL:
        case RT_GLOBAL_INTERNAL_ATTRIBUTE_FORCE_DEMAND_LOAD_WHOLE_MIP_LEVEL_BACK_COMPAT:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            const int forced = Context::getForceDemandLoadWholeMipLevel() ? 1 : 0;
            *static_cast<int*>( p ) = forced;
            return RT_SUCCESS;
        }
        case RT_GLOBAL_ATTRIBUTE_DEMAND_LOAD_NUM_VIRTUAL_PAGES:
        {
            if( size != sizeof( int ) )
                return RT_ERROR_ILWALID_VALUE;
            const int numPages      = static_cast<int>( Context::getDemandLoadNumVirtualPages() );
            *static_cast<int*>( p ) = numPages;
            return RT_SUCCESS;
        }
        default:
            break;
    }

    return RT_ERROR_ILWALID_GLOBAL_ATTRIBUTE;
}

RTresult _rtDeviceGetDeviceCount( unsigned int* count )
{

    // Exception handling for this call is different than the other
    // functions because we do not have a valid context object in which
    // to set error codes.
    if( !count )
    {
        lerr << "rtDeviceGetDeviceCount error: count parameter is NULL.\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        *count = 0;  // In case getDeviceCount fails
        *count = optix::DeviceManager::getDeviceCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME( "rtDeviceGetDeviceCount" );
}

RTresult _rtDeviceGetAttribute( int ordinal, RTdeviceattribute attrib, RTsize size, void* p )
{
    if( !p )
    {
        lerr << "rtDeviceGetAttribute error: return attribute parameter is NULL.\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        optix::DeviceManager::getDeviceAttribute( ordinal, attrib, size, p );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME( "rtDeviceGetAttribute" );
}

/************************************
 **
 **    Context object
 **
 ***********************************/

// Backwards compatibility with ABI version 16.
RTresult _rtContextCreateABI16( RTcontext* context )
{
    // Exception handling for create and destroy is different than the other
    // methods because we may not have a valid context object in which to
    // set error codes.
    if( !context )
    {
        lerr << "rtContextCreate error: context parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        *context = nullptr;
        std::unique_ptr<Context> ctx( new Context( Context::ABI_16_USE_MULTITHREADED_DEMAND_LOAD_CALLBACKS_BY_DEFAULT ) );
        *context = api_cast( ctx.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME_CODE( "rtContextCreate", RT_ERROR_CONTEXT_CREATION_FAILED );
}

RTresult _rtContextCreateABI17( RTcontext* context )
{
    // Exception handling for create and destroy is different than the other
    // methods because we may not have a valid context object in which to
    // set error codes.
    if( !context )
    {
        lerr << "rtContextCreate error: context parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        *context = nullptr;
        std::unique_ptr<Context> ctx( new Context( Context::ABI_17_USE_MAIN_THREAD_DEMAND_LOAD_CALLBACKS_BY_DEFAULT ) );
        *context = api_cast( ctx.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME_CODE( "rtContextCreate", RT_ERROR_CONTEXT_CREATION_FAILED );
}

RTresult _rtContextCreateABI18( RTcontext* context )
{
    // Exception handling for create and destroy is different than the other
    // methods because we may not have a valid context object in which to
    // set error codes.
    if( !context )
    {
        lerr << "rtContextCreate error: context parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        *context = nullptr;
        std::unique_ptr<Context> ctx( new Context( Context::ABI_18_USE_DEMAND_LOAD_CALLBACK_PER_TILE ) );
        *context = api_cast( ctx.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME_CODE( "rtContextCreate", RT_ERROR_CONTEXT_CREATION_FAILED );
}

RTresult _rtContextDestroy( RTcontext context_api )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    RTresult result = RT_SUCCESS;

    try
    {
        // Context internals cleaned up separately from its destructor.
        // If errors occur during tearDown, context may be left in an invalid state.
        // We want to catch and report errors, so keep the context around.
        context->tearDown();
    }
    HANDLE_EXCEPTIONS( context );

    // Exception handling for create and destroy is different than the other
    // methods because we may not have a valid context object in which to
    // set error codes.
    try
    {
        delete context;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME( "rtContextDestroy" );

    return result;
}

RTresult _rtContextValidate( RTcontext context_api )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->getValidationManager()->run();

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

void _rtContextGetErrorString( RTcontext context_api, RTresult code, const char** return_string )
{
    Context* context = api_cast( context_api );

    if( !return_string )
        return;
    if( !context )
    {
        try
        {
            *return_string = ErrorManager::getErrorString_static( code );
        }
        catch( ... )
        {
            *return_string = "Caught exception while processing error string";
        }
    }
    else
    {
        finishAsyncLaunches( context_api, false );
        // The only exceptions are likely to get caught here are memory allocation
        // failures.  Check anyway just for paranoia
        try
        {
            *return_string = context->getPublicString( context->getErrorManager()->getErrorString( code ) );
        }
        catch( ... )
        {
            *return_string = context->getPublicString( "Caught exception while processing error string" );
        }
    }
}

RTresult _rtContextSetDevices( RTcontext context_api, unsigned int count, const int* devices )
{
    Context* context = api_cast( context_api );

    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( devices, context );

    try
    {
        context->setDevices( count, devices );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetDevices( RTcontext context_api, int* devices )
{
    Context* context = api_cast( context_api );

    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( devices, context );

    try
    {
        std::vector<int> device_array;
        context->getDevices( device_array );
        for( size_t i = 0; i < device_array.size(); ++i )
        {
            devices[i] = device_array[i];
        }

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetDeviceCount( RTcontext context_api, unsigned int* count )
{
    if( count )
        *count = 0;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( count, context );

    try
    {
        std::vector<int> device_array;
        context->getDevices( device_array );
        *count = range_cast<unsigned int>( device_array.size() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetStackSize( RTcontext context_api, RTsize stack_size_bytes )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setStackSize( stack_size_bytes );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetStackSize( RTcontext context_api, RTsize* stack_size_bytes )
{
    if( stack_size_bytes )
        *stack_size_bytes = 0;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( stack_size_bytes, context );

    try
    {
        *stack_size_bytes = context->getStackSize();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetMaxCallableProgramDepth( RTcontext context_api, unsigned int max_depth )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setAPIMaxCallableProgramDepth( max_depth );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetMaxCallableProgramDepth( RTcontext context_api, unsigned int* max_depth )
{
    if( max_depth )
        *max_depth = 0;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( max_depth, context );

    try
    {
        *max_depth = context->getMaxCallableProgramDepth();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetMaxTraceDepth( RTcontext context_api, unsigned int max_depth )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setAPIMaxTraceDepth( max_depth );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetMaxTraceDepth( RTcontext context_api, unsigned int* max_depth )
{
    if( max_depth )
        *max_depth = 0;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( max_depth, context );

    try
    {
        *max_depth = context->getMaxTraceDepth();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetTimeoutCallback( RTcontext context_api, RTtimeoutcallback /*callback*/, double /*min_polling_seconds*/ )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    // no-op

    return RT_SUCCESS;
}

RTresult _rtContextSetUsageReportCallback( RTcontext context_api, RTusagereportcallback callback, int verbosity, void* cbdata )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setUsageReportCallback( callback, verbosity, cbdata );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetEntryPointCount( RTcontext context_api, unsigned int num_entry_points )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setEntryPointCount( num_entry_points );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetEntryPointCount( RTcontext context_api, unsigned int* num_entry_points )
{
    if( num_entry_points )
        *num_entry_points = 0;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( num_entry_points, context );

    try
    {
        *num_entry_points = context->getEntryPointCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetRayGenerationProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram program_api )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        if( entry_point_index >= context->getEntryPointCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid entry point index: ", entry_point_index );

        context->getGlobalScope()->setRayGenerationProgram( entry_point_index, program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetRayGenerationProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( program, context );

    try
    {
        if( entry_point_index >= context->getEntryPointCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid entry point index: ", entry_point_index );

        *program = api_cast( context->getGlobalScope()->getRayGenerationProgram( entry_point_index ), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetExceptionProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram program_api )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        if( entry_point_index >= context->getEntryPointCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid entry point index: ", entry_point_index );

        context->getGlobalScope()->setExceptionProgram( entry_point_index, program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetExceptionProgram( RTcontext context_api, unsigned int entry_point_index, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( program, context );

    try
    {
        if( entry_point_index >= context->getEntryPointCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid entry point index: ", entry_point_index );

        *program = api_cast( context->getGlobalScope()->getExceptionProgram( entry_point_index ), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetExceptionEnabled( RTcontext context_api, RTexception exception, int enabled )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setExceptionEnabled( exception, enabled != 0 );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetExceptionEnabled( RTcontext context_api, RTexception exception, int* enabled )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( enabled, context );

    try
    {
        *enabled = context->getExceptionEnabled( exception );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetRayTypeCount( RTcontext context_api, unsigned int num_ray_types )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setRayTypeCount( num_ray_types );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetRayTypeCount( RTcontext context_api, unsigned int* num_ray_types )
{
    if( num_ray_types )
        *num_ray_types = 0;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( num_ray_types, context );

    try
    {
        *num_ray_types = context->getRayTypeCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetMissProgram( RTcontext context_api, unsigned int ray_type_index, RTprogram program_api )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        if( ray_type_index >= context->getRayTypeCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid ray type index: ", ray_type_index );

        context->getGlobalScope()->setMissProgram( ray_type_index, program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetMissProgram( RTcontext context_api, unsigned int ray_type_index, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );

    CHECK_NULL( program, context );

    try
    {
        if( ray_type_index >= context->getRayTypeCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid entry point index: ", ray_type_index );

        *program = api_cast( context->getGlobalScope()->getMissProgram( ray_type_index ), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextCompile( RTcontext context_api )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    // This is a no-op since Goldenrod. See OP-738

    return RT_SUCCESS;
}

RTresult _rtContextLaunch1D( RTcontext context_api, unsigned int entry_point_index, RTsize width )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    TIMEVIZ_FLUSH;

    try
    {
        context->launchFromAPI( entry_point_index, 1, width, 1, 1 );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextLaunch2D( RTcontext context_api, unsigned int entry_point_index, RTsize width, RTsize height )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    TIMEVIZ_FLUSH;

    try
    {
        context->launchFromAPI( entry_point_index, 2, width, height, 1 );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextLaunch3D( RTcontext context_api, unsigned int entry_point_index, RTsize width, RTsize height, RTsize depth )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    TIMEVIZ_FLUSH;

    try
    {
        context->launchFromAPI( entry_point_index, 3, width, height, depth );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextLaunchProgressive2D( RTcontext context_api, unsigned int entry_point_index, RTsize width, RTsize height, unsigned int max_subframes )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );

    TIMEVIZ_FLUSH;

    try
    {
        context->launchProgressive( max_subframes, entry_point_index, 2, width, height );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextStopProgressive( RTcontext context_api )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );

    try
    {
        context->stopProgressiveLaunch( true );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/* running will be set to 1 if running, 0 otherwise */
RTresult _rtContextGetRunningState( RTcontext context_api, int* running )
{
    if( running )
        *running = 0;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );

    CHECK_NULL( running, context );

    try
    {
        bool ir  = context->isRunning();
        *running = ir ? 1 : 0;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}


RTresult _rtContextSetPrintEnabled( RTcontext context_api, int enabled )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setPrintEnabled( enabled != 0 );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetPrintEnabled( RTcontext context_api, int* enabled )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( enabled, context );

    try
    {
        *enabled = (int)context->getPrintEnabled();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetPrintBufferSize( RTcontext context_api, RTsize buffer_size_bytes )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setPrintBufferSize( buffer_size_bytes );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetPrintBufferSize( RTcontext context_api, RTsize* buffer_size_bytes )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( buffer_size_bytes, context );

    try
    {
        *buffer_size_bytes = context->getPrintBufferSize();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetPrintLaunchIndex( RTcontext context_api, int x, int y, int z )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    try
    {
        context->setPrintLaunchIndex( x, y, z );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetPrintLaunchIndex( RTcontext context_api, int* x, int* y, int* z )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );

    try
    {
        const int3 ri = context->getPrintLaunchIndex();
        if( x )
            *x = ri.x;
        if( y )
            *y = ri.y;
        if( z )
            *z = ri.z;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextDeclareVariable( RTcontext context_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( context->getGlobalScope()->declareVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextQueryVariable( RTcontext context_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( context->getGlobalScope()->queryVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}


RTresult _rtContextRemoveVariable( RTcontext context_api, RTvariable v_api )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );

    Variable* v = api_cast( v_api );
    finishAsyncLaunches( context_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );

    try
    {
        context->getGlobalScope()->removeVariable( v );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetVariableCount( RTcontext context_api, unsigned int* c )
{
    if( c )
        *c = 0;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( c, context );

    try
    {
        *c = context->getGlobalScope()->getVariableCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetVariable( RTcontext context_api, unsigned int index, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );

    CHECK_NULL( v, context );

    try
    {
        // assume throw on index out of range
        *v = api_cast( context->getGlobalScope()->getVariableByIndex( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetAttribute( RTcontext context_api, RTcontextattribute attrib, RTsize size, const void* p )
{
    // Interpret nullptr context attributes as global attributes
    if( context_api == nullptr )
    {
        switch( attrib )
        {
            case RT_CONTEXT_INTERNAL_ATTRIBUTE_RESERVE_PROGRAM_ID:
            case RT_CONTEXT_INTERNAL_ATTRIBUTE_RESERVE_BUFFER_ID:
            case RT_CONTEXT_INTERNAL_ATTRIBUTE_RESERVE_TEXTURE_SAMPLER_ID:
            case RT_CONTEXT_INTERNAL_ATTRIBUTE_FORCE_DEMAND_LOAD_WHOLE_MIP_LEVEL:
                return _rtGlobalSetAttribute( static_cast<RTglobalattribute>( attrib ), size, p );

            default:
                return RT_ERROR_ILWALID_VALUE;
        }
    }

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( p, context );

    try
    {
        context->setAttribute( attrib, size, p );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetAttribute( RTcontext context_api, RTcontextattribute attrib, RTsize size, void* p )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api, false );
    CHECK_NULL( p, context );

    try
    {
        context->getAttribute( attrib, size, p );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetTextureSamplerFromId( RTcontext context_api, int sampler_id, RTtexturesampler* sampler )
{
    if( sampler )
        *sampler = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );

    finishAsyncLaunches( context_api );
    CHECK_NULL( sampler, context );

    try
    {
        TextureSampler* result = nullptr;
        if( !context->getObjectManager()->getTextureSamplers().get( sampler_id, result ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Supplied sampler_id is not valid: ", sampler_id );
        *sampler = api_cast( result );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
 **
 **    Program object
 **
 ***********************************/

RTresult _rtProgramCreateFromPTXString( RTcontext context_api, const char* ptx, const char* program_name, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( program, context );
    CHECK_NULL( ptx, context );
    CHECK_NULL( program_name, context );

    try
    {
        std::unique_ptr<Program> prog( new Program( context ) );
        lwca::ComputeCapability  target_max( 9999 );  // externally created programs always target full range of compute
                                                      // capability
        prog->createFromString( ptx, "(api input string)", program_name, target_max );

        *program = api_cast( prog.release(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramCreateFromPTXStrings( RTcontext context_api, unsigned int n, const char** ptx_strings, const char* program_name, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( program, context );
    CHECK_NULL( ptx_strings, context );
    CHECK_EXPRESSION( n > 0, context, "Number of strings must be positive" );

    for( unsigned int i = 0; i < n; ++i )
    {
        CHECK_NULL( ptx_strings[i], context );
    }
    CHECK_NULL( program_name, context );

    try
    {
        std::unique_ptr<Program> prog( new Program( context ) );
        lwca::ComputeCapability  target_max( 9999 );  // externally created programs always target full range of compute
                                                      // capability
        std::vector<prodlib::StringView> strings( n );
        std::vector<std::string>         descriptions( n );
        for( unsigned int i = 0; i < n; ++i )
        {
            // TODO strlen_vectorized
            strings[i]      = {ptx_strings[i], strlen( ptx_strings[i] )};
            descriptions[i] = "(api input string)";
        }
        prog->createFromStrings( strings, descriptions, program_name, target_max );

        *program = api_cast( prog.release(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramCreateFromPTXFile( RTcontext context_api, const char* filename, const char* program_name, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( program, context );
    CHECK_NULL( filename, context );
    CHECK_NULL( program_name, context );

    try
    {
        std::unique_ptr<Program> prog( new Program( context ) );
        lwca::ComputeCapability  target_max( 9999 );  // externally created programs always target full range of compute
                                                      // capability
        prog->createFromFile( filename, program_name, target_max );

        *program = api_cast( prog.release(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramCreateFromPTXFiles( RTcontext context_api, unsigned int n, const char** filenames, const char* program_name, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( program, context );
    CHECK_EXPRESSION( n > 0, context, "number of files must be positive" );
    CHECK_NULL( filenames, context );

    for( unsigned int i = 0; i < n; ++i )
    {
        CHECK_NULL( filenames[i], context );
    }
    CHECK_NULL( program_name, context );

    try
    {
        std::unique_ptr<Program> prog( new Program( context ) );
        lwca::ComputeCapability  target_max( 9999 );  // externally created programs always target full range of compute
                                                      // capability
        std::vector<std::string> files( n );
        for( unsigned int i = 0; i < n; ++i )
        {
            files[i] = filenames[i];
        }
        prog->createFromFiles( files, program_name, target_max );

        *program = api_cast( prog.release(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramCreateFromProgram( RTcontext context_api, RTprogram program_in_api, RTprogram* program_out )
{
    if( program_out )
        *program_out = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );

    finishAsyncLaunches( context_api );

    Program* program_in = api_cast_ignore_null( program_in_api );
    CHECK_OBJECT_POINTER( program_in, RT_OBJECT_PROGRAM );

    CHECK_NULL( program_out, context );

    try
    {
        std::unique_ptr<Program> prog( new Program( context ) );
        prog->createFromProgram( program_in );
        *program_out = api_cast( prog.release(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}


RTresult _rtProgramDestroy( RTprogram program_api )
{
    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();

    try
    {
        // Remove this program from any object that references it
        program->detachFromParents();
        delete program;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramValidate( RTprogram program_api )
{
    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();

    try
    {
        program->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramGetContext( RTprogram program_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtProgramGetId( RTprogram program_api, int* program_id )
{
    if( program_id )
        *program_id = 0;

    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();
    CHECK_NULL( program_id, context );

    try
    {
        *program_id = program->getAPIId();
        llog( 40 ) << "GetId: program -> " << *program_id << std::endl;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetProgramFromId( RTcontext context_api, int program_id, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );

    finishAsyncLaunches( context_api );
    CHECK_NULL( program, context );

    try
    {
        Program* result = nullptr;
        if( !context->getObjectManager()->getPrograms().get( program_id, result ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Supplied program_id is not valid: ", program_id );
        *program = api_cast( result, context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramDeclareVariable( RTprogram program_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( program->declareVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramQueryVariable( RTprogram program_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( program->queryVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}


RTresult _rtProgramRemoveVariable( RTprogram program_api, RTvariable v_api )
{
    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );

    Variable* v = api_cast( v_api );
    finishAsyncLaunches( program_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    Context* context = program->getContext();

    try
    {
        program->removeVariable( v );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramGetVariableCount( RTprogram program_api, unsigned int* c )
{
    if( c )
        *c = 0;

    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();
    CHECK_NULL( c, context );

    try
    {
        *c = program->getVariableCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramGetVariable( RTprogram program_api, unsigned int index, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();
    CHECK_NULL( v, context );

    try
    {
        // assume throw on index out of range
        *v = api_cast( program->getVariableByIndex( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtProgramCallsiteSetPotentialCallees( RTprogram program_api, const char* name, const int* ids, int numIds )
{
    Program* program = api_cast_ignore_null( program_api );
    CHECK_OBJECT_POINTER( program, RT_OBJECT_PROGRAM );
    finishAsyncLaunches( program_api );
    Context* context = program->getContext();
    CHECK_NULL( name, context );
    CHECK_EXPRESSION( ids != nullptr || numIds == 0, context, "Illegal program ids" );  // is numIds == 0 valid?

    try
    {
        // assume throw on unknown callsite name
        program->setCallsitePotentialCallees( name, std::vector<int>{ids, ids + numIds} );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}


/************************************
 **
 **    Group object
 **
 ***********************************/

RTresult _rtGroupCreate( RTcontext context_api, RTgroup* group )
{
    if( group )
        *group = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( group, context );

    try
    {
        std::unique_ptr<Group> grp( new Group( context ) );

        *group = api_cast( grp.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupDestroy( RTgroup group_api )
{
    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();

    try
    {
        group->detachFromParents();
        // GOLDENROD: review and/or delete. ilwalidateLwrrentAccelerations not implemented
        // context->getASManager()->ilwalidateLwrrentAccelerations();
        delete group;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupValidate( RTgroup group_api )
{
    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();

    try
    {
        group->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupGetContext( RTgroup group_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtGroupSetAcceleration( RTgroup group_api, RTacceleration acceleration_api )
{
    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );

    Acceleration* acceleration = api_cast( acceleration_api );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_OBJECT_POINTER_CONTEXT( acceleration, RT_OBJECT_ACCELERATION, context );

    try
    {
        group->setAcceleration( acceleration );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupGetAcceleration( RTgroup group_api, RTacceleration* acceleration )
{
    if( acceleration )
        *acceleration = nullptr;

    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL( acceleration, context );

    try
    {
        *acceleration = api_cast( group->getAcceleration() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupSetChildCount( RTgroup group_api, unsigned int count )
{
    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();

    try
    {
        group->setChildCount( count );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupGetChildCount( RTgroup group_api, unsigned int* count )
{
    if( count )
        *count = 0;

    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL( count, context );

    try
    {
        const unsigned int cnt = group->getChildCount();
        if( count )
            *count = cnt;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupSetChild( RTgroup group_api, unsigned int index, RTobject child )
{
    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL_AND_CONTEXT_MATCHES( child, context );
    CHECK_EXPRESSION( isLegalGroupChild( child ), context, "Illegal child type" );

    try
    {
        group->setChild( index, static_cast<LexicalScope*>( child ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupGetChild( RTgroup group_api, unsigned int index, RTobject* child )
{
    if( child )
        *child = nullptr;

    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL( child, context );

    try
    {
        *child = group->getChild( index );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupGetChildType( RTgroup group_api, unsigned int index, RTobjecttype* type )
{
    Group* group = api_cast( group_api );
    if( type )
        *type = RT_OBJECTTYPE_UNKNOWN;
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL( type, context );

    try
    {
        *type = static_cast<RTobjecttype>( group->getChildType( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
 **
 **    Selector object
 **
 ***********************************/

RTresult _rtSelectorCreate( RTcontext context_api, RTselector* selector )
{
    if( selector )
        *selector = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( selector, context );

    try
    {
        std::unique_ptr<Selector> sel( new Selector( context ) );

        *selector = api_cast( sel.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorDestroy( RTselector selector_api )
{
    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();

    try
    {
        // Removes this selector from any variables and groups
        selector->detachFromParents();
        delete selector;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorValidate( RTselector selector_api )
{
    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();

    try
    {
        selector->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorGetContext( RTselector selector_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtSelectorSetVisitProgram( RTselector selector_api, RTprogram program_api )
{
    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );

    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        selector->setVisitProgram( program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorGetVisitProgram( RTselector selector_api, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( program, context );

    try
    {
        *program = api_cast( selector->getVisitProgram(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorSetChildCount( RTselector selector_api, unsigned int count )
{
    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();

    try
    {
        selector->setChildCount( count );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorGetChildCount( RTselector selector_api, unsigned int* count )
{
    if( count )
        *count = 0;

    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( count, context );

    try
    {
        const unsigned int cnt = selector->getChildCount();
        if( count )
            *count = cnt;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorSetChild( RTselector selector_api, unsigned int index, RTobject child )
{
    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL_AND_CONTEXT_MATCHES( child, context );
    CHECK_EXPRESSION( isLegalSelectorChild( child ), context, "Illegal child type" );

    try
    {
        selector->setChild( index, static_cast<GraphNode*>( child ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorGetChild( RTselector selector_api, unsigned int index, RTobject* child )
{
    if( child )
        *child = nullptr;

    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( child, context );

    try
    {
        *child = selector->getChild( index );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorGetChildType( RTselector selector_api, unsigned int index, RTobjecttype* type )
{
    Selector* selector = api_cast( selector_api );
    if( type )
        *type = RT_OBJECTTYPE_UNKNOWN;
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( type, context );

    try
    {
        *type = static_cast<RTobjecttype>( selector->getChildType( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorDeclareVariable( RTselector selector_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( selector->declareVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorQueryVariable( RTselector selector_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( selector->queryVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorRemoveVariable( RTselector selector_api, RTvariable v_api )
{
    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );

    Variable* v = api_cast( v_api );
    finishAsyncLaunches( selector_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    Context* context = selector->getContext();

    try
    {
        selector->removeVariable( v );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorGetVariableCount( RTselector selector_api, unsigned int* c )
{
    if( c )
        *c = 0;

    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( c, context );

    try
    {
        *c = selector->getVariableCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtSelectorGetVariable( RTselector selector_api, unsigned int index, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Selector* selector = api_cast( selector_api );
    CHECK_OBJECT_POINTER( selector, RT_OBJECT_SELECTOR );
    finishAsyncLaunches( selector_api );
    Context* context = selector->getContext();
    CHECK_NULL( v, context );

    try
    {
        // assume throw on index out of range
        *v = api_cast( selector->getVariableByIndex( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
 **
 **    Transform object
 **
 ***********************************/

RTresult _rtTransformCreate( RTcontext context_api, RTtransform* transform )
{
    if( transform )
        *transform = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( transform, context );

    try
    {
        std::unique_ptr<Transform> trn( new Transform( context ) );
        *transform = api_cast( trn.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformDestroy( RTtransform transform_api )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        // Remove transform from all parents and variables
        transform->detachFromParents();
        delete transform;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformValidate( RTtransform transform_api )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        transform->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformGetContext( RTtransform transform_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtTransformSetMatrix( RTtransform transform_api, int transpose, const float* matrix, const float* ilwerse_matrix )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    // At least one of the pointers must be non-null. If exactly one is valid,
    // we will compute the other matrix for the user. If both are valid, we just
    // take whatever the user specified.
    if( matrix == nullptr && ilwerse_matrix == nullptr )
    {
        if( context )
            context->getErrorManager()->setErrorString( RTAPI_FUNC, "matrix and ilwerse_matrix pointers are NULL.",
                                                        RT_ERROR_ILWALID_VALUE );
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {

        if( matrix )
        {
            transform->setMatrix( matrix, transpose != 0 );
        }
        else
        {
            const Matrix4x4* m = reinterpret_cast<const Matrix4x4*>( ilwerse_matrix );
            transform->setMatrix( m->ilwerse().getData(), transpose != 0 );
        }

        if( ilwerse_matrix )
        {
            transform->setIlwerseMatrix( ilwerse_matrix, transpose != 0 );
        }
        else
        {
            const Matrix4x4* m = reinterpret_cast<const Matrix4x4*>( matrix );
            transform->setIlwerseMatrix( m->ilwerse().getData(), transpose != 0 );
        }

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformGetMatrix( RTtransform transform_api, int transpose, float* matrix, float* ilwerse_matrix )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    // Don't check for null pointers - the user is allowed to set either one to null.

    try
    {
        if( matrix )
            transform->getMatrix( matrix, transpose != 0 );
        if( ilwerse_matrix )
            transform->getIlwerseMatrix( ilwerse_matrix, transpose != 0 );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformSetMotionRange( RTtransform transform_api, float timeBegin, float timeEnd )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        if( timeBegin > timeEnd )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Transform motion range timeBegin must be <= timeEnd:", timeBegin, timeEnd );

        transform->setMotionRange( timeBegin, timeEnd );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformGetMotionRange( RTtransform transform_api, float* timeBegin, float* timeEnd )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        transform->getMotionRange( *timeBegin, *timeEnd );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformSetMotionBorderMode( RTtransform transform_api, RTmotionbordermode beginMode, RTmotionbordermode endMode )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        transform->setMotionBorderMode( beginMode, endMode );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformGetMotionBorderMode( RTtransform transform_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        transform->getMotionBorderMode( *beginMode, *endMode );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformSetMotionKeys( RTtransform transform_api, unsigned int n, RTmotionkeytype type, const float* keys )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        if( n == 0 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Transform number of keys is 0; must be positive" );
        if( keys == nullptr )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Transform keys array is null" );
        if( type == RT_MOTIONKEYTYPE_NONE )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Transform key type is NONE" );

        transform->setKeys( n, type, keys );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );

    return RT_ERROR_NOT_SUPPORTED;
}

RTresult _rtTransformGetMotionKeyCount( RTtransform transform_api, unsigned int* n )
{
    if( n )
        *n = 1u;

    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();
    CHECK_NULL( n, context );

    try
    {
        int keycount = transform->getKeyCount();
        if( keycount > 0 )
            *n = keycount;
        else
            *n = 1;  // for static
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformGetMotionKeyType( RTtransform transform_api, RTmotionkeytype* type )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        *type = transform->getKeyType();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformGetMotionKeys( RTtransform transform_api, float* keys )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();

    try
    {
        transform->getKeys( keys );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformSetChild( RTtransform transform_api, RTobject child )
{
    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();
    CHECK_NULL_AND_CONTEXT_MATCHES( child, context );
    CHECK_EXPRESSION( isLegalTransformChild( child ), context, "Illegal child type" );

    try
    {
        transform->setChild( static_cast<GraphNode*>( child ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformGetChild( RTtransform transform_api, RTobject* child )
{
    if( child )
        *child = nullptr;

    Transform* transform = api_cast( transform_api );
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();
    CHECK_NULL( child, context );

    try
    {
        *child = transform->getChild();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTransformGetChildType( RTtransform transform_api, RTobjecttype* type )
{
    Transform* transform = api_cast( transform_api );
    if( type )
        *type = RT_OBJECTTYPE_UNKNOWN;
    CHECK_OBJECT_POINTER( transform, RT_OBJECT_TRANSFORM );
    finishAsyncLaunches( transform_api );
    Context* context = transform->getContext();
    CHECK_NULL( type, context );

    try
    {
        *type = static_cast<RTobjecttype>( transform->getChildType() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
**
**    GeometryGroup object
**
***********************************/

RTresult _rtGeometryGroupCreate( RTcontext context_api, RTgeometrygroup* geometrygroup )
{
    if( geometrygroup )
        *geometrygroup = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( geometrygroup, context );

    try
    {
        std::unique_ptr<GeometryGroup> grp( new GeometryGroup( context ) );

        *geometrygroup = api_cast( grp.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupDestroy( RTgeometrygroup geometrygroup_api )
{
    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();

    try
    {
        geometrygroup->detachFromParents();
        // GOLDENROD: review and/or delete. ilwalidateLwrrentAccelerations not implemented
        // context->getASManager()->ilwalidateLwrrentAccelerations();
        delete geometrygroup;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupValidate( RTgeometrygroup geometrygroup_api )
{
    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();

    try
    {
        geometrygroup->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupGetContext( RTgeometrygroup geometrygroup_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtGeometryGroupSetAcceleration( RTgeometrygroup geometrygroup_api, RTacceleration acceleration_api )
{
    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );

    Acceleration* acceleration = api_cast( acceleration_api );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();
    CHECK_OBJECT_POINTER_CONTEXT( acceleration, RT_OBJECT_ACCELERATION, context );

    try
    {
        geometrygroup->setAcceleration( acceleration );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupGetAcceleration( RTgeometrygroup geometrygroup_api, RTacceleration* acceleration )
{
    if( acceleration )
        *acceleration = nullptr;

    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();
    CHECK_NULL( acceleration, context );

    try
    {
        *acceleration = api_cast( geometrygroup->getAcceleration() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupSetChildCount( RTgeometrygroup geometrygroup_api, unsigned int count )
{
    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();

    try
    {
        geometrygroup->setChildCount( count );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupGetChildCount( RTgeometrygroup geometrygroup_api, unsigned int* count )
{
    if( count )
        *count = 0;

    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();
    CHECK_NULL( count, context );

    try
    {
        const unsigned int cnt = geometrygroup->getChildCount();
        if( count )
            *count = cnt;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupSetChild( RTgeometrygroup geometrygroup_api, unsigned int index, RTgeometryinstance geometryinstance_api )
{
    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );

    GeometryInstance* geometryinstance = api_cast( geometryinstance_api );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();
    CHECK_NULL_AND_CONTEXT_MATCHES( geometryinstance, context );
    CHECK_EXPRESSION( isLegalGeometryGroupChild( geometryinstance ), context, "Illegal child type" );

    try
    {
        geometrygroup->setChild( index, geometryinstance );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupGetChild( RTgeometrygroup geometrygroup_api, unsigned int index, RTgeometryinstance* geometryinstance )
{
    if( geometryinstance )
        *geometryinstance = nullptr;

    GeometryGroup* geometrygroup = api_cast( geometrygroup_api );
    CHECK_OBJECT_POINTER( geometrygroup, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( geometrygroup_api );
    Context* context = geometrygroup->getContext();
    CHECK_NULL( geometryinstance, context );

    try
    {
        *geometryinstance = api_cast( static_cast<GeometryInstance*>( geometrygroup->getChild( index ) ) );
        RT_ASSERT( api_cast( *geometryinstance )->getClass() == RT_OBJECT_GEOMETRY_INSTANCE );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
 **
 **    Acceleration object
 **
 ***********************************/

RTresult _rtAccelerationCreate( RTcontext context_api, RTacceleration* acceleration )
{
    if( acceleration )
        *acceleration = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( acceleration, context );

    try
    {
        std::unique_ptr<Acceleration> accel( new Acceleration( context ) );

        *acceleration = api_cast( accel.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationDestroy( RTacceleration acceleration_api )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();

    try
    {
        //    if ( acceleration->isAsyncEnabled() &&
        //         acceleration->getState() != Acceleration::BUILDSTATE_READY ) {
        //#ifdef ADOBE_BUILD
        //      lerr << "Cannot destroy acceleration while asynchronous building is in progress\n";
        //#else
        //      throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot destroy acceleration while asynchronous building is in
        //      progress" );
        //#endif
        //    }

        // Removes this acceleration from all groups
        acceleration->detachFromParents();
        // GOLDENROD: review and/or delete. ilwalidateLwrrentAccelerations not implemented
        // context->getASManager()->ilwalidateLwrrentAccelerations();
        delete acceleration;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAcceleratiolwalidate( RTacceleration acceleration_api )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();

    try
    {
        acceleration->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationGetContext( RTacceleration acceleration_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtAccelerationSetBuilder( RTacceleration acceleration_api, const char* builder )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( builder, context );

    try
    {
        acceleration->setBuilder( builder );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationGetBuilder( RTacceleration acceleration_api, const char** return_string )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( return_string, context );

    try
    {
        *return_string = context->getPublicString( acceleration->getBuilderName() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationSetTraverser( RTacceleration acceleration_api, const char* traverser )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( traverser, context );

    // no-op

    return RT_SUCCESS;
}

RTresult _rtAccelerationGetTraverser( RTacceleration acceleration_api, const char** return_string )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( return_string, context );

    try
    {
        *return_string = context->getPublicString( acceleration->getTraverser() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationSetProperty( RTacceleration acceleration_api, const char* name, const char* value )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( name, context );
    CHECK_NULL( value, context );

    try
    {
        acceleration->setProperty( name, value );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationGetProperty( RTacceleration acceleration_api, const char* name, const char** return_string )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( name, context );
    CHECK_NULL( return_string, context );

    try
    {
        *return_string = context->getPublicString( acceleration->getProperty( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationGetDataSize( RTacceleration acceleration_api, RTsize* size )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( size, context );

    try
    {
        *size = acceleration->getDataSize();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationGetData( RTacceleration acceleration_api, void* data )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( data, context );

    try
    {
        acceleration->getData( data );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationSetData( RTacceleration acceleration_api, const void* data, RTsize size )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( data, context );

    try
    {
        acceleration->setData( data, size );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationMarkDirty( RTacceleration acceleration_api )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();

    try
    {
        acceleration->markDirty();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtAccelerationIsDirty( RTacceleration acceleration_api, int* dirty )
{
    Acceleration* acceleration = api_cast( acceleration_api );
    CHECK_OBJECT_POINTER( acceleration, RT_OBJECT_ACCELERATION );
    finishAsyncLaunches( acceleration_api );
    Context* context = acceleration->getContext();
    CHECK_NULL( dirty, context );

    try
    {
        *dirty = acceleration->isDirtyExternal();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
 **
 **    GeometryInstance object
 **
 ***********************************/

RTresult _rtGeometryInstanceCreate( RTcontext context_api, RTgeometryinstance* geometryinstance )
{
    if( geometryinstance )
        *geometryinstance = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( geometryinstance, context );

    try
    {
        std::unique_ptr<GeometryInstance> gi( new GeometryInstance( context ) );

        *geometryinstance = api_cast( gi.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceDestroy( RTgeometryinstance geometryinstance_api )
{
    GeometryInstance* geometryinstance = api_cast( geometryinstance_api );
    CHECK_OBJECT_POINTER( geometryinstance, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( geometryinstance_api );
    Context* context = geometryinstance->getContext();

    try
    {
        geometryinstance->detachFromParents();
        delete geometryinstance;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceValidate( RTgeometryinstance geometryinstance_api )
{
    GeometryInstance* geometryinstance = api_cast( geometryinstance_api );
    CHECK_OBJECT_POINTER( geometryinstance, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( geometryinstance_api );
    Context* context = geometryinstance->getContext();

    try
    {
        geometryinstance->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceGetContext( RTgeometryinstance gi_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtGeometryInstanceSetGeometry( RTgeometryinstance gi_api, RTgeometry geo_api )
{
    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );

    Geometry* geo = api_cast( geo_api );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_OBJECT_POINTER_CONTEXT( geo, RT_OBJECT_GEOMETRY, context );

    try
    {
        gi->setGeometry( geo );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceGetGeometry( RTgeometryinstance gi_api, RTgeometry* geo )
{
    if( geo )
        *geo = nullptr;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( geo, context );

    try
    {
        Geometry*          geom     = gi->getGeometry();
        GeometryTriangles* geomTris = managedObjectCast<GeometryTriangles>( geom );
        if( geom && geomTris )
            // invalid to return anything other than Geometry (such as GeometryTriangles)
            throw IlwalidValue( RT_EXCEPTION_INFO, "Attached object is not a Geometry" );
        *geo = api_cast( geom );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceSetGeometryTriangles( RTgeometryinstance gi_api, RTgeometrytriangles geo_api )
{
    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );

    GeometryTriangles* geo = api_cast( geo_api );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_OBJECT_POINTER_CONTEXT( geo, RT_OBJECT_GEOMETRY, context );

    try
    {
        gi->setGeometry( geo );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceGetGeometryTriangles( RTgeometryinstance gi_api, RTgeometrytriangles* geo )
{
    if( geo )
        *geo = nullptr;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( geo, context );

    try
    {
        Geometry*          geom     = gi->getGeometry();
        GeometryTriangles* geomTris = managedObjectCast<GeometryTriangles>( geom );
        if( geom && !geomTris )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Attached object is not a GeometryTriangles" );
        *geo = api_cast( geomTris );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceSetMaterialCount( RTgeometryinstance gi_api, unsigned int num_materials )
{
    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();

    try
    {
        gi->setMaterialCount( num_materials );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceGetMaterialCount( RTgeometryinstance gi_api, unsigned int* num_materials )
{
    if( num_materials )
        *num_materials = 0;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( num_materials, context );

    try
    {
        const unsigned int num = gi->getMaterialCount();
        if( num_materials )
            *num_materials = num;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceSetMaterial( RTgeometryinstance gi_api, unsigned int material_idx, RTmaterial mat_api )
{
    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );

    Material* mat = api_cast( mat_api );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_OBJECT_POINTER_CONTEXT( mat, RT_OBJECT_MATERIAL, context );

    try
    {
        gi->setMaterial( material_idx, mat );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceGetMaterial( RTgeometryinstance gi_api, unsigned int material_idx, RTmaterial* mat )
{
    if( mat )
        *mat = nullptr;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( mat, context );

    try
    {
        *mat = api_cast( gi->getMaterial( material_idx ) );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceDeclareVariable( RTgeometryinstance gi_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( gi->declareVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceQueryVariable( RTgeometryinstance gi_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( gi->queryVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceRemoveVariable( RTgeometryinstance gi_api, RTvariable v_api )
{
    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    Variable* v = api_cast( v_api );
    finishAsyncLaunches( gi_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    Context* context = gi->getContext();

    try
    {
        gi->removeVariable( v );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceGetVariableCount( RTgeometryinstance gi_api, unsigned int* c )
{
    if( c )
        *c = 0;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( c, context );

    try
    {
        *c = gi->getVariableCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryInstanceGetVariable( RTgeometryinstance gi_api, unsigned int index, RTvariable* v )
{
    if( v )
        *v = nullptr;

    GeometryInstance* gi = api_cast( gi_api );
    CHECK_OBJECT_POINTER( gi, RT_OBJECT_GEOMETRY_INSTANCE );
    finishAsyncLaunches( gi_api );
    Context* context = gi->getContext();
    CHECK_NULL( v, context );

    try
    {
        // assume throw on index out of range
        *v = api_cast( gi->getVariableByIndex( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
 **
 **    GeometryObject
 **
 ***********************************/

RTresult _rtGeometryCreate( RTcontext context_api, RTgeometry* geometry )
{
    if( geometry )
        *geometry = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( geometry, context );

    try
    {
        std::unique_ptr<Geometry> geom( new Geometry( context, false ) );

        *geometry = api_cast( geom.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryDestroy( RTgeometry geometry_api )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        // Removes this geometry from all geometry instances
        geometry->detachFromParents();
        delete geometry;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryValidate( RTgeometry geometry_api )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        geometry->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetContext( RTgeometry geometry_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtGeometrySetPrimitiveCount( RTgeometry geometry_api, unsigned int num_primitives )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        geometry->setPrimitiveCount( num_primitives );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetPrimitiveCount( RTgeometry geometry_api, unsigned int* num_primitives )
{
    if( num_primitives )
        *num_primitives = 0u;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( num_primitives, context );

    try
    {
        *num_primitives = geometry->getPrimitiveCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometrySetPrimitiveIndexOffset( RTgeometry geometry_api, unsigned int index_offset )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        geometry->setPrimitiveIndexOffset( index_offset );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetPrimitiveIndexOffset( RTgeometry geometry_api, unsigned int* index_offset )
{
    if( index_offset )
        *index_offset = 0u;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( index_offset, context );

    try
    {
        *index_offset = geometry->getPrimitiveIndexOffset();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometrySetMotionRange( RTgeometry geometry_api, float timeBegin, float timeEnd )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        if( timeBegin > timeEnd )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Geometry motion range timeBegin must be <= timeEnd:", timeBegin, timeEnd );

        geometry->setMotionRange( timeBegin, timeEnd );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetMotionRange( RTgeometry geometry_api, float* timeBegin, float* timeEnd )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        geometry->getMotionRange( *timeBegin, *timeEnd );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometrySetMotionBorderMode( RTgeometry geometry_api, RTmotionbordermode beginMode, RTmotionbordermode endMode )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        geometry->setMotionBorderMode( beginMode, endMode );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetMotionBorderMode( RTgeometry geometry_api, RTmotionbordermode* beginMode, RTmotionbordermode* endMode )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        geometry->getMotionBorderMode( *beginMode, *endMode );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometrySetMotionSteps( RTgeometry geometry_api, unsigned int n )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    try
    {
        if( n == 0 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Geometry number of motion steps is 0; must be positive" );

        geometry->setMotionSteps( n );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetMotionSteps( RTgeometry geometry_api, unsigned int* n )
{
    if( n )
        *n = 0u;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( n, context );

    try
    {
        *n = geometry->getMotionSteps();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometrySetBoundingBoxProgram( RTgeometry geometry_api, RTprogram program_api )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        geometry->setBoundingBoxProgram( program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetBoundingBoxProgram( RTgeometry geometry_api, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( program, context );

    try
    {
        *program = api_cast( geometry->getBoundingBoxProgram(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometrySetIntersectionProgram( RTgeometry geometry_api, RTprogram program_api )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        geometry->setIntersectionProgram( program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetIntersectionProgram( RTgeometry geometry_api, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( program, context );

    try
    {
        *program = api_cast( geometry->getIntersectionProgram(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryMarkDirty( RTgeometry geometry_api )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        geometry->markDirty();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryIsDirty( RTgeometry geometry_api, int* dirty )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( dirty, context );

    try
    {
        *dirty = geometry->isDirty();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryDeclareVariable( RTgeometry geometry_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( geometry->declareVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryQueryVariable( RTgeometry geometry_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( geometry->queryVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryRemoveVariable( RTgeometry geometry_api, RTvariable v_api )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    Variable* v = api_cast( v_api );
    finishAsyncLaunches( geometry_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    Context* context = geometry->getContext();

    try
    {
        geometry->removeVariable( v );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetVariableCount( RTgeometry geometry_api, unsigned int* c )
{
    if( c )
        *c = 0;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( c, context );

    try
    {
        *c = geometry->getVariableCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetVariable( RTgeometry geometry_api, unsigned int index, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( v, context );

    try
    {
        // assume throw on index out of range
        *v = api_cast( geometry->getVariableByIndex( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
**
**    GeometryTriangles Object
**
***********************************/

RTresult _rtGeometryTrianglesCreate( RTcontext context_api, RTgeometrytriangles* geometrytriangles )
{
    if( geometrytriangles )
        *geometrytriangles = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( geometrytriangles, context );

    try
    {
        std::unique_ptr<GeometryTriangles> geom( new GeometryTriangles( context ) );

        *geometrytriangles = api_cast( geom.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesDestroy( RTgeometrytriangles geometrytriangles_api )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        // Removes this geometry from all geometry instances
        geometrytriangles->detachFromParents();
        delete geometrytriangles;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesValidate( RTgeometrytriangles geometrytriangles_api )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetContext( RTgeometrytriangles geometrytriangles_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtGeometryTrianglesSetPrimitiveIndexOffset( RTgeometrytriangles geometrytriangles_api, unsigned int index_offset )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setPrimitiveIndexOffset( index_offset );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetPrimitiveIndexOffset( RTgeometrytriangles geometrytriangles_api, unsigned int* index_offset )
{
    if( index_offset )
        *index_offset = 0u;

    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( index_offset, context );

    try
    {
        *index_offset = geometrytriangles->getPrimitiveIndexOffset();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetPreTransformMatrix( RTgeometrytriangles geometrytriangles_api, int transpose, const float* matrix )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setPreTransformMatrix( matrix, transpose != 0 );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetPreTransformMatrix( RTgeometrytriangles geometrytriangles_api, int transpose, float* matrix )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->getPreTransformMatrix( matrix, transpose != 0 );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetPrimitiveCount( RTgeometrytriangles geometrytriangles_api, unsigned int num_triangles )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setPrimitiveCount( num_triangles );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetPrimitiveCount( RTgeometrytriangles geometrytriangles_api, unsigned int* num_triangles )
{
    if( num_triangles )
        *num_triangles = 0u;

    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( num_triangles, context );

    try
    {
        *num_triangles = geometrytriangles->getPrimitiveCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetTriangleIndices( RTgeometrytriangles geometrytriangles_api,
                                                 RTbuffer            index_buffer_api,
                                                 RTsize              index_buffer_byte_offset,
                                                 RTsize              tri_indices_byte_stride,
                                                 RTformat            tri_indices_format )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Buffer*  index_buffer = api_cast( index_buffer_api );
    Context* context      = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setTriangleIndices( index_buffer, index_buffer_byte_offset, tri_indices_byte_stride, tri_indices_format );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetVertices( RTgeometrytriangles geometrytriangles_api,
                                          unsigned int        num_vertices,
                                          RTbuffer            vertex_buffer_api,
                                          RTsize              vertex_buffer_byte_offset,
                                          RTsize              vertex_byte_stride,
                                          RTformat            position_format )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Buffer*  vertex_buffer = api_cast( vertex_buffer_api );
    Context* context       = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setVertices( num_vertices, vertex_buffer, vertex_buffer_byte_offset, vertex_byte_stride, position_format );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetMotiolwertices( RTgeometrytriangles geometrytriangles_api,
                                                unsigned int        num_vertices,
                                                RTbuffer            vertex_buffer_api,
                                                RTsize              vertex_buffer_byte_offset,
                                                RTsize              vertex_byte_stride,
                                                RTsize              vertex_motion_step_byte_stride,
                                                RTformat            position_format )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Buffer*  vertex_buffer = api_cast( vertex_buffer_api );
    Context* context       = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setMotiolwertices( num_vertices, vertex_buffer, vertex_buffer_byte_offset,
                                              vertex_byte_stride, vertex_motion_step_byte_stride, position_format );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetMotiolwerticesMultiBuffer( RTgeometrytriangles geometrytriangles_api,
                                                           unsigned int        num_vertices,
                                                           RTbuffer*           vertex_buffers_api,
                                                           unsigned int        vertex_buffer_count,
                                                           RTsize              vertex_buffer_byte_offset,
                                                           RTsize              vertex_byte_stride,
                                                           RTformat            position_format )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Buffer** vertex_buffers = api_cast( vertex_buffers_api );
    Context* context        = geometrytriangles->getContext();
    CHECK_NULL( vertex_buffers_api, context );

    try
    {
        geometrytriangles->setMotiolwerticesMultiBuffer( num_vertices, vertex_buffers, vertex_buffer_count,
                                                         vertex_buffer_byte_offset, vertex_byte_stride, position_format );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetMotionSteps( RTgeometrytriangles geometrytriangles_api, unsigned int num_motion_steps )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setMotionSteps( num_motion_steps );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetMotionSteps( RTgeometrytriangles geometrytriangles_api, unsigned int* num_motion_steps )
{
    if( num_motion_steps )
        *num_motion_steps = 0u;

    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( num_motion_steps, context );

    try
    {
        *num_motion_steps = geometrytriangles->getMotionSteps();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetMotionRange( RTgeometrytriangles geometrytriangles_api, float timeBegin, float timeEnd )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        if( timeBegin > timeEnd )
            throw IlwalidValue( RT_EXCEPTION_INFO, "GeometryTriangles motion range timeBegin must be <= timeEnd:", timeBegin, timeEnd );

        geometrytriangles->setMotionRange( timeBegin, timeEnd );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetMotionRange( RTgeometrytriangles geometrytriangles_api, float* timeBegin, float* timeEnd )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->getMotionRange( *timeBegin, *timeEnd );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetMotionBorderMode( RTgeometrytriangles geometrytriangles_api, RTmotionbordermode beginMode, RTmotionbordermode endMode )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setMotionBorderMode( beginMode, endMode );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetMotionBorderMode( RTgeometrytriangles geometrytriangles_api,
                                                  RTmotionbordermode* beginMode,
                                                  RTmotionbordermode* endMode )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->getMotionBorderMode( *beginMode, *endMode );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetBuildFlags( RTgeometrytriangles geometrytriangles_api, RTgeometrybuildflags build_flags )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setBuildFlags( build_flags );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetMaterialCount( RTgeometrytriangles geometrytriangles_api, unsigned int num_materials )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setMaterialCount( num_materials );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetMaterialCount( RTgeometrytriangles geometrytriangles_api, unsigned int* num_materials )
{
    if( num_materials )
        *num_materials = 0u;

    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( num_materials, context );

    try
    {
        *num_materials = geometrytriangles->getMaterialCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetMaterialIndices( RTgeometrytriangles geometrytriangles_api,
                                                 RTbuffer            material_index_buffer_api,
                                                 RTsize              material_index_buffer_byte_offset,
                                                 RTsize              material_index_byte_stride,
                                                 RTformat            material_index_format )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Buffer*  material_index_buffer = api_cast( material_index_buffer_api );
    Context* context               = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setMaterialIndices( material_index_buffer, material_index_buffer_byte_offset,
                                               material_index_byte_stride, material_index_format );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetFlagsPerMaterial( RTgeometrytriangles geometrytriangles_api, unsigned int material_index, RTgeometryflags flags )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->setFlagsPerMaterial( material_index, flags );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetFlagsPerMaterial( RTgeometrytriangles geometrytriangles_api, unsigned int material_index, RTgeometryflags* flags )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( flags, context );

    try
    {
        *flags = geometrytriangles->getFlagsPerMaterial( material_index );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupSetVisibilityMask( RTgroup group_api, RTvisibilitymask mask )
{
    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();

    try
    {
        group->setVisibilityMask( mask );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGroupGetVisibilityMask( RTgroup group_api, RTvisibilitymask* mask )
{
    Group* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL( mask, context );

    try
    {
        *mask = group->getVisibilityMask();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupSetFlags( RTgeometrygroup group_api, RTinstanceflags flags )
{
    GeometryGroup* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();

    try
    {
        group->setFlags( flags );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupGetFlags( RTgeometrygroup group_api, RTinstanceflags* flags )
{
    GeometryGroup* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL( flags, context );

    try
    {
        *flags = group->getFlags();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupSetVisibilityMask( RTgeometrygroup group_api, RTvisibilitymask mask )
{
    GeometryGroup* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();

    try
    {
        group->setVisibilityMask( mask );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGroupGetVisibilityMask( RTgeometrygroup group_api, RTvisibilitymask* mask )
{
    GeometryGroup* group = api_cast( group_api );
    CHECK_OBJECT_POINTER( group, RT_OBJECT_GEOMETRY_GROUP );
    finishAsyncLaunches( group_api );
    Context* context = group->getContext();
    CHECK_NULL( mask, context );

    try
    {
        *mask = group->getVisibilityMask();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometrySetFlags( RTgeometry geometry_api, RTgeometryflags flags )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();

    try
    {
        geometry->setFlags( flags );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryGetFlags( RTgeometry geometry_api, RTgeometryflags* flags )
{
    Geometry* geometry = api_cast( geometry_api );
    CHECK_OBJECT_POINTER( geometry, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometry_api );
    Context* context = geometry->getContext();
    CHECK_NULL( flags, context );

    try
    {
        *flags = geometry->getFlags();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesSetAttributeProgram( RTgeometrytriangles geometrytriangles_api, RTprogram program_api )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        geometrytriangles->setAttributeProgram( program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetAttributeProgram( RTgeometrytriangles geometrytriangles_api, RTprogram* program_api )
{
    if( program_api )
        *program_api = nullptr;

    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    if( !managedObjectCast<GeometryTriangles>( static_cast<Geometry*>( geometrytriangles ) ) )
        return RT_ERROR_ILWALID_VALUE;
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( program_api, context );

    try
    {
        *program_api = api_cast( geometrytriangles->getAttributeProgram(), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesDeclareVariable( RTgeometrytriangles geometrytriangles_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    CHECK_OBJECT_POINTER( geometrytriangles, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( geometrytriangles->declareVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesQueryVariable( RTgeometrytriangles geometrytriangles_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    CHECK_OBJECT_POINTER( geometrytriangles, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( geometrytriangles->queryVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesRemoveVariable( RTgeometrytriangles geometrytriangles_api, RTvariable v_api )
{
    GeometryTriangles* geometrytriangles = api_cast( geometrytriangles_api );
    CHECK_OBJECT_POINTER( geometrytriangles, RT_OBJECT_GEOMETRY );
    Variable* v = api_cast( v_api );
    finishAsyncLaunches( geometrytriangles_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    Context* context = geometrytriangles->getContext();

    try
    {
        geometrytriangles->removeVariable( v );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetVariableCount( RTgeometrytriangles geometrytriangles_api, unsigned int* count )
{
    if( count )
        *count = 0;

    Geometry* geometrytriangles = api_cast( geometrytriangles_api );
    CHECK_OBJECT_POINTER( geometrytriangles, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( count, context );

    try
    {
        *count = geometrytriangles->getVariableCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtGeometryTrianglesGetVariable( RTgeometrytriangles geometrytriangles_api, unsigned int index, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Geometry* geometrytriangles = api_cast( geometrytriangles_api );
    CHECK_OBJECT_POINTER( geometrytriangles, RT_OBJECT_GEOMETRY );
    finishAsyncLaunches( geometrytriangles_api );
    Context* context = geometrytriangles->getContext();
    CHECK_NULL( v, context );

    try
    {
        // assume throw on index out of range
        *v = api_cast( geometrytriangles->getVariableByIndex( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}


/************************************
 **
 **    Material Object
 **
 ***********************************/

RTresult _rtMaterialCreate( RTcontext context_api, RTmaterial* material )
{
    if( material )
        *material = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( material, context );

    try
    {
        std::unique_ptr<Material> matl( new Material( context ) );

        *material = api_cast( matl.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialDestroy( RTmaterial material_api )
{
    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();

    try
    {
        // Removes this material from all geometry instances
        material->detachFromParents();
        delete material;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialValidate( RTmaterial material_api )
{
    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();

    try
    {
        material->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialGetContext( RTmaterial material_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtMaterialSetClosestHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram program_api )
{
    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        if( ray_type_index >= context->getRayTypeCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid ray type index: ", ray_type_index );

        material->setClosestHitProgram( ray_type_index, program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialGetClosestHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    CHECK_NULL( program, context );

    try
    {
        if( ray_type_index >= context->getRayTypeCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid ray type index: ", ray_type_index );

        *program = api_cast( material->getClosestHitProgram( ray_type_index ), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialSetAnyHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram program_api )
{
    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    Program* program = api_cast( program_api, context );
    CHECK_OBJECT_POINTER_CONTEXT( program, RT_OBJECT_PROGRAM, context );

    try
    {
        if( ray_type_index >= context->getRayTypeCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid ray type index: ", ray_type_index );

        material->setAnyHitProgram( ray_type_index, program );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialGetAnyHitProgram( RTmaterial material_api, unsigned int ray_type_index, RTprogram* program )
{
    if( program )
        *program = nullptr;

    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    CHECK_NULL( program, context );

    try
    {
        if( ray_type_index >= context->getRayTypeCount() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid ray type index: ", ray_type_index );

        *program = api_cast( material->getAnyHitProgram( ray_type_index ), context );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialDeclareVariable( RTmaterial material_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( material->declareVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialQueryVariable( RTmaterial material_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( material->queryVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}


RTresult _rtMaterialRemoveVariable( RTmaterial material_api, RTvariable v_api )
{
    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    Variable* v = api_cast( v_api );
    finishAsyncLaunches( material_api );
    CHECK_OBJECT_POINTER( v, RT_OBJECT_VARIABLE );
    Context* context = material->getContext();

    try
    {
        material->removeVariable( v );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialGetVariableCount( RTmaterial material_api, unsigned int* c )
{
    if( c )
        *c = 0;

    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    CHECK_NULL( c, context );

    try
    {
        *c = material->getVariableCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtMaterialGetVariable( RTmaterial material_api, unsigned int index, RTvariable* v )
{
    if( v )
        *v = nullptr;

    Material* material = api_cast( material_api );
    CHECK_OBJECT_POINTER( material, RT_OBJECT_MATERIAL );
    finishAsyncLaunches( material_api );
    Context* context = material->getContext();
    CHECK_NULL( v, context );

    try
    {
        // assume throw on index out of range
        *v = api_cast( material->getVariableByIndex( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
 **
 **    Texture sampler Object
 **
 ***********************************/

RTresult _rtTextureSamplerCreate( RTcontext context_api, RTtexturesampler* textureSampler )
{
    if( textureSampler )
        *textureSampler = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( textureSampler, context );

    try
    {
        std::unique_ptr<TextureSampler> ts( new TextureSampler( context ) );

        *textureSampler = api_cast( ts.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerDestroy( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        textureSampler->detachFromParents();
        delete textureSampler;  // Removes this texture sampler from all variables

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerValidate( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        textureSampler->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetContext( RTtexturesampler textureSampler_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

// deprecated
RTresult _rtTextureSamplerSetMipLevelCount( RTtexturesampler textureSampler_api, unsigned int deprecated )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_EXPRESSION( deprecated == 1, context,
                      "rtTextureSamplerSetMipLevelCount parameter must be 1. rtTextureSamplerSetMipLevelCount is "
                      "deprecated and can be omitted" );

    try
    {
        if( textureSampler->isInteropTexture() )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "The number of mip levels for an RTtexturesampler created from an interop texture is "
                                "determined automatically by OptiX." );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );

    return RT_SUCCESS;
}

// deprecated
RTresult _rtTextureSamplerGetMipLevelCount( RTtexturesampler textureSampler_api, unsigned int* deprecated )
{
    if( deprecated )
        *deprecated = 0;

    TextureSampler* textureSampler = api_cast( textureSampler_api );

    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( deprecated, context );

    try
    {
        *deprecated = ( textureSampler->isInteropTexture() || textureSampler->getBuffer() ) ? 1 : 0;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

// deprecated
RTresult _rtTextureSamplerSetArraySize( RTtexturesampler textureSampler_api, unsigned int deprecated )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_EXPRESSION( deprecated == 1, context,
                      "rtTextureSamplerSetArraySize parameter must be 1. rtTextureSamplerSetArraySize is deprecated "
                      "and can be omitted." );

    try
    {
        if( textureSampler->isInteropTexture() )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "The array size for an RTtexturesampler created from an interop texture is determined "
                                "automatically by OptiX." );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context )
}

// deprecated
RTresult _rtTextureSamplerGetArraySize( RTtexturesampler textureSampler_api, unsigned int* deprecated )
{
    if( deprecated )
        *deprecated = 0;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( deprecated, context );

    try
    {
        *deprecated = ( textureSampler->isInteropTexture() || textureSampler->getBuffer() ) ? 1 : 0;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerSetWrapMode( RTtexturesampler textureSampler_api, unsigned int dim, RTwrapmode wm )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        if( dim >= 3 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal dimension for wrap mode: ", dim );
        textureSampler->setWrapMode( dim, wm );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetWrapMode( RTtexturesampler textureSampler_api, unsigned int dim, RTwrapmode* wm )
{
    if( wm )
        *wm = RT_WRAP_REPEAT;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( wm, context );

    try
    {
        if( dim >= 3 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal dimension for wrap mode: ", dim );
        *wm = textureSampler->getWrapMode( dim );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerSetFilteringModes( RTtexturesampler textureSampler_api, RTfiltermode minFilter, RTfiltermode magFilter, RTfiltermode mipFilter )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        textureSampler->setFilterModes( minFilter, magFilter, mipFilter );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetFilteringModes( RTtexturesampler textureSampler_api,
                                             RTfiltermode*    minFilter,
                                             RTfiltermode*    magFilter,
                                             RTfiltermode*    mipFilter )
{
    if( minFilter )
        *minFilter = RT_FILTER_NEAREST;
    if( magFilter )
        *magFilter = RT_FILTER_NEAREST;
    if( mipFilter )
        *mipFilter = RT_FILTER_NEAREST;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( minFilter, context );
    CHECK_NULL( magFilter, context );
    CHECK_NULL( mipFilter, context );

    try
    {
        textureSampler->getFilterModes( *minFilter, *magFilter, *mipFilter );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerSetMaxAnisotropy( RTtexturesampler textureSampler_api, float maxAnisotropy )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        textureSampler->setMaxAnisotropy( maxAnisotropy );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetMaxAnisotropy( RTtexturesampler textureSampler_api, float* maxAnisotropy )
{
    if( maxAnisotropy )
        *maxAnisotropy = 0;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( maxAnisotropy, context );

    try
    {
        *maxAnisotropy = textureSampler->getMaxAnisotropy();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerSetMipLevelClamp( RTtexturesampler textureSampler_api, float minLevel, float maxLevel )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );

    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        textureSampler->setMinMipLevelClamp( minLevel );
        textureSampler->setMaxMipLevelClamp( maxLevel );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetMipLevelClamp( RTtexturesampler textureSampler_api, float* minLevel, float* maxLevel )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );

    if( minLevel )
        *minLevel = 0;
    if( maxLevel )
        *maxLevel = 0;
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( minLevel, context );
    CHECK_NULL( maxLevel, context );

    try
    {
        *minLevel = textureSampler->getMinMipLevelClamp();
        *maxLevel = textureSampler->getMaxMipLevelClamp();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerSetMipLevelBias( RTtexturesampler textureSampler_api, float bias )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );

    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        textureSampler->setMipLevelBias( bias );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetMipLevelBias( RTtexturesampler textureSampler_api, float* bias )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );

    if( bias )
        *bias = 0;
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( bias, context );

    try
    {
        *bias = textureSampler->getMipLevelBias();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerSetReadMode( RTtexturesampler textureSampler_api, RTtexturereadmode readmode )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        textureSampler->setReadMode( readmode );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetReadMode( RTtexturesampler textureSampler_api, RTtexturereadmode* readmode )
{
    if( readmode )
        *readmode = RT_TEXTURE_READ_ELEMENT_TYPE;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( readmode, context );

    try
    {
        *readmode = textureSampler->getReadMode();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerSetIndexingMode( RTtexturesampler textureSampler_api, RTtextureindexmode indexmode )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        textureSampler->setIndexMode( indexmode );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetIndexingMode( RTtexturesampler textureSampler_api, RTtextureindexmode* indexmode )
{
    if( indexmode )
        *indexmode = RT_TEXTURE_INDEX_NORMALIZED_COORDINATES;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( indexmode, context );

    try
    {
        *indexmode = textureSampler->getIndexMode();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerSetBuffer( RTtexturesampler textureSampler_api, unsigned int deprecated0, unsigned int deprecated1, RTbuffer buffer_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    Buffer*  buffer  = api_cast( buffer_api );
    CHECK_OBJECT_POINTER_CONTEXT( buffer, RT_OBJECT_BUFFER, context );

    try
    {
        if( deprecated0 != 0 )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Parameter deprecated0 of function rtTextureSamplerSetBuffer must be 0." );
        if( deprecated1 != 0 )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Parameter deprecated1 of function rtTextureSamplerSetBuffer must be 0." );

        if( textureSampler->isInteropTexture() )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "A buffer cannot be added to an RTtexturesampler when it is has been created from an "
                                "interop "
                                "texture." );

        textureSampler->setBuffer( buffer );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetBuffer( RTtexturesampler textureSampler_api, unsigned int deprecated0, unsigned int deprecated1, RTbuffer* buffer )
{
    if( buffer )
        *buffer = nullptr;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( buffer, context );

    try
    {
        if( deprecated0 != 0 )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Parameter deprecated0 of function rtTextureSamplerGetBuffer must be 0." );
        if( deprecated1 != 0 )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Parameter deprecated1 of function rtTextureSamplerGetBuffer must be 0." );


        *buffer = api_cast( textureSampler->getBuffer() );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetId( RTtexturesampler textureSampler_api, int* texture_id )
{
    if( texture_id )
        *texture_id = 0;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( texture_id, context );

    try
    {
        *texture_id = textureSampler->getAPIId();
        llog( 40 ) << "GetId: tex sampler -> " << *texture_id << std::endl;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

/************************************
 **
 **    Buffer object
 **
 ***********************************/

RTresult _rtBufferCreate( RTcontext context_api, unsigned int type, RTbuffer* buffer )
{
    if( buffer )
        *buffer = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( buffer, context );


    try
    {
        if( type & RT_BUFFER_PROGRESSIVE_STREAM )
        {
            if( type & ~RT_BUFFER_PROGRESSIVE_STREAM )
            {
                throw IlwalidValue( RT_EXCEPTION_INFO, "The specified buffer type is not valid: ", type );
            }
            std::unique_ptr<StreamBuffer> b( new StreamBuffer( context ) );
            *buffer = (RTbuffer)b.release();
        }
        else
        {
            Buffer::checkBufferType( type, false );
            context->getDeviceManager()->enableActiveDevices();
            std::unique_ptr<Buffer> b( new Buffer( context, type ) );

            *buffer = api_cast( b.release() );
        }

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferCreateForLWDA( RTcontext context_api, unsigned int type, RTbuffer* buffer )
{
    return _rtBufferCreate( context_api, type, buffer );
}

RTresult _rtBufferCreateFromCallback( RTcontext context_api, unsigned int type, RTbuffercallback callback, void* callbackContext, RTbuffer* buffer )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( buffer, context );
    CHECK_NULL( callback, context );
    CHECK_EXPRESSION( type == RT_BUFFER_INPUT, context, "Only RT_BUFFER_INPUT buffers can be demand loaded." );
    // TODO: check other flags or enhance checkBufferType?
    try
    {
        Buffer::checkBufferType( type, false );
        context->getDeviceManager()->enableActiveDevices();
        std::unique_ptr<Buffer> b( new Buffer( context, type, callback, callbackContext ) );
        *buffer = reinterpret_cast<RTbuffer>( b.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferSetDevicePointer( RTbuffer buffer_api, int optix_device_ordinal, void* device_pointer )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        CHECK_EXPRESSION( false, stream->getContext(), "Operation not valid for stream buffers" );
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( device_pointer, context );
    CHECK_EXPRESSION( !buffer->isDemandLoad(), context, "Operation not valid for demand buffers" );

    try
    {
        const DeviceArray& devices = context->getDeviceManager()->visibleDevices();
        if( optix_device_ordinal < 0 || optix_device_ordinal >= static_cast<int>( devices.size() ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid device ordinal." );
        Device* device = devices[optix_device_ordinal];
        if( !device->isActive() )
            throw IlwalidValue( RT_EXCEPTION_INFO,
                                "Setting buffer device pointers for devices on which OptiX isn't being run is "
                                "disallowed." );

        buffer->setInteropPointer( device_pointer, device );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferMarkDirty( RTbuffer buffer_api )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        CHECK_EXPRESSION( false, stream->getContext(), "Operation not valid for stream buffers" );
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();

    finishAsyncLaunches( buffer_api );
    try
    {
        buffer->markDirty();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferCreateFromGLBO( RTcontext context_api, unsigned int type, unsigned int glId, RTbuffer* buffer )
{
    if( buffer )
        *buffer = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( buffer, context );

#if 0 && !defined( __APPLE__ )
  // These give spurious failures on mac
  // glIsBuffer has started wrongly returning GL_FALSE in R304. See http://lwbugs/1158215
  CHECK_EXPRESSION( GLIsBuffer, context, "Failed to initialize OpenGL extensions" );
  CHECK_EXPRESSION( GLIsBuffer(glId), context, "Not a valid OpenGL buffer" );
#endif

    try
    {
        Buffer::checkBufferType( type, true );

        context->getDeviceManager()->enableActiveDevices();
        GfxInteropResource      resource( GfxInteropResource::OGL_BUFFER_OBJECT, glId );
        Device*                 device = context->getDeviceManager()->glInteropDevice();
        std::unique_ptr<Buffer> b( new Buffer( context, type, resource, device ) );

        *buffer = api_cast( b.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferDestroy( RTbuffer buffer_api )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        Context* context = stream->getContext();
        try
        {

            stream->detachFromParents();
            delete stream;
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        buffer->detachFromParents();
        delete buffer;  // Removes this buffer from texturesamplers and variables

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferValidate( RTbuffer buffer_api )
{
    if( getStreamBuffer( buffer_api ) )
    {
        return RT_SUCCESS;
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        buffer->validate();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetProgressiveUpdateReady( RTbuffer buffer_api, int* ready, unsigned int* subframe_count, unsigned int* max_subframes )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        Context* context = stream->getContext();
        try
        {
            stream->updateReady( ready, subframe_count, max_subframes );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    CHECK_EXPRESSION( false, context, "Buffer must be of stream type" );
}

RTresult _rtBufferBindProgressiveStream( RTbuffer buffer_api, RTbuffer source )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        Buffer*  srcbuf  = api_cast( source );
        try
        {
            if( srcbuf )
            {
                if( srcbuf->getType() & RT_BUFFER_INPUT )
                    throw IlwalidValue( RT_EXCEPTION_INFO, "Buffers of type INPUT are not allowed as stream source" );
            }
            stream->bindSource( srcbuf );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
        return RT_SUCCESS;
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_EXPRESSION( false, context, "Target must be a stream buffer" );
}

RTresult _rtBufferSetAttribute( RTbuffer buffer_api, RTbufferattribute attrib, RTsize size, const void* p )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );

        Context* context = stream->getContext();
        switch( attrib )
        {
            case RT_BUFFER_ATTRIBUTE_STREAM_FORMAT:
            {
                char* str = (char*)alloca( size + 1 );  // size doesn't include terminating \0
                strncpy( str, (const char*)p, size );
                str[size]                    = 0;
                stream->stream_attrib_format = str;
                break;
            }
            case RT_BUFFER_ATTRIBUTE_STREAM_BITRATE:
            {
                CHECK_EXPRESSION( size == sizeof( int ), context, "Invalid buffer attribute size" );
                stream->stream_attrib_bitrate = std::max( 1, *( (int*)p ) );
                break;
            }
            case RT_BUFFER_ATTRIBUTE_STREAM_FPS:
            {
                CHECK_EXPRESSION( size == sizeof( int ), context, "Invalid buffer attribute size" );
                stream->stream_attrib_fps = std::max( 1, *( (int*)p ) );
                break;
            }
            case RT_BUFFER_ATTRIBUTE_STREAM_GAMMA:
            {
                CHECK_EXPRESSION( size == sizeof( float ), context, "Invalid buffer attribute size" );
                stream->stream_attrib_gamma = std::max( 0.0f, *( (float*)p ) );
                break;
            }
            case RT_BUFFER_ATTRIBUTE_PAGE_SIZE:
                // This attribute is for demand loaded buffers only.
                return RT_ERROR_ILWALID_VALUE;
        }
        return RT_SUCCESS;
    }

    if( attrib == RT_BUFFER_ATTRIBUTE_PAGE_SIZE )
        // This attribute is read only.
        return RT_ERROR_ILWALID_VALUE;

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_EXPRESSION( false, context, "Invalid attribute for non-stream buffer" );
}

RTresult _rtBufferGetAttribute( RTbuffer buffer_api, RTbufferattribute attrib, RTsize size, void* p )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        switch( attrib )
        {
            case RT_BUFFER_ATTRIBUTE_STREAM_FORMAT:
            {
                if( size > 0 )
                {
                    strncpy( (char*)p, stream->stream_attrib_format.c_str(), size );
                    ( (char*)p )[size - 1] = 0;
                }
                break;
            }
            case RT_BUFFER_ATTRIBUTE_STREAM_BITRATE:
            {
                CHECK_EXPRESSION( size == sizeof( int ), context, "Invalid buffer attribute size" );
                *( (int*)p ) = stream->stream_attrib_bitrate;
                break;
            }
            case RT_BUFFER_ATTRIBUTE_STREAM_FPS:
            {
                CHECK_EXPRESSION( size == sizeof( int ), context, "Invalid buffer attribute size" );
                *( (int*)p ) = stream->stream_attrib_fps;
                break;
            }
            case RT_BUFFER_ATTRIBUTE_STREAM_GAMMA:
            {
                CHECK_EXPRESSION( size == sizeof( float ), context, "Invalid buffer attribute size" );
                *( (float*)p ) = stream->stream_attrib_gamma;
                break;
            }
            case RT_BUFFER_ATTRIBUTE_PAGE_SIZE:
                // This attribute is for demand loaded buffers only.
                return RT_ERROR_ILWALID_VALUE;
        }
        return RT_SUCCESS;
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    if( buffer->isDemandLoad() && attrib == RT_BUFFER_ATTRIBUTE_PAGE_SIZE )
    {
        CHECK_EXPRESSION( size == sizeof( int ), context, "Invalid buffer attribute size" );
        *static_cast<int*>( p ) = buffer->getPageSize();
        return RT_SUCCESS;
    }
    else
        CHECK_EXPRESSION( false, context, "Invalid attribute for non-stream buffer" );
}

RTresult _rtBufferGetContext( RTbuffer buffer_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( c, context );
        *c = api_cast( stream->getContext() );
        return RT_SUCCESS;
    }


    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtBufferSetFormat( RTbuffer buffer_api, RTformat type )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        try
        {
            stream->setFormat( type );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        buffer->setFormat( type );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetFormat( RTbuffer buffer_api, RTformat* format )
{
    if( format )
        *format = RT_FORMAT_UNKNOWN;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( format, context );
        try
        {
            *format = stream->getFormat();
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( format, context );

    try
    {
        *format = buffer->getFormat();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferSetElementSize( RTbuffer buffer_api, RTsize size_of_element )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        try
        {
            stream->setElementSize( size_of_element );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        buffer->setElementSize( size_of_element );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetElementSize( RTbuffer buffer_api, RTsize* size_of_element )
{
    if( size_of_element )
        *size_of_element = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( size_of_element, context );
        try
        {
            *size_of_element = range_cast<RTsize>( stream->getElementSize() );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }


    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( size_of_element, context );

    try
    {
        *size_of_element = range_cast<RTsize>( buffer->getElementSize() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferSetSize1D( RTbuffer buffer_api, RTsize width )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        try
        {
            stream->setSize1D( width );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        buffer->setSize1D( width );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetSize1D( RTbuffer buffer_api, RTsize* width )
{
    if( width )
        *width = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( width, context );
        try
        {
            *width = range_cast<RTsize>( stream->getWidth() );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }


    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( width, context );

    try
    {
        *width = range_cast<RTsize>( buffer->getWidth() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferSetSize2D( RTbuffer buffer_api, RTsize width, RTsize height )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        try
        {
            stream->setSize2D( width, height );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        buffer->setSize2D( width, height );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetSize2D( RTbuffer buffer_api, RTsize* width, RTsize* height )
{
    if( width )
        *width = 0;
    if( height )
        *height = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( width, context );
        CHECK_NULL( height, context );
        try
        {
            *width  = range_cast<RTsize>( stream->getWidth() );
            *height = range_cast<RTsize>( stream->getHeight() );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }


    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( width, context );
    CHECK_NULL( height, context );

    try
    {
        *width  = range_cast<RTsize>( buffer->getWidth() );
        *height = range_cast<RTsize>( buffer->getHeight() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferSetSize3D( RTbuffer buffer_api, RTsize width, RTsize height, RTsize depth )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        try
        {
            stream->setSize3D( width, height, depth );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        buffer->setSize3D( width, height, depth );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetSize3D( RTbuffer buffer_api, RTsize* width, RTsize* height, RTsize* depth )
{
    if( width )
        *width = 0;
    if( height )
        *height = 0;
    if( depth )
        *depth = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( width, context );
        CHECK_NULL( height, context );
        CHECK_NULL( depth, context );
        try
        {
            *width  = range_cast<RTsize>( stream->getWidth() );
            *height = range_cast<RTsize>( stream->getHeight() );
            *depth  = range_cast<RTsize>( stream->getDepth() );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }


    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( width, context );
    CHECK_NULL( height, context );
    CHECK_NULL( depth, context );

    try
    {
        *width  = range_cast<RTsize>( buffer->getWidth() );
        *height = range_cast<RTsize>( buffer->getHeight() );
        *depth  = range_cast<RTsize>( buffer->getDepth() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetMipLevelSize1D( RTbuffer buffer_api, unsigned int level, RTsize* width )
{
    if( width )
        *width = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( width, context );
        try
        {
            *width = range_cast<RTsize>( stream->getLevelWidth( level ) );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( width, context );

    try
    {
        *width = range_cast<RTsize>( buffer->getLevelWidth( level ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetMipLevelSize2D( RTbuffer buffer_api, unsigned int level, RTsize* width, RTsize* height )
{
    if( width )
        *width = 0;
    if( height )
        *height = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( width, context );
        CHECK_NULL( height, context );
        try
        {
            *width  = range_cast<RTsize>( stream->getLevelWidth( level ) );
            *height = range_cast<RTsize>( stream->getLevelHeight( level ) );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( width, context );
    CHECK_NULL( height, context );

    try
    {
        *width  = range_cast<RTsize>( buffer->getLevelWidth( level ) );
        *height = range_cast<RTsize>( buffer->getLevelHeight( level ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetMipLevelSize3D( RTbuffer buffer_api, unsigned int level, RTsize* width, RTsize* height, RTsize* depth )
{
    if( width )
        *width = 0;
    if( height )
        *height = 0;
    if( depth )
        *depth = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( width, context );
        CHECK_NULL( height, context );
        CHECK_NULL( depth, context );
        try
        {
            *width  = range_cast<RTsize>( stream->getLevelWidth( level ) );
            *height = range_cast<RTsize>( stream->getLevelHeight( level ) );
            *depth  = range_cast<RTsize>( stream->getLevelDepth( level ) );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( width, context );
    CHECK_NULL( height, context );
    CHECK_NULL( depth, context );

    try
    {
        *width  = range_cast<RTsize>( buffer->getLevelWidth( level ) );
        *height = range_cast<RTsize>( buffer->getLevelHeight( level ) );
        *depth  = range_cast<RTsize>( buffer->getLevelDepth( level ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferSetSizev( RTbuffer buffer_api, unsigned int dimensionality, const RTsize* indims )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( indims, context );
        try
        {
            if( dimensionality > 3 )
                throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal buffer dimensionality: ", dimensionality );
            stream->setSize( dimensionality, indims, 1 );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( indims, context );

    try
    {
        if( dimensionality > 3 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal buffer dimensionality: ", dimensionality );
        size_t dims[3];
        for( unsigned int i = 0; i < dimensionality; ++i )
            dims[i]         = indims[i];
        buffer->setSize( dimensionality, dims );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetSizev( RTbuffer buffer_api, unsigned int maxdim, RTsize* outdims )
{
    if( outdims )
    {
        unsigned int max = maxdim > 3 ? 3 : maxdim;
        for( unsigned int i = 0; i < max; ++i )
            outdims[i]      = 0;
    }

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( outdims, context );
        try
        {
            if( maxdim > 3 )
                throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal buffer dimensionality: ", maxdim );
            stream->getSize( maxdim, outdims );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( outdims, context );

    try
    {
        if( maxdim > 3 )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal buffer dimensionality: ", maxdim );
        if( buffer->getDimensionality() > maxdim )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal buffer dimensionality: ", maxdim );
        size_t dims[3];
        buffer->getSize( dims );
        size_t dimensionality = buffer->getDimensionality();
        for( unsigned int i = 0; i < dimensionality; ++i )
            outdims[i]      = range_cast<RTsize>( dims[i] );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetDimensionality( RTbuffer buffer_api, unsigned int* dimensionality )
{
    if( dimensionality )
        *dimensionality = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( dimensionality, context );
        try
        {
            *dimensionality = stream->getDimensionality();
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }


    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( dimensionality, context );

    try
    {
        *dimensionality = buffer->getDimensionality();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetGLBOId( RTbuffer buffer_api, unsigned int* glId )
{
    if( glId )
        *glId = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        CHECK_EXPRESSION( false, stream->getContext(), "Operation not valid for stream buffers" );
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( glId, context );
    CHECK_EXPRESSION( !buffer->isDemandLoad(), context, "Operation not valid for demand buffers" );

    try
    {
        const GfxInteropResource& resource = buffer->getGfxInteropResource();
        if( resource.isOGL() )
        {
            if( resource.kind != GfxInteropResource::OGL_BUFFER_OBJECT )
                throw IlwalidValue( RT_EXCEPTION_INFO, "GL interop Buffer does not contain a GLBuffer binding." );

            *glId = resource.gl.glId;
        }

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetId( RTbuffer buffer_api, int* buffer_id )
{
    if( buffer_id )
        *buffer_id = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        CHECK_EXPRESSION( false, stream->getContext(), "Operation not valid for stream buffers" );
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( buffer_id, context );

    try
    {
        *buffer_id = buffer->getAPIId();
        llog( 40 ) << "GetId: buffer -> " << *buffer_id << std::endl;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextGetBufferFromId( RTcontext context_api, int buffer_id, RTbuffer* buffer )
{
    if( buffer )
        *buffer = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );

    finishAsyncLaunches( context_api );
    CHECK_NULL( buffer, context );

    try
    {
        Buffer* result = nullptr;
        if( !context->getObjectManager()->getBuffers().get( buffer_id, result ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Supplied buffer_id is not valid: ", buffer_id );
        *buffer = api_cast( result );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetDevicePointer( RTbuffer buffer_api, int optix_device_ordinal, void** device_pointer )
{
    if( device_pointer )
        *device_pointer = nullptr;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        CHECK_EXPRESSION( false, stream->getContext(), "Operation not valid for stream buffers" );
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( device_pointer, context );
    CHECK_EXPRESSION( !buffer->isDemandLoad(), context, "Operation not valid for demand buffers" );
    try
    {
        const DeviceArray& devices = context->getDeviceManager()->visibleDevices();
        if( optix_device_ordinal < 0 || optix_device_ordinal >= static_cast<int>( devices.size() ) )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid device ordinal." );

        Device* device = devices[optix_device_ordinal];

        const GfxInteropResource& resource = buffer->getGfxInteropResource();
        if( resource.kind != GfxInteropResource::NONE )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot get device pointers from non-LWCA interop buffers." );

        if( !device->isEnabled() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot get device pointers from non-enabled device." );

        void* ptr = buffer->getInteropPointer( device );
        if( !ptr )
            throw IlwalidValue( RT_EXCEPTION_INFO, "No device memory present for given device in buffer." );

        *device_pointer = ptr;

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferSetMipLevelCount( RTbuffer buffer_api, unsigned int levels )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        try
        {
            stream->setMipLevelCount( levels );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        buffer->setMipLevelCount( levels );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetMipLevelCount( RTbuffer buffer_api, unsigned int* level )
{
    if( level )
        *level = 0;

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        finishAsyncLaunches( buffer_api );
        Context* context = stream->getContext();
        CHECK_NULL( level, context );
        try
        {
            *level = stream->getMipLevelCount();
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    CHECK_NULL( level, context );

    try
    {
        *level = buffer->getMipLevelCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

static RTresult bufferMap( RTbuffer buffer_api, unsigned int map_flags, unsigned int level, void* user_owned, void** optix_owned )
{
    if( optix_owned )
        *optix_owned = nullptr;

    if( user_owned )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Mapping buffers to user owned memory is not yet supported." );

    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        Context* context = stream->getContext();
        CHECK_NULL( optix_owned, context );
        try
        {
            *optix_owned = stream->map( level );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );

    finishAsyncLaunches( buffer_api );  // stop only if this isn't a stream buffer

    Context* context = buffer->getContext();
    CHECK_NULL( optix_owned, context );

    try
    {
        // If the buffer is a stream source, the user expects the aclwmulated output rather than the actual buffer
        if( StreamBuffer* target_stream = context->getObjectManager()->getTargetStream( buffer ) )
        {
            *optix_owned = target_stream->map_aclwm( level );
            return RT_SUCCESS;
        }

        const MapMode mode = buffer->getMapModeForAPI( map_flags );
        *optix_owned       = buffer->map( mode, level );

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferMap( RTbuffer buffer_api, void** optix_owned )
{
    return bufferMap( buffer_api, RT_BUFFER_MAP_READ_WRITE, 0, nullptr, optix_owned );
}

RTresult _rtBufferMapEx( RTbuffer buffer_api, unsigned int map_flags, unsigned int level, void* user_owned, void** optix_owned )
{
    return bufferMap( buffer_api, map_flags, level, user_owned, optix_owned );
}

RTresult static bufferUnmap( RTbuffer buffer_api, unsigned int level )
{
    if( StreamBuffer* stream = getStreamBuffer( buffer_api ) )
    {
        Context* context = stream->getContext();
        try
        {
            stream->unmap( level );
            return RT_SUCCESS;
        }
        HANDLE_EXCEPTIONS( context );
    }

    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );  // stop only if this isn't a stream buffer
    Context* context = buffer->getContext();

    try
    {
        if( StreamBuffer* target_stream = context->getObjectManager()->getTargetStream( buffer ) )
        {
            target_stream->unmap_aclwm( level );
            return RT_SUCCESS;
        }

        buffer->unmap( level );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferUnmap( RTbuffer buffer_api )
{
    return bufferUnmap( buffer_api, 0 );
}

RTresult _rtBufferUnmapEx( RTbuffer buffer_api, unsigned int level )
{
    return bufferUnmap( buffer_api, level );
}

RTresult _rtTextureSamplerCreateFromGLImage( RTcontext context_api, unsigned int gl_id, RTgltarget target, RTtexturesampler* textureSampler )
{
    if( textureSampler )
        *textureSampler = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( textureSampler, context );

    if( target == RT_TARGET_GL_RENDER_BUFFER )
    {
        CHECK_EXPRESSION( GL::IsRenderbuffer( gl_id ), context, "Not a valid OpenGL renderbuffer" );
    }
    else
    {
        CHECK_EXPRESSION( GL::IsTexture( gl_id ), context, "Not a valid OpenGL texture" );
    }

    try
    {
        GfxInteropResource resource( target == RT_TARGET_GL_RENDER_BUFFER ? GfxInteropResource::OGL_RENDER_BUFFER :
                                                                            GfxInteropResource::OGL_TEXTURE,
                                     gl_id, target );
        Device*                         device = context->getDeviceManager()->glInteropDevice();
        std::unique_ptr<TextureSampler> ts( new TextureSampler( context, resource, device ) );

        *textureSampler = api_cast( ts.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetGLImageId( RTtexturesampler textureSampler_api, unsigned int* glId )
{
    if( glId )
        *glId = 0;

    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();
    CHECK_NULL( glId, context );

    try
    {
        const GfxInteropResource& resource = textureSampler->getGfxInteropResource();
        if( resource.isOGL() )
        {
            if( !( resource.kind == GfxInteropResource::OGL_TEXTURE || resource.kind == GfxInteropResource::OGL_RENDER_BUFFER ) )
                throw IlwalidValue( RT_EXCEPTION_INFO, "GL Texture does not contain a GL image binding." );

            *glId = resource.gl.glId;
        }

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );

    return RT_SUCCESS;
}

RTresult _rtBufferGLRegister( RTbuffer buffer_api )
{
    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();

    try
    {
        const GfxInteropResource& resource = buffer->getGfxInteropResource();
        if( !resource.isOGL() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Buffer is not bound to an OpenGL object." );
        buffer->registerGfxInteropResource();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGLUnregister( RTbuffer buffer_api )
{
    Buffer* buffer = api_cast( buffer_api );

    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    finishAsyncLaunches( buffer_api );
    Context* context = buffer->getContext();
    try
    {
        const GfxInteropResource& resource = buffer->getGfxInteropResource();
        if( !resource.isOGL() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Buffer is not bound to an OpenGL object." );
        buffer->unregisterGfxInteropResource();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGLRegister( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );

    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        const GfxInteropResource& resource = textureSampler->getGfxInteropResource();
        if( !resource.isOGL() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "TextureSampler is not created from an OpenGL object." );
        textureSampler->registerGfxInteropResource();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGLUnregister( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );

    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    finishAsyncLaunches( textureSampler_api );
    Context* context = textureSampler->getContext();

    try
    {
        const GfxInteropResource& resource = textureSampler->getGfxInteropResource();
        if( !resource.isOGL() )
            throw IlwalidValue( RT_EXCEPTION_INFO, "TextureSampler is not created from an OpenGL object." );
        textureSampler->unregisterGfxInteropResource();

        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListCreate( RTcontext context_api, RTcommandlist* list )
{
    if( list )
        *list = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( list, context );

    try
    {
        std::unique_ptr<CommandList> cl( new CommandList( context ) );
        *list = api_cast( cl.release() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListDestroy( RTcommandlist list_api )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api );
    Context* context = list->getContext();

    try
    {
        // The command list doesn't have any parents so no need to detachFromParents.
        delete list;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListAppendPostprocessingStage( RTcommandlist list_api, RTpostprocessingstage stage_api, RTsize launch_width, RTsize launch_height )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    PostprocessingStage* stage = api_cast( stage_api );
    CHECK_OBJECT_POINTER( stage, RT_OBJECT_POSTPROCESSINGSTAGE );
    finishAsyncLaunches( list_api );
    Context* context = list->getContext();
    try
    {
        list->appendPostprocessingStage( stage, launch_width, launch_height );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListAppendLaunch1D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api );
    Context* context = list->getContext();
    try
    {
        list->appendLaunch( entry_point_index, launch_width );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListAppendLaunch2D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width, RTsize launch_height )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api );
    Context* context = list->getContext();
    try
    {
        list->appendLaunch( entry_point_index, launch_width, launch_height );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListAppendLaunch3D( RTcommandlist list_api, unsigned int entry_point_index, RTsize launch_width, RTsize launch_height, RTsize launch_depth )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api );
    Context* context = list->getContext();
    try
    {
        list->appendLaunch( entry_point_index, launch_width, launch_height, launch_depth );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListSetDevices( RTcommandlist list_api, unsigned int count, const int* devices )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api, false );
    Context* context = list->getContext();

    try
    {
        std::vector<unsigned int> device_array( count );
        for( size_t i       = 0; i < count; ++i )
            device_array[i] = devices[i];
        list->setDevices( device_array );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListGetDevices( RTcommandlist list_api, int* devices )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api, false );
    Context* context = list->getContext();

    try
    {
        std::vector<unsigned int> device_array = list->getDevices();
        for( size_t i = 0; i < device_array.size(); ++i )
        {
            devices[i] = device_array[i];
        }
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListGetDeviceCount( RTcommandlist list_api, unsigned int* count )
{
    if( count )
        *count = 0;

    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api, false );
    Context* context = list->getContext();
    CHECK_NULL( count, context );

    try
    {
        std::vector<unsigned int> device_array = list->getDevices();
        *count                                 = static_cast<unsigned int>( device_array.size() );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListFinalize( RTcommandlist list_api )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api );
    Context* context = list->getContext();
    try
    {
        list->finalize();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListExelwte( RTcommandlist list_api )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api, false );
    Context* context = list->getContext();
    try
    {
        list->execute();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListSetLwdaStream( RTcommandlist list_api, void* stream )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api );
    Context* context = list->getContext();
    try
    {
        list->setLwdaStream( stream );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListGetLwdaStream( RTcommandlist list_api, void** stream )
{
    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api, false );
    Context* context = list->getContext();
    try
    {
        list->getLwdaStream( stream );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtCommandListGetContext( RTcommandlist list_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    CommandList* list = api_cast( list_api );
    CHECK_OBJECT_POINTER( list, RT_OBJECT_COMMANDLIST );
    finishAsyncLaunches( list_api, false );
    Context* context = list->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtPostProcessingStageCreateBuiltin( RTcontext context_api, const char* builtin_name, void* denoiser, void* ssim_predictor, RTpostprocessingstage* stage )
{
    if( stage )
        *stage = nullptr;

    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    finishAsyncLaunches( context_api );
    CHECK_NULL( stage, context );
    CHECK_NULL( builtin_name, context );

    try
    {
        std::string name( builtin_name );
        if( name == "DLDenoiser" )
        {
            std::unique_ptr<PostprocessingStage> st( new PostprocessingStageDenoiser( context, denoiser ) );
            *stage = api_cast( st.release() );
            return RT_SUCCESS;
        }
        else if( name == "TonemapperSimple" )
        {
            std::unique_ptr<PostprocessingStage> st( new PostprocessingStageTonemap( context ) );
            *stage = api_cast( st.release() );
            return RT_SUCCESS;
        }
        else if( name == "DLSSIMPredictor" )
        {
            std::unique_ptr<PostprocessingStage> st( new PostprocessingStageSSIMPredictor( context, ssim_predictor ) );
            *stage = api_cast( st.release() );
            return RT_SUCCESS;
        }
        else
        {
            char buff[512];
            snprintf( buff, sizeof( buff ),
                      "Failed to create built-in postprocessing stage. No stage with name %s exists.", builtin_name );
            throw IlwalidValue( RT_EXCEPTION_INFO, buff );
        }
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtPostProcessingStageDeclareVariable( RTpostprocessingstage stage_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    PostprocessingStage* stage = api_cast( stage_api );
    CHECK_OBJECT_POINTER( stage, RT_OBJECT_POSTPROCESSINGSTAGE );
    finishAsyncLaunches( stage_api );
    Context* context = stage->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( stage->declareVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}


RTresult _rtPostProcessingStageDestroy( RTpostprocessingstage stage_api )
{
    PostprocessingStage* stage = api_cast( stage_api );
    CHECK_OBJECT_POINTER( stage, RT_OBJECT_POSTPROCESSINGSTAGE );
    finishAsyncLaunches( stage_api );
    Context* context = stage->getContext();

    try
    {
        // Removes this stage from all Command lists
        stage->detachFromParents();
        delete stage;
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtPostProcessingStageGetContext( RTpostprocessingstage stage_api, RTcontext* c )
{
    if( c )
        *c = nullptr;

    PostprocessingStage* stage = api_cast( stage_api );
    CHECK_OBJECT_POINTER( stage, RT_OBJECT_POSTPROCESSINGSTAGE );
    finishAsyncLaunches( stage_api, false );
    Context* context = stage->getContext();
    CHECK_NULL( c, context );

    *c = api_cast( context );

    return RT_SUCCESS;
}

RTresult _rtPostProcessingStageQueryVariable( RTpostprocessingstage stage_api, const char* name, RTvariable* v )
{
    if( v )
        *v = nullptr;

    PostprocessingStage* stage = api_cast( stage_api );
    CHECK_OBJECT_POINTER( stage, RT_OBJECT_POSTPROCESSINGSTAGE );
    finishAsyncLaunches( stage_api );
    Context* context = stage->getContext();
    CHECK_NULL( v, context );
    CHECK_NULL( name, context );

    try
    {
        *v = api_cast( stage->queryVariable( name ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtPostProcessingStageGetVariableCount( RTpostprocessingstage stage_api, unsigned int* c )
{
    if( c )
        *c = 0;

    PostprocessingStage* stage = api_cast( stage_api );
    CHECK_OBJECT_POINTER( stage, RT_OBJECT_POSTPROCESSINGSTAGE );
    finishAsyncLaunches( stage_api, false );
    Context* context = stage->getContext();
    CHECK_NULL( c, context );

    try
    {
        *c = stage->getVariableCount();
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtPostProcessingStageGetVariable( RTpostprocessingstage stage_api, unsigned int index, RTvariable* v )
{
    if( v )
        *v = nullptr;

    PostprocessingStage* stage = api_cast( stage_api );
    CHECK_OBJECT_POINTER( stage, RT_OBJECT_POSTPROCESSINGSTAGE );
    finishAsyncLaunches( stage_api );
    Context* context = stage->getContext();
    CHECK_NULL( v, context );

    try
    {
        // assume throw on index out of range
        *v = api_cast( stage->getVariableByIndex( index ) );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

namespace {

/// Indicates whether this DLL was loaded from the SDK or from the drivers.
bool g_isLibraryFromSdk;

/// The build version.
const char* g_buildVersion = OPTIX_BUILD_VERSION;

RTresult splitVersionIntoBranchAndCL( const char* version, int& branch, int& CL )
{
    if( !version )
        return RT_ERROR_ILWALID_VALUE;

    // Find '.'
    const std::string& versionStr = version;
    size_t             pos        = versionStr.find( '.' );
    if( pos == std::string::npos )
        return RT_ERROR_ILWALID_VALUE;

    // Parse string before '.' into tmpBranch
    std::istringstream s( std::string( version, version + pos ) );
    int                tmpBranch;
    s >> tmpBranch;
    if( s.fail() || !s.eof() )
        return RT_ERROR_ILWALID_VALUE;

    // Parse string after '.' into tmpCL
    s.clear();
    s.str( version + pos + 1 );
    int tmpCL;
    s >> tmpCL;
    if( s.fail() || !s.eof() )
        return RT_ERROR_ILWALID_VALUE;

    branch = tmpBranch;
    CL     = tmpCL;

    return RT_SUCCESS;
}
}

RTresult _rtGetBuildVersion( const char** result )
{
    *result = g_buildVersion;
    return RT_SUCCESS;
}

RTresult _rtOverridesOtherVersion( const char* otherVersion, int* result )
{
    RTresult res;

    const char* ourVersion;
    res = _rtGetBuildVersion( &ourVersion );
    if( res != RT_SUCCESS )
        return res;

    int ourBranch, ourCL;
    res = splitVersionIntoBranchAndCL( ourVersion, ourBranch, ourCL );
    if( res != RT_SUCCESS )
        return res;

    // The wrapper should not call that method if it cannot provide otherVersion, but let's be
    // explicit here.
    if( !otherVersion )
        return RT_ERROR_ILWALID_VALUE;

    int otherBranch, otherCL;
    res = splitVersionIntoBranchAndCL( otherVersion, otherBranch, otherCL );

    // If we do not understand otherVersion, let's hope that the other DLL understands our version
    // and returns something meaningful. If neither DLL understands the others version, the
    // wrapper needs to make a decision.
    if( res != RT_SUCCESS )
        return res;

    // Right now, we use the same implemention for the driver and the SDK. If different
    // implementations need to be used, you can make use of the g_isLibraryFromSdk variable:
    //
    // if( g_isLibraryFromSdk ) { ... } else { ... }.
    //
    // Make sure that the variable is not removed by accident.
    (void)g_isLibraryFromSdk;

    // Below is the default implementation of the comparison function, but it can be changed as
    // needed. The default implementation implements lexicographic comparison for
    // (ourBranch, ourCL) > (otherBranch, otherCL).
    if( ourBranch > otherBranch )
        *result = 1;
    else if( ourBranch < otherBranch )
        *result = 0;
    else
        *result = ourCL > otherCL ? 1 : 0;

    return RT_SUCCESS;
}

RTresult _rtSetLibraryVariant( bool isLibraryFromSdk )
{
    g_isLibraryFromSdk = isLibraryFromSdk;
    RTCore::setRtcoreLibraryVariant( isLibraryFromSdk );
    return RT_SUCCESS;
}

RTresult _rtSupportsLwrrentDriver()
{
    DriverVersion lwrrentDriverVersion;
    DriverVersion requiredDriverVersion( DeviceManager::getMinimumRequiredDriverVersion() );
    if( lwrrentDriverVersion.isValid() && lwrrentDriverVersion < requiredDriverVersion )
        return RT_ERROR_ILWALID_DRIVER_VERSION;

    return RT_SUCCESS;
}

#ifdef _WIN32  ////////////////////////// Windows-only section ///////////////////////////////

RTresult _rtDeviceGetWGLDevice( int* device, HGPULW hGpu )
{
    if( !device )
    {
        lerr << "rtDeviceGetWGLDevice error: device parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        *device = optix::DeviceManager::wglDevice( hGpu );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME( "rtDeviceGetWGLDevice" );
}

RTresult _rtDeviceGetD3D10Device( int* device, IDXGIAdapter* pAdapter )
{
    if( !device )
    {
        lerr << "rtDeviceGetD3D10Device error: device parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    if( !pAdapter )
    {
        lerr << "rtDeviceGetD3D10Device error: pszAdapterName parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        lerr << "DirectX interop is not supported in this version of OptiX\n";
        return RT_ERROR_NOT_SUPPORTED;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME( "rtDeviceGetD3D10Device" );
}

RTresult _rtDeviceGetD3D11Device( int* device, IDXGIAdapter* pAdapter )
{
    if( !device )
    {
        lerr << "rtDeviceGetD3D11Device error: device parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    if( !pAdapter )
    {
        lerr << "rtDeviceGetD3D11Device error: pszAdapterName parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        lerr << "DirectX interop is not supported in this version of OptiX\n";
        return RT_ERROR_NOT_SUPPORTED;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME( "rtDeviceGetD3D11Device" );
}

RTresult _rtDeviceGetD3D9Device( int* device, const char* pszAdapterName )
{
    if( !device )
    {
        lerr << "rtDeviceGetD3D9Device error: device parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    if( !pszAdapterName )
    {
        lerr << "rtDeviceGetD3D9Device error: pszAdapterName parameter is null\n";
        return RT_ERROR_ILWALID_VALUE;
    }

    try
    {
        lerr << "DirectX interop is not supported in this version of OptiX\n";
        return RT_ERROR_NOT_SUPPORTED;
    }
    HANDLE_EXCEPTIONS_NO_CONTEXT_NAME( "rtDeviceGetD3D9Device" );
}

RTresult _rtContextSetD3D9Device( RTcontext context_api, IDirect3DDevice9* matchingDevice )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferCreateFromD3D9Resource( RTcontext context_api, unsigned int type, IDirect3DResource9* pResource, RTbuffer* buffer )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetD3D9Resource( RTtexturesampler textureSampler_api, IDirect3DResource9** pResource )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerCreateFromD3D9Resource( RTcontext context_api, IDirect3DResource9* pResource, RTtexturesampler* textureSampler )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerCreateFromD3D10Resource( RTcontext context_api, ID3D10Resource* pResource, RTtexturesampler* textureSampler )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerCreateFromD3D11Resource( RTcontext context_api, ID3D11Resource* pResource, RTtexturesampler* textureSampler )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetD3D9Resource( RTbuffer buffer_api, IDirect3DResource9** pResource )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetD3D10Device( RTcontext context_api, ID3D10Device* matchingDevice )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferCreateFromD3D10Resource( RTcontext context_api, unsigned int type, ID3D10Resource* pResource, RTbuffer* buffer )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetD3D10Resource( RTbuffer buffer_api, ID3D10Resource** pResource )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetD3D10Resource( RTtexturesampler textureSampler_api, ID3D10Resource** pResource )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtContextSetD3D11Device( RTcontext context_api, ID3D11Device* matchingDevice )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferCreateFromD3D11Resource( RTcontext context_api, unsigned int type, ID3D11Resource* pResource, RTbuffer* buffer )
{
    Context* context = api_cast( context_api );
    CHECK_NULL( context, context );
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferGetD3D11Resource( RTbuffer buffer_api, ID3D11Resource** pResource )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerGetD3D11Resource( RTtexturesampler textureSampler_api, ID3D11Resource** pResource )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferD3D9Register( RTbuffer buffer_api )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferD3D10Register( RTbuffer buffer_api )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferD3D11Register( RTbuffer buffer_api )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferD3D9Unregister( RTbuffer buffer_api )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferD3D10Unregister( RTbuffer buffer_api )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtBufferD3D11Unregister( RTbuffer buffer_api )
{
    Buffer* buffer = api_cast( buffer_api );
    CHECK_OBJECT_POINTER( buffer, RT_OBJECT_BUFFER );
    Context* context = buffer->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerD3D9Register( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerD3D10Register( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerD3D11Register( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerD3D9Unregister( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerD3D10Unregister( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

RTresult _rtTextureSamplerD3D11Unregister( RTtexturesampler textureSampler_api )
{
    TextureSampler* textureSampler = api_cast( textureSampler_api );
    CHECK_OBJECT_POINTER( textureSampler, RT_OBJECT_TEXTURE_SAMPLER );
    Context* context = textureSampler->getContext();
    try
    {
        throw IlwalidValue( RT_EXCEPTION_INFO, "DirectX interop is not supported in this version of OptiX" );
        return RT_SUCCESS;
    }
    HANDLE_EXCEPTIONS( context );
}

#endif  ////////////////////////// END Windows-only section ///////////////////////////////
