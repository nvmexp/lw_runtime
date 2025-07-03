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

#include <Objects/Buffer.h>     // LinkedPtr<,Buffer> needs to know about Buffer
#include <Objects/GraphNode.h>  // LinkedPtr<,GraphNode> needs to know about GraphNode
#include <Objects/LexicalScope.h>
#include <Objects/Program.h>         // LinkedPtr<,Program> needs to know about Program
#include <Objects/TextureSampler.h>  // LinkedPtr<,TextureSampler> needs to know about TextureSampler
#include <Objects/VariableType.h>
#include <Util/LinkedPtr.h>
#include <corelib/misc/Concepts.h>
#include <o6/optix.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_matrix.h>

#include <string>

namespace optix {

// SGP: retire NonCopyable in favor of c++-11. Why should IsRemote be needed? No point in this being a virtual class
class Variable : public corelib::NonCopyable
{
  public:
    Variable( LexicalScope* scope, const std::string& name, unsigned int index, unsigned short token );
    virtual ~Variable();

    /* Identifier for C API santity checking */
    ObjectClass getClass() const;

    /* Parent */
    LexicalScope* getScope() const;

    /* Name */
    const std::string& getName() const;
    unsigned short     getToken() const;

    /* Index and offset within parent scope */
    static const unsigned int ILWALID_OFFSET = ~0u;

    unsigned int getIndex() const;
    unsigned int getScopeOffset() const;

    /* Type information */
    unsigned int getSize() const;
    unsigned int getAlignment() const;
    VariableType getType() const;
    bool         isMatrix() const;
    bool         isBuffer() const;
    bool         isTextureSampler() const;
    bool         isProgram() const;
    bool         isGraphNode() const;

    void setOrCheckType( const VariableType& type );


    /* Set values */
    template <typename T>
    void set( const T& data );

    template <typename T, unsigned int N>
    void set( const T* data );

    void setUserData( size_t size, const void* ptr );

    template <unsigned int R, unsigned int C>
    void setMatrix( bool tranpose, const float* m );

    void setGraphNode( GraphNode* );
    void setProgram( Program* );
    void setBindlessProgram( Program* );
    void setBuffer( Buffer* );
    void setTextureSampler( TextureSampler* );

    /* Gets - note that gets do not perform colwersion */
    template <typename T>
    T get() const;

    template <typename T, int N>
    void get( T* data ) const;

    void getUserData( size_t size, void* ptr ) const;

    template <unsigned int N, unsigned int M>
    void getMatrix( bool transpose, float* m ) const;

    uint2           getMatrixDim() const;
    GraphNode*      getGraphNode() const;
    Program*        getProgram() const;
    Buffer*         getBuffer() const;
    TextureSampler* getTextureSampler() const;

    /* Used only by LexicalScope to manage variable placement within the object record */
    void setIndex( unsigned int index );
    void setScopeOffset( unsigned int offset );

    // Write the value of the variable into the scope
    void writeRecord( char* scope_base ) const;

    // Notifiers for buffer and texture sampler variables
    void bufferFormatDidChange();
    void textureSamplerFormatDidChange();

    // LinkedPtr relationship mangement
    void detachLinkedChild( const LinkedPtr_Link* link );
    void graphNodeOffsetDidChange();


  private:
    void checkType( const VariableType& type ) const;

    // Object class for identifying to the API
    ObjectClass m_class;

    // Parent scope
    LexicalScope* m_scope;

    // Name
    std::string    m_name;
    unsigned short m_token;

    // Index and offset within parent scope
    unsigned int m_index;
    unsigned int m_scopeOffset;

    // Type
    VariableType m_type;
    uint2        m_matrixDim;

    // Data - only one of {m_ptr, m_graphNode, m_program, m_bindlessProgram, m_buffer, or m_textureSampler} will be valid
    char m_databuf[sizeof( float ) * 4 * 4];  // Enough to hold mat4x4.   Anything larger needs to be dynamically allocated
    void* m_ptr;
    LinkedPtr<Variable, Buffer>         m_buffer;
    LinkedPtr<Variable, GraphNode>      m_graphNode;
    LinkedPtr<Variable, Program>        m_program;
    LinkedPtr<Variable, Program>        m_bindlessProgram;
    LinkedPtr<Variable, TextureSampler> m_textureSampler;
};

template <typename T>
void optix::Variable::set( const T& data )
{
    set<T, 1>( &data );
}

namespace {
// The purpose of these functions is to facilitate comparisons with NaN.  VS's cl.exe
// /fp:fast doesn't deal with NaN's correctly and the user should expect the value of
// the variable to be exactly as they set it (NaN's and all).  By using these comparison
// functions we can use a bitwise comparison for floats.
template <typename T>
bool compareEQ( T* dst, const T* src )
{
    return *dst == *src;
}

template <>
bool compareEQ<float>( float* dst, const float* src )
{
    return compareEQ( reinterpret_cast<int*>( dst ), reinterpret_cast<const int*>( src ) );
}
}

// Avoid warning about unused function.
inline void fakeUseOfCompareEQFloat()
{
    float f = 42.0f;
    compareEQ( &f, &f );
}

template <typename T, unsigned int N>
void Variable::set( const T* data )
{
    bool changed = !m_type.isValid();
    setOrCheckType( VariableType( ValueType<T>::vtype, N ) );

    // Copy data
    T* dst = static_cast<T*>( m_ptr );
    for( unsigned int i = 0; i < N; ++i )
        if( !compareEQ( dst + i, data + i ) )
        {
            changed = true;
            dst[i]  = data[i];
        }

    if( changed )
        getScope()->variableValueDidChange( this );
}

template <>
inline void Variable::set<int, 1>( const int* data )
{
    bool changed = !m_type.isValid();
    setOrCheckType( VariableType( ValueType<int>::vtype, 1 ) );

    // Copy data
    int* dst      = static_cast<int*>( m_ptr );
    int  oldValue = dst[0];  // what about initial value?
    if( !compareEQ( dst, data ) )
    {
        changed = true;
        dst[0]  = data[0];
    }

    if( changed )
        getScope()->variableValueDidChange( this, oldValue, dst[0] );
}

template <typename T>
T optix::Variable::get() const
{
    T data;
    get<T, 1>( &data );
    return data;
}

template <typename T, int N>
void Variable::get( T* data ) const
{
    checkType( VariableType( ValueType<T>::vtype, N ) );

    const T* src = static_cast<const T*>( m_ptr );
    for( unsigned int i = 0; i < N; ++i )
        data[i]         = src[i];
}

template <unsigned int R, unsigned int C>
void Variable::setMatrix( bool transpose, const float* m )
{
    bool changed = !m_type.isValid();
    setOrCheckType( VariableType( VariableType::UserData, R * C * sizeof( float ) ) );
    m_matrixDim.y = R;
    m_matrixDim.x = C;
    const Matrix<R, C>* source = reinterpret_cast<const Matrix<R, C>*>( m );
    if( !transpose )
    {
        Matrix<R, C>* dst = reinterpret_cast<Matrix<R, C>*>( m_ptr );
        for( unsigned int i = 0; i < R * C; ++i )
            if( !compareEQ( dst->getData() + i, source->getData() + i ) )
            {
                changed = true;
                break;
            }
        if( changed )
            *dst = *source;
    }
    else
    {
        Matrix<C, R>* dst   = reinterpret_cast<Matrix<C, R>*>( m_ptr );
        Matrix<C, R>  trans = source->transpose();
        for( unsigned int i = 0; i < R * C; ++i )
            if( !compareEQ( dst->getData() + i, trans.getData() + i ) )
            {
                changed = true;
                break;
            }
        if( changed )
            *dst = trans;
    }

    if( changed )
        getScope()->variableValueDidChange( this );
}

template <unsigned int R, unsigned int C>
void Variable::getMatrix( bool transpose, float* m ) const
{
    checkType( VariableType( VariableType::UserData, R * C * sizeof( float ) ) );

    const Matrix<R, C>* source = reinterpret_cast<const Matrix<R, C>*>( m_ptr );
    if( !transpose )
    {
        Matrix<R, C>* dst = reinterpret_cast<Matrix<R, C>*>( m );
        *dst = *source;
    }
    else
    {
        Matrix<C, R>* dst = reinterpret_cast<Matrix<C, R>*>( m );
        *dst = source->transpose();
    }
}

}  // namespace optix
