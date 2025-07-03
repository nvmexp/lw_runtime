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


namespace optix {
class PersistentStream;
class VariableType;

void readOrWrite( PersistentStream* stream, VariableType* vtype, const char* label );

/*
   * Compact representation of variable types.  Keep sizeof(VariableType) == 4!
   */
class VariableType
{
    //
    // WARNING: This is a persistent class. If you change anything you
    // should also update the readOrWrite function.
    //

  public:
    enum Type
    {
        Float,
        Int,
        Uint,
        LongLong,
        ULongLong,
        Ray,
        Buffer,
        BufferId,
        DemandBuffer,
        GraphNode,
        Program,
        ProgramId,
        TextureSampler,
        UserData,
        Unknown /* Make sure this stays last. */
    };
    VariableType();

    // size is:
    //   - element count for Float, Int, or Uint types
    //   - element size for Buffer
    //   - program signature id for Program
    VariableType( Type type, unsigned int size, unsigned int dimensionality = 0 );

    // Buffer and TextureSampler Variables shouldn't use a generalized VariableType constructor.
    // Their sizes and dimensionality never change in the Variable, and are instead tracked in the
    // Buffer/TextureSampler itself.
    // For Variables only, not varrefs!
    static VariableType createForBufferVariable( bool isDemandLoaded );
    static VariableType createForTextureSamplerVariable();

    static VariableType createForProgramVariable();
    static VariableType createForCallableProgramVariable( unsigned sig );

    ~VariableType();

    bool operator==( const VariableType& ) const;
    bool operator!=( const VariableType& ) const;

    Type         baseType() const;
    unsigned int numElements() const;
    unsigned int programSignatureToken() const;
    unsigned int bufferDimensionality() const;

    bool   isValid() const;
    bool   isBuffer() const;
    bool   isDemandBuffer() const;
    bool   isGraphNode() const;
    bool   isTextureSampler() const;
    bool   isProgram() const;
    bool   isProgramId() const;
    bool   isBufferId() const;
    bool   isTypeWithValidDefaultValue() const;
    bool   isProgramOrProgramId() const;
    size_t computeSize() const;
    size_t computeAlignment() const;

    std::string toString() const;

    friend void optix::readOrWrite( PersistentStream* stream, VariableType* vtype, const char* label );

  private:
    void pack( Type t, unsigned int size, unsigned int dimensionality );
    unsigned int m_packedType;
};


// This is used to compute the VariableType enum from the type
template <typename T>
struct ValueType
{
};

template <>
struct ValueType<float>
{
    static const VariableType::Type vtype = VariableType::Float;
};

template <>
struct ValueType<int>
{
    static const VariableType::Type vtype = VariableType::Int;
};

template <>
struct ValueType<unsigned int>
{
    static const VariableType::Type vtype = VariableType::Uint;
};

template <>
struct ValueType<long long>
{
    static const VariableType::Type vtype = VariableType::LongLong;
};

template <>
struct ValueType<unsigned long long>
{
    static const VariableType::Type vtype = VariableType::ULongLong;
};

}  // namespace optix
