#pragma once

#include <unordered_map>
#include <map>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <list>

#include <lwtensor.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/operators.h>
#include <lwtensor/internal/defines.h>

#include <lwtensor/internal/operators.h>

namespace ReferencePW
{
    using namespace LWTENSOR_NAMESPACE;

lwtensorStatus_t initExtent(const TensorDescriptor *desc, const mode_type* mode,
    std::unordered_map<mode_type, extent_type> &extent)
{
    if ( desc == NULL || mode == NULL ) return LWTENSOR_STATUS_SUCCESS;

    for ( int i = 0; i < desc->getNumModes(); ++ i )
    {
        auto modeSize = desc->getExtent( i );
        if ( desc->getExtent( i ) == 0 || extent.find( mode[ i ] ) != extent.end() &&
                extent.at( mode[ i ] ) != modeSize )
        {
            fprintf(stderr, "LWTENSOR ERROR: extent of mode %c does not match.\n",mode[i]);
            return LWTENSOR_STATUS_ILWALID_VALUE;
        }
        else
        {
            extent[ mode[ i ] ] = modeSize;
        }
    }
    return LWTENSOR_STATUS_SUCCESS;
}

lwtensorStatus_t initStride( const TensorDescriptor *desc, const mode_type* mode,
    std::unordered_map<mode_type, stride_type> &stride )
{
    if ( desc == NULL || mode == NULL ) return LWTENSOR_STATUS_SUCCESS;

    for ( int i=0; i < desc->getNumModes(); ++ i )
    {
        stride[ mode[ i ] ] = desc->getStride( i );
    }
    return LWTENSOR_STATUS_SUCCESS;
}

template<typename typeA, typename typeB, typename typeC, typename typeCompute>
lwtensorStatus_t lwtensorElementwise(
        int lwrrentModeIdx, const std::vector<mode_type> &modes,
        const typeCompute* alpha, const typeA *A, const std::unordered_map<mode_type, stride_type> &strideA,
        const typeCompute* beta,  const typeB *B, const std::unordered_map<mode_type, stride_type> &strideB,
        const typeCompute* gamma, const typeC *C, const std::unordered_map<mode_type, stride_type> &strideC,
        const std::unordered_map<mode_type, extent_type> &extent,
        mode_type vectorModeA, uint8_t vectorWidthA, uint8_t vectorOffsetA,
        mode_type vectorModeB, uint8_t vectorWidthB, uint8_t vectorOffsetB,
        mode_type vectorModeC, uint8_t vectorWidthC, uint8_t vectorOffsetC, bool useZeroPaddingC,
        const lwtensorOperator_t opA, const lwtensorOperator_t opB, const lwtensorOperator_t opC,
        const lwtensorOperator_t opAB, const lwtensorOperator_t opUnaryAfterBinary, const lwtensorOperator_t opABC, typeC *D )
{
    bool useA = A != nullptr && alpha != nullptr && !lwIsEqual( *alpha, lwGet<typeCompute>( 0 ) );
    bool useB = B != nullptr && beta  != nullptr && !lwIsEqual( *beta,  lwGet<typeCompute>( 0 ) );
    bool useC = C != nullptr && gamma != nullptr && !lwIsEqual( *gamma, lwGet<typeCompute>( 0 ) );

    if ( modes.size() <= lwrrentModeIdx )
    {
        return LWTENSOR_STATUS_ILWALID_VALUE;
    }

    mode_type lwrrentMode = modes[ lwrrentModeIdx ];
    if ( extent.find( lwrrentMode ) == extent.end() )
    {
        return LWTENSOR_STATUS_ILWALID_VALUE;
    }
    /* The extent of the current mode. */
    extent_type size = extent.at( lwrrentMode );
    /* If a tensor does not have the current mode, set the stride to 0. */
    stride_type strideA_ = (strideA.find(lwrrentMode) != strideA.end()) ?  strideA.at(lwrrentMode) : 0;
    stride_type strideB_ = (strideB.find(lwrrentMode) != strideB.end()) ?  strideB.at(lwrrentMode) : 0;
    stride_type strideC_ = (strideC.find(lwrrentMode) != strideC.end()) ?  strideC.at(lwrrentMode) : 0;
    //printf("%c %d %d %d %d %d %d %d %d\n", lwrrentMode, maxNumModes, size, strideA_, strideB_, strideC_, useA, useB, useC );

    /* If the current mode is vectorized, then set the vector width. */
    extent_type vectorA_ = ( vectorModeA == lwrrentMode ) ? vectorWidthA : 1;
    extent_type vectorB_ = ( vectorModeB == lwrrentMode ) ? vectorWidthB : 1;
    extent_type vectorC_ = ( vectorModeC == lwrrentMode ) ? vectorWidthC : 1;
    /* If the current mode is vectorized, then set the vector offset. */
    extent_type offsetA_ = ( vectorModeA == lwrrentMode ) ? vectorOffsetA : 0;
    extent_type offsetB_ = ( vectorModeB == lwrrentMode ) ? vectorOffsetB : 0;
    extent_type offsetC_ = ( vectorModeC == lwrrentMode ) ? vectorOffsetC : 0;

    /* Recur if this is not the last mode. */
    if ( lwrrentModeIdx < modes.size() - 1 )
    {
        for ( size_t i = 0; i < size; ++ i )
        {
            size_t shiftA = ( ( i + offsetA_ ) / vectorA_ ) * strideA_ + ( i + offsetA_ ) % vectorA_;
            size_t shiftB = ( ( i + offsetB_ ) / vectorB_ ) * strideB_ + ( i + offsetB_ ) % vectorB_;
            size_t shiftC = ( ( i + offsetC_ ) / vectorC_ ) * strideC_ + ( i + offsetC_ ) % vectorC_;
            auto ret = ReferencePW::lwtensorElementwise(
                    lwrrentModeIdx + 1, modes,
                    alpha, A + shiftA, strideA,
                    beta,  B + shiftB, strideB,
                    gamma, C + shiftC, strideC, extent,
                    vectorModeA, vectorWidthA, vectorOffsetA,
                    vectorModeB, vectorWidthB, vectorOffsetB,
                    vectorModeC, vectorWidthC, vectorOffsetC, useZeroPaddingC,
                    opA, opB, opC, opAB, opUnaryAfterBinary, opABC, D + shiftC );
            if ( ret != LWTENSOR_STATUS_SUCCESS )
            {
                return ret;
            }
        }
    }
    else
    {
        /* The size we will be using for tensor C with padding. */
        extent_type padding_size = size;

        /* Compute the padding size for tensor C. */
        if ( vectorModeC == lwrrentMode && useZeroPaddingC )
        {
            auto pad = ( size + vectorOffsetC ) % vectorWidthC;
            if ( pad ) pad = vectorWidthC - pad;
            padding_size += pad;
            //printf( "padding size %d\n", padding_size );
        }

        /* Loop over the padding size (the same as size while C is not padded). */
        for ( size_t i = 0; i < padding_size; ++ i )
        {
            size_t shiftA = ( ( i + offsetA_ ) / vectorA_ ) * strideA_ + ( i + offsetA_ ) % vectorA_;
            size_t shiftB = ( ( i + offsetB_ ) / vectorB_ ) * strideB_ + ( i + offsetB_ ) % vectorB_;
            size_t shiftC = ( ( i + offsetC_ ) / vectorC_ ) * strideC_ + ( i + offsetC_ ) % vectorC_;
            /* Initialize with zero. */
            auto a = lwGet<typeCompute>( 0 );
            auto b = lwGet<typeCompute>( 0 );
            auto c = lwGet<typeCompute>( 0 );
            auto ab = lwGet<typeCompute>( 0 );
            auto abc = lwGet<typeCompute>( 0 );

            if ( i < size )
            {
                /* Load elements from tensor A. */
                if ( useA )
                {
                    /* Type cast from typeA to typeCompute. */
                    a = lwGet<typeCompute>( A[ shiftA ] );
                    /* Unary operator and scaling. */
                    a = lwMul( (*alpha), lwtensorUnaryOp( a, opA ) );
                }
                if ( useB )
                {
                    /* Type cast from typeB to typeCompute. */
                    b = lwGet<typeCompute>( B[ shiftB ] );
                    /* Unary operator and scaling. */
                    b = lwMul( (*beta), lwtensorUnaryOp( b, opB ) );
                }
                /* Binary operator opAB. */
                ab = lwtensorUnaryOp(lwtensorBinaryOp( a, b, opAB ), opUnaryAfterBinary);
                if ( useC )
                {
                    /* Type cast from typeC to typeCompute. */
                    c = lwGet<typeCompute>( C[ shiftC ] );
                    /* Unary operator and scaling. */
                    c = lwMul( (*gamma), lwtensorUnaryOp( c, opC ) );
                }
                abc = lwtensorBinaryOp( ab, c, opABC );
            }
            D[ shiftC ] = lwGet<typeC>( abc );
        }
    }
    /* Return with no error. */
    return LWTENSOR_STATUS_SUCCESS;
}

lwtensorStatus_t validateElementwiseInputs(
    const void* alpha, const void* A, const TensorDescriptor* descA, const mode_type* modeA,
    const void* beta,  const void* B, const TensorDescriptor* descB, const mode_type* modeB,
    const void* gamma, const void* C, const TensorDescriptor* descC, const mode_type* modeC, void* D )
{
    if ( descC == nullptr || descC->getNumModes() > 0 && modeC == nullptr )
    {
        printf( "ERROR: descriptor for C must not be null.\n" );
        return LWTENSOR_STATUS_ILWALID_VALUE;
    }
    (void)gamma; (void) C;// gamma and C may be arbitrary based on the subsequent definition of useC (see below)
    if( D == nullptr)
    {
        return LWTENSOR_STATUS_ILWALID_VALUE;
    }


    /* Sanity check whether modeC is a superset of modeA. */
    if ( descA )
    {
        if(  descA->getNumModes() > 0 && modeA == nullptr )
        {
            printf( "modeA is null.\n" );
            return LWTENSOR_STATUS_ILWALID_VALUE;
        }
        if( A == nullptr || alpha == nullptr)
        {
            return LWTENSOR_STATUS_ILWALID_VALUE;
        }

        for ( int i = 0; i < descA->getNumModes(); ++ i )
        {
            bool found = false;
            for ( int j = 0; j < descC->getNumModes(); ++ j )
            {
                if ( modeC[ j ] == modeA[ i ] ) found = true;
            }
            if ( !found )
            {
                printf( "ERROR: each mode of A or B must be part of C too.\n" );
                return LWTENSOR_STATUS_ILWALID_VALUE;
            }
        }
    }


    /* Sanity check whether modeC is a superset of modeA. */
    if ( descB )
    {
        if(  descB->getNumModes() > 0 && modeB == nullptr )
        {
            printf( "modeB is null.\n" );
            return LWTENSOR_STATUS_ILWALID_VALUE;
        }
        if( B == nullptr || beta == nullptr)
        {
            return LWTENSOR_STATUS_ILWALID_VALUE;
        }
        for ( int i = 0; i < descB->getNumModes(); ++ i )
        {
            bool found = false;
            for ( int j = 0; j < descC->getNumModes(); ++ j )
            {
                if ( modeC[ j ] == modeB[ i ] ) found = true;
            }
            if ( !found )
            {
                printf( "ERROR: each mode of A or B must be part of C too.\n" );
                return LWTENSOR_STATUS_ILWALID_VALUE;
            }
        }
    }
    /* Return with no error. */
    return LWTENSOR_STATUS_SUCCESS;
}

}; /* end namespace ReferencePW */



extern "C"
lwtensorStatus_t lwtensorElementwiseReference( const lwtensorHandle_t* handle,
        const void* alpha, const void *A, const lwtensorTensorDescriptor_t* descA_, const int* modeA,
        const void* beta,  const void *B, const lwtensorTensorDescriptor_t* descB_, const int* modeB,
        const void* gamma, const void *C, const lwtensorTensorDescriptor_t* descC_, const int* modeC,
        const lwtensorOperator_t opAB, const lwtensorOperator_t opUnaryAfterBinary, const lwtensorOperator_t opABC, void *D, lwdaDataType_t typeCompute )
{
    (void) handle; // surpress warning

    /* Use STL namespace. */
    using namespace std;
    using namespace LWTENSOR_NAMESPACE;
    /* Colwert public struct to internal struct. */
    auto *descA = reinterpret_cast<const TensorDescriptor*>(descA_);
    auto *descB = reinterpret_cast<const TensorDescriptor*>(descB_);
    auto *descC = reinterpret_cast<const TensorDescriptor*>(descC_);

    /* Check if inputs are valid? */
    auto err =  ReferencePW::validateElementwiseInputs( alpha, A, descA, modeA,
                beta,  B, descB, modeB, gamma, C, descC, modeC, D );
    if( err != LWTENSOR_STATUS_SUCCESS )
        return err;


    lwtensorOperator_t opA = (descA != nullptr) ? descA->getOp() : LWTENSOR_OP_UNKNOWN;
    lwtensorOperator_t opB = (descB != nullptr) ? descB->getOp() : LWTENSOR_OP_UNKNOWN;
    lwtensorOperator_t opC = (descC != nullptr) ? descC->getOp() : LWTENSOR_OP_UNKNOWN;
    //bool useA = A != nullptr && descA != nullptr && alpha != nullptr && !isZero(alpha, typeCompute);
    //bool useB = B != nullptr && descB != nullptr && beta  != nullptr && !isZero(beta,  typeCompute);
    //bool useC = C != nullptr && descC != nullptr && gamma != nullptr && !isZero(gamma, typeCompute);
    bool useA = A != nullptr && descA != nullptr && alpha != nullptr;
    bool useB = B != nullptr && descB != nullptr && beta  != nullptr;
    bool useC = C != nullptr && descC != nullptr && gamma != nullptr;
    if( !useA )
    {
        A = nullptr;
        descA = nullptr;
        alpha = nullptr;
    }
    if( !useB )
    {
        B = nullptr;
        descB = nullptr;
        beta = nullptr;
    }

    std::vector<int> modeC_;
    for ( int i = 0; i < descC->getNumModes(); ++ i )
        modeC_.push_back(modeC[i]);

    unordered_map<mode_type, stride_type> strideA;
    unordered_map<mode_type, stride_type> strideB;
    unordered_map<mode_type, stride_type> strideC;
    unordered_map<mode_type, extent_type> extent;

    /* Add new mode if tensors are 0-order tensors */
    if( useA && descA->getNumModes() == 0 )
    {
        strideA[RESERVED_A_MODE_PW] = 0;
        strideC[RESERVED_A_MODE_PW] = 0;
        extent[RESERVED_A_MODE_PW] = 1;
        modeC_.push_back(RESERVED_A_MODE_PW);
    }
    if( useB && descB->getNumModes() == 0 )
    {
        strideB[RESERVED_B_MODE_PW] = 0;
        strideC[RESERVED_B_MODE_PW] = 0;
        extent[RESERVED_B_MODE_PW] = 1;
        modeC_.push_back(RESERVED_B_MODE_PW);
    }
    if( useC && descC->getNumModes() == 0 )
    {
        modeC_.push_back(RESERVED_C_MODE_PW);
        strideA[RESERVED_C_MODE_PW] = 0;
        strideB[RESERVED_C_MODE_PW] = 0;
        strideC[RESERVED_C_MODE_PW] = 0;
        extent[RESERVED_C_MODE_PW] = 1;
    }

    ReferencePW::initStride( descA, modeA, strideA );
    ReferencePW::initStride( descB, modeB, strideB );
    ReferencePW::initStride( descC, modeC_.data(), strideC );
    ReferencePW::initExtent( descA, modeA, extent );
    ReferencePW::initExtent( descB, modeB, extent );
    ReferencePW::initExtent( descC, modeC_.data(), extent );

    /* Leave vectorization info unset. */
//    VectorInfo info( modeA, descA, modeB, descB, modeC_.data(), descC );

    auto vectorModeA = (descA != nullptr) ? descA->getVectorMode(modeA) : LWTENSOR_ILWALID_MODE;
    auto vectorWidthA = (descA != nullptr) ? descA->getVectorWidth() : 1U;
    auto vectorOffsetA = (descA != nullptr) ? descA->getVectorOffset() : 0U;

    auto vectorModeB = (descB != nullptr) ? descB->getVectorMode(modeB) : LWTENSOR_ILWALID_MODE;
    auto vectorWidthB = (descB != nullptr) ? descB->getVectorWidth() : 1U;
    auto vectorOffsetB = (descB != nullptr) ? descB->getVectorOffset() : 0U;

    auto vectorModeC = (descC != nullptr) ? descC->getVectorMode(modeC_.data()) : LWTENSOR_ILWALID_MODE;
    auto vectorWidthC = (descC != nullptr) ? descC->getVectorWidth() : 1U;
    auto vectorOffsetC = (descC != nullptr) ? descC->getVectorOffset() : 0U;
    bool useZeroPaddingC = (descC != nullptr) ? descC->getZeroPadding() : false;

    /* Decide the mode order. */
    vector<mode_type> modes;

    /* While C is vectorized, the vectorized mode will be the last mode. */
    for ( int i = 0; i < modeC_.size(); ++ i )
    {
        auto mode = modeC_[ modeC_.size() - 1 - i ];
        /* Push the mode if it is not vectorized. */
        if ( mode != vectorModeC )
        {
            modes.push_back( mode );
        }
    }

    /* If C is vectorized, push the vectorized mode at the end. */
    if ( vectorModeC != LWTENSOR_ILWALID_MODE )
    {
        auto mode = vectorModeC;
        modes.push_back( mode );
    }


    lwdaDataType_t typeA = (descA) ? descA->getDataType() : LWDA_R_32F;
    lwdaDataType_t typeB = (descB) ? descB->getDataType() : LWDA_R_32F;
    lwdaDataType_t typeC = descC->getDataType();



#define TENSOR_PW_HELPER(a, a2, b, b2, c, c2, comp, comp2) \
    if( (!useA || typeA == a) && (!useB || typeB == b) && typeC == c && typeCompute == comp) \
    return ReferencePW::lwtensorElementwise<a2,b2,c2,comp2>(\
            0, modes, \
            (const comp2*) alpha, (const a2*) A, strideA, \
            (const comp2*) beta,  (const b2*) B, strideB, \
            (const comp2*) gamma, (const c2*) C, strideC, \
            extent,\
            vectorModeA, vectorWidthA, vectorOffsetA,\
            vectorModeB, vectorWidthB, vectorOffsetB,\
            vectorModeC, vectorWidthC, vectorOffsetC, useZeroPaddingC,\
            opA, opB, opC, opAB, opUnaryAfterBinary, opABC, (c2*) D );





    /* Public release uni-precision */
    TENSOR_PW_HELPER(LWDA_R_16F, half,  LWDA_R_16F, half,  LWDA_R_16F, half,  LWDA_R_16F, half);
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
    TENSOR_PW_HELPER(LWDA_R_16BF, BFloat16,  LWDA_R_16BF, BFloat16,  LWDA_R_16BF, BFloat16,  LWDA_R_16BF, BFloat16);
    TENSOR_PW_HELPER(LWDA_R_16BF, BFloat16,  LWDA_R_16BF, BFloat16,  LWDA_R_16BF, BFloat16,  LWDA_R_32F, float);
#endif
    TENSOR_PW_HELPER(LWDA_R_32F,   float, LWDA_R_32F,   float, LWDA_R_32F,   float, LWDA_R_32F,   float);
    TENSOR_PW_HELPER(LWDA_R_64F, double, LWDA_R_64F, double, LWDA_R_64F, double, LWDA_R_64F, double);
    TENSOR_PW_HELPER(LWDA_C_32F, lwComplex, LWDA_C_32F, lwComplex, LWDA_C_32F, lwComplex, LWDA_C_32F, lwComplex);
    TENSOR_PW_HELPER(LWDA_C_64F, lwDoubleComplex, LWDA_C_64F, lwDoubleComplex, LWDA_C_64F, lwDoubleComplex, LWDA_C_64F, lwDoubleComplex);


    TENSOR_PW_HELPER(LWDA_R_16F, half,  LWDA_R_16F, half,  LWDA_R_16F, half,  LWDA_R_32F, float);
    TENSOR_PW_HELPER(LWDA_R_32F,   float, LWDA_R_32F,   float, LWDA_R_16F,  half, LWDA_R_32F,   float);
    TENSOR_PW_HELPER(LWDA_R_64F, double, LWDA_R_64F, double, LWDA_R_32F, float, LWDA_R_64F, double);
    TENSOR_PW_HELPER(LWDA_C_64F, lwDoubleComplex, LWDA_C_64F, lwDoubleComplex, LWDA_C_32F, lwComplex, LWDA_C_64F, lwDoubleComplex);


    /* Mixed-precision not used. */
    TENSOR_PW_HELPER(LWDA_R_32F, float, LWDA_R_32F, float, LWDA_R_16F, half,  LWDA_R_16F, half);
    TENSOR_PW_HELPER(LWDA_R_16F, half,  LWDA_R_16F, half,  LWDA_R_16F, half,  LWDA_R_32F, float);
    TENSOR_PW_HELPER(LWDA_R_32F, float, LWDA_R_32F, float, LWDA_R_64F, double,  LWDA_R_64F, double);
    TENSOR_PW_HELPER(LWDA_R_64F, double,  LWDA_R_64F, double,  LWDA_R_32F, float, LWDA_R_32F, float);
    /* TensorRT: 16F and 32F. */
    TENSOR_PW_HELPER(LWDA_R_32F, float, LWDA_R_32F, float, LWDA_R_16F,  half, LWDA_R_32F, float);
    TENSOR_PW_HELPER(LWDA_R_16F,  half, LWDA_R_32F, float, LWDA_R_32F, float, LWDA_R_32F, float);
    TENSOR_PW_HELPER(LWDA_R_16F,  half, LWDA_R_32F, float, LWDA_R_16F,  half, LWDA_R_32F, float);
    /* TensorRT: 8I */
    TENSOR_PW_HELPER(LWDA_R_8I, int8_t, LWDA_R_32F, float, LWDA_R_32F, float, LWDA_R_32F, float);
    TENSOR_PW_HELPER(LWDA_R_8I, int8_t, LWDA_R_32F, float, LWDA_R_16F, half, LWDA_R_32F, float);
    TENSOR_PW_HELPER(LWDA_R_16F, half, LWDA_R_32F, float, LWDA_R_8I, int8_t, LWDA_R_32F, float);
    TENSOR_PW_HELPER(LWDA_R_32F, float, LWDA_R_32F, float, LWDA_R_8I, int8_t, LWDA_R_32F, float);
    /** lwDNN:  */
    TENSOR_PW_HELPER(LWDA_R_16F, half,  LWDA_R_16F, half,  LWDA_R_32F, float, LWDA_R_32F, float);
    TENSOR_PW_HELPER(LWDA_R_16F, half, LWDA_R_16F, half, LWDA_R_16F, half, LWDA_R_16F, half);
    TENSOR_PW_HELPER(LWDA_R_8U, uint8_t, LWDA_R_8U, uint8_t, LWDA_R_8U, uint8_t, LWDA_R_8U, uint8_t);
    TENSOR_PW_HELPER(LWDA_R_8U, uint8_t, LWDA_R_8U, uint8_t, LWDA_R_32U, uint32_t, LWDA_R_32U, uint32_t);
    /* Complete ("LWDA_R_8U",  "LWDA_R_8U",  "LWDA_R_32U", "LWDA_R_32U", 256, 32, 1) */
    TENSOR_PW_HELPER(LWDA_R_32U, uint32_t, LWDA_R_8U, uint8_t, LWDA_R_32U, uint32_t, LWDA_R_32U, uint32_t);
    /* ("LWDA_R_16F", "LWDA_R_16F", "LWDA_R_32F", "LWDA_R_32F", 256, 32, 1) */
    TENSOR_PW_HELPER(LWDA_R_32F, float, LWDA_R_16F, half, LWDA_R_32F, float, LWDA_R_32F, float);



#undef TENSOR_PW_HELPER

    printf( "WARNING: Reference does not suppport data type combination.\n" );
    return LWTENSOR_STATUS_NOT_SUPPORTED;
}
