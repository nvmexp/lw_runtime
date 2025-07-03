#include <stdio.h>
#include <assert.h>
#include <limits>
#include <cmath>

#include <lwComplex.h>

#include <lwtensor/internal/operators.h>
#include <lwtensor/internal/utilEx.h>
#include <lwtensor/internal/types.h>
#include <lwtensor/internal/exceptions.h>
//#define DEBUG

#include<lwtensor/internal/defines.h>
namespace LWTENSOR_NAMESPACE
{
const void* lwtensorGetOnePtr(lwdaDataType_t type)
{
   switch(type)
   {
      case LWDA_R_8I:
         return &lwtensorOneI8__ ;
      case LWDA_R_8U:
         return &lwtensorOneU8__ ;
      case LWDA_R_16F:
         return &lwtensorOneFP16__ ;
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
      case LWDA_R_16BF:
         return &lwtensorOneBF16__ ;
#endif
      case LWDA_R_32F:
         return &lwtensorOneFP32__ ;
      case LWDA_R_64F:
         return &lwtensorOneFP64__ ;
      case LWDA_C_32F:
         return &lwtensorOneC32__ ;
      case LWDA_C_64F:
         return &lwtensorOneC64__ ;
      default:
         {
            throw NotSupported("Datatype is not yet supported.\n");
         }
   }
}

const void* lwtensorGetZeroPtr(lwdaDataType_t type)
{
   switch(type)
   {
      case LWDA_R_8I:
         return &lwtensorZeroI8__ ;
      case LWDA_R_8U:
         return &lwtensorZeroU8__ ;
      case LWDA_R_16F:
         return &lwtensorZeroFP16__ ;
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
      case LWDA_R_16BF:
         return &lwtensorZeroBF16__ ;
#endif
      case LWDA_R_32F:
         return &lwtensorZeroFP32__ ;
      case LWDA_R_64F:
         return &lwtensorZeroFP64__ ;
      case LWDA_C_32F:
         return &lwtensorZeroC32__ ;
      case LWDA_C_64F:
         return &lwtensorZeroC64__ ;
      default:
      {
         throw NotSupported("Datatype is not yet supported.\n");
      }
   }
}

#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
lwblasComputeType_t getLwblasComputeType(lwtensorComputeType_t typeCompute)
{
    if( typeCompute == LWTENSOR_COMPUTE_16F ) {
        return LWBLAS_COMPUTE_32F_FAST_16F; // 32bit accumulate
    }
    else if( typeCompute == LWTENSOR_COMPUTE_32F ) {
        return LWBLAS_COMPUTE_32F;
    }
    else if( (typeCompute == LWTENSOR_COMPUTE_16BF) ) {
        return LWBLAS_COMPUTE_32F_FAST_16BF;
    }
    else if( typeCompute == LWTENSOR_COMPUTE_TF32 ) {
        return LWBLAS_COMPUTE_32F_FAST_TF32;
    }
    else if( typeCompute == LWTENSOR_COMPUTE_64F ) {
        return LWBLAS_COMPUTE_64F;
    }
    else {
        throw InternalError("Colwersion not possible.\n");
    }
}
#endif

lwdaDataType_t getScalarType(const lwdaDataType_t typeOutput, const lwtensorComputeType_t typeCompute)
{
    if( isComplex(typeOutput) )
    {
        if( typeCompute == LWTENSOR_COMPUTE_64F ||
                typeOutput == LWDA_C_64F )
        {
            return LWDA_C_64F;
        }else if( (typeCompute == LWTENSOR_COMPUTE_16F) ||
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
                (typeCompute == LWTENSOR_COMPUTE_16BF) ||
                (typeCompute == LWTENSOR_COMPUTE_TF32) ||
#endif
                (typeCompute == LWTENSOR_COMPUTE_32F)
                )
        {
            return LWDA_C_32F;
        }
    } else {
        if( typeCompute == LWTENSOR_COMPUTE_64F ||
                (typeOutput == LWDA_R_64F))
        {
            return LWDA_R_64F;
        }else if( (typeCompute == LWTENSOR_COMPUTE_16F) || 
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
                (typeCompute == LWTENSOR_COMPUTE_16BF) ||
                (typeCompute == LWTENSOR_COMPUTE_TF32) ||
#endif
                (typeCompute == LWTENSOR_COMPUTE_32F))
        {
            return LWDA_R_32F;
        }
    }
    throw InternalError("scalar type not defined.\n");
}

inline int numBitsMantissa(const lwdaDataType_t aclwmulatorType)
{
    if (aclwmulatorType == LWDA_R_16F)
    {
        return 10;
    }
    else if (aclwmulatorType == LWDA_R_16BF)
    {
        return 7;
    }
    else if ((aclwmulatorType == LWDA_R_TF32) || (aclwmulatorType == LWDA_C_TF32))
    {
        return 10;
    }
    else if (aclwmulatorType == LWDA_R_32F || (aclwmulatorType == LWDA_C_32F))
    {
        return 23;
    }
    else if (aclwmulatorType == LWDA_R_64F || (aclwmulatorType == LWDA_C_64F))
    {
        return 52;
    }
    throw InternalError("type not supported yet.\n");
}

inline int numBitsExponent(const lwdaDataType_t aclwmulatorType)
{
    if (aclwmulatorType == LWDA_R_16F)
    {
        return 5;
    }
    else if (aclwmulatorType == LWDA_R_16BF)
    {
        return 8;
    }
    else if ((aclwmulatorType == LWDA_R_TF32) || (aclwmulatorType == LWDA_C_TF32))
    {
        return 8;
    }
    else if ((aclwmulatorType == LWDA_R_32F) || (aclwmulatorType == LWDA_C_32F))
    {
        return 8;
    }
    else if ((aclwmulatorType == LWDA_R_64F) || (aclwmulatorType == LWDA_C_64F))
    {
        return 11;
    }
    throw InternalError("type not supported yet.\n");
}

/**
 * \pre types must be real-valued
 */
bool lwdaTypeAsAclwrateAs(const lwdaDataType_t typeA, const lwdaDataType_t typeB)
{
    return numBitsMantissa(typeA) >= numBitsMantissa(typeB) &&
           numBitsExponent(typeA) >= numBitsExponent(typeB);
}

lwdaDataType_t lwdaDataTypeToReal(const lwdaDataType_t type)
{
    if (type == LWDA_C_16F)
    {
        return LWDA_R_16F;
    }
    else if (type == LWDA_C_TF32)
    {
        return LWDA_R_TF32;
    }
    else if (type == LWDA_C_32F)
    {
        return LWDA_R_32F;
    }
    else if (type == LWDA_C_64F)
    {
        return LWDA_R_64F;
    }
    return type;
}

lwdaDataType_t computeTypeToLwda(const lwtensorComputeType_t typeCompute, const bool isComplexValued)
{
    if( isComplexValued )
    {
        if(typeCompute == LWTENSOR_COMPUTE_16F)
        {
            return LWDA_C_16F;
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
        }
        else if(typeCompute == LWTENSOR_COMPUTE_16BF)
        {
            return LWDA_C_16BF;
        }
        else if(typeCompute == LWTENSOR_COMPUTE_TF32)
        {
            return LWDA_C_TF32;
#endif
        }
        else if(typeCompute == LWTENSOR_COMPUTE_32F)
        {
            return LWDA_C_32F;
        }
        else if(typeCompute == LWTENSOR_COMPUTE_64F)
        {
            return LWDA_C_64F;
        }
    } else {
        if(typeCompute == LWTENSOR_COMPUTE_16F)
        {
            return LWDA_R_16F;
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
        }
        else if(typeCompute == LWTENSOR_COMPUTE_16BF)
        {
            return LWDA_R_16BF;
        }
        else if(typeCompute == LWTENSOR_COMPUTE_TF32)
        {
            return LWDA_R_TF32;
#endif
        }
        else if(typeCompute == LWTENSOR_COMPUTE_32F)
        {
            return LWDA_R_32F;
        }
        else if(typeCompute == LWTENSOR_COMPUTE_64F)
        {
            return LWDA_R_64F;
        }
    }
    throw InternalError("compute type not defined.\n");
}

lwtensorComputeType_t normalizeComputeType(const lwtensorComputeType_t typeCompute) noexcept
{
    switch(typeCompute)
    {
        case LWTENSOR_R_MIN_16BF:
            return LWTENSOR_COMPUTE_16BF;
        case LWTENSOR_R_MIN_TF32:
        case LWTENSOR_C_MIN_TF32:
            return LWTENSOR_COMPUTE_TF32;
        case LWTENSOR_R_MIN_16F:
        case LWTENSOR_C_MIN_16F:
            return LWTENSOR_COMPUTE_16F;
        case LWTENSOR_R_MIN_32F:
        case LWTENSOR_C_MIN_32F:
            return LWTENSOR_COMPUTE_32F;
        case LWTENSOR_R_MIN_64F:
        case LWTENSOR_C_MIN_64F:
            return LWTENSOR_COMPUTE_64F;
        case LWTENSOR_R_MIN_8U:
            return LWTENSOR_COMPUTE_8U;
        case LWTENSOR_R_MIN_32U:
            return LWTENSOR_COMPUTE_32U;
        case LWTENSOR_R_MIN_8I:
            return LWTENSOR_COMPUTE_8I;
        case LWTENSOR_R_MIN_32I:
            return LWTENSOR_COMPUTE_32I;
        default:
            return typeCompute;
    }
}

bool isValidComputeType(const lwtensorComputeType_t typeCompute) noexcept
{
    return ( (typeCompute == LWTENSOR_COMPUTE_16F) ||
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
        (typeCompute == LWTENSOR_COMPUTE_16BF) ||
        (typeCompute == LWTENSOR_COMPUTE_TF32) ||
#endif
        (typeCompute == LWTENSOR_COMPUTE_32F)  ||
        (typeCompute == LWTENSOR_COMPUTE_64F)  ||
        (typeCompute == LWTENSOR_COMPUTE_8U )  ||
        (typeCompute == LWTENSOR_COMPUTE_8I )  ||
        (typeCompute == LWTENSOR_COMPUTE_32U)  ||
        (typeCompute == LWTENSOR_COMPUTE_32I));
}

bool isValidLwdaDataType( const lwdaDataType_t computeType) noexcept
{
 return ( computeType == LWDA_R_16F  ||
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
          computeType == LWDA_R_16BF ||
#endif
          computeType == LWDA_C_16F  ||
          computeType == LWDA_R_32F  ||
          computeType == LWDA_C_32F  ||
          computeType == LWDA_R_64F  ||
          computeType == LWDA_C_64F  ||
          computeType == LWDA_R_8I   ||
          computeType == LWDA_C_8I   ||
          computeType == LWDA_R_8U   ||
          computeType == LWDA_C_8U   ||
          computeType == LWDA_R_32I  ||
          computeType == LWDA_C_32I  ||
          computeType == LWDA_R_32U  ||
          computeType == LWDA_C_32U  );
}

extent_type findGoodSplitK(const extent_type extent, const int numCTAs, const int numSMs, const extent_type vec)
{
    if(extent % vec != 0)
    {
        throw InternalError("Split not possible due to vectorization.\n");
    }

    const extent_type numDesiredCTAs = numSMs * 2;
    const extent_type desiredExtent = (numDesiredCTAs + numCTAs - 1) / numCTAs;

    // this algorithm could be improved by first finding all prime factors of
    // extentK that are smaller than 2*numSMs
    for(extent_type i = 0; i < desiredExtent; ++i)
    {
        // the extent --after the split-- must still be divisible by both newExtent and vec
        if (extent % ((desiredExtent - i) * vec) == 0 )
        {
            return desiredExtent - i;
        }
        else if (extent % ((desiredExtent + i) * vec) == 0 )
        {
            return desiredExtent + i;
        }
        
    }
    
    throw InternalError("Split.\n");
}

void splitAndInsert(const mode_type modeK,
                    const mode_type newMode,
                    const extent_type newExtent, 
                    int pos,
                    ModeList &modeOrderK,
                    ExtentMap &extent,
                    StrideMap &strideA,
                    StrideMap &strideB)
{
    assert( extent[modeK] % newExtent == 0 );

    extent[modeK] = extent[modeK] / newExtent;
    modeOrderK.insert(std::next(modeOrderK.begin(), pos), newMode);
    extent[newMode] = newExtent;
    strideA[newMode] = extent[modeK] * strideA.at(modeK);
    strideB[newMode] = extent[modeK] * strideB.at(modeK);
}

bool hasDuplicates(const mode_type *a, const uint32_t na,
   std::unordered_map<mode_type, uint32_t> &count)
{
   std::unordered_map<mode_type,bool> duplicates;
   for(uint32_t i=0; i < na; ++i)
   {
      if(duplicates.find(a[i]) != duplicates.end() )
      {
         fprintf(stderr, "LWTENSOR ERROR: the same mode %d may not be duplicated in the same tensor.\n",a[i]);
         return true;
      }
      if( count.find(a[i]) == count.end() )
      {
         count[a[i]] = 1U;
      }
      else
      {
         count[a[i]] += 1U;
      }
      duplicates[a[i]] = true;
   }
   return false;
}

lwtensorStatus_t validateModes(const mode_type *a, const uint32_t na,
                               const mode_type *b, const uint32_t nb,
                               const mode_type *c, const uint32_t nc,
                               const bool isReduction /*= false*/)
{
   std::unordered_map<mode_type, uint32_t> count;
   if( hasDuplicates(a, na, count) ){
      return handleError(LWTENSOR_STATUS_ILWALID_VALUE, "Mode A has duplicated values.");
   }
   if( hasDuplicates(b, nb, count) ){
      return handleError(LWTENSOR_STATUS_ILWALID_VALUE, "Mode B has duplicated values.");
   }
   if( hasDuplicates(c, nc, count) ){
      return handleError(LWTENSOR_STATUS_ILWALID_VALUE, "Mode C has duplicated values.");
   }
   if( !isReduction )
   {
       for(const auto it : count )
       {
           if(it.second == 1)
           {
               return handleError(LWTENSOR_STATUS_ILWALID_VALUE, "Mode " + std::to_string(it.first) + "only oclwres once.");;
           }
       }
   }
   return LWTENSOR_STATUS_SUCCESS;
}

void intersect(const mode_type *a, int na, const mode_type *b, int nb,
               ModeList &intersection)
{
   intersection.clear();
   for(int i=0; i < na; ++i)
      for(int j=0; j < nb; ++j)
         if( a[i] == b[j] )
         {
            intersection.emplace_back(a[i]);
            break;
         }
}
void intersect(const ModeList &a,
               const ModeList &b,
               ModeList &intersection)
{
   intersection.clear();
   for(auto ai : a )
      for(auto bi : b )
         if( ai == bi )
         {
            intersection.emplace_back(ai);
            break;
         }
}

mode_type getMaxMode(const ModeList &modes, const ExtentMap &extent)
{
   mode_type maxMode = LWTENSOR_ILWALID_MODE;
   extent_type maxExtent = 0;
   for( auto mode : modes )
      if( maxExtent < extent.at(mode) )
      {
         maxMode = mode;
         maxExtent = extent.at(mode);
      }
   return maxMode;
}

float getFlopMultiplier(const bool useComplex)
{
  if( useComplex )
     return 8.f;
  else
     return 2.f;
}

extent_type getTotalModeExtent(const ModeList &modes, const ExtentMap &extent)
{
   uint64_t k = 1;
   for(auto mode : modes)
      k *= (uint64_t) extent.at(mode);

   if( k > std::numeric_limits<extent_type>::max() )
       handleError(LWTENSOR_STATUS_NOT_SUPPORTED, "total extent exceeds int limit.");

   return (extent_type) k;
}

bool isComplex(lwdaDataType_t type)
{
   return type == LWDA_C_64F || type == LWDA_C_32F || type == LWDA_C_16F;
}

lwtensorStatus_t colwert(const void* src, lwdaDataType_t typeSrc,
                   void* dst, lwdaDataType_t typeDst)
{
   if(typeDst == LWDA_R_16F ){
      if(typeSrc == LWDA_R_16F ){
         *(half*)dst = lwGet<half>(*(half*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32U ){
         *(half*)dst = lwGet<half>(*(uint32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_64F ){
         *(half*)dst = lwGet<half>(*(double*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8I ){
         *(half*)dst = lwGet<half>(*(int8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8U ){
         *(half*)dst = lwGet<half>(*(uint8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32F ){
         *(half*)dst = lwGet<half>(*(float*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32I ){
         *(half*)dst = lwGet<half>(*(int32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   } else if(typeDst == LWDA_R_32U ){
      if(typeSrc == LWDA_R_16F ){
         *(uint32_t*)dst = lwGet<uint32_t>(*(half*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32U ){
         *(uint32_t*)dst = lwGet<uint32_t>(*(uint32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_64F ){
         *(uint32_t*)dst = lwGet<uint32_t>(*(double*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8I ){
         *(uint32_t*)dst = lwGet<uint32_t>(*(int8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8U ){
         *(uint32_t*)dst = lwGet<uint32_t>(*(uint8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32F ){
         *(uint32_t*)dst = lwGet<uint32_t>(*(float*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32I ){
         *(uint32_t*)dst = lwGet<uint32_t>(*(int32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   } else if(typeDst == LWDA_R_64F ){
      if(typeSrc == LWDA_R_16F ){
         *(double*)dst = lwGet<double>(*(half*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32U ){
         *(double*)dst = lwGet<double>(*(uint32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_64F ){
         *(double*)dst = lwGet<double>(*(double*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8I ){
         *(double*)dst = lwGet<double>(*(int8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8U ){
         *(double*)dst = lwGet<double>(*(uint8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32F ){
         *(double*)dst = lwGet<double>(*(float*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32I ){
         *(double*)dst = lwGet<double>(*(int32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   } else if(typeDst == LWDA_R_8I ){
      if(typeSrc == LWDA_R_16F ){
         *(int8_t*)dst = lwGet<int8_t>(*(half*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32U ){
         *(int8_t*)dst = lwGet<int8_t>(*(uint32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_64F ){
         *(int8_t*)dst = lwGet<int8_t>(*(double*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8I ){
         *(int8_t*)dst = lwGet<int8_t>(*(int8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8U ){
         *(int8_t*)dst = lwGet<int8_t>(*(uint8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32F ){
         *(int8_t*)dst = lwGet<int8_t>(*(float*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32I ){
         *(int8_t*)dst = lwGet<int8_t>(*(int32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   } else if(typeDst == LWDA_R_8U ){
      if(typeSrc == LWDA_R_16F ){
         *(uint8_t*)dst = lwGet<uint8_t>(*(half*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32U ){
         *(uint8_t*)dst = lwGet<uint8_t>(*(uint32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_64F ){
         *(uint8_t*)dst = lwGet<uint8_t>(*(double*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8I ){
         *(uint8_t*)dst = lwGet<uint8_t>(*(int8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8U ){
         *(uint8_t*)dst = lwGet<uint8_t>(*(uint8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32F ){
         *(uint8_t*)dst = lwGet<uint8_t>(*(float*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32I ){
         *(uint8_t*)dst = lwGet<uint8_t>(*(int32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   } else if(typeDst == LWDA_R_32F ){
      if(typeSrc == LWDA_R_16F ){
         *(float*)dst = lwGet<float>(*(half*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32U ){
         *(float*)dst = lwGet<float>(*(uint32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_64F ){
         *(float*)dst = lwGet<float>(*(double*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8I ){
         *(float*)dst = lwGet<float>(*(int8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8U ){
         *(float*)dst = lwGet<float>(*(uint8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32F ){
         *(float*)dst = lwGet<float>(*(float*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32I ){
         *(float*)dst = lwGet<float>(*(int32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   } else if(typeDst == LWDA_R_32I ){
      if(typeSrc == LWDA_R_16F ){
         *(int32_t*)dst = lwGet<int32_t>(*(half*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32U ){
         *(int32_t*)dst = lwGet<int32_t>(*(uint32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_64F ){
         *(int32_t*)dst = lwGet<int32_t>(*(double*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8I ){
         *(int32_t*)dst = lwGet<int32_t>(*(int8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_8U ){
         *(int32_t*)dst = lwGet<int32_t>(*(uint8_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32F ){
         *(int32_t*)dst = lwGet<int32_t>(*(float*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_R_32I ){
         *(int32_t*)dst = lwGet<int32_t>(*(int32_t*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   }
   if(typeDst == LWDA_C_64F ){
      if(typeSrc == LWDA_C_64F ){
         *(lwDoubleComplex*)dst = lwGet<lwDoubleComplex>(*(lwDoubleComplex*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_C_32F ){
         *(lwDoubleComplex*)dst = lwGet<lwDoubleComplex>(*(lwComplex*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   } else if(typeDst == LWDA_C_32F ){
      if(typeSrc == LWDA_C_64F ){
         *(lwComplex*)dst = lwGet<lwComplex>(*(lwDoubleComplex*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
      else if(typeSrc == LWDA_C_32F ){
         *(lwComplex*)dst = lwGet<lwComplex>(*(lwComplex*)src);
         return LWTENSOR_STATUS_SUCCESS;
      }
   }
   RETURN_STATUS(LWTENSOR_STATUS_INTERNAL_ERROR)
}

//this is a very crude approximation
float getPeakGFlops(lwtensorComputeType_t typeCompute)
{
    // TODO this function is super lwrde and should depent on the deviceProp (of the ctx)
   float baseGFlops = 14000;

   if(typeCompute == LWTENSOR_COMPUTE_16F )
   {
      return baseGFlops * 4;
   }
   else if(typeCompute == LWTENSOR_COMPUTE_64F )
   {
      return baseGFlops / 2; // TODO
   }
   else if(typeCompute == LWTENSOR_COMPUTE_8I )
   {
      return baseGFlops * 4;
   }
   else if(typeCompute == LWTENSOR_COMPUTE_8U )
   {
      return baseGFlops * 4;
   }
   return baseGFlops;
}
 
lwtensorStatus_t isHostPtr(const void *ptr, bool *isHost)
{
    lwdaPointerAttributes attr;
    auto err = lwdaPointerGetAttributes( &attr, ptr);
    if( err == lwdaSuccess )
    {
        // we assume that UVM memory resides on host
#if LWDART_VERSION >= 10000
        if ( (attr.type == lwdaMemoryTypeHost) || (attr.type == lwdaMemoryTypeManaged) ) //TODO we assume that managed memory resides on the CPU
#else
        if ( attr.memoryType == lwdaMemoryTypeHost )
#endif
        {
            *isHost = true;
        }
        else
        {
            *isHost = false;
        }
        return LWTENSOR_STATUS_SUCCESS;
    }
    else
    {
        printf("ERR: %s\n", lwdaGetErrorString(err));
        RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE)
    }
}

int getVectorization(const lwdaDataType_t floatType)
{
    if( floatType == LWDA_R_8I || floatType == LWDA_R_8U ) 
    {
        return 16;
    }
    else if( floatType == LWDA_R_16F
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
            || floatType == LWDA_R_16BF 
#endif
           )
    {
        return 8;
    }
    else if( floatType == LWDA_R_32I || floatType == LWDA_R_32U ) 
    {
        return 4;
    }
    else if( floatType == LWDA_R_32F || floatType == LWDA_C_16F ) 
    {
        return 4;
    }
    else if( floatType == LWDA_R_64F || floatType == LWDA_C_32F  ) 
    {
        return 2;
    }
    else if( floatType == LWDA_C_64F ) 
    {
        return 1;
    }
    else 
    {
        throw NotSupported("Datatype is not yet supported.\n");
    }
} 

    std::string reproduceCommand(
        const TensorDescriptor *descA, const int* modeA, const uint32_t alignmentA,
        const TensorDescriptor *descB, const int* modeB, const uint32_t alignmentB,
        const TensorDescriptor *descC, const int* modeC, const uint32_t alignmentC,
        int typeCompute, lwtensorAlgo_t algo, const uint64_t workspaceSize, const int32_t partitionsK, lwtensorRoutine_t routine)
    {
        /* Using an unordered map to store the extents. */
        ExtentMap extents;
        /* The output descriptor. */
        std::string descr = "";

        if( descC && modeC )
        {
            auto numModes = descC->getNumModes();
            descr += std::string(" -Pc") + lwdaDataTypeToString(descC->getDataType());
            std::string stride = " -strideC";
            for ( uint32_t i = 0; i < numModes; i ++ )
            {
                auto mode = modeC[i];
                auto search = extents.find(mode);
                if( search != extents.end() && search->second != descC->getExtent(i) )
                    throw IlwalidArgument("Extents do not match.\n");
                extents[mode] = descC->getExtent(i);
                stride += std::to_string(descC->getStride(i)) + std::string(",");
            }
            descr += stride;
            if( routine == LWTENSOR_ROUTINE_EW )
            {
                descr += std::string(" -gamma") + std::to_string(0.7);
            }
            descr += std::string(" -opC") + std::to_string(descC->getOp());
            if (routine != LWTENSOR_ROUTINE_REDUCTION)
                descr += std::string( " -alignmentC" ) + std::to_string(alignmentC);
            if( descC->isVectorized() )
            {
                descr += std::string( " -vectorModeC" )   + std::to_string( descC->getVectorModeIndex());
                descr += std::string( " -vectorWidthC" )  + std::to_string( descC->getVectorWidth());
                descr += std::string( " -vectorOffsetC" ) + std::to_string( descC->getVectorOffset() );
                descr += std::string( " -paddingC" )      + std::to_string( descC->getZeroPadding() );
            }
        }

        if( descA && modeA )
        {
            descr += std::string(" -Pa") + lwdaDataTypeToString(descA->getDataType());
            auto numModes = descA->getNumModes();
            std::string stride = " -strideA";
            for ( uint32_t i = 0; i < numModes; i ++ )
            {
                auto mode = modeA[i];
                auto search = extents.find(mode);
                if( search != extents.end() && search->second != descA->getExtent(i) )
                    throw IlwalidArgument("Extents do not match.\n");
                extents[mode] = descA->getExtent(i);
                stride += std::to_string(descA->getStride(i)) + std::string(",");
            }
            descr += stride;
            descr += std::string(" -opA") + std::to_string(descA->getOp());
            if (routine != LWTENSOR_ROUTINE_REDUCTION)
                descr += std::string( " -alignmentA" ) + std::to_string(alignmentA);
            if( descA->isVectorized() )
            {
                descr += std::string(" -vectorModeA") + std::to_string(descA->getVectorModeIndex());
                descr += std::string(" -vectorWidthA") + std::to_string(descA->getVectorWidth());
                descr += std::string(" -vectorOffsetA") + std::to_string(descA->getVectorOffset() );
                descr += std::string(" -paddingA") + std::to_string(descA->getZeroPadding() );
            }
        }
        if( descB && modeB )
        {
            descr += std::string(" -Pb") + lwdaDataTypeToString(descB->getDataType());
            auto numModes = descB->getNumModes();
            std::string stride = " -strideB";
            for ( uint32_t i = 0; i < numModes; i ++ )
            {
                auto mode = modeB[i];
                auto search = extents.find(mode);
                if( search != extents.end() && search->second != descB->getExtent(i) )
                    throw IlwalidArgument("Extents do not match.\n");
                extents[mode] = descB->getExtent(i);
                stride += std::to_string(descB->getStride(i)) + std::string(",");
            }
            descr += stride;
            descr += std::string(" -opB") + std::to_string(descB->getOp());
            if (routine != LWTENSOR_ROUTINE_REDUCTION)
                descr += std::string( " -alignmentB" ) + std::to_string(alignmentB);
            if( descA->isVectorized() )
            {
                descr += std::string(" -vectorModeB") + std::to_string(descB->getVectorModeIndex());
                descr += std::string(" -vectorWidthB") + std::to_string(descB->getVectorWidth());
                descr += std::string(" -vectorOffsetB") + std::to_string(descB->getVectorOffset() );
                descr += std::string(" -paddingB") + std::to_string(descB->getZeroPadding() );
            }
        }

        descr += std::string(" -algo") + std::to_string(static_cast<int32_t>(algo));

        // print modes last
        if( descA && modeA )
        {
            auto numModes = descA->getNumModes();
            descr += std::string( " -modeA" );
            for ( uint32_t i = 0; i < numModes; i ++ )
            {
                auto mode = modeA[i];
                descr += isprint(mode) ? std::string( 1, mode ) : std::to_string( mode );
                if ( i != numModes - 1 ) descr += std::string( "," );
            }
        }

        if( descB && modeB )
        {
            auto numModes = descB->getNumModes();
            descr += std::string( " -modeB" );
            for ( uint32_t i = 0; i < numModes; i ++ )
            {
                auto mode = modeB[i];
                descr += isprint(mode) ? std::string( 1, mode ) : std::to_string( mode );
                if ( i != numModes - 1 ) descr += std::string( "," );
            }
        }

        if( descC && modeC )
        {
            auto numModes = descC->getNumModes();
            descr += std::string( " -modeC" );
            for ( uint32_t i = 0; i < numModes; i ++ )
            {
                auto mode = modeC[i];
                descr += isprint(mode) ? std::string( 1, mode ) : std::to_string( mode );
                if ( i != numModes - 1 ) descr += std::string( "," );
            }
        }

        descr += std::string(" -extent");
        for ( auto i : extents )
        {
            descr += isprint(i.first) ? std::string( 1, i.first ) : std::to_string( i.first );
            descr += std::string( "=" );
            descr += std::to_string( i.second ) + std::string( "," );
        }
        descr += std::string(" -Pcomp");
        switch(typeCompute)
        {
            case LWTENSOR_COMPUTE_16F:
                descr += 'h';
                break;
            case LWTENSOR_COMPUTE_32F:
                descr += 's';
                break;
            case LWTENSOR_COMPUTE_64F:
                descr += 'd';
                break;
            default:
                descr += std::to_string(typeCompute);
        }
        descr += std::string(" -workspace");
        descr += std::to_string(workspaceSize);
        if (routine == LWTENSOR_ROUTINE_TC)
            descr += std::string(" -Rcontraction");
        else if (routine == LWTENSOR_ROUTINE_REDUCTION)
            descr += std::string(" -Rreduction");
        else if (routine == LWTENSOR_ROUTINE_EW)
            descr += std::string(" -Relementwise");

        if (partitionsK != -1)
        {
            descr += std::string(" -partitionsK") + std::to_string(partitionsK);
        }
        descr += std::string("\n");

        return descr;
    }

lwtensorStatus_t initModeOrderContraction(const ModeList& modeA,
                               const ModeList& modeB,
                               const ModeList& modeC,
                               const ExtentMap& extent,
                               ModeList& modeM,
                               ModeList& modeN,
                               ModeList& modeK,
                               ModeList& modeL,
                               bool& stridedLoadsA, bool& stridedLoadsB, bool &contiguousModeIsBatchedA, bool &contiguousModeIsBatchedB)
{
    intersect(modeA, modeC, modeM);
    intersect(modeB, modeC, modeN);

    const bool isGEMVlike = getTotalModeExtent(modeN, extent) <= 16;
    if (isGEMVlike) // GEMV-like TC
    {
        intersect(modeA, modeB, modeK); // favor reads of A over B (since A is larger)
    }
    else
    {
        intersect(modeB, modeA, modeK);
    }
    intersect(modeM, modeN, modeL);
    assert(modeM.size() > 0);
    assert(modeN.size() > 0);

    stridedLoadsA = false;
    stridedLoadsB = false;
    contiguousModeIsBatchedA = false;
    contiguousModeIsBatchedB = false;

    bool strideOneIsBatchedA = false;
    bool strideOneIsBatchedB = false;
    bool strideOneIsBatchedC = false;
    for (auto l : modeL)
    {
        if( modeC.front() == l)
        {
            strideOneIsBatchedC = true;
        }
        if (modeA.front() == l )
        {
            strideOneIsBatchedA = true;
            contiguousModeIsBatchedA = true;
            stridedLoadsA = true;
        }
        if (modeB.front() == l )
        {
            strideOneIsBatchedB = true;
            contiguousModeIsBatchedB = true;
            stridedLoadsB = true;
        }

        modeM.remove(l);
        modeN.remove(l);
        modeK.remove(l);
    }

    const bool transA = !strideOneIsBatchedA && (std::find(modeK.begin(), modeK.end(), modeA.front()) != modeK.end());
    const bool transB = !strideOneIsBatchedB && (std::find(modeK.begin(), modeK.end(), modeB.front()) == modeK.end());

    if (isGEMVlike && !transB && !modeK.empty() && modeA.front() != modeK.front() && !strideOneIsBatchedA && !strideOneIsBatchedB)
    {
        // move modeB[0] to front of modeK
        modeK.remove(modeB.front());
        modeK.push_front(modeB.front());
    }
    const bool strideOneCisMmode =  !strideOneIsBatchedC && (std::find(modeM.begin(), modeM.end(), modeC.front()) != modeM.end());
    auto *myModes = strideOneCisMmode ? &modeM : &modeN;

    if (myModes->front() != modeA.front() && myModes->front() != modeC.front())
    {
        // move stride-1 of C to the front of the M set
        if (strideOneCisMmode)
        {
            myModes->remove(modeC.front());
            myModes->push_front(modeC.front());
        }
    }

    if (!strideOneIsBatchedC && modeC.front() != myModes->front()) // multi-dim blocking
    {
        myModes->remove(modeC.front());
        auto secondPos = std::next(myModes->begin());
        myModes->insert( secondPos,
                modeC.front()); // make sure that C's stride-1 mode is the second mode of the M set: favor reads over writes
    }

    /* **************************************
     *   Reorder modes (both for perf & correctness)
     * **************************************/

    /** If transA, then the leading mode of A is from modeK. */
    if (transA)
    {
        /** If !transB, then the leading mode of B is from mode K. Given that transA and
         * !transB, if modeB.front() != modeA.front(), then we need to use strided load on either tensor.  */
        if (!transB && modeB.front() != modeA.front())
        {
            // this would require blocking along two k-modes (which are really small within the GEMM-like kernel)
            // this case should really be avoided at the application level.
            // TODO: this should be fixed according the total extents of M and N.
            if (isGEMVlike)
            {
                stridedLoadsB = true; // in this cases modeK will be modeK will be sorted w.r.t. A
            }
            else if (extent.at(modeA.front()) > extent.at(modeB.front()))
            {
                stridedLoadsB = true;
                auto it = std::find(modeK.begin(), modeK.end(),
                                    modeA.front()); // keep largest contracted index in first position
                if (!modeK.empty() && modeK.front() != modeA.front())
                {
                    std::swap(*it, *modeK.begin());
                }
            }
            else
            {
                stridedLoadsA = true;
                modeK.remove(modeA.front());
                auto it = modeK.begin();
                if (it != modeK.end())
                {
                    it = std::next(it);
                }
                modeK.insert(it, modeA.front()); // for perf reasons
            }
        }
        else
        {
            auto it = std::find(modeK.begin(), modeK.end(), modeA.front());
            if (!modeK.empty() && modeK.front() != modeA.front())
            {
                std::swap(*it, *modeK.begin());
            }
        }
    }
    else if (transB && modeK.size() > 1)
    {
        // we may choose to reorder the k-modes to increase performance (since no k-mode is also stride-1 mode)
        auto largest = modeK.begin();
        extent_type largestExtent = extent.at(*largest);
        for (auto it = std::next(modeK.begin()); it != modeK.end(); it++)
        {
            if (extent.at(*it) > largestExtent)
            {
                largestExtent = extent.at(*it);
                largest = it;
            }
        }
        if (largestExtent > 0) std::swap(*largest, *modeK.begin());
    }

    auto maximizeBlockedExtent = [modeA, modeB, modeC, extent] (ModeList &freeModes)
    {
        constexpr int kNumBlockedModes = 2;
        if (freeModes.size() <= kNumBlockedModes )
        {
            return;
        }

        auto *data = freeModes.data();
        const int numModes = freeModes.size();

        // loop over blocked (free) modes
        for(int i=0; i < kNumBlockedModes; ++i)
        {
            auto mode = data[i];
            if (mode != modeA.front() &&
                mode != modeB.front() &&
                mode != modeC.front())
            {
                auto maxExtent = 0;
                auto maxJ = -1;
                // we have some choice: find a larger mode
                for(int j=kNumBlockedModes; j < numModes; ++j)
                {
                    if (extent.at(data[j]) > maxExtent)
                    {
                        maxExtent = extent.at(data[j]);
                        maxJ = j;
                    }
                }
                if (maxExtent > extent.at(mode))
                {
                    // swap modes
                    auto tmp = data[i];
                    data[i] = data[maxJ];
                    data[maxJ] = tmp;
                }
            }
        }
    };

    /*
     * Try to incease the stride of the blocked modes as much as possible to reduce the
     * remainer
     */
    maximizeBlockedExtent(modeM);
    maximizeBlockedExtent(modeN);
    return LWTENSOR_STATUS_SUCCESS;
}

}













