#pragma once

#include <stdio.h>
#include <lwtensor/types.h>

size_t getDataTypeSize( const lwdaDataType_t type )
{
    if ( (type == LWDA_R_8I) || (type == LWDA_R_8U) )
    {
        return 1U;
    }
    else if( type == LWDA_R_16F )
    {
        return 2U;
    }
    else if( (type == LWDA_R_32I) || (type == LWDA_R_32U) )
    {
        return 4U;
    }
    else if( (type == LWDA_R_32F) || (type == LWDA_C_16F) )
    {
        return 4U;
    }
    else if( (type == LWDA_R_64F) || (type == LWDA_C_32F)  )
    {
        return 8U;
    }
    else if( type == LWDA_C_64F )
    {
        return 16U;
    }
    else if( type == LWDA_R_16BF )
    {
        return 2U;
    }
    else
    {
        printf("Error: Datatype is not yet supported.\n" );
        exit(-1);
    }
}

bool isComplex(const lwdaDataType_t typeOutput)
{
    return (typeOutput == LWDA_C_16F) || (typeOutput == LWDA_C_32F) || (typeOutput == LWDA_C_64F);
}

lwdaDataType_t getScalarType(const lwdaDataType_t typeOutput, const lwtensorComputeType_t typeCompute)
{
    if( isComplex(typeOutput) )
    {
        if( typeCompute == LWTENSOR_COMPUTE_64F ||
                typeOutput == LWDA_C_64F )
        {
            return LWDA_C_64F;
        }else if( (typeCompute == LWTENSOR_COMPUTE_16F) ||
                (typeCompute == LWTENSOR_COMPUTE_TF32) ||
                (typeCompute == LWTENSOR_COMPUTE_32F))
        {
            return LWDA_C_32F;
        }
    } else {
        if( typeCompute == LWTENSOR_COMPUTE_64F ||
                (typeOutput == LWDA_R_64F))
        {
            return LWDA_R_64F;
        }else if( (typeCompute == LWTENSOR_COMPUTE_16F) || 
                (typeCompute == LWTENSOR_COMPUTE_16BF) ||
                (typeCompute == LWTENSOR_COMPUTE_TF32) ||
                (typeCompute == LWTENSOR_COMPUTE_32F))
        {
            return LWDA_R_32F;
        }
    }
    printf("Error: scalar type not defined.\n" );
    exit(-1);
}

lwdaDataType_t computeTypeToLwda(const lwtensorComputeType_t typeCompute, const bool isComplexValued)
{
    if( isComplexValued )
    {
        if(typeCompute == LWTENSOR_COMPUTE_16F)
        {
            return LWDA_C_16F;
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
        }else if(typeCompute == LWTENSOR_COMPUTE_16BF)
        {
            return LWDA_C_16BF;
        }else if(typeCompute == LWTENSOR_COMPUTE_TF32)
        {
            return LWDA_C_TF32;
#endif
        }else if(typeCompute == LWTENSOR_COMPUTE_32F)
        {
            return LWDA_C_32F;
        }else if(typeCompute == LWTENSOR_COMPUTE_64F)
        {
            return LWDA_C_64F;
        }
    } else {

        if(typeCompute == LWTENSOR_COMPUTE_16F)
        {
            return LWDA_R_16F;
#if LWTENSOR_LWDA_VERSION_MAJOR >= 11
        }else if(typeCompute == LWTENSOR_COMPUTE_16BF)
        {
            return LWDA_R_16BF;
        }else if(typeCompute == LWTENSOR_COMPUTE_TF32)
        {
            return LWDA_R_TF32;
#endif
        }else if(typeCompute == LWTENSOR_COMPUTE_32F)
        {
            return LWDA_R_32F;
        }else if(typeCompute == LWTENSOR_COMPUTE_64F)
        {
            return LWDA_R_64F;
        }
    }
    printf("Error: compute type not defined.\n" );
    exit(-1);
}
