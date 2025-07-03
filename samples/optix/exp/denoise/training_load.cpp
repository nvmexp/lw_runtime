/***************************************************************************************************
 * Copyright 2020 LWPU Corporation. All rights reserved.
 **************************************************************************************************/

#include <vector>
#include <memory>
#include <stdio.h>
#include <string.h>
#include <include/optix_types.h>
#include <corelib/misc/String.h>
#include <exp/context/ErrorHandling.h>
#include <prodlib/system/Logger.h>
#include <Util/ProgressiveHash.h>

#include <embed_weights.h>

#if LW_OPTIX_EMBED_DENOISER_WEIGHTS
#ifdef __aarch64__
#include <denoise/models/optixDenoiserWeights_noUpscale2x_bin.h>
#else
#include <denoise/models/optixDenoiserWeights_bin.h>
#endif
#endif

namespace optix_exp {

struct DenoiserWeightsHeader
{
    unsigned int magicNumber = 0x07da7a;
    unsigned int fileVersion = 10001; // major * 10000 + minor * 100 + micro
    unsigned int o7version   = 70400; // taken from optix.h
    unsigned int reserved    = 0;
    struct {
        size_t KPN5_STATIC_BIN_OFFSET;
        size_t KPN5_STATIC_BIN_SIZE;
        size_t KPN5_TEMPORAL_BIN_OFFSET;
        size_t KPN5_TEMPORAL_BIN_SIZE;
        size_t KPN5_UPSCALE2X_BIN_OFFSET;
        size_t KPN5_UPSCALE2X_BIN_SIZE;
        size_t DIRECTPREDICTION_STATIC_BIN_OFFSET;
        size_t DIRECTPREDICTION_STATIC_BIN_SIZE;
        size_t DIRECTPREDICTION_TEMPORAL_BIN_OFFSET;
        size_t DIRECTPREDICTION_TEMPORAL_BIN_SIZE;
    } v_10000;
};

inline unsigned int WeightsMajor( unsigned int v ) { return ( v / 10000 ) * 10000; }

static const char* const ErrorMessage = "Unable to load denoiser weights";

static OptixResult getOffsetSizeAndXXH3( const DenoiserWeightsHeader& header, OptixDenoiserModelKind modelKind, size_t& offset, size_t& size, size_t& xxh3, ErrorDetails& errDetails )
{
    switch( modelKind )
    {
        case OPTIX_DENOISER_MODEL_KIND_AOV:
            offset = header.v_10000.KPN5_STATIC_BIN_OFFSET;
            size = header.v_10000.KPN5_STATIC_BIN_SIZE;
            xxh3 = 15725033021714209819u;
            break;
        case OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV:
            offset = header.v_10000.KPN5_TEMPORAL_BIN_OFFSET;
            size = header.v_10000.KPN5_TEMPORAL_BIN_SIZE;
            xxh3 = 327616210813003748u;
            break;
        case OPTIX_DENOISER_MODEL_KIND_UPSCALE2X:
            offset = header.v_10000.KPN5_UPSCALE2X_BIN_OFFSET;
            size = header.v_10000.KPN5_UPSCALE2X_BIN_SIZE;
            xxh3 = 16754490582415735455u;
            break;
        case OPTIX_DENOISER_MODEL_KIND_LDR:
        case OPTIX_DENOISER_MODEL_KIND_HDR:
            offset = header.v_10000.DIRECTPREDICTION_STATIC_BIN_OFFSET;
            size = header.v_10000.DIRECTPREDICTION_STATIC_BIN_SIZE;
            xxh3 = 17553692865896955132u;
            break;
        case OPTIX_DENOISER_MODEL_KIND_TEMPORAL:
            offset = header.v_10000.DIRECTPREDICTION_TEMPORAL_BIN_OFFSET;
            size = header.v_10000.DIRECTPREDICTION_TEMPORAL_BIN_SIZE;
            xxh3 = 6379114914766775485u;
            break;
        default:
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                              corelib::stringf( "Unknown model kind: 0x%x", modelKind ) );
            break;
    }
    if( size == 0 )
    {
        lerr << "Weights not defined for model kind: " << modelKind;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }

    return OPTIX_SUCCESS;
}

OptixResult denoiseGetBuiltinTrainingSet( const std::string& libPath, OptixDenoiserModelKind modelKind, std::vector<char>& data, ErrorDetails& errDetails )
{
#if !LW_OPTIX_EMBED_DENOISER_WEIGHTS
    // Load optixDenoiserWeights.bin file from the same directory where the optix shared library is installed
    std::string wname = libPath.substr( 0, libPath.find_last_of("/\\") + 1 ) + DRIVER_BASE_NAME + ".bin";

    FILE* fp = fopen( wname.c_str(), "rb" );
    if( !fp )
    {
        lwarn << "Could not open optix denoiser weights file \"" << wname << "\"\nTrying another filename.\n";
        wname = libPath.substr( 0, libPath.find_last_of("/\\") + 1 ) + "optixDenoiserWeights.bin";

        fp = fopen( wname.c_str(), "rb" );
        if( !fp )
        {
            lerr << "Could not open optix denoiser weights file " << wname;
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
        }
    }

    std::shared_ptr<FILE> fptr( fp, fclose );
    DenoiserWeightsHeader header;

    if( fread( &header, 1, sizeof(DenoiserWeightsHeader), fp ) != sizeof(DenoiserWeightsHeader) )
    {
        lerr << "Could not read header of denoiser weights file " << wname;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }

    if( header.magicNumber != DenoiserWeightsHeader().magicNumber )
    {
        lerr << "Header magic number " << header.magicNumber << " wrong, expected " << DenoiserWeightsHeader().magicNumber;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }

    if( WeightsMajor( header.fileVersion ) != WeightsMajor( DenoiserWeightsHeader().fileVersion ) )
    {
        lerr << "Header file version " << header.fileVersion << " wrong, expected " << DenoiserWeightsHeader().fileVersion;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }

    size_t offset, size, xxh3;
    if( OptixResult res = getOffsetSizeAndXXH3( header, modelKind, offset, size, xxh3, errDetails ) )
        return res;

    if( fseek( fp, offset, SEEK_SET ) == -1 )
    {
        lerr << "Cannot set file pointer to weight offset in denoiser weights file " << wname;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }

    data.resize( size );
    if( fread( &data[0], 1, size, fp ) != size )
    {
        lerr << "Could not read data of denoiser weights file " << wname;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }

    optix::ProgressiveHash ph;
    ph.update( (const void*)&data[0], size );
    if( ph.digest() != xxh3 )
    {
        lerr << "Checksum failed for denoiser weights, model kind " << modelKind;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }
    return OPTIX_SUCCESS;
#else
#if defined __aarch64__
    const char* trainingData       = optix::data::getoptixDenoiserWeights_noUpscale2xData();
    size_t      trainingDataLength = optix::data::getoptixDenoiserWeights_noUpscale2xDataLength();
#else
    const char* trainingData       = optix::data::getoptixDenoiserWeightsData();
    size_t      trainingDataLength = optix::data::getoptixDenoiserWeightsDataLength();
#endif

    const DenoiserWeightsHeader* header = ( const DenoiserWeightsHeader* )trainingData;

    if( header->magicNumber != DenoiserWeightsHeader().magicNumber )
    {
        lerr << "Header magic number " << header->magicNumber << " wrong, expected " << DenoiserWeightsHeader().magicNumber;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }

    if( WeightsMajor( header->fileVersion ) != WeightsMajor( DenoiserWeightsHeader().fileVersion ) )
    {
        lerr << "Header file version " << header->fileVersion << " wrong, expected " << DenoiserWeightsHeader().fileVersion;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }

    size_t offset, size, xxh3;
    if( OptixResult res = getOffsetSizeAndXXH3( *header, modelKind, offset, size, xxh3, errDetails ) )
        return res;

    data.resize( size );
    memcpy( &data[0], trainingData + offset, size );

    optix::ProgressiveHash ph;
    ph.update( (const void*)&data[0], size );
    if( ph.digest() != xxh3 )
    {
        lerr << "Checksum failed for denoiser weights, model kind " << modelKind;
        return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, ErrorMessage );
    }
    return OPTIX_SUCCESS;
#endif
}

};  // namespace optix_exp
