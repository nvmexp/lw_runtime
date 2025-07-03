//
// Copyright 2020 LWPU Corporation. All rights reserved.
//

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include "fp16_emu.h"
#include "layerdata_load.h"
#ifndef IRAY_BUILD
#include <corelib/misc/String.h>
#endif

#if defined( _WIN32 ) || defined( _WIN64 )
#define strtok_r strtok_s
#endif

namespace optix_exp {

const unsigned int OPTION_BIT_FOLDING = 1;
const unsigned int OPTION_BIT_FP16    = 2;
const unsigned int OPTION_BIT_SQRT    = 4;
const unsigned int OPTION_BIT_PQ      = 8;
const unsigned int OPTION_BIT_KPN     = 32;
const unsigned int OPTION_BIT_UPSCALE = 128;
const unsigned int OPTION_BIT_KPNCLIP = 256;

static inline void colwert_float_to_fp16( __half* result, const float* in, int sz )
{
    for( int i    = 0; i < sz; i++ )
        result[i] = cpu_float2half_rn( in[i] );
}

bool Layerdata::isUpscale() const
{
    return bool( ( m_options & OPTION_BIT_UPSCALE ) != 0 );
}

bool Layerdata::hKPNClipWeights() const
{
    return bool( ( m_options & OPTION_BIT_KPNCLIP ) != 0 );
}

bool Layerdata::needsS2D() const
{
    if( ( m_options & OPTION_BIT_FOLDING ) == 0 )
        return false;
    else if( isUpscale() )
        return false;
    return true;
}

bool Layerdata::needsD2S() const
{
    if( ( m_options & OPTION_BIT_FOLDING ) == 0 )
        return false;
    return true;
}

bool Layerdata::hKPN() const
{
    return bool( ( m_options & OPTION_BIT_KPN ) != 0 );
}

int Layerdata::alphaImportOp() const
{
    if( m_importOperations.size() == 0 )
        return IMPORT_A;
    else if( m_importOperations[0] == IMPORT_LOG )
        return IMPORT_A_LOG;
    else if( m_importOperations[0] == IMPORT_LOG_SQRT )
        return IMPORT_A_LOG_SQRT;
    else if( m_importOperations[0] == IMPORT_PRELOG )
        return IMPORT_A_PRELOG;
    return IMPORT_A;
}

int Layerdata::exportOp() const
{
    return hKPN() ? IMPORT_HDR : m_importOperations[0];
}

OptixResult Layerdata::setImportOperations( const char* const trainingId, ErrorDetails& errDetails )
{
    if( !trainingId || strlen( trainingId ) >= 1024 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( "invalid training set id %s", trainingId ) );

    m_importOperations.clear();

    m_temporalLayerIndex = -1;

    char tstring[1024];
    strcpy( tstring, trainingId );
    char* saveptr;
    char* ptr = strtok_r( tstring, "-", &saveptr );
    while( ptr )
    {
        if( !strcmp( ptr, "rgb" ) || !strcmp( ptr, "flow" ) )
        {
            if( !strcmp( ptr, "flow" ) )
                m_temporalLayerIndex = m_importOperations.size();

            if( m_isHdr )
            {
                if( m_options & OPTION_BIT_KPN )
                    m_importOperations.push_back( IMPORT_PRELOG );
                else if( m_options & OPTION_BIT_SQRT )
                    m_importOperations.push_back( IMPORT_LOG_SQRT );
                else if( m_options & OPTION_BIT_PQ )
                    m_importOperations.push_back( IMPORT_PQ );
                else
                    m_importOperations.push_back( IMPORT_LOG );
            }
            else
            {
                m_importOperations.push_back( IMPORT_RGB );
            }
        }
        else if( !strcmp( ptr, "albedo" ) )
            m_importOperations.push_back( IMPORT_RGB );
        else if( !strcmp( ptr, "normal" ) )
            m_importOperations.push_back( IMPORT_XY );
        else if( !strcmp( ptr, "normal3" ) )
            m_importOperations.push_back( IMPORT_NORM3 );
        else
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          corelib::stringf( "training set id contains unrecognized feature name (%s)", ptr ) );
        ptr = strtok_r( nullptr, "-", &saveptr );
    }
    return OPTIX_SUCCESS;
}

///// file

OptixResult Layerdata::load( const char* const filename, const char* const trainingId, ErrorDetails& errDetails )
{
    FILE* fp = fopen( filename, "rb" );
    if( !fp )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, corelib::stringf( "cannot open layerdata file %s", filename ) );

    std::shared_ptr<FILE> fptr( fp, fclose );

    m_isHdr    = 0;
    m_hdrScale = 1.f;
    m_leakyReluAlpha = 0.f;
    m_numHiddenChannels = 0;

    std::string errfile = corelib::stringf( "reading layerdata file %s failed", filename );

    if( !readFile( &m_version, sizeof( int ), fp ) )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
    if( !readFile( &m_options, sizeof( int ), fp ) )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

    int ndir;
    if( !readFile( &ndir, sizeof( int ), fp ) )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

    for( int i = 0; i < ndir; i++ )
    {
        char name[32];
        if( !readFile( name, 32, fp ) )
            return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
        name[31] = 0;

        // get offset for this training set
        unsigned int offset;
        if( !readFile( &offset, sizeof( int ), fp ) )
            return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

        if( !strcmp( name, trainingId ) )
        {
            if( fseek( fp, offset, SEEK_SET ) != 0 )
                return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
            int nlayer;
            if( !readFile( &nlayer, sizeof( int ), fp ) )
                return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
            m_data.resize( nlayer );
            if( m_version >= 2 )
            {
                if( !readFile( &m_isHdr, sizeof( int ), fp ) )
                    return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
                if( !readFile( &m_hdrScale, sizeof( float ), fp ) )
                    return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
                if( m_version >= 3 )
                {
                    if( !readFile( m_hdrTransform, 6 * sizeof( float ), fp ) )
                        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
                }
                if( m_version >= 4 )
                {
                    float ext_options[3];
                    readFile( &m_leakyReluAlpha, sizeof( float ), fp );
                    readFile( &m_numHiddenChannels, sizeof( int ), fp );
                    readFile( ext_options, 3 * sizeof( int ), fp );
                }
            }
            for( int k = 0; k < nlayer; k++ )
            {
                Weights& cw = m_data[k];

                if( !readFile( (void*)( &cw.m_tdim[0] ), sizeof( int ) * 4, fp ) )
                    return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
                size_t flattened_size = ( size_t )( cw.m_tdim[0] * cw.m_tdim[1] * cw.m_tdim[2] * cw.m_tdim[3] );

                cw.m_weights.resize( flattened_size );
                if( !( m_options & OPTION_BIT_FP16 ) )
                {
                    std::vector<float> tmp( flattened_size );
                    if( !readFile( (void*)( &tmp[0] ), sizeof( float ) * flattened_size, fp ) )
                        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
                    colwert_float_to_fp16( (__half*)&cw.m_weights[0], &tmp[0], flattened_size );
                }
                else if( !readFile( (void*)( &cw.m_weights[0] ), sizeof( short ) * flattened_size, fp ) )
                    return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

                cw.m_bias.resize( cw.m_tdim[0] );
                if( !( m_options & OPTION_BIT_FP16 ) )
                {
                    std::vector<float> tmp( cw.m_tdim[0] );
                    if( !readFile( (void*)( &tmp[0] ), sizeof( float ) * cw.m_tdim[0], fp ) )
                        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
                    colwert_float_to_fp16( (__half*)&cw.m_bias[0], &tmp[0], cw.m_tdim[0] );
                }
                else if( !readFile( (void*)( &cw.m_bias[0] ), sizeof( short ) * cw.m_tdim[0], fp ) )
                    return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
            }
            OptixResult res = setImportOperations( trainingId, errDetails );
            if( res )
                return res;
            return OPTIX_SUCCESS;
        }
    }
    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( "cannot find training set %s in "
                                                                               "layerdata "
                                                                               "file %s (version %d)",
                                                                               trainingId, filename, m_version ) );
}

///// memory

static inline int get_int( const char** address )
{
    int i = **(int**)address;
    *address += sizeof( int );
    return i;
}

OptixResult Layerdata::load( const void* data, size_t dataSize, const char* const trainingId, ErrorDetails& errDetails )
{
    if( !data )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "null data argument in Layerdata::load" );

    m_isHdr    = 0;
    m_hdrScale = 1.f;
    m_leakyReluAlpha = 0.f;
    m_numHiddenChannels = 0;

    const char* cdata = (const char*)data;

    m_version = get_int( &cdata );
    m_options = (unsigned)get_int( &cdata );
    int ndir  = get_int( &cdata );

    for( int i = 0; i < ndir; i++ )
    {
        char name[32];
        memcpy( name, cdata, sizeof( name ) );
        name[31] = 0;
        cdata += 32;

        // get offset for this training set
        int offset = get_int( &cdata );

        if( !strcmp( name, trainingId ) )
        {
            const char* cdata  = &( (const char*)data )[offset];
            int         nlayer = get_int( &cdata );
            m_data.resize( nlayer );
            if( m_version >= 2 )
            {
                memcpy( &m_isHdr, cdata, sizeof( int ) );
                cdata += sizeof( int );
                memcpy( &m_hdrScale, cdata, sizeof( float ) );
                cdata += sizeof( float );
                if( m_version >= 3 )
                {
                    memcpy( m_hdrTransform, cdata, 6 * sizeof( float ) );
                    cdata += 6 * sizeof( float );
                }
                if( m_version >= 4 )
                {
                    memcpy( &m_leakyReluAlpha, cdata, sizeof( float ) );
                    cdata += sizeof( float );
                    memcpy( &m_numHiddenChannels, cdata, sizeof( int ) );
                    cdata += 4 * sizeof( float );
                }
            }
            for( int k = 0; k < nlayer; k++ )
            {
                Weights& cw = m_data[k];

                memcpy( (void*)&cw.m_tdim[0], cdata, 4 * sizeof( int ) );
                cdata += 4 * sizeof( int );
                size_t flattened_size = ( size_t )( cw.m_tdim[0] * cw.m_tdim[1] * cw.m_tdim[2] * cw.m_tdim[3] );

                cw.m_weights.resize( flattened_size );
                if( !( m_options & OPTION_BIT_FP16 ) )
                {
                    std::vector<float> tmp( flattened_size );
                    memcpy( (void*)( &tmp[0] ), cdata, sizeof( float ) * flattened_size );
                    colwert_float_to_fp16( (__half*)&cw.m_weights[0], &tmp[0], flattened_size );
                    cdata += sizeof( float ) * flattened_size;
                }
                else
                {
                    memcpy( (void*)( &cw.m_weights[0] ), cdata, sizeof( short ) * flattened_size );
                    cdata += sizeof( short ) * flattened_size;
                }

                cw.m_bias.resize( cw.m_tdim[0] );
                if( !( m_options & OPTION_BIT_FP16 ) )
                {
                    std::vector<float> tmp( cw.m_tdim[0] );
                    memcpy( (void*)( &tmp[0] ), cdata, sizeof( float ) * cw.m_tdim[0] );
                    colwert_float_to_fp16( (__half*)&cw.m_bias[0], &tmp[0], cw.m_tdim[0] );
                    cdata += sizeof( float ) * cw.m_tdim[0];
                }
                else
                {
                    memcpy( (void*)( &cw.m_bias[0] ), cdata, sizeof( short ) * cw.m_tdim[0] );
                    cdata += sizeof( short ) * cw.m_tdim[0];
                }
            }
            OptixResult res = setImportOperations( trainingId, errDetails );
            if( res )
                return res;
            return OPTIX_SUCCESS;
        }
    };
    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                  corelib::stringf( "cannot find training set %s in database", trainingId ) );
}

///// file

OptixResult Layerdata::hasTrainingSet( const char* const filename,
                                       const char* const trainingId,
                                       bool&             hasSet,
                                       bool&             isHdr,
                                       float&            hdrScale,
                                       ErrorDetails&     errDetails )
{
    hasSet = false;

    FILE* fp = fopen( filename, "rb" );
    if( !fp )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, corelib::stringf( "cannot open layerdata file %s", filename ) );

    std::shared_ptr<FILE> fptr( fp, fclose );

    isHdr    = false;
    hdrScale = 1.f;

    std::string errfile = corelib::stringf( "reading layerdata file %s failed", filename );

    int version, options;
    if( fread( &version, sizeof( int ), 1, fp ) != 1 )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
    if( fread( &options, sizeof( int ), 1, fp ) != 1 )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

    int ndir;
    if( fread( &ndir, sizeof( int ), 1, fp ) != 1 )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

    for( int i = 0; i < ndir; i++ )
    {
        char name[32];
        if( fread( name, 1, 32, fp ) != 32 )
            return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
        name[31] = 0;

        unsigned int offset;
        if( fread( &offset, sizeof( int ), 1, fp ) != 1 )
            return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

        if( !strcmp( name, trainingId ) )
        {
            if( fseek( fp, offset, SEEK_SET ) != 0 )
                return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
            if( fseek( fp, sizeof( int ), SEEK_LWR ) != 0 )
                return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );  // nlayer
            if( version >= 2 )
            {
                int hdr;
                if( fread( &hdr, sizeof( int ), 1, fp ) != 1 )
                    return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
                isHdr = bool( hdr );
                if( fread( &hdrScale, sizeof( float ), 1, fp ) != 1 )
                    return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
            }
            hasSet = true;
            return OPTIX_SUCCESS;
        }
    }
    return OPTIX_SUCCESS;
}


OptixResult Layerdata::getTrainingIds( const char* const filename, std::vector<std::string>& ids, ErrorDetails& errDetails )
{
    FILE* fp = fopen( filename, "rb" );
    if( !fp )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, corelib::stringf( "cannot open layerdata file %s", filename ) );

    std::shared_ptr<FILE> fptr( fp, fclose );

    std::string errfile = corelib::stringf( "reading layerdata file %s failed", filename );

    if( fseek( fp, sizeof( int ), SEEK_LWR ) != 0 )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
    ;  // version
    if( fseek( fp, sizeof( int ), SEEK_LWR ) != 0 )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
    ;  // options

    int ndir;
    if( fread( &ndir, sizeof( int ), 1, fp ) != 1 )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

    for( int i = 0; i < ndir; i++ )
    {
        char name[32];
        if( fread( name, 1, 32, fp ) != 32 )
            return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
        name[31] = 0;
        ids.push_back( std::string( name ) );

        if( fseek( fp, sizeof( int ), SEEK_LWR ) != 0 )
            return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
    }
    return OPTIX_SUCCESS;
}

OptixResult Layerdata::getOptions( const char* const filename, bool& upscale, ErrorDetails& errDetails )
{
    FILE* fp = fopen( filename, "rb" );
    if( !fp )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, corelib::stringf( "cannot open layerdata file %s", filename ) );

    std::shared_ptr<FILE> fptr( fp, fclose );

    std::string errfile = corelib::stringf( "reading layerdata file %s failed", filename );

    if( fseek( fp, sizeof( int ), SEEK_LWR ) != 0 )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );
    ;  // version

    unsigned int options;
    if( fread( &options, sizeof( unsigned int ), 1, fp ) != 1 )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, errfile );

    upscale = bool( options & OPTION_BIT_UPSCALE );

    return OPTIX_SUCCESS;
}

// memory

OptixResult Layerdata::hasTrainingSet( const void* data, const char* const trainingId, bool& hasSet, bool& isHdr, float& hdrScale, ErrorDetails& errDetails )
{
    hasSet = false;

    if( !data )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "null data argument in Layerdata::hasTrainingSet" );

    isHdr    = false;
    hdrScale = 1.f;

    const char* cdata = (const char*)data;

    int version = get_int( &cdata );
    cdata += sizeof( int );  // skip options
    int ndir = get_int( &cdata );

    for( int i = 0; i < ndir; i++ )
    {
        char name[32];
        memcpy( name, cdata, sizeof( name ) );
        name[31] = 0;
        cdata += 32;

        int offset = get_int( &cdata );

        if( !strcmp( name, trainingId ) )
        {
            const char* cdata = &( (const char*)data )[offset];
            cdata += sizeof( int );  // skip nlayer
            if( version >= 2 )
            {
                int hdr;
                memcpy( &hdr, cdata, sizeof( int ) );
                cdata += sizeof( int );
                isHdr = bool( hdr );
                memcpy( &hdrScale, cdata, sizeof( float ) );
                cdata += sizeof( float );
            }
            hasSet = true;
            return OPTIX_SUCCESS;
        }
    }
    return OPTIX_SUCCESS;
}


OptixResult Layerdata::getTrainingIds( const void* data, std::vector<std::string>& ids, ErrorDetails& errDetails )
{
    if( !data )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "null data argument in Layerdata::getTrainingIds" );

    const char* cdata = (const char*)data;
    cdata += 2 * sizeof( int );  // skip version, options
    int ndir = get_int( &cdata );

    for( int i = 0; i < ndir; i++ )
    {
        char name[32];
        memcpy( name, cdata, sizeof( name ) );
        name[31] = 0;
        ids.push_back( std::string( name ) );
        cdata += 32;
        cdata += sizeof( int );  // skip offset
    }
    return OPTIX_SUCCESS;
}

OptixResult Layerdata::getOptions( const void* data, bool& upscale, ErrorDetails& errDetails )
{
    if( !data )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "null data argument in Layerdata::getTrainingIds" );

    const char* cdata = (const char*)data;
    cdata += sizeof( int );  // skip version
    upscale = bool( (unsigned int)get_int( &cdata ) & OPTION_BIT_UPSCALE );

    return OPTIX_SUCCESS;
}

};  // namespace optix_exp
