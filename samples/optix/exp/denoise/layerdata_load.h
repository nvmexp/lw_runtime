//
// Copyright 2020 LWPU Corporation. All rights reserved.
//

#pragma once

#include <string>
#include <vector>

#ifndef IRAY_BUILD
#include <corelib/misc/String.h>
#include <exp/context/ErrorHandling.h>
#endif

namespace optix_exp {

#define IMPORT_RGB          0       // no operation (rgb input, albedo input)
#define IMPORT_XYZ          1       // no operation (normals 3d)
#define IMPORT_LOG          2       // log(x+1)/log(100), exp(x*log(100))-1
#define IMPORT_LOG_SQRT     3       // log(sqrt(x)+1)/log(100), exp(x*x*log(100))-1
#define IMPORT_A            4       // in: A->AAA, out: A=A
#define IMPORT_XY           5       // no operation (normals 2d)
#define IMPORT_PQ           6
#define IMPORT_PRELOG       7
#define IMPORT_HDR          8
#define IMPORT_A_LOG        9       // in: A->AAA, out: A=A
#define IMPORT_A_LOG_SQRT  10       // in: A->AAA, out: A=A
#define IMPORT_NORM3       11       // xyz normalization
#define IMPORT_A_PRELOG    12

class Layerdata
{
  public:
    static const size_t MinUserDataSize = 4 * sizeof(int);
    struct Weights
    {
        std::vector<short> m_weights;  // actually __half
        std::vector<short> m_bias;     // actually __half
        unsigned int       m_tdim[4];
    };

    // load DL filter parameters from given training file. The training set is selected with
    // training_id, which is "rgb", "rgb-albedo" etc, depending on the feature channels used.

    Layerdata()
        : m_version( 0 )
    {
    }

    OptixResult load( const char* const trainingFile, const char* const trainingId, ErrorDetails& errDetails );
    OptixResult load( const void* data, size_t dataSize, const char* const trainingId, ErrorDetails& errDetails );

    bool valid() const { return m_data.size() > 0; }

    unsigned int getFeatureSize( int i ) const { return m_data[i].m_tdim[0]; }
    unsigned int getNumLayers() const { return (unsigned int)m_data.size(); }
    unsigned int getNumInfChannels() const { return needsS2D() ? m_data[0].m_tdim[1] / 4 : m_data[0].m_tdim[1]; }

    int getVersion() const { return m_version; }

    bool isUpscale() const;
    bool needsS2D() const;
    bool needsD2S() const;
    bool hKPNClipWeights() const;
    bool hKPN() const;
    bool isHDR() const { return bool( m_isHdr ); }

    int alphaImportOp() const;
    int exportOp() const;
    int getTemporalLayerIndex() const { return m_temporalLayerIndex; }
    int getHiddenLayerIndex() const { return getNumHiddenChannels() ? getTemporalLayerIndex()+1 : -1; }

    float getLeakyReluAlpha() const { return m_leakyReluAlpha; }
    unsigned int getNumHiddenChannels() const { return m_numHiddenChannels; }

    // check whether training_id is contained in the training database file
    static OptixResult hasTrainingSet( const char* const trainingFile,
                                       const char* const trainingId,
                                       bool&             hasSet,
                                       bool&             hdr,
                                       float&            hdrScale,
                                       ErrorDetails&     errDetails );
    static OptixResult getTrainingIds( const char* const trainingFile, std::vector<std::string>& ids, ErrorDetails& errDetails );
    static OptixResult getOptions( const char* const trainingFile, bool& upscale, ErrorDetails& errDetails );

    static OptixResult hasTrainingSet( const void* data, const char* const trainingId, bool& hasSet, bool& hdr, float& hdrScale, ErrorDetails& errDetails );
    static OptixResult getTrainingIds( const void* data, std::vector<std::string>& ids, ErrorDetails& errDetails );
    static OptixResult getOptions( const void* data, bool& upscale, ErrorDetails& errDetails );

    std::vector<Weights> m_data;
    std::vector<int>     m_importOperations;
    int                  m_isHdr;
    float                m_hdrScale;
    int                  m_version;
    unsigned int         m_options;
    float                m_hdrTransform[6];
    int                  m_temporalLayerIndex;
    float                m_leakyReluAlpha;
    unsigned int         m_numHiddenChannels;

  private:
    OptixResult setImportOperations( const char* const trainingId, ErrorDetails& errDetails );
    bool readFile( void* ptr, size_t sz, FILE* fp )
    {
        if( fread( ptr, 1, sz, fp ) != sz )
            return false;
        return true;
    }
};

};  // namespace optix_exp
