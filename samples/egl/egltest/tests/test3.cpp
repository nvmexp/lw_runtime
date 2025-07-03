/*
 * Copyright (c) 2016-2017, LWPU CORPORATION. All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "tests.h"


// Common methods to create and fill metadata blocks
constexpr LwU32 metadataSize = sizeof(EglTestMetadata);

static bool setMetaDataBlocks(EglTestMetadata **metadata, LwU32 frameIdx)
{
    if (!gEglState.metadataCount) {
        LOG_INFO("metadataCount is zero, doing nothing.\n");
        return true;
    }

    LOG_INFO("set metadata for frame %d.\n", frameIdx);

    for (int i = 0; i < gEglState.metadataCount; ++i) {
        if (!metadata[i]) {
            metadata[i] = (EglTestMetadata*)malloc(metadataSize);
            if (!metadata[i]) {
                LOG_ERR("malloc of EglTestMetadata failed for metadata[%d].\n", i);
                return false;
            }
        }

        // populate meta data block
        getMetaDataFor(metadata[i], frameIdx, i);
        LOG_INFO("metadata[%d] = (%.1f %lld)\n", i, metadata[i]->data1, metadata[i]->data2);
    }

    return true;
}


Producer3Stream1::~Producer3Stream1()
{
    for (int i = 0; i < gEglState.metadataCount; ++i) {
        if (metadata[i]) {
            free(metadata[i]);
        }
    }
}

void Producer3Stream1::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    if (gEglState.metadataCount > LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS) {
        LOG_ERR("metadataCount is %d, should be less than or equal to %d.\n",
            gEglState.metadataCount, LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS);
            return;
    }

    ProducerStream1::init(dsp, stream);

    for (int i = 0; i < gEglState.metadataCount; ++i) {
        metadata[i] = NULL;
        producerInfo.caps.metadataSize[i] = metadataSize;
        producerInfo.caps.metadataType[i] = 0;
    }
}

bool Producer3Stream1::run(void)
{
    bool ret = ProducerStream1::run();

    LOG_INFO("Producer3Stream1 run\n");

    return ret;
}

bool Producer3Stream1::setMetadata(int frameIdx)
{
    LwU32 offset = 0;

    if (!setMetaDataBlocks(metadata, frameIdx)) {
        LOG_ERR("failed to set metadata for frame %d\n", frameIdx);
        return false;
    }

    for (int i = 0; i < gEglState.metadataCount; ++i) {
        if (!TestExpStream1ProducerSetMetadata(display, eglStream, i, offset,
                                              metadataSize, metadata[i])) {
            LOG_ERR("failed to set metadata[%d]\n", i);
            return false;
        }
    }

    return true;
}

void Consumer3Stream1::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ConsumerStream1::init(dsp, stream);
}

bool Consumer3Stream1::run(void)
{
    bool ret = ConsumerStream1::run();

    LOG_INFO("Consumer3Stream1 run\n");

    return ret;
}

bool Consumer3Stream1::queryMetadata(int frameIdx)
{
    EglTestMetadata data;
    EglTestMetadata expectedData;
    LwU32 offset = 0;
    int i, j;

    struct MetadataNameMap {
        const LwU32 eglName;
        const char* label;
    } constexpr names[3] = {
        { EGL_PRODUCER_METADATA_LW, "EGL_PRODUCER_METADATA_LW"},
        { EGL_CONSUMER_METADATA_LW, "EGL_CONSUMER_METADATA_LW"},
        { EGL_PENDING_METADATA_LW, "EGL_PENDING_METADATA_LW"},
    };

    for (i = 0; i < 3; ++i) {
        LOG_INFO("Consumer query metadata for %s\n", names[i].label);

        for (j = 0; j < gEglState.metadataCount; ++j) {
            if (!TestExpStream1ConsumerQueryMetadata(display, eglStream, names[i].eglName,
                                                    j, offset, metadataSize, &data)) {
                LOG_ERR("Consumer failed to query metadata\n");
                return false;
            }

            // Verify against expected data
            if (names[i].eglName == EGL_CONSUMER_METADATA_LW) {
                // Consumer acquired frames go from 1-N. So adjust frameIdx for expected value
                getMetaDataFor(&expectedData, frameIdx - 1, j);
                if (!isMetadataBlockEqual(&data, &expectedData)) {
                    LOG_ERR("Mismatch between expected queried and expected Consumer metadata[%d] for frame %d\n", j, frameIdx);
                    LOG_ERR("\tExpected: data1=%.1f data2=%lld\n"
                            "\tReceived: data1=%.1f data2=%lld\n",
                                expectedData.data1, expectedData.data2, data.data1, data.data2);
                    return false;
                }

                LOG_INFO("Metadata: %d offset %d data1=%.1f data2=%lld\n",
                            j, offset, data.data1, data.data2);
            }
        }
    }

    return true;
}

// Note: For producer, metadata type is 32 bit vs 16 bit for consumer
//       So you cannot memcpy this into producer type
constexpr LwU16 metaDataTypes[LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS] = {
    // Graphics	0x1000 - 0x1FFF
    0x110F,
    // Camera	0x2000 - 0x2FFF
    0x2040,
    // Multimedia	0x3000 - 0x3FFF
    0x3005,
    // Compute	0x4000 - 0x4FFF
    0x477F,
};

Producer3Stream2::~Producer3Stream2()
{
}

void Producer3Stream2::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ProducerStream2::init(dsp, stream);

    if (gEglState.metadataCount > LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS) {
        LOG_ERR("metadataCount is %d, should be less than or equal to %d.\n",
            gEglState.metadataCount, LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS);
            return;
    }

    for (int i = 0; i < gEglState.metadataCount; ++i) {
        metadata[i] = NULL;
        // These are not required to be in the order requested by the consumer. So shift shuffle.
        caps.metadata.type[i] = metaDataTypes[(i + 3) % LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS];
        caps.metadata.size[i] = metadataSize;
    }
}

bool Producer3Stream2::run(void)
{
    bool ret = ProducerStream2::run();

    LOG_INFO("Producer3Stream2 run\n");

    return ret;
}

bool Producer3Stream2::setMetadata(int frameIdx)
{
    LwU32 offset = 0;

    if (!setMetaDataBlocks(metadata, frameIdx)) {
        LOG_ERR("failed to set metadata for frame %d\n", frameIdx);
        return false;
    }

    for (int i = 0; i < gEglState.metadataCount; ++i) {
        if (!TestExpStream2ProducerMetaDataSet(pvtEglStream, i, offset,
                                              metadataSize, metadata[i])) {
            LOG_ERR("failed to set metadata[%d]\n", i);
            return false;
        }
    }

    return true;
}

void Consumer3Stream2::init(EGLDisplay dsp, EGLStreamKHR stream)
{
    ConsumerStream2::init(dsp, stream);

    // setup supported metadata types
    static_assert(sizeof(caps.metadata.supportedTypes) == sizeof(metaDataTypes), "Mismatch metadata size");
    memcpy(caps.metadata.supportedTypes, metaDataTypes, sizeof(caps.metadata.supportedTypes));
}

bool Consumer3Stream2::run(void)
{
    bool ret = ConsumerStream2::run();

    LOG_INFO("Consumer3Stream2 run\n");

    return ret;
}

bool Consumer3Stream2::producerCapsMatch(void)
{
    int i, j, numMatchingTypes = 0;

    if(!ConsumerStream2::producerCapsMatch()) {
        LOG_ERR("Consumer: producer caps match failed.\n");
        return false;
    }

    // Verify metadata types
    for(i = 0; i < LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS; i++) {
        for(j = 0; j < LW_EGL_STREAM_METADATA_INTERNAL_BLOCKS; j++) {
            if(caps.metadata.supportedTypes[j] &&
              (prodCaps.metadata.type[i] == caps.metadata.supportedTypes[j])) {
                numMatchingTypes++;
            }
        }
    }

    // The test requires at least one supported type matches
    if (!numMatchingTypes) {
        LOG_ERR("Consumer: Failed to match producer metadata types.\n");
        return false;
    }

    LOG_INFO("Consumer: Matching Metadata types supported\n");

    return true;
}

bool Consumer3Stream2::queryMetadata(const LwEglApiStream2Frame* acquiredFrame)
{
    const LwEglApiClientBuffer clientBuffer = (*acquiredFrame).buffer;
    const int frameIdx = (int) (*acquiredFrame).number;
    EglTestMetadata data;
    EglTestMetadata expectedData;
    LwU32 offset = 0;
    int mdIdx;

    for (mdIdx = 0; mdIdx < gEglState.metadataCount; ++mdIdx) {
        if (!TestExpStream2ConsumerMetaDataGet(pvtEglStream,
                                               clientBuffer,
                                               mdIdx,
                                               offset,
                                               metadataSize,
                                               &data)) {
            LOG_ERR("Consumer: Failed to query metadata[%d], frame %d\n", mdIdx, frameIdx);
            return false;
        }

        // Verify frame metadata
        // Consumer acquired frames go from 1-N. So adjust frameIdx for expected value
        getMetaDataFor(&expectedData, frameIdx - 1, mdIdx);
        if (!isMetadataBlockEqual(&data, &expectedData)) {
            LOG_ERR("Consumer: Mismatch between expected queried and expected Consumer metadata[%d] for frame %d\n", mdIdx, frameIdx);
            LOG_ERR("\tExpected: data1=%.1f data2=%lld\n"
                    "\tReceived: data1=%.1f data2=%lld\n",
                        expectedData.data1, expectedData.data2, data.data1, data.data2);
            return false;
        }

        LOG_INFO("Consumer: Metadata: %d offset %d data1=%.1f data2=%lld\n",
                    mdIdx, offset, data.data1, data.data2);
    }

    return true;
}
