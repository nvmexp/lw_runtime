/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/

#include "lwntest_c.h"
#include "lwnTest/lwnTest_Framebuffer.h"

#include "lwn_utils.h"

namespace lwnTest {

void Framebuffer::alloc(LWNdevice *device)
{
    LWNtextureBuilder tb;
    lwnTextureBuilderSetDefaults(&tb);
    lwnTextureBuilderSetDevice(&tb, device);
    lwnTextureBuilderSetSize3D(&tb, m_width, m_height, m_depth);
    lwnTextureBuilderSetFlags(&tb, m_flags);

    LWNsizeiptr frameBufferSize = 0;

    // two pass over color and depth targets, first pass collect
    // sizes, ssecond pass allocate memory pool and targets
    for (int pass = 0; pass < 2; pass++) {
        if (pass) {
            m_allocator = new lwnUtil::MemoryPoolAllocator(device, NULL, frameBufferSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
        }

        for (int i = 0; i < MaxColorBuffers; i++) {
            if (m_colorFormats[i] == LWN_FORMAT_NONE) {
                continue;
            }

            lwnTextureBuilderSetFormat(&tb, m_colorFormats[i]);

            // Create a single-sample texture whether or not we're multisampled.
            if (m_depth == 1) {
                lwnTextureBuilderSetTarget(&tb, LWN_TEXTURE_TARGET_2D);
            } else {
                lwnTextureBuilderSetTarget(&tb, LWN_TEXTURE_TARGET_2D_ARRAY);
            }
            lwnTextureBuilderSetSamples(&tb, 0);

            if (!pass) {
                frameBufferSize += lwnTextureBuilderGetStorageSize(&tb);
                frameBufferSize += lwnTextureBuilderGetStorageAlignment(&tb);
            } else {
                m_colorTextures[i] = m_allocator->allocTexture(&tb);
                m_allocColorCount = i+1;
            }

            // If multisampled, also create a multisample texture.
            if (m_samples) {
                lwnTextureBuilderSetTarget(&tb, LWN_TEXTURE_TARGET_2D_MULTISAMPLE);
                lwnTextureBuilderSetSamples(&tb, m_samples);

                if (!pass) {
                    frameBufferSize += lwnTextureBuilderGetStorageSize(&tb);
                    frameBufferSize += lwnTextureBuilderGetStorageAlignment(&tb);
                } else {
                    m_msTextures[i] = m_allocator->allocTexture(&tb);
                }
            }
        }

        if (m_depthFormat != LWN_FORMAT_NONE) {
            LWNint effectiveSamples = m_depthSamples ? m_depthSamples : m_samples;
            if (m_depth == 1) {
                lwnTextureBuilderSetTarget(&tb, (effectiveSamples ?
                                                 LWN_TEXTURE_TARGET_2D_MULTISAMPLE :
                                                 LWN_TEXTURE_TARGET_2D));
            } else {
                lwnTextureBuilderSetTarget(&tb, (effectiveSamples ?
                                                 LWN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY :
                                                 LWN_TEXTURE_TARGET_2D_ARRAY));
            }
            lwnTextureBuilderSetFormat(&tb, m_depthFormat);
            lwnTextureBuilderSetFlags(&tb, lwnTextureBuilderGetFlags(&tb) |
                                      LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT);
            lwnTextureBuilderSetSamples(&tb, effectiveSamples);

            if (!pass) {
                frameBufferSize += lwnTextureBuilderGetStorageSize(&tb);
                frameBufferSize += lwnTextureBuilderGetStorageAlignment(&tb);
            }
            else {
                m_depthTexture = m_allocator->allocTexture(&tb);
            }
        }
    }
}

void Framebuffer::bind(LWNcommandBuffer *cmdBuf) const
{
    LWNtexture * const *colorTextures = m_samples ? m_msTextures : m_colorTextures;
    lwnCommandBufferSetRenderTargets(cmdBuf, m_allocColorCount, colorTextures, NULL, m_depthTexture, NULL);
}

void Framebuffer::destroy(LWNboolean tag /*= LWN_TRUE*/)
{
    for (LWNuint i = 0; i < m_allocColorCount; i++) {
        if (m_colorTextures[i]) {
            m_allocator->freeTexture(m_colorTextures[i]);
            m_colorTextures[i] = NULL;
        }
        if (m_msTextures[i]) {
            m_allocator->freeTexture(m_msTextures[i]);
            m_msTextures[i] = NULL;
        }
    }
    if (m_depthTexture) {
        m_allocator->freeTexture(m_depthTexture);
        m_depthTexture = NULL;
    }

    delete m_allocator;
    m_allocator = NULL;
}

void Framebuffer::downsample(LWNcommandBuffer *cmdBuf, LWNuint index /*= 0*/,
                             bool discard /*= true*/)
{
    lwnCommandBufferDownsample(cmdBuf, m_msTextures[index], m_colorTextures[index]);
    if (discard) {
        // Assumes the Framebuffer's m_msTextures & m_colorTextures
        // are lwrrently bound by a prior call to Framebuffer::bind().
        lwnCommandBufferDiscardColor(cmdBuf, 0);
        if (m_depthFormat != LWN_FORMAT_NONE) {
            lwnCommandBufferDiscardDepthStencil(cmdBuf);
        }
    }
}

void Framebuffer::setViewportScissor() const
{
    LWNcommandBuffer *cmdBuf = g_lwnQueueCB;
    lwnCommandBufferSetViewport(cmdBuf, 0, 0, m_width, m_height);
    lwnCommandBufferSetScissor(cmdBuf, 0, 0, m_width, m_height);
}

} // namespace lwnTest
