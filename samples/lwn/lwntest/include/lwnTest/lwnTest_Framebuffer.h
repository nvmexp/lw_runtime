/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_Framebuffer_h__
#define __lwnTest_Framebuffer_h__

#include "lwnUtil/lwnUtil_Interface.h"

#include "lwn/lwn.h"
#include "lwnUtil/lwnUtil_PoolAllocator.h"

namespace lwnTest {
//
//                      LWN FRAMEBUFFER CLASS
//
// Utility class provided to allocate and manage the attachments of
// user-defined framebuffers.  To use, program the framebuffer size and the
// format of each attachment point and then call class methods to allocate,
// bind, and free textures with the specified size and format.
//
class Framebuffer {
public:
    static const int MaxColorBuffers = 8;

private:
    LWNuint                         m_width, m_height, m_depth;
    LWNuint                         m_samples;
    LWNuint                         m_depthSamples;          // mixed samples support
    LWNuint                         m_allocColorCount;       // (derived) color attachment count
    LWNbitfield                     m_flags;
    lwnUtil::MemoryPoolAllocator    *m_allocator;
    LWNformat                       m_colorFormats[MaxColorBuffers];
    LWNtexture                      *m_colorTextures[MaxColorBuffers];
    LWNtexture                      *m_msTextures[MaxColorBuffers];
    LWNformat                       m_depthFormat;
    LWNtexture                      *m_depthTexture;

public:
    explicit Framebuffer(LWNuint w = 0, LWNuint h = 0) :
        m_width(w), m_height(h), m_depth(1),
        m_samples(0), m_depthSamples(0), m_allocColorCount(0),
        m_flags(LWN_TEXTURE_FLAGS_COMPRESSIBLE_BIT),
        m_allocator(NULL),
        m_depthFormat(LWN_FORMAT_NONE),
        m_depthTexture(NULL)
    {
        for (int i = 0; i < MaxColorBuffers; i++) {
            m_colorFormats[i] = LWN_FORMAT_NONE;
            m_colorTextures[i] = NULL;
            m_msTextures[i] = NULL;
        }
    }
    ~Framebuffer()
    {
        assert(!m_allocator); // make sure destroy was called
    }

    void setSize(LWNuint width, LWNuint height)         { m_width = width; m_height = height; }
    void setSamples(LWNuint ms)                         { m_samples = ms; }
    void setDepthStencilFormat(LWNformat f)             { m_depthFormat = f; }
    void setDepth(LWNuint depth)                        { m_depth = depth; }
    void setColorFormat(LWNuint index, LWNformat f)     { m_colorFormats[index] = f; }
    void setColorFormat(LWNformat f)                    { m_colorFormats[0] = f; }
    void setFlags(LWNbitfield flags)                    { m_flags = flags; }
    void setDepthSamples(LWNuint ms)                    { m_depthSamples = ms; }

    //
    // Methods to query framebuffer attachments have separate C and C++
    // implementations.  They can't coexist as overloads because they are
    // different only in return type.  The C++ methods use reinterpret_cast to
    // colwert to/from native C types.
    //
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_C
    LWNtexture *getColorTexture(int index, bool msTexture = false) const
    {
        if (msTexture) {
            return m_msTextures[index];
        } else {
            return m_colorTextures[index];
        }
    }
    LWNtexture *getDepthTexture() const
    {
        return m_depthTexture;
    }
#else
    lwn::Texture *getColorTexture(int index, bool msTexture = false) const
    {
        LWNtexture *ctexture;
        if (msTexture) {
            ctexture = m_msTextures[index];
        } else {
            ctexture = m_colorTextures[index];
        }
        return reinterpret_cast<lwn::Texture *>(ctexture);
    }
    lwn::Texture *getDepthTexture() const
    {
        return reinterpret_cast<lwn::Texture *>(m_depthTexture);
    }
#endif

    LWNuint getWidth() const        { return m_width; }
    LWNuint getHeight() const       { return m_height; }
    LWNuint getDepth() const        { return m_depth; }
    LWNuint getSamples() const      { return m_samples; }
    LWNuint getColorCount() const   { return m_allocColorCount; }

    // Allocate textures for the framebuffer according to the size and formats
    // programmed in the class.
    void alloc(LWNdevice *device);

    // Free the textures created for the framebuffer, using automatic tagging
    // or not according to <tag>.
    void destroy(LWNboolean tag = LWN_TRUE);

    // Bind the textures attached to the framebuffer to <queue> for rendering.
    void bind(LWNcommandBuffer *cmdBuf) const;

    // For multisample framebuffers, downsample from the multisample color
    // attachment <index> to the single-sample attachment using <queue>,
    // optionally discarding the multisample attachment.
    void downsample(LWNcommandBuffer *cmdBuf, LWNuint index = 0, bool discard = true);

    // Set the viewport and scissor to the full framebuffer size.
    void setViewportScissor() const;

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    //
    // Methods to provide a native C++ interface to the core Framebuffer
    // class, using reinterpret_cast to colwert between C and C++ object
    // types.
    //
    void alloc(lwn::Device *device)
    {
        LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);
        alloc(cdevice);
    }
    void bind(lwn::CommandBuffer *cmdBuf) const
    {
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
        bind(ccb);
    }
    void downsample(lwn::CommandBuffer *cmdBuf, LWNuint index = 0, bool discard = true)
    {
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
        downsample(ccb, index, discard);
    }
    void setDepthStencilFormat(lwn::Format f)
    {
        setDepthStencilFormat(LWNformat(int(f)));
    }
    void setColorFormat(LWNuint index, lwn::Format f)
    {
        setColorFormat(index, LWNformat(int(f)));
    }
    void setColorFormat(lwn::Format f)
    {
        setColorFormat(0, LWNformat(int(f)));
    }
#endif
};

} // namespace lwnTest

#endif // #ifndef __lwnTest_Framebuffer_h__
