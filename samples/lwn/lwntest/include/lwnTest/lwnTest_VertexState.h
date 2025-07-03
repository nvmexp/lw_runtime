/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_VertexState_h__
#define __lwnTest_VertexState_h__

#include <vector>

#include "lwnUtil/lwnUtil_Interface.h"

#include "lwn/lwn.h"
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
#include "lwn/lwn_Cpp.h"
#endif

// Pick up lwnTool::texpkg::dt data type utility classes.
#include "lwnTool/texpkg/lwnTool_DataTypes.h"

#include "lwnUtil/lwnUtil_PoolAllocator.h"

namespace lwnTest {

//////////////////////////////////////////////////////////////////////////
//
//                   VERTEX ARRAY STATE CLASS
//
// The VertexArrayState class wraps VertexAttribState and VertexStreamState
// arrays into one bigger state object that can be bound to a command buffer
// easily.
//
class VertexArrayState
{
    friend class VertexStream;

    int numAttribs;
    int numStreams;
    LWLwertexAttribState attribs[16];
    LWLwertexStreamState streams[16];

public:

    VertexArrayState() : numAttribs(0), numStreams(0) { }

    void bind(LWNcommandBuffer *cmd) const
    {
        lwnCommandBufferBindVertexAttribState(cmd, numAttribs, attribs);
        lwnCommandBufferBindVertexStreamState(cmd, numStreams, streams);
    }

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    // Native C++ implementation of bind().
    void bind(lwn::CommandBuffer *cmd)
    {
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmd);
        bind(ccb);
    }
#endif
};

//////////////////////////////////////////////////////////////////////////
//
//                      VERTEX STREAM CLASS
//
// The VertexStream class is used to define the format of a vertex stream,
// which has one or multiple attributes.  To specify the layout of a stream,
// you first pass a stride to the constructor.  Then, you use the template
// methods to provide the type and offset of each attribute in the stream,
// using datatypes from the lwnTool::texpkg::dt namespace.  Each of those data
// types has an LWN format associated with it, which is used for setting up
// the attribute.  Once all attributes are added, the stream can be added to
// VertexStreamSet objects, or used directly to build a vertex state object
// and allocate vertex buffers.
//
//      struct Vertex {
//          dt::vec3 position;
//          dt::vec3 color;
//      };
//      VertexStream stream(sizeof(Vertex));
//      stream.addAttribute<dt::vec3>(offsetof(Vertex, position));
//      stream.addAttribute<dt::vec3>(offsetof(Vertex, color));
//
//      VertexState vertex = stream.CreateVertexState();
//      lwn::Buffer vbo = stream.AllocateVertexBuffer(device, 3, vertexData);
//
// The attributes in the stream are assigned conselwtively increasing
// attribute numbers in the order they were assigned to the stream.
//
class VertexStream
{
    struct Attribute {
        LWNformat               format;
        LWNuint                 offset;
    };
    std::vector<Attribute>      m_attribs;
    LWNuint                     m_stride;
    LWNuint                     m_divisor;
public:
    VertexStream(LWNuint stride = 0, LWNuint divisor = 0) : m_stride(stride), m_divisor(divisor) {}

    void setStride(LWNuint stride)          { m_stride = stride; }
    void setDivisor(LWNuint divisor)        { m_divisor = divisor; }
    LWNuint getStride() const               { return m_stride; }
    LWNuint getDivisor() const              { return m_divisor; }

    // Template methods add an attribute of type <A> to the stream at offset
    // <relativeOffset>.  The type must one of the lwnTool::texpkg::dt classes
    // above, with a ::format() method returning an LWN format enum
    // corresponding to the data type.
    template <typename A> void addAttribute(LWNuint relativeOffset)
    {
        Attribute attrib;
        attrib.format = lwnTool::texpkg::dt::traits<A>::lwnFormat();
        attrib.offset = relativeOffset;
        m_attribs.push_back(attrib);
    }
    template <typename A> void addAttribute(const A &a, LWNuint relativeOffset)
    {
        Attribute attrib;
        attrib.format = lwnTool::texpkg::dt::traits<A>::lwnFormat();
        attrib.offset = relativeOffset;
        m_attribs.push_back(attrib);
    }
    template <typename A> void addAttribute(const A *a, LWNuint relativeOffset)
    {
        Attribute attrib;
        attrib.format = lwnTool::texpkg::dt::traits<A>::lwnFormat();
        attrib.offset = relativeOffset;
        m_attribs.push_back(attrib);
    }

    // Hacky method to add an attribute to the vertex stream
    // without an lwnTool::texpkg::dt:: type
    void addAttributeExplicit(LWNformat format, LWNuint relativeOffset)
    {
        Attribute attrib;
        attrib.format = format;
        attrib.offset = relativeOffset;
        m_attribs.push_back(attrib);
    }

    // Adds all the attributes in the vertex stream to the state object <state>.
    void addToVertexArrayState(VertexArrayState &state) const
    {
        int count = (int) m_attribs.size();
        int first = state.numAttribs;
        int stream = state.numStreams;

        for (int i = 0; i < count; i++) {
            const Attribute &a = m_attribs[i];
            lwlwertexAttribStateSetDefaults(&state.attribs[first + i]);
            lwlwertexAttribStateSetFormat(&state.attribs[first + i], a.format, a.offset);
            lwlwertexAttribStateSetStreamIndex(&state.attribs[first + i], stream);
        }
        lwlwertexStreamStateSetDefaults(&state.streams[stream]);
        lwlwertexStreamStateSetStride(&state.streams[stream], m_stride);
        lwlwertexStreamStateSetDivisor(&state.streams[stream], m_divisor);

        state.numAttribs += count;
        state.numStreams++;
    }

    // Creates and compiles a vertex state object with a single stream (this
    // object).
    VertexArrayState CreateVertexArrayState()
    {
        VertexArrayState state;
        addToVertexArrayState(state);
        return state;
    }

    // Allocates a vertex buffer sufficient to hold <lwertices> worth of data
    // for the stream.  <data> specifies values to fill in the buffer if
    // non-NULL.  <allowMap> and <allowCopy> specify whether the buffer can be
    // mapped or copied into.
    LWNbuffer *AllocateVertexBuffer(LWNdevice *device, int lwertices,
                                    lwnUtil::MemoryPoolAllocator& allocator,
                                    const void *data = NULL);

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    lwn::Buffer *AllocateVertexBuffer(lwn::Device *device, int lwertices,
                                      lwnUtil::MemoryPoolAllocator& allocator,
                                      const void *data = NULL)
    {
        LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);
        LWNbuffer *cbuffer = AllocateVertexBuffer(cdevice, lwertices, allocator, data);
        return reinterpret_cast<lwn::Buffer *>(cbuffer);
    }
    void addAttributeExplicit(lwn::Format format, LWNuint relativeOffset)
    {
        LWNformat cFormat = LWNformat(int(format));
        addAttributeExplicit(cFormat, relativeOffset);
    }
#endif
};

// LWN_VERTEX_STREAM_ADD_MEMBER:  Hacky macro to deal with the fact that our
// dt::vec* classes are not considered POD in C++, making the use of
// offsetof() to identify structure member offsets illegal.  In practice, our
// classes are straightforward and should have no non-POD behavior.  So we
// roll our own offsetof() using a bogus pointer-to-structure.
#define LWN_VERTEX_STREAM_ADD_MEMBER(_stream, _structtype, _member) \
    do {                                                            \
        _structtype *s = (_structtype *) 0x10000000;                \
        LWNuint offset = (char *) (&s->_member) - (char *) s;       \
        _stream.addAttribute(s->_member, offset);                   \
    } while (0)

// The VertexStreamSet class is simply a collection of vertex streams.
class VertexStreamSet {
private:
    std::vector<const VertexStream *>  m_streams;
public:
    VertexStreamSet() {}

    // Methods to add a new stream to the set.
    void addStream(const VertexStream &stream)
        { m_streams.push_back(&stream); }
    inline VertexStreamSet& operator << (const VertexStream &stream)
        { m_streams.push_back(&stream); return *this; }

    // Constructors that accept small numbers of streams and append them all
    // to the set.
    VertexStreamSet(const VertexStream &s0)
        { addStream(s0); }
    VertexStreamSet(const VertexStream &s0, const VertexStream &s1)
        { addStream(s0); addStream(s1); }
    VertexStreamSet(const VertexStream &s0, const VertexStream &s1,
                 const VertexStream &s2)
        { addStream(s0); addStream(s1); addStream(s2); }
    VertexStreamSet(const VertexStream &s0, const VertexStream &s1,
                 const VertexStream &s2, const VertexStream &s3)
        { addStream(s0); addStream(s1); addStream(s2); addStream(s3); }

    // Method to create, compile, and return a vertex state object built from
    // the streams in the set.  Attribute and stream numbers are assigned
    // sequentially.
    VertexArrayState CreateVertexArrayState()
    {
        VertexArrayState state;
        int count = (int) m_streams.size();
        for (int i = 0; i < count; i++) {
            m_streams[i]->addToVertexArrayState(state);
        }
        return state;
    }
};

} // namespace lwnTest

#endif // #ifndef __lwnTest_VertexState_h__
