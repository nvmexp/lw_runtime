
//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//
//
//

#pragma once

#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>

#include <optix.h>

// TODO namespace for helpers?

// Adapts an std::ostream to the log callback interface used by the OptiX API.
//
// It forwards all log messages to the ostream irrespective of their log level. To make use of this
// class, pass OptixLogBuffer::callback as log callback and the address of your instance as log
// callback data.
class OptixLogger
{
  public:
    OptixLogger( std::ostream& s )
        : m_stream( s )
    {
    }

    static void callback( unsigned int level, const char* tag, const char* message, void* cbdata )
    {
        OptixLogger* self = static_cast<OptixLogger*>( cbdata );
        self->callback( level, tag, message );
    }

    void callback( unsigned int /*level*/, const char* tag, const char* message )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_stream << tag << ":" << ( message ? message : "(no message)" ) << "\n";
    }


  private:
    // Mutex that protects m_stream.
    std::mutex m_mutex;

    // Needs m_mutex.
    std::ostream& m_stream;
};

// This variant of OptixLogger records all messages. For a given regular expression, it allows to count the number
// of matching messages and whether at least one message matches
class OptixRecordingLogger
{
  public:
    OptixRecordingLogger( std::ostream* s = nullptr )
        : m_stream( s ? *s : std::cerr )
    {
    }

    static void callback( unsigned int level, const char* tag, const char* message, void* cbdata )
    {
        OptixRecordingLogger* self = static_cast<OptixRecordingLogger*>( cbdata );
        self->callback( level, tag, message );
    }

    void callback( unsigned int /*level*/, const char* tag, const char* message )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_stream << tag << ":" << ( message ? message : "(no message)" ) << "\n";
        m_messages.push_back( message ? message : "(no message)" );
    }

    std::string getMessagesAsOneString()
    {
        return std::accumulate( m_messages.begin(), m_messages.end(), std::string() );
    }

    void clearMessages() { m_messages.clear(); }

  private:
    // Mutex that protects m_stream and m_messages.
    std::mutex m_mutex;

    // Needs m_mutex.
    std::ostream& m_stream;

    // Needs m_mutex.
    std::vector<std::string> m_messages;
};

// Implements a buffer for the log callback interface used by the OptiX API.
//
// It buffers all log messages along with their log level. For colwenience, it offers methods to
// forward the buffered messages to std::ostream, to instances of OptixLogger, and to other
// callbacks defined by a pair of OptixLogCallback and an opaque pointer.
class OptixLogBuffer
{
  public:
    static void callback( unsigned int level, const char* tag, const char* message, void* cbdata )
    {
        OptixLogBuffer* self = static_cast<OptixLogBuffer*>( cbdata );
        self->callback( level, tag, message );
    }

    void callback( unsigned int level, const char* tag, const char* message )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        m_messages.emplace_back( Message{level, tag, message ? message : "(no message)"} );
    }

    void forward( OptixLogCallback logCallback, void* logCallbackData )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        for( const auto& m : m_messages )
            logCallback( m.level, m.tag.c_str(), m.message.c_str(), logCallbackData );
    }

    void forward( std::ostream& s )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        for( const auto& m : m_messages )
            s << m.tag << ":" << m.message << "\n";
    }

    void forward( OptixLogger& logger )
    {
        std::lock_guard<std::mutex> lock( m_mutex );
        for( const auto& m : m_messages )
            logger.callback( m.level, m.tag.c_str(), m.message.c_str() );
    }

    // Mutex that protects m_messages.
    std::mutex m_mutex;

    struct Message
    {
        unsigned int level;
        std::string  tag;
        std::string  message;
    };

    // Needs m_mutex.
    std::vector<Message> m_messages;
};
