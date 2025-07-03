/*
 * Copyright (c) 2009 - 2012 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __CPPSTRING_H__
#define __CPPSTRING_H__
//
// cppstring.h
//
// lwString is a generic resizeable string class, loosely modeled after
// std::string.
//
// lwStringBuf is a class loosely modeled after std::ostringstream that
// provides an ostream-like "<<" operator to append strings, numbers, and
// such.
//

#include "cppsharedobject.h"

class lwString {

    // lwStringBuffer is a helper class wrapping a dynamically-resizeable
    // string allocation used to hold a string.  The allocation starts out
    // empty (NULL pointer), and grows as data is added to the buffer.
    class lwStringBuffer {
    public:
        // Minimum allocation granularity; intended to limit reallocs().
        static const int granularity = 1024;

        char *m_data;       // allocated storage
        size_t m_capacity;  // size (in bytes) of allocated storage
        size_t m_length;    // length of current string (not counting terminator)

        // The default constructor leaves us with a NULL buffer.
        lwStringBuffer() : m_data(NULL), m_capacity(0), m_length(0) {}

        // Constructing one string buffer from another creates a new buffer to
        // hold the old buffer contents, if any.
        lwStringBuffer(const lwStringBuffer &old)
            : m_data(NULL), m_capacity(0), m_length(0)
        {
          this->operator = (old);
        }

        ~lwStringBuffer()
        {
            __LWOG_FREE(m_data);
        }

        // Empty any contents of the buffer.
        void clear()
        {
            __LWOG_FREE(m_data);
            m_data = NULL;
            m_capacity = 0;
            m_length = 0;
        }

        // Reserve capacity to ensure that the capacity of the string buffer
        // is at least <size> bytes.
        bool reserve(size_t needed)
        {
            // Don't implement shrinking behavior for now.
            if (needed <= m_capacity) return true;

            // Otherwise allocate a new buffer of the requested size, and copy
            // over any old data.
            char *newData = (char *) __LWOG_MALLOC(needed);
            if (newData == NULL) {
                return false;
            }
            if (m_data) {
                memcpy(newData, m_data, m_length + 1);
                __LWOG_FREE(m_data);
            }
            m_data = newData;
            m_capacity = needed;
            return true;
        }

        // Probe to ensure that we have space to add <size> extra characters
        // and a terminator to the string buffer.
        bool probe(size_t size)
        {
            if (m_capacity > m_length + size) return true;

            size_t needed = m_length + size + 1;
            if (needed < m_capacity + granularity) {
                needed = m_capacity + granularity;
            }
            return reserve(needed);
        }

        char *begin()               { return m_data; }
        const char *begin() const   { return m_data; }
        char *end()                 { return m_data + m_length; }
        const char *end() const     { return m_data + m_length; }
        size_t capacity() const     { return m_capacity; }
        size_t length() const       { return m_length; }
        size_t freeSpace() const    { return m_capacity - m_length; }

    private:
        lwStringBuffer& operator = (const lwStringBuffer &other)
        {
            clear();
            m_length=other.m_length;
            if (m_length){
                reserve(m_length+1);
                memcpy(m_data, other.m_data, m_length+1);
            }
            return *this;
        }
    };

    // The only data stored directly in an lwString object is a shared object
    // wrapping the string buffer data object.
    lwSharedObject<lwStringBuffer> m_data;

    // Minimum amount of space to reserve for new append() operations that
    // use sprintf.
    static const size_t append_format_space = 80;

public:
    lwString() : m_data() {}
    lwString(const lwString &old) : m_data(old.m_data) {}

    // Constructors taking a C string or array of strings builds an empty
    // object and then appends the provided strings.
    lwString(const char *str) : m_data()
    {
        append(str);
    }
    lwString(int nStrings, const char * const * const strings) : m_data()
    {
        for (int i = 0; i < nStrings; i++) {
            append(strings[i]);
        }
    }

    // Various overloaded methods to append formatted versions of
    // different data types to the current string.
    void append(const char data)
    {
        lwStringBuffer *sb = m_data.edit();
        const char *format = "%c";
        if (sb->probe(append_format_space)) {
            char *cp = sb->end();
            lwog_snprintf(cp, sb->freeSpace(), format, data);
            sb->m_length += strlen(cp);
        }
    }

    void append(const int data)
    {
        lwStringBuffer *sb = m_data.edit();
        const char *format = "%d";
        if (sb->probe(append_format_space)) {
            char *cp = sb->end();
            lwog_snprintf(cp, sb->freeSpace(), format, data);
            sb->m_length += strlen(cp);
        }
    }

    void append(const signed long long data)
    {
        lwStringBuffer *sb = m_data.edit();
        const char *format = "%lld";
        if (sb->probe(append_format_space)) {
            char *cp = sb->end();
            lwog_snprintf(cp, sb->freeSpace(), format, data);
            sb->m_length += strlen(cp);
        }
    }

    void append(const unsigned char data)
    {
        lwStringBuffer *sb = m_data.edit();
        const char *format = "%c";
        if (sb->probe(append_format_space)) {
            char *cp = sb->end();
            lwog_snprintf(cp, sb->freeSpace(), format, data);
            sb->m_length += strlen(cp);
        }
    }

    void append(const unsigned int data)
    {
        lwStringBuffer *sb = m_data.edit();
        const char *format = "%u";
        if (sb->probe(append_format_space)) {
            char *cp = sb->end();
            lwog_snprintf(cp, sb->freeSpace(), format, data);
            sb->m_length += strlen(cp);
        }
    }

    void append(const unsigned long long data)
    {
        lwStringBuffer *sb = m_data.edit();
        const char *format = "%llu";
        if (sb->probe(append_format_space)) {
            char *cp = sb->end();
            lwog_snprintf(cp, sb->freeSpace(), format, data);
            sb->m_length += strlen(cp);
        }
    }

    void append(const float data)
    {
        lwStringBuffer *sb = m_data.edit();
        const char *format = "%f";
        if (sb->probe(append_format_space)) {
            char *cp = sb->end();
            lwog_snprintf(cp, sb->freeSpace(), format, data);
            sb->m_length += strlen(cp);
        }
    }

    void append(const double data)
    {
        lwStringBuffer *sb = m_data.edit();
        const char *format = "%lf";
        if (sb->probe(append_format_space)) {
            char *cp = sb->end();
            lwog_snprintf(cp, sb->freeSpace(), format, data);
            sb->m_length += strlen(cp);
        }
    }

    void append(const char *str)
    {
        lwStringBuffer *sb = m_data.edit();
        size_t len = strlen(str);
        if (sb->probe(len)) {
            char *cp = sb->end();
            memcpy(cp, str, len + 1);
            sb->m_length += len;
        }
    }

    void append(const lwString& str)
    {
        lwStringBuffer *sb = m_data.edit();
        size_t len = str.length();
        if (sb->probe(len)) {
            char *cp = sb->end();
            memcpy(cp, str.c_str(), len + 1);
            sb->m_length += len;
        }
    }

    // The "+=" operator can also be used like append().
    template <typename T> lwString & operator +=(const T value)
    {
        append(value);
        return *this;
    }

    // c_str() returns a C string corresponding to the current string
    // contents.  This pointer is not protected from subsequent updates to the
    // lwString object, which may modify or even free the C string.
    const char *c_str() const {
        const char *cp = m_data->begin();
        return cp ? cp : "";
    }

    // length() returns the length of the string, not counting the null
    // terminator.
    size_t length() const { return m_data->length(); }

    // capacity() returns the amount of storage used for the string buffer.
    size_t capacity() const { return m_data->capacity(); }

    // refcount() returns the number of lwString objects lwrrently referencing
    // the string buffer.
    int refcount() const { return m_data.refcount(); }
};

class lwStringBuf
{
private:
    lwString        m_string;
public:
    lwStringBuf() {}
    ~lwStringBuf() {}

    // The str() method can be used to get or retrieve the internal string
    // object.
    lwString str() const { return m_string; }
    void str(const lwString &s) { m_string = s; }

    // The "<<" operator can be used to add various objects to the string.
    template <typename T> lwStringBuf & operator <<(T data)
    {
        m_string.append(data);
        return *this;
    }
};

#endif // #ifndef __CPPSTRING_H__
