// Copyright LWPU Corporation 2013
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include "ParameterList.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <prodlib/exceptions/IlwalidValue.h>

using namespace prodlib::bvhtools;

//------------------------------------------------------------------------

ParameterList::ParameterList(void)
{
}

//------------------------------------------------------------------------

ParameterList::ParameterList(const char* string)
{
    set(string);
}

//------------------------------------------------------------------------

ParameterList::~ParameterList(void)
{
}

//------------------------------------------------------------------------

void ParameterList::clear(void)
{
    m_names.clear();
    m_values.clear();
    m_pool.clear();
}

//------------------------------------------------------------------------

void ParameterList::set(const char* string)
{
    const char* ptr = string;
    bool ok = false;

    for (int i = 0;; i++)
    {
        // Skip whitespace.

        while (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')
            ptr++;

        // End-of-string before first parameter => done.

        if (!*ptr && !i)
        {
            ok = true;
            break;
        }

        // Parse parameter name.

        const char* nameBegin = ptr;
        while ((*ptr >= 'a' && *ptr <= 'z') || (*ptr >= 'A' && *ptr <= 'Z') || *ptr == '_')
            ptr++;
        const char* nameEnd = ptr;
        if (nameBegin == nameEnd)
            break;

        // Skip whitespace.

        while (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')
            ptr++;

        // Parse equals sign.

        if (*ptr != '=')
            break;
        ptr++;

        // Skip whitespace.

        while (*ptr == ' ' || *ptr == '\t' || *ptr == '\n' || *ptr == '\r')
            ptr++;

        // Parse value.

        const char* valueBegin = ptr;
        while (*ptr && *ptr != ',')
            ptr++;
        const char* valueEnd = ptr;

        // Trim trailing whitespace.

        while (valueBegin < valueEnd && (valueEnd[-1] == ' ' || valueEnd[-1] == '\t' || valueEnd[-1] == '\n' || valueEnd[-1] == '\r'))
            valueEnd--;

        // Record the name/value pair.

        set(nameBegin, nameEnd, valueBegin, valueEnd);

        // End-of-string => done.

        if (!*ptr)
        {
            ok = true;
            break;
        }

        // Parse comma.

        if (*ptr != ',')
            break;
        ptr++;
    }

    // Report errors.

    if (!ok)
        throw IlwalidValue( RT_EXCEPTION_INFO, "Syntax error in parameter string!" );
}

//------------------------------------------------------------------------

void ParameterList::set(const char* name, const char* value)
{
    set(name, name + strlen(name), value, value + strlen(value));
}

//------------------------------------------------------------------------

void ParameterList::set(const char* nameBegin, const char* nameEnd, const char* valueBegin, const char* valueEnd)
{
    // Append value and name to the pool.

    int valueOfs = (int)m_pool.size();
    m_pool.insert(m_pool.end(), valueBegin, valueEnd);
    m_pool.push_back('\0');

    int nameOfs = (int)m_pool.size();
    m_pool.insert(m_pool.end(), nameBegin, nameEnd);
    m_pool.push_back('\0');

    // Search for existing parameter of the same name.

    int idx = -1;
    for (int i = 0; i < (int)m_names.size() && idx == -1; i++)
        if (strcmp(m_pool.data() + nameOfs, m_pool.data() + m_names[i]) == 0)
            idx = i;

    // Not found => add new parameter.

    if (idx == -1)
    {
        m_names.push_back(nameOfs);
        m_values.push_back(valueOfs);
    }

    // Existing value not changed => revert changes to the pool.

    else if (strcmp(m_pool.data() + valueOfs, m_pool.data() + m_values[idx]) == 0)
        m_pool.resize(valueOfs);

    // Otherwise => update the value.

    else
    {
        m_values[idx] = valueOfs;
        m_pool.resize(nameOfs);
    }
}

//------------------------------------------------------------------------

const char* ParameterList::get(const char* name, const char* defaultValue) const
{
    for (int i = 0; i < (int)m_names.size(); i++)
        if (strcmp(name, m_pool.data() + m_names[i]) == 0)
            return m_pool.data() + m_values[i];
    return defaultValue;
}

//------------------------------------------------------------------------

int ParameterList::get(const char* name, int defaultValue) const
{
    const char* str = get(name, (const char*)NULL);
    if (!str)
        return defaultValue;

    int value = 0;
    int length = 0;
    sscanf(str, "%d%n", &value, &length);
    if (length && (size_t)length == strlen(str))
        return value;

    throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid integer parameter specified!" );
}

//------------------------------------------------------------------------

size_t ParameterList::get(const char* name, size_t defaultValue) const
{
  const char* str = get(name, (const char*)NULL);
  if (!str)
    return defaultValue;

  std::string inputString = str;
  std::istringstream istr(inputString);

  size_t value = 0;
  istr >> value;

  return value;
}

namespace {

// This is a quite limited colwersion function and is relying on the fact that the bvh build spec
// contains floats in its basic form [-][X]*.[Y]* only. The sole reason for having it at all is
// that the ParameterList::get() might be called numerous times and having a full-blown float
// colwersion with the required additional (local) locale setting gets expensive, performance-wise.
float stringToFloat(const char* str, bool& failure)
{
  float value = 0.f;
  failure = false;

  bool neg = false;
  if (*str == '-') {
    neg = true;
    ++str;
  }
  // "error handling"
  if (!(*str >= '0' && *str <= '9') && *str != '.') {
    failure = true;
    return value;
  }

  while (*str >= '0' && *str <= '9') {
    value = (value*10.f) + (*str - '0');
    ++str;
  }
  // fractional part
  if (*str == '.') {
    float fr = 0.f;
    int n = 0;
    ++str;
    while (*str >= '0' && *str <= '9') {
      fr = (fr*10.f) + (*str - '0');
      ++str;
      ++n;
    }
    value += fr / std::pow(10.f, n);
  }
  if (neg)
    value = -value;

  return value;
}

}

//------------------------------------------------------------------------

float ParameterList::get(const char* name, float defaultValue) const
{
    const char* str = get(name, (const char*)NULL);
    if (!str)
        return defaultValue;

    // Tell the string stream that we want to use the standard c language convention for locals.
    // This makes sure that we can parse a float with "." separator and do not need
    // to care about international formating (e.g. German floats with a "," separator!
    bool hasFailure = false;
    float value = stringToFloat(str, hasFailure);

    if( hasFailure )
      throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid floating point parameter specified!" );


    return value;
}

//------------------------------------------------------------------------

bool ParameterList::get(const char* name, bool defaultValue) const
{
    int value = get(name, (defaultValue) ? 1 : 0);
    if (value == 0)
        return false;
    if (value == 1)
        return true;

    throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid boolean parameter specified!" );
}

//------------------------------------------------------------------------

namespace
{
  struct Comparator
  {
    char* pool;
    bool operator()(int i, int j) { return (strcmp(pool + i, pool + j) < 0); }
  };
}

void ParameterList::sort(void)
{
    Comparator comparator;
    comparator.pool = m_pool.data();
    std::sort(m_names.begin(), m_names.end(), comparator );
}

//------------------------------------------------------------------------

void ParameterList::toString(std::vector<char>& out) const
{
    for (int i = 0; i < (int)m_names.size(); i++)
    {
        if (i != 0)
        {
            out.push_back(',');
            out.push_back(' ');
        }

        const char* name = m_pool.data() + m_names[i];
        const char* value = m_pool.data() + m_values[i];
        out.insert(out.end(), name, name + strlen(name));
        out.push_back('=');
        out.insert(out.end(), value, value + strlen(value));
    }
    out.push_back('\0');
}

//------------------------------------------------------------------------
