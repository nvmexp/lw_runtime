//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

#ifndef _COMMON_INCLUDED_
#define _COMMON_INCLUDED_


#if defined(__ANDROID__) || (defined(_MSC_VER) && _MSC_VER < 1700)
#include <sstream>
namespace std {
template<typename T>
std::string to_string(const T& val) {
  std::ostringstream os;
  os << val;
  return os.str();
}
}
#endif

#if (defined(_MSC_VER) && _MSC_VER < 1900 /*vs2015*/) || defined MINGW_HAS_SELWRE_API
    #include <basetsd.h>
    #ifndef snprintf
    #define snprintf sprintf_s
    #endif
    #define safe_vsprintf(buf,max,format,args) vsnprintf_s((buf), (max), (max), (format), (args))
#elif defined (solaris)
    #define safe_vsprintf(buf,max,format,args) vsnprintf((buf), (max), (format), (args))
    #include <sys/int_types.h>
    #define UINT_PTR uintptr_t
#else
    #define safe_vsprintf(buf,max,format,args) vsnprintf((buf), (max), (format), (args))
    #include <stdint.h>
    #define UINT_PTR uintptr_t
#endif

#if defined(_MSC_VER) && _MSC_VER < 1800
    #include <stdlib.h>
    inline long long int strtoll (const char* str, char** endptr, int base)
    {
        return _strtoi64(str, endptr, base);
    }
    inline unsigned long long int strtoull (const char* str, char** endptr, int base)
    {
        return _strtoui64(str, endptr, base);
    }
    inline long long int atoll (const char* str)
    {
        return strtoll(str, NULL, 10);
    }
#endif

#if defined(_MSC_VER)
#define strdup _strdup
#endif

/* windows only pragma */
#ifdef _MSC_VER
    #pragma warning(disable : 4786) // Don't warn about too long identifiers
    #pragma warning(disable : 4514) // unused inline method
    #pragma warning(disable : 4201) // nameless union
#endif

#include <set>
#include <unordered_set>
#include <vector>
#include <map>
#include <unordered_map>
#include <list>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "PoolAlloc.h"

//
// Put POOL_ALLOCATOR_NEW_DELETE in base classes to make them use this scheme.
//
#define POOL_ALLOCATOR_NEW_DELETE(A)                                  \
    void* operator new(size_t s) { return (A).allocate(s); }          \
    void* operator new(size_t, void *_Where) { return (_Where); }     \
    void operator delete(void*) { }                                   \
    void operator delete(void *, void *) { }                          \
    void* operator new[](size_t s) { return (A).allocate(s); }        \
    void* operator new[](size_t, void *_Where) { return (_Where); }   \
    void operator delete[](void*) { }                                 \
    void operator delete[](void *, void *) { }

namespace glslang {

    //
    // Pool version of string.
    //
    typedef pool_allocator<char> TStringAllocator;
    typedef std::basic_string <char, std::char_traits<char>, TStringAllocator> TString;

} // end namespace glslang

// Repackage the std::hash for use by unordered map/set with a TString key.
namespace std {

    template<> struct hash<glslang::TString> {
        std::size_t operator()(const glslang::TString& s) const
        {
            const unsigned _FLW_offset_basis = 2166136261U;
            const unsigned _FLW_prime = 16777619U;
            unsigned _Val = _FLW_offset_basis;
            size_t _Count = s.size();
            const char* _First = s.c_str();
            for (size_t _Next = 0; _Next < _Count; ++_Next)
            {
                _Val ^= (unsigned)_First[_Next];
                _Val *= _FLW_prime;
            }

            return _Val;
        }
    };
}

namespace glslang {

inline TString* NewPoolTString(const char* s)
{
    void* memory = GetThreadPoolAllocator().allocate(sizeof(TString));
    return new(memory) TString(s);
}

template<class T> inline T* NewPoolObject(T*)
{
    return new(GetThreadPoolAllocator().allocate(sizeof(T))) T;
}

template<class T> inline T* NewPoolObject(T, int instances)
{
    return new(GetThreadPoolAllocator().allocate(instances * sizeof(T))) T[instances];
}

//
// Pool allocator versions of vectors, lists, and maps
//
template <class T> class TVector : public std::vector<T, pool_allocator<T> > {
public:
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    typedef typename std::vector<T, pool_allocator<T> >::size_type size_type;
    TVector() : std::vector<T, pool_allocator<T> >() {}
    TVector(const pool_allocator<T>& a) : std::vector<T, pool_allocator<T> >(a) {}
    TVector(size_type i) : std::vector<T, pool_allocator<T> >(i) {}
    TVector(size_type i, const T& val) : std::vector<T, pool_allocator<T> >(i, val) {}
};

template <class T> class TList  : public std::list<T, pool_allocator<T> > {
};

template <class K, class D, class CMP = std::less<K> >
class TMap : public std::map<K, D, CMP, pool_allocator<std::pair<K const, D> > > {
};

template <class K, class D, class HASH = std::hash<K>, class PRED = std::equal_to<K> >
class TUnorderedMap : public std::unordered_map<K, D, HASH, PRED, pool_allocator<std::pair<K const, D> > > {
};

//
// Persistent string memory.  Should only be used for strings that survive
// across compiles/links.
//
typedef std::basic_string<char> TPersistString;

//
// templatized min and max functions.
//
template <class T> T Min(const T a, const T b) { return a < b ? a : b; }
template <class T> T Max(const T a, const T b) { return a > b ? a : b; }

//
// Create a TString object from an integer.
//
#if defined _MSC_VER || defined MINGW_HAS_SELWRE_API
inline const TString String(const int i, const int base = 10)
{
    char text[16];     // 32 bit ints are at most 10 digits in base 10
    _itoa_s(i, text, sizeof(text), base);
    return text;
}
#else
inline const TString String(const int i, const int /*base*/ = 10)
{
    char text[16];     // 32 bit ints are at most 10 digits in base 10

    // we assume base 10 for all cases
    snprintf(text, sizeof(text), "%d", i);

    return text;
}
#endif

struct TSourceLoc {
    void init()
    {
        name = nullptr; string = 0; line = 0; column = 0;
    }
    void init(int stringNum) { init(); string = stringNum; }
    // Returns the name if it exists. Otherwise, returns the string number.
    std::string getStringNameOrNum(bool quoteStringName = true) const
    {
        if (name != nullptr) {
            TString qstr = quoteStringName ? ("\"" + *name + "\"") : *name;
            std::string ret_str(qstr.c_str());
            return ret_str;
        }
        return std::to_string((long long)string);
    }
    const char* getFilename() const
    {
        if (name == nullptr)
            return nullptr;
        return name->c_str();
    }
    const char* getFilenameStr() const { return name == nullptr ? "" : name->c_str(); }
    TString* name; // descriptive name for this string, when a textual name is available, otherwise nullptr
    int string;
    int line;
    int column;
};

class TPragmaTable : public TMap<TString, TString> {
public:
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())
};

const int MaxTokenLength = 1024;

template <class T> bool IsPow2(T powerOf2)
{
    if (powerOf2 <= 0)
        return false;

    return (powerOf2 & (powerOf2 - 1)) == 0;
}

// Round number up to a multiple of the given powerOf2, which is not
// a power, just a number that must be a power of 2.
template <class T> void RoundToPow2(T& number, int powerOf2)
{
    assert(IsPow2(powerOf2));
    number = (number + powerOf2 - 1) & ~(powerOf2 - 1);
}

template <class T> bool IsMultipleOfPow2(T number, int powerOf2)
{
    assert(IsPow2(powerOf2));
    return ! (number & (powerOf2 - 1));
}

} // end namespace glslang

#endif // _COMMON_INCLUDED_
