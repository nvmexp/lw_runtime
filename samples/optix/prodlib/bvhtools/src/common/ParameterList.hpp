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

#pragma once
#include <vector>
#include <sstream>

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

class ParameterList
{
public:
                            ParameterList   (void);
    explicit                ParameterList   (const char* string);
                            ~ParameterList  (void);

    void                    clear           (void);
    void                    set             (const char* string); // e.g. "builder=TRBVH, splitBeta=0.30, optRounds=1"
    void                    set             (const char* name, const char* value);
    void                    set             (const char* nameBegin, const char* nameEnd, const char* valueBegin, const char* valueEnd);

    const char*             get             (const char* name, const char* defaultValue) const;
    int                     get             (const char* name, int defaultValue) const;
    size_t                  get             (const char* name, size_t defaultValue) const;
    float                   get             (const char* name, float defaultValue) const;
    bool                    get             (const char* name, bool defaultValue) const;

    template <class T> T&   get             (T& value, const char* name) const { value = get(name, value); return value; }

    void                    sort            (void);
    void                    toString        (std::vector<char>& out) const; // Does not append a terminating NULL character.

private:
    std::vector<int>        m_names;        // Start offsets in m_pool.
    std::vector<int>        m_values;       // Start offsets in m_pool.
    std::vector<char>       m_pool;         // Stores the strings referenced by m_names and m_values.
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
