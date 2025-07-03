/*
 *  Copyright (c) 2012, LWPU Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of LWPU Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __TRACERNAIVE_HPP__
#define __TRACERNAIVE_HPP__

#include "Tracer.hpp"

namespace prodlib
{
namespace bvhtools
{
//------------------------------------------------------------------------

class TracerNaive : public Tracer
{
public:
                        TracerNaive            (void);
    virtual             ~TracerNaive           (void);

    virtual const char* getBackendName         (void) const    { return "Naive CPU fallback"; }
    virtual Layout      getBVHLayout           (void) const    { return m_bvhLayout; }

    virtual void        traceFromHostMem       (const TracerData& bvh, const void* const rays, const int rayFormat, void* const hits, const int hitFormat, const int numRays, const bool anyHit, const bool watertight, const void* const stream = 0);

private:

    TracerNaive         (const TracerNaive&);                        // forbidden
    TracerNaive&        operator=              (const TracerNaive&); // forbidden

private:
    Layout              m_bvhLayout;
};

//------------------------------------------------------------------------
} // namespace bvhtools
} // namespace prodlib
#endif // __TRACERNAIVE_HPP__
