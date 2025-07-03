// Copyright (c) 2017, LWPU CORPORATION.
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

#include <ExelwtionStrategy/Common/VariableSpecialization.h>
#include <map>
#include <string>

namespace optix {
class ProgramManager;

using VariableSpecializations = std::map<VariableReferenceID, VariableSpecialization>;
struct Specializations
{
    //
    // WARNING: This is a persistent class. If you change anything you
    // should also update the readOrWrite function and bump the the
    // version number.
    //

    // Helper functions
    std::string summaryString( const ProgramManager* pm ) const;
    bool isCompatibleWith( const Specializations& other ) const;
    bool operator<( const Specializations& other ) const;
    void mergeVariableSpecializations( const Specializations& other );

    // Global exceptions across each function
    uint64_t m_exceptionFlags    = 0;
    bool     m_printEnabled      = false;
    int      m_minTransformDepth = 0;
    int      m_maxTransformDepth = 0;

    // Variable specializations (including buffer, texture sampler, and
    // regular variables), indexed by ref id
    VariableSpecializations m_varspec;
};

// Persistent support
void readOrWrite( PersistentStream* stream, Specializations* spec, const char* label );
}
