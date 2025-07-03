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

#include <AS/Traversers.h>

#include <corelib/misc/Concepts.h>
#include <string>

#include <map>
#include <vector>


namespace optix {

class CanonicalProgram;
class Context;
class Program;

// A factory and cache for internal shared programs.
//
// Because the Programs returned by SharedProgramManager are potentially used in
// multiple locations on the node graph, it is important to remember that
// any Variables declared on the programs will appear at all locations.
//
// Programs are not owned by SharedProgramManager.
class SharedProgramManager : private corelib::NonCopyable
{
  public:
    SharedProgramManager( Context* context );

    // ptxList - a null terminated list of string pairs for different architectures.
    //           The first string of the pair is the file name, second is the PTX contents.
    // func    - the name of the function
    Program* getProgram( const char** ptxList, const std::string& func, bool useDiskCache = true );
    // same as getProgram, but only if the program was created before. Returns nullptr otherwise
    Program* getCachedProgram( const char** ptxList, const std::string& func );

    // Return a traversal program from the runtime
    Program* getTraverserRuntimeProgram( const std::string& traverser, bool isGeom, bool hasMotion, bool bakedChildPtrs );

    // Return a bounds program from the runtime
    Program* getBoundsRuntimeProgram( const std::string& traverser, bool isGeom, bool hasMotion );

    // Returns the program representing a null Program (does nothing).
    Program* getNullProgram();

    // Returns a program that just calls rtPrintExceptionDetails(), see knob context.forceTrivialExceptionProgram.
    Program* getTrivialExceptionProgram();

    // Returns the program used by RtcBvh as visit program (does nothing).
    //
    // We cannot use the null program instead because that is special-cased in various places.
    Program* getRtcBvhDummyTraverserProgram();

  private:
    // Return a bounds/traversal program from the runtime
    Program* getRuntimeProgram( const std::string& traverser, bool isGeom, bool hasMotion, bool bakedChildPtrs );

    struct PtxListParams
    {
        const char** ptxList;  // ptxList contains pairs of strings. First is the file name, second is the PTX contents
        std::string  func;

        bool operator<( const PtxListParams& other ) const;
    };


    Context* m_context;
    std::map<TraverserParams, Program*> m_programMap;
    std::map<PtxListParams, Program*>   m_ptxListMap;
    Program* m_nullProgram                 = nullptr;
    Program* m_trivialExceptionProgram     = nullptr;
    Program* m_rtcBvhDummyTraverserProgram = nullptr;

    Program* createPtx( const PtxListParams& p, bool useDiskCache = true );
    Program* createTraverser( const TraverserParams& p, const std::vector<unsigned>& smVersions );
};
}
