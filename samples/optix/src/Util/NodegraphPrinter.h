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

#include <FrontEnd/Canonical/CanonicalProgramID.h>
#include <Objects/SemanticType.h>
#include <corelib/misc/Concepts.h>
#include <map>
#include <sstream>
#include <string>

namespace optix {
class BindingManager;
class CanonicalProgram;
class GlobalScope;
class LexicalScope;
class ObjectManager;
class Program;
class ProgramManager;
class VariableReferenceSet;
class VariableReferenceSetWithCount;

class NodegraphPrinter : private corelib::NonCopyable
{
  public:
    NodegraphPrinter( ObjectManager* objectManager, ProgramManager* programManager, BindingManager* bindingManager );
    void setPrintReferences( bool val );
    void        run();
    std::string str();

  private:
    std::ostringstream out;
    std::ostringstream edges;
    ObjectManager*     m_objectManager;
    ProgramManager*    m_programManager;
    BindingManager*    m_bindingManager;
    int                m_unique                  = 0;
    bool               m_printReferences         = true;
    bool               m_printFullBindingManager = false;

    void emitScope( const LexicalScope* scope );
    void emitCanonicalProgram( const CanonicalProgram* cp );
    enum EdgeKind
    {
        Scoped,
        ReverseScoped,
        VirtualChild,
        Unscoped
    };
    void emitEdge( const std::string& name, const LexicalScope* from, const LexicalScope* to, EdgeKind edgeKind );
    void emitEdge( const std::string& name, const LexicalScope* from, const LexicalScope* to, SemanticType stype, unsigned int index );
    void emitEdge( const Program* from, const CanonicalProgram* to );
    void emitEdge( const GlobalScope* from, const Program* to );
    void emitMap( const std::string& name, const std::map<VariableReferenceID, VariableReferenceID>& map );
    template <typename T>
    void emitGraphProperty( const std::string& name, const T& prop );
    template <typename T>
    void emitGraphPropertySingle( const std::string& name, const T& prop );
    template <typename T>
    void emitGraphPropertyMulti( const std::string& name, const T& prop );
    void emitVirtualParents( const std::string& label, const Program* program );

    void emitBindingManager();
};
}  // end namespace optix
