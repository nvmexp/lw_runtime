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

#include <iosfwd>
#include <string>
#include <vector>


namespace cort {
struct Buffer;
struct TextureSampler;
struct ProgramHeader;
};  // namespace cort

namespace optix {

class Buffer;
class Device;
class GeometryTriangles;
class LexicalScope;
class ObjectManager;
class TableManager;
class TextureSampler;
class Variable;
class Program;

//
// LayoutPrinter prints the contents of an object record buffer for debugging
// purposes.
//
class LayoutPrinter
{
  public:
    LayoutPrinter( std::ostream&               out,
                   const std::vector<Device*>& activeDevices,
                   ObjectManager*              objectManager,
                   TableManager*               tableManager,
                   bool                        useRtxDataModel );
    ~LayoutPrinter();

    void run();

  private:
    void printHeader( size_t objectsSize, size_t varTableSize, size_t traversablesSize, size_t programsSize, size_t buffersSize, size_t texturesSize, int lw );
    void printFooter();

    void printTableSpacer();

    void printShortColumnEntry( const std::string& entry );
    void printColumnEntry( const std::string& entry );

    // Print a dumb dump of a memory area with each word interpreted as
    // a float, uint, hex uint, and int
    void dumpMemory( const void* ptr, size_t size );

    void printBufferHeader( int id, const std::vector<unsigned int>& devs );
    void printTextureHeader( int id, const std::vector<unsigned int>& devs );
    void printProgramHeader( int id, const std::vector<unsigned int>& devs );
    void printTraversableHandles( int id, const std::vector<unsigned int>& devs );

    // Format and pretty-print routines
    std::string addressString( const char* base, const char* ptr, size_t offset );
    void printRow( const std::string& c1, const std::string& c2, const std::string& c3, const std::string& c4 );

    // Routines for printing different lexical scopes.
    void printLexicalScope( const LexicalScope* scope, const std::vector<unsigned int>& devs );
    void printScopeHeader( void* ptr, const LexicalScope* scope );
    void printLexicalScopeSpacer( char* p );
    void printLexicalScopeDynamicVariableTableOffset( const LexicalScope* scope, char* p );
    void printAcceleration( const LexicalScope* scope, char* p );
    void printGlobalScope( const LexicalScope* scope, char* p );
    void printGeometry( const LexicalScope* scope, char* p );
    void printGeometryTriangles( const GeometryTriangles* gt, char* p );
    void printGeometryInstance( const LexicalScope* scope, char* p );
    void printAbstractGroup( const LexicalScope* scope, char* p );
    void printInstanceDescriptorRow( size_t offset, size_t size, const std::string& text, const std::string& value );
    void printGroup( const LexicalScope* scope, char* p, const std::vector<unsigned int>& devs );
    void printMaterial( const LexicalScope* scope, char* p );
    void printProgram( const LexicalScope* scope, char* p );
    void printSelector( const LexicalScope* scope, char* p );
    void printTransform( const LexicalScope* scope, char* p );
    void printLexicalScopeVariables( const LexicalScope* scope, char* p );

    void printBoundVariable( char* ptr, Variable* v );
    void printMemberVariable( char* base, const void* addr, size_t size, const std::string& name, const std::string& type, const std::string& value );
    void printGap( const char* from, const char* to );

    std::ostream&               m_out;
    const std::vector<Device*>& m_activeDevices;
    ObjectManager*              m_objectManager = nullptr;
    TableManager*               m_tableManager  = nullptr;
    char*                       m_objectData    = nullptr;

    // Internally used variables to track gaps
    const char* m_nextAddress = nullptr;

    bool m_useRtxDataModel = false;
};

}  // namespace optix
