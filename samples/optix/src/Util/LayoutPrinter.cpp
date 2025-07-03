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

#include <Util/LayoutPrinter.h>

#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/TableManager.h>
#include <Device/Device.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <Objects/Acceleration.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/GlobalScope.h>
#include <Objects/Group.h>
#include <Objects/LexicalScope.h>
#include <Objects/ObjectClass.h>
#include <Objects/Variable.h>
#include <Util/ContainerAlgorithm.h>

#include <corelib/misc/String.h>
#include <prodlib/system/Knobs.h>

#include <cstddef>
#include <iomanip>
#include <sstream>

#include <iostream>

using namespace optix;
using namespace corelib;


namespace {
// clang-format off
  Knob<bool> k_dumpRawObjectRecord( RT_DSTRING( "launch.dumpRawObjectRecords" ), false, RT_DSTRING( "When dumping the object record, also dump the raw bytes" ) );
// clang-format on
}  // namespace

namespace {
static const unsigned int VALUE_LEFT_WIDTH  = 14;
static const unsigned int VALUE_RIGHT_WIDTH = 8;
static const unsigned int HOLE_WIDTH        = 6u;
static const unsigned int COL_WIDTH         = 32u + HOLE_WIDTH;
static const unsigned int SHORT_COL_WIDTH   = COL_WIDTH / 2 + 6;

// Returns space to be used between list elements
inline const char* spacer( int pos )
{
    if( pos > 0 )
        return " ";
    else
        return "";
}

std::string hexString( size_t u, unsigned int width = 8 )
{
    std::stringstream ss;
    ss << "0x" << std::setw( width ) << std::setfill( '0' ) << std::hex << u;
    return ss.str();
}

std::string offsetString( size_t o, unsigned width )
{
    if( o == 0 )
        return "(null)";
    else
        return "@" + hexString( o, width );
}
std::string offsetString( const unsigned int& val )
{
    return offsetString( val, sizeof( val ) * 2 );
}

template <typename T>
std::string offsetString( const T& val )
{
    return offsetString( val.o, sizeof( T ) * 2 );
}

bool RecordOffsetCompare( LexicalScope* t1, LexicalScope* t2 )
{
    return t1->getRecordOffset() < t2->getRecordOffset();
}

std::string to_string( cort::Matrix4x4 val )
{
    std::ostringstream str;
    str << "{ ";
    for( size_t y = 0; y < 4; ++y )
        for( size_t x = 0; x < 4; ++x )
            str << val.matrix[x][y] << " ";
    str << "}";
    return str.str();
}

std::string valuePairString( const std::string& type, const std::string& value )
{
    std::stringstream ss;
    ss << std::setw( VALUE_LEFT_WIDTH ) << std::left << type << ": " << std::setw( VALUE_RIGHT_WIDTH ) << std::left << value;
    return ss.str();
}

std::string byteString( char* ptr, size_t size )
{
    std::ostringstream type, vals;
    type << "byte[" << size << "]";
    for( size_t i = 0; i < size; ++i )
        vals << spacer( i ) << std::setw( 2 ) << std::setfill( '0' ) << std::hex << ( (unsigned int)ptr[i] & 0xFF );
    return valuePairString( type.str(), vals.str() );
}
}  // namespace

LayoutPrinter::LayoutPrinter( std::ostream&               out,
                              const std::vector<Device*>& activeDevices,
                              ObjectManager*              objectManager,
                              TableManager*               tableManager,
                              bool                        useRtxDataModel )
    : m_out( out )
    , m_activeDevices( activeDevices )
    , m_objectManager( objectManager )
    , m_tableManager( tableManager )
    , m_useRtxDataModel( useRtxDataModel )
{
}

LayoutPrinter::~LayoutPrinter()
{
}

void LayoutPrinter::run()
{
    // Print active devices
    m_out << "Devices:";
    std::vector<unsigned int> devs;
    for( std::vector<Device*>::const_iterator iter = m_activeDevices.begin(); iter != m_activeDevices.end(); ++iter )
    {
        devs.push_back( ( *iter )->allDeviceListIndex() );
        if( iter != m_activeDevices.begin() )
            m_out << ", ";
        m_out << ( *iter )->allDeviceListIndex();
    }
    m_out << "\n\n";

    // Print header
    const size_t objectsSize  = m_tableManager->getObjectRecordSize();
    const size_t varTableSize = m_tableManager->getDynamicVariableTableSize();
    const int    ntrv         = (int)m_objectManager->getTraversables().linearArraySize();
    const int    nprg         = (int)m_objectManager->getPrograms().linearArraySize();
    const int    nbuf         = (int)m_objectManager->getBuffers().linearArraySize();
    const int    ntex         = (int)m_objectManager->getTextureSamplers().linearArraySize();
    const int    ntok         = (int)m_objectManager->getVariableTokenCount();
    printHeader( objectsSize, varTableSize, ntrv * sizeof( RtcTraversableHandle ), nprg * sizeof( cort::ProgramHeader ),
                 nbuf * sizeof( cort::Buffer ), ntex * sizeof( cort::TextureSamplerHost ), ntok );

    // Sort scopes by offset for printing
    std::vector<LexicalScope*> sorted;
    int                        ilwalidObjectRecords = 0;
    for( const auto& scope : m_objectManager->getLexicalScopes() )
    {
        if( scope->recordIsAllocated() )
            sorted.push_back( scope );
        else
            ilwalidObjectRecords++;
    }
    algorithm::sort( sorted, RecordOffsetCompare );

    if( ilwalidObjectRecords != 0 )
        m_out << "Invalid/unallocated object records: " << ilwalidObjectRecords << '\n';

    // Print scopes
    m_objectData  = m_tableManager->getObjectRecordHostPointer();
    m_nextAddress = m_objectData;
    for( LexicalScope* iter : sorted )
        printLexicalScope( iter, devs );

    // Print footer
    printFooter();

    // Print traversables
    m_out << "======================================================\n";
    m_out << "Traversable Headers (header slots: " << ntrv << ", in use: " << m_objectManager->getTraversables().size() << ")\n";
    for( int i = 0; i < ntrv; ++i )
        printTraversableHandles( i, devs );
    m_out << '\n';

    // Print program headers
    m_out << "======================================================\n";
    m_out << "Program Headers (header slots: " << nprg << ", in use: " << m_objectManager->getPrograms().size()
          << ")\n";
    for( int i = 1; i < nprg; ++i )
        printProgramHeader( i, devs );
    m_out << '\n';

    // Print buffer headers
    m_out << "======================================================\n";
    m_out << "Buffer Headers (header slots: " << nbuf << ", in use: " << m_objectManager->getBuffers().size() << ")\n";
    for( int i = 1; i < nbuf; ++i )
        printBufferHeader( i, devs );
    m_out << '\n';

    // Print texture headers
    m_out << "======================================================\n";
    m_out << "Texture Headers (header slots: " << ntex << ", in use: " << m_objectManager->getTextureSamplers().size() << ")\n";
    for( int i = 1; i < ntex; ++i )
        printTextureHeader( i, devs );
    m_out << '\n';

    if( k_dumpRawObjectRecord.get() )
    {
        // And a raw dump
        m_out << "======================================================\n";
        m_out << "Raw object record dump\n";
        dumpMemory( m_objectData, objectsSize );
    }
}

void LayoutPrinter::printHeader( size_t objectsSize,
                                 size_t varTableSize,
                                 size_t traversablesSize,
                                 size_t programsSize,
                                 size_t buffersSize,
                                 size_t texturesSize,
                                 int    ntok )
{
    m_out << "Object record total size: " << objectsSize << '\n';
    m_out << "Variable Table total size: " << varTableSize << '\n';
    m_out << "Traversable handles total size: " << traversablesSize << '\n';
    m_out << "Program headers total size: " << programsSize << '\n';
    m_out << "Buffer headers total size: " << buffersSize << '\n';
    m_out << "Texture headers total size: " << texturesSize << '\n';
    m_out << "Total unique variable names: " << ntok << '\n';
    m_out << '\n';

    // Print variable strings
    m_out << "Variables tokens: ";
    for( int i = 0; i < ntok; i++ )
    {
        if( i != 0 )
            m_out << ", ";
        m_out << m_objectManager->getVariableNameForToken( i ) << '=' << i;
    }
    m_out << '\n';

    printRow( "Offset           size", "Class", "Variables", "Value" );
}

void LayoutPrinter::printFooter()
{
    printTableSpacer();
    m_out << "\n\n";
}

void LayoutPrinter::printScopeHeader( void* ptr, const LexicalScope* scope )
{
    RT_ASSERT( scope );

    std::string       type = getNameForClass( scope->getClass() );
    std::stringstream scopeIdAndName;
    scopeIdAndName << "scopeID=" << scope->getScopeID() << " ";
    if( const Program* prog = dynamic_cast<const Program*>( scope ) )
    {
        scopeIdAndName << prog->getInputFunctionName();
    }

    size_t            offset = (char*)ptr - m_objectData;
    std::stringstream ss;
    ss << hexString( offset, 4 ) << "         " << std::setw( 6 ) << scope->getRecordSize();
    printRow( ss.str(), type, scopeIdAndName.str(), "" );
}

void LayoutPrinter::printMemberVariable( char*              scope,
                                         const void*        p,
                                         size_t             size,
                                         const std::string& name,
                                         const std::string& type,
                                         const std::string& value )
{
    const char* ptr = static_cast<const char*>( p );
    if( ptr != m_nextAddress )
        printGap( m_nextAddress, ptr );
    m_nextAddress = ptr + size;

    printRow( addressString( scope, ptr, size ), "", "(m) " + name, valuePairString( type, value ) );
}

void LayoutPrinter::printGap( const char* from, const char* to )
{
    RT_ASSERT( from != nullptr );
    RT_ASSERT( to != nullptr );
    if( from < to )
    {
        size_t size = to - from;
        printRow( addressString( nullptr, from, size ), "", "--- padding ---", "" );  // use byteString( from, size ) to see padding
    }
    else
    {
        size_t size = from - to;
        printRow( addressString( nullptr, from, size ), "", "!!! backward !!!", "" );
    }
}

std::string LayoutPrinter::addressString( const char* base, const char* ptr, size_t size )
{
    size_t            offset = ptr - m_objectData;
    std::stringstream ss;
    ss << "  " << hexString( offset, 4 );
    if( base )
        ss << "  " << hexString( ptr - base, 2 );
    else
        ss << "      ";
    ss << " " << std::setw( 6 ) << size;
    return ss.str();
}

void LayoutPrinter::printRow( const std::string& c1, const std::string& c2, const std::string& c3, const std::string& c4 )
{
    printShortColumnEntry( c1 );
    printShortColumnEntry( c2 );
    printColumnEntry( c3 );
    printColumnEntry( c4 );
    m_out << '\n';
}

void LayoutPrinter::printLexicalScopeSpacer( char* p )
{
    if( m_nextAddress != p )
    {
        printTableSpacer();
        printGap( m_nextAddress, p );
        m_nextAddress = p;
    }
}

void LayoutPrinter::printLexicalScopeDynamicVariableTableOffset( const LexicalScope* scope, char* p )
{
    cort::LexicalScopeRecord* ls = scope->getObjectRecord<cort::LexicalScopeRecord>();
    printTableSpacer();
    printScopeHeader( ls, scope );
    printMemberVariable( p, &ls->dynamicVariableTable, sizeof( ls->dynamicVariableTable ), "dynamicVariableTable",
                         "Offset", offsetString( ls->dynamicVariableTable ) );
}

void LayoutPrinter::printAcceleration( const LexicalScope* scope, char* p )
{
    const Acceleration* accel = static_cast<const Acceleration*>( scope );
    if( accel->hasMotionAabbs() )
    {
        cort::MotionAccelerationRecord* mar = scope->getObjectRecord<cort::MotionAccelerationRecord>();
        printMemberVariable( p, &mar->timeBegin, sizeof( mar->timeBegin ), "timeBegin", "float", to_string( mar->timeBegin ) );
        printMemberVariable( p, &mar->timeEnd, sizeof( mar->timeEnd ), "timeEnd", "float", to_string( mar->timeEnd ) );
        printMemberVariable( p, &mar->motionSteps, sizeof( mar->motionSteps ), "motionSteps", "int", to_string( mar->motionSteps ) );
        printMemberVariable( p, &mar->motionStride, sizeof( mar->motionStride ), "motionStride", "int",
                             to_string( mar->motionStride ) );
    }
}

void LayoutPrinter::printGlobalScope( const LexicalScope* scope, char* p )
{
    cort::GlobalScopeRecord* gs  = scope->getObjectRecord<cort::GlobalScopeRecord>();
    size_t                   nep = scope->getContext()->getGlobalScope()->getAllEntryPointCount();
    size_t                   nrt = scope->getContext()->getRayTypeCount();
    size_t                   n   = std::max( nep, nrt );
    for( size_t i = 0; i < n; ++i )
    {
        if( i < nep )
            printMemberVariable( p, &gs->programs[i].raygen, sizeof( gs->programs[i].raygen ),
                                 "programs[" + ::to_string( i ) + "].raygen", "Program", offsetString( gs->programs[i].raygen ) );
        if( i < nep )
            printMemberVariable( p, &gs->programs[i].exception, sizeof( gs->programs[i].exception ),
                                 "programs[" + ::to_string( i ) + "].exception", "Program",
                                 offsetString( gs->programs[i].exception ) );
        if( i < nrt )
            printMemberVariable( p, &gs->programs[i].miss, sizeof( gs->programs[i].miss ),
                                 "programs[" + ::to_string( i ) + "].miss", "Program", offsetString( gs->programs[i].miss ) );
    }
}

void LayoutPrinter::printGeometry( const LexicalScope* scope, char* p )
{
    cort::GeometryRecord* g = scope->getObjectRecord<cort::GeometryRecord>();
    printMemberVariable( p, &g->indexOffset, sizeof( g->indexOffset ), "indexOffset", "uint", ::to_string( g->indexOffset ) );
    printMemberVariable( p, &g->intersectOrAttribute, sizeof( g->intersectOrAttribute ), "intersect", "Program",
                         offsetString( g->intersectOrAttribute ) );
    printMemberVariable( p, &g->aabb, sizeof( g->aabb ), "aabb", "Program", offsetString( g->aabb ) );
    printMemberVariable( p, &g->attributeKind, sizeof( g->attributeKind ), "attributeKind", "uint", ::to_string( g->attributeKind ) );
}

void LayoutPrinter::printGeometryTriangles( const GeometryTriangles* gt, char* p )
{
    cort::GeometryRecord* g = gt->getObjectRecord<cort::GeometryRecord>();
    printMemberVariable( p, &g->indexOffset, sizeof( g->indexOffset ), "indexOffset", "uint", ::to_string( g->indexOffset ) );
    printMemberVariable( p, &g->intersectOrAttribute, sizeof( g->intersectOrAttribute ), "attribute", "Program",
                         offsetString( g->intersectOrAttribute ) );
    printMemberVariable( p, &g->aabb, sizeof( g->aabb ), "aabb", "Program", offsetString( g->aabb ) );
    printMemberVariable( p, &g->attributeKind, sizeof( g->attributeKind ), "attributeKind", "uint", ::to_string( g->attributeKind ) );
}

void LayoutPrinter::printGeometryInstance( const LexicalScope* scope, char* p )
{
    cort::GeometryInstanceRecord* gi = scope->getObjectRecord<cort::GeometryInstanceRecord>();
    printMemberVariable( p, &gi->geometry, sizeof( gi->geometry ), "geometry", "Geometry", offsetString( gi->geometry ) );
    printMemberVariable( p, &gi->numMaterials, sizeof( gi->numMaterials ), "numMaterials", "int", offsetString( gi->numMaterials ) );
    const GeometryInstance* gis = static_cast<const GeometryInstance*>( scope );
    for( int i = 0; i < gis->getMaterialCount(); ++i )
    {
        printMemberVariable( p, &gi->materials[i], sizeof( gi->materials[0] ), "materials[" + ::to_string( i ) + "]",
                             "Material", offsetString( gi->materials[i] ) );
    }
}

void LayoutPrinter::printAbstractGroup( const LexicalScope* scope, char* p )
{
    cort::AbstractGroupRecord* g = scope->getObjectRecord<cort::AbstractGroupRecord>();
    printMemberVariable( p, &g->traverse, sizeof( g->traverse ), "traverse", "Program", offsetString( g->traverse ) );
    printMemberVariable( p, &g->bounds, sizeof( g->bounds ), "bounds", "Program", offsetString( g->bounds ) );
    printMemberVariable( p, &g->traversableId, sizeof( g->traversableId ), "traversableId", "int", to_string( g->traversableId ) );
    printMemberVariable( p, &g->accel, sizeof( g->accel ), "accel", "Acceleration", offsetString( g->accel ) );
    printMemberVariable( p, &g->children, sizeof( g->children ), "children", "Buffer", to_string( g->children ) );
}

static std::string devicePointersForInstanceDescriptors( const Group* group, const std::vector<unsigned int>& devs )
{
    std::ostringstream text;
    bool               first = true;
    for( unsigned int allDevicesIndex : devs )
    {
        if( !first )
            text << ' ';
        text << hexString( reinterpret_cast<size_t>( group->getInstanceDescriptorTableDevicePtr( allDevicesIndex ) ), 16 );
    }
    return text.str();
}

static std::string matrix4x3ToString( const float transform[12] )
{
    std::ostringstream text;
    text << "Matrix4x3     : {";
    for( int r = 0; r < 3; ++r )
    {
        for( int c = 0; c < 4; ++c )
        {
            text << ' ' << transform[r * 4 + c];
        }
        if( r < 2 )
            text << ',';
    }
    text << " }";
    return text.str();
}

static std::string traversableHandlesForGroupChild( const Group* group, unsigned int child, const std::vector<unsigned int>& devs )
{
    std::ostringstream text;
    using DD   = InstanceDescriptorHost::DeviceDependent;
    bool first = true;
    for( unsigned int allDevicesIndex : devs )
    {
        const DD dd = group->getInstanceDescriptorDeviceDependent( child, allDevicesIndex );
        if( !first )
            text << ' ';
        text << hexString( dd.accelOrTraversableHandle, 16 );
        first = false;
    }
    return text.str();
}

static std::string instanceOffsetString( size_t offset, size_t size )
{
    std::stringstream ss;
    ss << "          " << hexString( offset, 2 ) << ' ' << std::setw( 6 ) << size;
    return ss.str();
}

void LayoutPrinter::printInstanceDescriptorRow( size_t offset, size_t size, const std::string& text, const std::string& value )
{
    printRow( instanceOffsetString( offset, size ), "", "      " + text, value );
}

void LayoutPrinter::printGroup( const LexicalScope* scope, char* p, const std::vector<unsigned int>& devs )
{
    printAbstractGroup( scope, p );
    printLexicalScopeVariables( scope, p );
    const Group* group = static_cast<const Group*>( scope );
    if( !m_useRtxDataModel || group->getChildCount() == 0 )
        return;

    using DI = InstanceDescriptorHost::DeviceIndependent;
    printRow( "", "", "", "" );
    printRow( "", "Associated Instance Data", "",
              valuePairString( "Device ptr", devicePointersForInstanceDescriptors( group, devs ) ) );
    for( unsigned int i = 0; i < group->getChildCount(); ++i )
    {
        printRow( "  " + to_string( i ), "", "    Instance", "" );
        const DI di = group->getInstanceDescriptor( i );

        printInstanceDescriptorRow( offsetof( DI, transform ), sizeof DI::transform, "Transform", matrix4x3ToString( di.transform ) );
        size_t offset = offsetof( DI, transform ) + sizeof( di.transform );
        printInstanceDescriptorRow( offset, 3, "Instance id: ", valuePairString( "int", to_string( di.instanceId ) ) );
        offset += 3;
        printInstanceDescriptorRow( offset, 1, "Mask:", valuePairString( "Mask", hexString( di.mask, 2 ) ) );
        offset += 1;
        printInstanceDescriptorRow( offset, 3, "Instance offset:", valuePairString( "Offset", to_string( di.instanceOffset ) ) );
        offset += 3;
        printInstanceDescriptorRow( offset, 1, "Flags:", valuePairString( "Flags", hexString( di.flags, 2 ) ) );
        printInstanceDescriptorRow( offsetof( InstanceDescriptorHost, dd ), sizeof( InstanceDescriptorHost::dd ),
                                    "Traversable [per-device]",
                                    valuePairString( "TravHandle", traversableHandlesForGroupChild( group, i, devs ) ) );
    }
}

void LayoutPrinter::printMaterial( const LexicalScope* scope, char* p )
{
    cort::MaterialRecord* m   = scope->getObjectRecord<cort::MaterialRecord>();
    size_t                nrt = scope->getContext()->getRayTypeCount();
    for( size_t i = 0; i < nrt; ++i )
    {
        printMemberVariable( p, &m->programs[i].closestHit, sizeof( m->programs[i].closestHit ),
                             "programs[" + ::to_string( i ) + "].closestHit", "Program",
                             offsetString( m->programs[i].closestHit ) );
        printMemberVariable( p, &m->programs[i].anyHit, sizeof( m->programs[i].anyHit ),
                             "programs[" + ::to_string( i ) + "].anyHit", "Program", offsetString( m->programs[i].anyHit ) );
    }
}

void LayoutPrinter::printProgram( const LexicalScope* scope, char* p )
{
    cort::ProgramRecord* program = scope->getObjectRecord<cort::ProgramRecord>();
    printMemberVariable( p, &program->programID, sizeof( program->programID ), "programID", "int", ::to_string( program->programID ) );
}

void LayoutPrinter::printSelector( const LexicalScope* scope, char* p )
{
    cort::SelectorRecord* s = scope->getObjectRecord<cort::SelectorRecord>();
    printMemberVariable( p, &s->traverse, sizeof( s->traverse ), "traverse", "Program", offsetString( s->traverse ) );
    printMemberVariable( p, &s->bounds, sizeof( s->bounds ), "bounds", "Program", offsetString( s->bounds ) );
    printMemberVariable( p, &s->traversableId, sizeof( s->traversableId ), "traversableId", "int", to_string( s->traversableId ) );
    printMemberVariable( p, &s->children, sizeof( s->children ), "children", "Buffer", std::to_string( s->children ) );
}

void LayoutPrinter::printTransform( const LexicalScope* scope, char* p )
{
    if( !m_useRtxDataModel )
    {
        cort::TransformRecord* t = scope->getObjectRecord<cort::TransformRecord>();
        printMemberVariable( p, &t->traverse, sizeof( t->traverse ), "traverse", "Program", offsetString( t->traverse ) );
        printMemberVariable( p, &t->bounds, sizeof( t->bounds ), "bounds", "Program", offsetString( t->bounds ) );
        printMemberVariable( p, &t->traversableId, sizeof( t->traversableId ), "traversableId", "int", to_string( t->traversableId ) );
        printMemberVariable( p, &t->child, sizeof( t->child ), "child", "GraphNode", offsetString( t->child ) );
        printMemberVariable( p, &t->matrix, sizeof( t->matrix ), "matrix", "Matrix4x4", ::to_string( t->matrix ) );
        printMemberVariable( p, &t->ilwerse_matrix, sizeof( t->ilwerse_matrix ), "ilwerse_matrix", "Matrix4x4",
                             ::to_string( t->ilwerse_matrix ) );
        printMemberVariable( p, &t->motionData, sizeof( t->motionData ), "motionData", "Buffer", ::to_string( t->motionData ) );
    }
    else if( managedObjectCast<const Transform>( scope )->requiresDirectTraversable() )
    {
        cort::GraphNodeRecord* t = scope->getObjectRecord<cort::GraphNodeRecord>();
        printMemberVariable( p, &t->traverse, sizeof( t->traverse ), "traverse", "Program", offsetString( t->traverse ) );
        printMemberVariable( p, &t->bounds, sizeof( t->bounds ), "bounds", "Program", offsetString( t->bounds ) );
        printMemberVariable( p, &t->traversableId, sizeof( t->traversableId ), "traversableId", "int", to_string( t->traversableId ) );
    }
}

void LayoutPrinter::printLexicalScopeVariables( const LexicalScope* scope, char* p )
{
    // Now print variables belonging to this scope
    unsigned int lwar = scope->getVariableCount();
    for( unsigned int i = 0; i < lwar; ++i )
        printBoundVariable( p, scope->getVariableByIndex( i ) );

    // Print the dynamic variable table - offset from the beginning of the dylwarTable memory, not from the scope
    std::ostringstream        tbl;
    cort::LexicalScopeRecord* ls = scope->getObjectRecord<cort::LexicalScopeRecord>();
    unsigned short*           dyn_offsets =
        reinterpret_cast<unsigned short*>( m_tableManager->getDynamicVariableTableHostPointer() + ls->dynamicVariableTable );
    bool first = true;
    int  ntok  = scope->getVariableCount();
    for( int i = 0; i < ntok; ++i )
    {
        unsigned short token  = cort::getUnmarkedVariableTokenId( dyn_offsets++ );
        unsigned short offset = *dyn_offsets++;
        if( !first )
            tbl << ", ";
        first                   = false;
        const std::string& name = m_objectManager->getVariableNameForToken( token );
        tbl << "[" << i << "]" << name << ":" << offsetString( offset, 2 );
    }
    if( first )
        tbl << "(empty)";
    //printMemberVariable( p, dyn_offsets, sizeof( unsigned short ) * ntok, "dynamic variable tables",
    //                     "ushort[" + ::to_string( ntok ) + "]", tbl.str() );
    const char* ptr = static_cast<const char*>( p );
    //std::string s = addressString(p, ptr, sizeof(unsigned short) * ntok);
    std::string s;
    {
        unsigned int offset   = ls->dynamicVariableTable;
        bool         hasValue = true;
        if( offset == (unsigned int)( -1 ) )
            hasValue = false;
        std::stringstream ss;
        if( hasValue )
            ss << "**" << hexString( offset, 4 );
        else
            ss << "**<none>";

        if( p )
            ss << "  " << hexString( ptr - p, 2 );
        else
            ss << "      ";
        ss << " " << std::setw( 6 ) << sizeof( unsigned short ) * ntok;
        s = ss.str();
        s += "**";
    }

    printRow( s, "", "(m) dynamic variable table", valuePairString( "ushort[" + ::to_string( ntok ) + "]", tbl.str() ) );
}

void LayoutPrinter::printLexicalScope( const LexicalScope* scope, const std::vector<unsigned int>& devs )
{
    char* p = scope->getObjectRecord<char>();
    printLexicalScopeSpacer( p );
    printLexicalScopeDynamicVariableTableOffset( scope, p );

    // Group handles these by itself so it can print instance data afterwards
    bool variablesPrinted = false;
    switch( scope->getClass() )
    {
        case RT_OBJECT_ACCELERATION:
            printAcceleration( scope, p );
            break;

        case RT_OBJECT_GLOBAL_SCOPE:
            printGlobalScope( scope, p );
            break;

        case RT_OBJECT_GEOMETRY:
            if( const GeometryTriangles* gt = dynamic_cast<const GeometryTriangles*>( scope ) )
                printGeometryTriangles( gt, p );
            else
                printGeometry( scope, p );
            break;

        case RT_OBJECT_GEOMETRY_INSTANCE:
            printGeometryInstance( scope, p );
            break;

        case RT_OBJECT_GEOMETRY_GROUP:
            printAbstractGroup( scope, p );
            break;

        case RT_OBJECT_GROUP:
            printGroup( scope, p, devs );
            variablesPrinted = true;
            break;

        case RT_OBJECT_MATERIAL:
            printMaterial( scope, p );
            break;

        case RT_OBJECT_PROGRAM:
            printProgram( scope, p );
            break;

        case RT_OBJECT_SELECTOR:
            printSelector( scope, p );
            break;

        case RT_OBJECT_TRANSFORM:
            printTransform( scope, p );
            break;

        default:
            RT_ASSERT( !!!"Illegal scope" );
            break;
    }

    if( !variablesPrinted )
        printLexicalScopeVariables( scope, p );
}

void LayoutPrinter::printBoundVariable( char* objPtr, Variable* v )
{
    RT_ASSERT( v );
    char*                    varPtr = objPtr + v->getScopeOffset();
    std::string              address_string;
    std::vector<std::string> textRows;

    VariableType vt       = v->getType();
    unsigned int num_elem = vt.numElements();
    size_t       size     = vt.computeSize();

    switch( vt.baseType() )
    {
        case VariableType::Float:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss, name;
            name << "float[" << num_elem << "]";
            float* fdata = reinterpret_cast<float*>( varPtr );
            for( size_t i = 0; i < num_elem; ++i )
                ss << spacer( i ) << *fdata++;
            textRows.push_back( valuePairString( name.str(), ss.str() ) );
        }
        break;

        case VariableType::Int:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss, name;
            name << "int[" << num_elem << "]";
            int* idata = reinterpret_cast<int*>( varPtr );
            for( size_t i = 0; i < num_elem; ++i )
                ss << spacer( i ) << idata[i];
            textRows.push_back( valuePairString( name.str(), ss.str() ) );
        }
        break;

        case VariableType::Uint:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss, name;
            name << "uint[" << num_elem << "]";
            unsigned int* udata = reinterpret_cast<unsigned int*>( varPtr );
            for( size_t i = 0; i < num_elem; ++i )
                ss << spacer( i ) << udata[i];
            textRows.push_back( valuePairString( name.str(), ss.str() ) );
        }
        break;

        case VariableType::LongLong:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss, name;
            name << "longlong[" << num_elem << "]";
            long long* idata = reinterpret_cast<long long*>( varPtr );
            for( size_t i = 0; i < num_elem; ++i )
                ss << spacer( i ) << idata[i];
            textRows.push_back( valuePairString( name.str(), ss.str() ) );
        }
        break;

        case VariableType::ULongLong:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss, name;
            name << "ulonglong[" << num_elem << "]";
            unsigned long long* udata = reinterpret_cast<unsigned long long*>( varPtr );
            for( size_t i = 0; i < num_elem; ++i )
                ss << spacer( i ) << udata[i];
            textRows.push_back( valuePairString( name.str(), ss.str() ) );
        }
        break;

        case VariableType::Buffer:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss;
            int*              idata = reinterpret_cast<int*>( varPtr );
            ss << idata[0];  // Just the id is stored
            textRows.push_back( valuePairString( "Buffer", ss.str() ) );
        }
        break;

        case VariableType::DemandBuffer:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss;
            int*              idata = reinterpret_cast<int*>( varPtr );
            ss << idata[0];  // Just the id is stored
            textRows.push_back( valuePairString( "DemandBuffer", ss.str() ) );
        }
        break;

        case VariableType::GraphNode:
        {
            GraphNode* node = v->getGraphNode();
            address_string  = node ? addressString( objPtr, varPtr, size ) : "(invalid)";
            std::stringstream ss;
            unsigned int*     idata = reinterpret_cast<unsigned int*>( varPtr );
            for( size_t i = 0; i < num_elem; ++i )
                ss << spacer( i ) << offsetString( idata[i] );
            textRows.push_back( valuePairString( node ? getNameForClass( node->getClass() ) : "unknown graphnode", ss.str() ) );
        }
        break;

        case VariableType::Program:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss;
            char*             ls = v->getProgram()->getObjectRecord<char>();
            ss << offsetString( static_cast<unsigned int>( std::ptrdiff_t( ls - m_objectData ) ) );
            // as only callable Programs get here, we should print the id too
            unsigned int* idata = reinterpret_cast<unsigned int*>( varPtr );
            ss << ", programID " << idata[0];
            textRows.push_back( valuePairString( "Program", ss.str() ) );
        }
        break;

        case VariableType::TextureSampler:
        {
            address_string = addressString( objPtr, varPtr, size );
            std::stringstream ss;
            int*              idata = reinterpret_cast<int*>( varPtr );
            ss << idata[0];  // Just the id is stored
            textRows.push_back( valuePairString( "TextureSampler", ss.str() ) );
        }
        break;

        case VariableType::UserData:
        {
            address_string = addressString( objPtr, varPtr, size );
            textRows.push_back( valuePairString( "User Data", byteString( varPtr, size ) ) );
        }
        break;

        default:
            textRows.push_back( "UNKNOWN" );
    }

    if( varPtr != m_nextAddress )
        printGap( m_nextAddress, varPtr );
    m_nextAddress = varPtr + size;

    std::vector<std::string>::iterator iter = textRows.begin();
    printRow( address_string, "", "(v) " + v->getName(), *iter );
    for( ++iter; iter != textRows.end(); ++iter )
    {
        printRow( "", "", "", *iter );
    }
}

void LayoutPrinter::printColumnEntry( const std::string& entry )
{

    m_out << "| " << std::left << std::setw( COL_WIDTH - 2 ) << entry;
}

void LayoutPrinter::printShortColumnEntry( const std::string& entry )
{

    m_out << "| " << std::left << std::setw( SHORT_COL_WIDTH ) << entry;
}

void LayoutPrinter::printTableSpacer()
{
    m_out << std::setfill( '-' ) << std::setw( COL_WIDTH * 3 + 20 + 1 ) << "" << std::setfill( ' ' ) << '\n';
}

void LayoutPrinter::dumpMemory( const void* ptr, const size_t size )
{
    const int*          iptr = reinterpret_cast<const int*>( ptr );
    const float*        fptr = reinterpret_cast<const float*>( ptr );
    const unsigned int* uptr = reinterpret_cast<const unsigned int*>( ptr );

    m_out << std::right << std::setw( 2 * sizeof( void* ) + 1 ) << "Address" << std::right << std::setw( 20 + 1 )
          << "float" << std::right << std::setw( 20 + 1 ) << "hex uint" << std::right << std::setw( 20 + 1 ) << "uint"
          << std::right << std::setw( 20 + 1 ) << "int\n"
          << std::setfill( '-' ) << std::setw( 93 ) << "-" << std::setfill( ' ' ) << '\n';

    for( unsigned int i = 0; i < size; i += 4 )
    {
        int          the_int   = *iptr++;
        unsigned int the_uint  = *uptr++;
        float        the_float = *fptr++;
        m_out << std::right << std::setw( 2 * sizeof( void* ) + 1 ) << std::hex << i << " " << std::right
              << std::setw( 20 ) << std::setprecision( 8 ) << std::fixed << the_float << " " << std::right
              << std::setw( 20 ) << stringf( "%08x", the_uint ) << " " << std::right << std::setw( 20 ) << std::dec
              << the_uint << " " << std::right << std::setw( 20 ) << std::dec << the_int << std::endl;
    }
}

void LayoutPrinter::printBufferHeader( int id, const std::vector<unsigned int>& devs )
{
    auto bhOffsString = []( int index, size_t offset ) {
        return hexString( index * sizeof( cort::Buffer ) + offset, 4 );
    };

    const cort::Buffer::DeviceIndependent* diBuf = m_tableManager->getBufferHeaderDiHostPointerReadOnly( id );
    m_out << bhOffsString( id, 0U ) << "\tbuf id " << std::right << std::setw( 3 ) << id << ":\n"
          << bhOffsString( id, offsetof( cort::Buffer, di.size ) ) << "\t\t(width, height, depth)  = (" << diBuf->size.x
          << ", " << diBuf->size.y << ", " << diBuf->size.z << ")\n"
          << bhOffsString( id, offsetof( cort::Buffer, di.pageSize ) ) << "\t\tpage (w, h, d)          = ("
          << diBuf->pageSize.x << ", " << diBuf->pageSize.y << ", " << diBuf->pageSize.z << ")\n";

    for( unsigned int devIndex : devs )
    {
        const cort::Buffer::DeviceDependent* ddBuf = m_tableManager->getBufferHeaderDdHostPointerReadOnly( id, devIndex );
        m_out << bhOffsString( id, offsetof( cort::Buffer, dd.data ) )
              << "\t\tdata[per device]        = " << hexString( reinterpret_cast<size_t>( ddBuf->data ) ) << '\n'
              << bhOffsString( id, offsetof( cort::Buffer, dd.texUnit ) ) << "\t\ttexUnit[per device]     = ";

        if( ddBuf->texUnit == -3 )
        {
            m_out << "          - ";
        }
        else
        {
            m_out << std::right << std::setw( 10 ) << ddBuf->texUnit << ' ';
        }
        m_out << '\n';
    }
}

static inline const char* formatStr( int format )
{
    switch( format )
    {
        case cort::TEX_FORMAT_UNSIGNED_BYTE1:
            return "ubyte1";
        case cort::TEX_FORMAT_UNSIGNED_SHORT1:
            return "ushort1";
        case cort::TEX_FORMAT_UNSIGNED_INT1:
            return "uint1";
        case cort::TEX_FORMAT_BYTE1:
            return "byte1 ";
        case cort::TEX_FORMAT_SHORT1:
            return "short1";
        case cort::TEX_FORMAT_INT1:
            return "int1";
        case cort::TEX_FORMAT_FLOAT1:
            return "float1";

        case cort::TEX_FORMAT_UNSIGNED_BYTE2:
            return "ubyte2";
        case cort::TEX_FORMAT_UNSIGNED_SHORT2:
            return "ushort2";
        case cort::TEX_FORMAT_UNSIGNED_INT2:
            return "uint2";
        case cort::TEX_FORMAT_BYTE2:
            return "byte2";
        case cort::TEX_FORMAT_SHORT2:
            return "short2";
        case cort::TEX_FORMAT_INT2:
            return "int2";
        case cort::TEX_FORMAT_FLOAT2:
            return "float2";

        case cort::TEX_FORMAT_UNSIGNED_BYTE4:
            return "ubyte4";
        case cort::TEX_FORMAT_UNSIGNED_SHORT4:
            return "ushort4";
        case cort::TEX_FORMAT_UNSIGNED_INT4:
            return "uint4";
        case cort::TEX_FORMAT_BYTE4:
            return "byte4";
        case cort::TEX_FORMAT_SHORT4:
            return "short4";
        case cort::TEX_FORMAT_INT4:
            return "int4";
        case cort::TEX_FORMAT_FLOAT4:
            return "float4";
    }
    return "<unknown format>";
}

static inline const char* wrapModeStr( int mode )
{
    switch( mode )
    {
        case cort::TEX_WRAP_REPEAT:
            return "repeat";
        case cort::TEX_WRAP_CLAMP_TO_EDGE:
            return "clamp";
        case cort::TEX_WRAP_MIRROR:
            return "mirror";
        case cort::TEX_WRAP_CLAMP_TO_BORDER:
            return "border";
    }
    return "<unknown wrap mode>";
}

static inline const char* boolStr( int b )
{
    if( b )
        return "true";
    else
        return "false";
}

static inline const char* filterStr( int f )
{
    if( f )
        return "linear";
    else
        return "nearest";
}

void LayoutPrinter::printTextureHeader( int id, const std::vector<unsigned int>& devs )
{
    auto thOffsString = []( int index, size_t offset ) {
        return hexString( index * sizeof( cort::TextureSamplerHost ) + offset, 4 );
    };

    const cort::TextureSamplerHost::DeviceIndependent* ts = m_tableManager->getTextureHeaderDiHostPointerReadOnly( id );
    m_out << thOffsString( id, 0 ) << "\ttex id " << std::right << std::setw( 3 ) << id << ":\n";
    m_out << thOffsString( id, offsetof( cort::TextureSamplerHost, di.width ) ) << "\t\t(width, depth, height)  = ("
          << ts->width << ", " << ts->height << ", " << ts->depth << ")\n";
    m_out << thOffsString( id, offsetof( cort::TextureSamplerHost, di.mipLevels ) )
          << "\t\tmipLevels               = " << ts->mipLevels << "\n";
    std::string bitfieldOffset = thOffsString( id, offsetof( cort::TextureSamplerHost, di.mipLevels )
                                                       + sizeof( cort::TextureSamplerHost::di.mipLevels ) );
    m_out << bitfieldOffset << "\t\tformat                  = " << formatStr( ts->format ) << '\n';
    m_out << bitfieldOffset << "\t\twrapModes               = " << wrapModeStr( ts->wrapMode0 ) << ", "
          << wrapModeStr( ts->wrapMode1 ) << ", " << wrapModeStr( ts->wrapMode2 ) << '\n';
    m_out << bitfieldOffset << "\t\tnormCoord               = " << boolStr( ts->normCoord ) << '\n';
    m_out << bitfieldOffset << "\t\tfilterMode              = " << filterStr( ts->filterMode ) << '\n';
    m_out << bitfieldOffset << "\t\tnormRet                 = " << boolStr( ts->normRet ) << '\n';
    m_out << bitfieldOffset << "\t\tisDemandLoad            = " << boolStr( ts->isDemandLoad ) << '\n';

    // Demand texture fields
    m_out << thOffsString( id, offsetof( cort::TextureSamplerHost, di.mipTailFirstLevel ) )
          << "\t\tmipTailFirstLevel           = " << ts->mipTailFirstLevel << "\n";
    m_out << thOffsString( id, offsetof( cort::TextureSamplerHost, di.ilwAnisotropy ) )
          << "\t\tilwAnisotropy               = " << ts->ilwAnisotropy << "\n";
    bitfieldOffset = thOffsString( id, offsetof( cort::TextureSamplerHost, di.ilwAnisotropy )
                                           + sizeof( cort::TextureSamplerHost::di.ilwAnisotropy ) );
    m_out << bitfieldOffset << "\t\ttileWidth               = " << ts->tileWidth << '\n';
    m_out << bitfieldOffset << "\t\ttileHeight              = " << ts->tileHeight << '\n';
    m_out << bitfieldOffset << "\t\ttileGutterWidth         = " << ts->tileGutterWidth << '\n';
    m_out << bitfieldOffset << "\t\tisInitialized           = " << boolStr( ts->isInitialized ) << '\n';
    m_out << bitfieldOffset << "\t\tisSquarePowerOfTwo      = " << boolStr( ts->isSquarePowerOfTwo ) << '\n';
    m_out << bitfieldOffset << "\t\tmipmapFilterMode        = " << ts->mipmapFilterMode << '\n';

    // Device-dependent fields
    m_out << thOffsString( id, offsetof( cort::TextureSamplerHost, dd.texref ) ) << "\t\ttexref                  = ";
    for( unsigned int devIndex : devs )
        m_out << m_tableManager->getTextureHeaderDdHostPointerReadOnly( id, devIndex )->texref << ' ';
    m_out << '\n';
    m_out << thOffsString( id, offsetof( cort::TextureSamplerHost, dd.swptr ) ) << "\t\tsw ptr                  = ";
    for( unsigned int devIndex : devs )
        m_out << stringf( "%p", m_tableManager->getTextureHeaderDdHostPointerReadOnly( id, devIndex )->swptr ) << ' ';
    m_out << '\n';
}

void LayoutPrinter::printProgramHeader( int id, const std::vector<unsigned int>& devs )
{
    auto phOffsString = []( int index, size_t offset ) {
        return hexString( index * sizeof( cort::ProgramHeader ) + offset, 4 );
    };

    const cort::ProgramHeader::DeviceIndependent* diBuf = m_tableManager->getProgramHeaderDiHostPointerReadOnly( id );
    m_out << phOffsString( id, 0 ) << "\tprogram id " << std::right << std::setw( 3 ) << id << ":\n";
    m_out << phOffsString( id, offsetof( cort::ProgramHeader, di.programOffset ) )
          << "\t\tprogramOffset                  = " << offsetString( diBuf->programOffset ) << '\n';
    m_out << phOffsString( id, offsetof( cort::ProgramHeader, dd.canonicalProgramID ) )
          << "\t\tcanonicalProgramID[per device] = ";
    for( unsigned int devIndex : devs )
        m_out << m_tableManager->getProgramHeaderDdHostPointerReadOnly( id, devIndex )->canonicalProgramID << ' ';
    m_out << '\n';
}

void LayoutPrinter::printTraversableHandles( int id, const std::vector<unsigned int>& devs )
{
    auto thOffsString = []( int index, size_t offset ) {
        return hexString( index * sizeof( cort::TraversableHeader ) + offset, 4 );
    };

    m_out << thOffsString( id, offsetof( cort::TraversableHeader, traversable ) )
          << "\t\ttraversableHandle[per device] = ";
    for( unsigned int devIndex : devs )
        m_out << hexString( m_tableManager->getTraversableHeaderDdHostPointerReadOnly( id, devIndex )->traversable, 16 ) << ' ';
    m_out << '\n';
}
