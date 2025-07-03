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

#include <Util/NodegraphPrinter.h>

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/Geometry.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/GlobalScope.h>
#include <Objects/Group.h>
#include <Objects/LexicalScope.h>
#include <Objects/Material.h>
#include <Objects/PostprocessingStage.h>
#include <Objects/Program.h>
#include <Objects/Selector.h>
#include <Objects/Transform.h>
#include <Objects/Variable.h>

#include <prodlib/exceptions/Assert.h>

using namespace optix;
using namespace prodlib;

static std::string hexString( size_t u, unsigned int width = 8 )
{
    std::stringstream ss;
    ss << "0x" << std::setw( width ) << std::setfill( '0' ) << std::hex << std::uppercase << u;
    return ss.str();
}

NodegraphPrinter::NodegraphPrinter( ObjectManager* objectManager, ProgramManager* programManager, BindingManager* bindingManager )
    : m_objectManager( objectManager )
    , m_programManager( programManager )
    , m_bindingManager( bindingManager )
{
}


void NodegraphPrinter::setPrintReferences( bool val )
{
    m_printReferences = val;
}

std::string NodegraphPrinter::str()
{
    return out.str();
}

void NodegraphPrinter::run()
{
    // Print header
    out << "digraph g {\n";
    out << "graph [\n";
    out << "];\n";
    out << "node [\n";
    out << "fontsize = \"12\"\n";
    out << "labeljust = \"l\"\n";
    out << "]\n";
    out << "edge [\n";
    out << "]\n";

    emitBindingManager();

    // Emit scopes
    for( const auto& scope : m_objectManager->getLexicalScopes() )
        emitScope( scope );

    // Emit canonical programs
    for( const auto& cp : m_programManager->getCanonicalProgramMap() )
        emitCanonicalProgram( cp );

    // Emit the edges that were aclwmulated above
    out << edges.str();

    out << "}\n";
}

void NodegraphPrinter::emitScope( const LexicalScope* scope )
{
    // Print header and label
    int id = scope->getScopeID();
    out << "s" << id << " [\n";
    out << "shape = \"record\"\n";
    std::string className = getNameForClass( scope->getClass() );
    if( dynamic_cast<const GeometryTriangles*>( scope ) != nullptr )
        className = "GeometryTriangles";
    out << "label = \"{" << className << ": " << id << "\\n"
        << "offs = " << ( scope->recordIsAllocated() ? hexString( scope->getRecordOffset(), 4 ) : "(unallocated)" ) << "\\n"
        << "addr = " << scope << "\\n";
    if( scope->getClass() == RT_OBJECT_POSTPROCESSINGSTAGE )
        out << "name = " << dynamic_cast<const PostprocessingStage*>( scope )->getName() << "\\n";
    if( scope->getClass() == RT_OBJECT_PROGRAM )
        out << "progid:" << dynamic_cast<const Program*>( scope )->getId() << "\\n";
    if( scope->getClass() == RT_OBJECT_GEOMETRY )
    {
        out << "prim count: " << dynamic_cast<const Geometry*>( scope )->getPrimitiveCount() << "\\n";
        out << "prime offset: " << dynamic_cast<const Geometry*>( scope )->getPrimitiveIndexOffset() << "\\n";
        out << "motion steps: " << dynamic_cast<const Geometry*>( scope )->getMotionSteps() << "\\n";
    }
    if( scope->getClass() == RT_OBJECT_GROUP || scope->getClass() == RT_OBJECT_GEOMETRY_GROUP )
    {
        const AbstractGroup* grp = dynamic_cast<const AbstractGroup*>( scope );
        out << "SBT Index: " << ( grp->m_SBTIndex ? std::to_string( grp->getSBTRecordIndex() ) : "none" ) << "\\n";
    }


    // Properties
    out << " | <props> ";
    out << "attached count = " << scope->m_attachment.count() << "\\n";
    if( scope->getClass() == RT_OBJECT_GLOBAL_SCOPE )
    {
    }
    else if( isGraphNode( scope->getClass() ) )
    {
        const GraphNode* gn = dynamic_cast<const GraphNode*>( scope );
        emitGraphProperty( "xform height", gn->m_transformHeight );
        emitGraphProperty( "acceleration height", gn->m_accelerationHeight );
        emitGraphProperty( "trace caller", gn->m_traceCaller );
        emitGraphPropertySingle( "has motion aabbs", gn->m_hasMotionAabbs );
        emitGraphPropertySingle( "requires traversable", gn->m_requiresTraversable );
        if( scope->getClass() == RT_OBJECT_TRANSFORM )
        {
            out << "motion steps: " << dynamic_cast<const Transform*>( scope )->getKeyCount() << "\\n";
        }
    }
    else if( scope->getClass() == RT_OBJECT_ACCELERATION )
    {
        const Acceleration* as = dynamic_cast<const Acceleration*>( scope );
        emitGraphProperty( "acceleration height", as->m_accelerationHeight );
        emitGraphPropertySingle( "has motion aabbs", as->m_hasMotionAabbs );
    }
    else if( scope->getClass() == RT_OBJECT_PROGRAM )
    {
        const Program* program = dynamic_cast<const Program*>( scope );
        if( !program->m_inheritedSemanticType.empty() )
            emitGraphProperty( "inherited stype", program->m_inheritedSemanticType );
        emitGraphPropertySingle( "used as bounding box program", program->m_usedAsBoundingBoxProgram );
    }
    else if( scope->getClass() == RT_OBJECT_MATERIAL )
    {
        const Material* material = dynamic_cast<const Material*>( scope );
        emitGraphProperty( "trace caller", material->m_traceCaller );
    }
    else if( scope->getClass() == RT_OBJECT_GEOMETRY_INSTANCE )
    {
        const GeometryInstance* gi = dynamic_cast<const GeometryInstance*>( scope );
        emitGraphProperty( "trace caller", gi->m_traceCaller );
    }
    emitGraphProperty( "direct caller", scope->m_directCaller );

    // Virtual parents
    if( scope->getClass() == RT_OBJECT_PROGRAM )
    {
        const Program* program = dynamic_cast<const Program*>( scope );
        emitVirtualParents( "vparents", program );
    }

    // Print variables including their valid reference ids
    out << " | <vars> ";
    unsigned int lwar = scope->getVariableCount();
    for( unsigned int i = 0; i < lwar; ++i )
    {
        Variable* var = scope->getVariableByIndex( i );
        out << var->getName() << " (" << var->getToken() << "): ";
        if( m_printReferences )
        {
            const ProgramManager::VariableReferenceIDListType& refids =
                m_programManager->getReferencesForVariable( var->getToken() );
            for( ProgramManager::VariableReferenceIDListType::const_iterator iter = refids.begin(); iter != refids.end(); ++iter )
            {
                if( iter != refids.begin() )
                    out << ',';
                out << *iter;
            }
        }
        out << "\\n";
    }

    // Print reference sets
    if( m_printReferences )
    {
        out << " | <refs>";

        // Compute derived sets for debugging
        GraphProperty<VariableReferenceID, false> unresolved;
        scope->computeUnresolvedOutputForDebugging( unresolved );

        switch( scope->getClass() )
        {
            case RT_OBJECT_PROGRAM:
            {
                const Program* p = dynamic_cast<const Program*>( scope );
                emitGraphProperty( "IN", p->m_unresolvedSet );
                emitGraphProperty( "R", p->m_resolvedSet );
                emitGraphProperty( "OUT", unresolved );

                emitGraphProperty( "INA", p->m_unresolvedAttributeSet );
            }
            break;
            case RT_OBJECT_MATERIAL:
            {
                const Material* m = dynamic_cast<const Material*>( scope );
                GraphProperty<VariableReferenceID, false> INP;
                m->computeUnresolvedGIOutputForDebugging( INP );
                emitGraphProperty( "IN", m->m_unresolvedSet );
                emitGraphProperty( "C", unresolved );
                emitGraphProperty( "PC", m->m_unresolvedSet_giCantResolve );
                emitGraphProperty( "IN\'", INP );
                emitGraphProperty( "R", m->m_resolvedSet );

                emitGraphProperty( "INA", m->m_unresolvedAttributeSet );
            }
            break;
            case RT_OBJECT_GEOMETRY:
            {
                const Geometry* g = dynamic_cast<const Geometry*>( scope );
                GraphProperty<VariableReferenceID, false> INP;
                g->computeUnresolvedGIOutputForDebugging( INP );
                emitGraphProperty( "IN", g->m_unresolvedSet );
                emitGraphProperty( "PC", g->m_unresolvedSet_giCantResolve );
                emitGraphProperty( "IN\'", INP );
                emitGraphProperty( "R", g->m_resolvedSet );
                emitGraphProperty( "C", unresolved );

                emitGraphProperty( "INA", g->m_unresolvedAttributeSet );
            }
            break;
            case RT_OBJECT_GEOMETRY_INSTANCE:
            {
                const GeometryInstance* gi = dynamic_cast<const GeometryInstance*>( scope );
                GraphProperty<VariableReferenceID, false> C;
                gi->computeUnresolvedChildOutputForDebugging( C );
                emitGraphProperty( "IN", gi->m_unresolvedSet );
                emitGraphProperty( "R", gi->m_resolvedSet );
                emitGraphProperty( "C", C );
                emitGraphProperty( "IN\'", gi->m_unresolvedSet_childCantResolve );
                emitGraphProperty( "OUT", unresolved );

                emitGraphProperty( "UNA", gi->m_unresolvedAttributesRemaining );
                emitMap( "RA", gi->m_resolvedAttributes );
            }
            break;
            case RT_OBJECT_GLOBAL_SCOPE:
            {
                const GlobalScope* gs = static_cast<const GlobalScope*>( scope );
                emitGraphProperty( "IN", gs->m_unresolvedSet );
                emitGraphProperty( "R", gs->m_resolvedSet );
                emitGraphProperty( "OUT", gs->getRemainingUnresolvedReferences() );

                emitGraphProperty( "INA", gs->getRemainingUnresolvedReferences() );
            }
            break;
            default:
                emitGraphProperty( "IN", scope->m_unresolvedSet );
                emitGraphProperty( "R", scope->m_resolvedSet );
                emitGraphProperty( "OUT", unresolved );
                break;
        }
    }

    // Print children and program links including variables
    out << "| {";
    switch( scope->getClass() )
    {
        case RT_OBJECT_ACCELERATION:
        {
        }
        break;
        case RT_OBJECT_GLOBAL_SCOPE:
        {
            const GlobalScope* gs  = dynamic_cast<const GlobalScope*>( scope );
            size_t             nep = scope->getContext()->getEntryPointCount();
            size_t             nrt = scope->getContext()->getRayTypeCount();
            size_t             n   = std::max( nep, nrt );
            for( size_t i = 0; i < n; ++i )
            {
                if( i < nep )
                    emitEdge( "RG" + std::to_string( i ), scope, gs->getRayGenerationProgram( i ), ST_RAYGEN, i );
                if( i < nep )
                    emitEdge( "EX" + std::to_string( i ), scope, gs->getExceptionProgram( i ), ST_EXCEPTION, i );
                if( i < nrt )
                    emitEdge( "MI" + std::to_string( i ), scope, gs->getMissProgram( i ), ST_MISS, i );
            }
            emitEdge( "AAI", scope, gs->getAabbComputeProgram(), Scoped );
            emitEdge( "AAE", scope, gs->getAabbExceptionProgram(), Scoped );
            for( const auto& program : m_objectManager->getPrograms() )
            {
                if( program->isBindless() )
                    emitEdge( gs, program );
            }
        }
        break;

        case RT_OBJECT_GEOMETRY:
        {
            if( const GeometryTriangles* gt = dynamic_cast<const GeometryTriangles*>( scope ) )
            {
                emitEdge( "ATTR", scope, gt->getAttributeProgram(), ST_ATTRIBUTE, 0 );
                emitEdge( "IS", scope, gt->getIntersectionProgram(), ST_INTERSECTION, 0 );
                emitEdge( "BB", scope, gt->getBoundingBoxProgram(), ST_BOUNDING_BOX, 0 );
            }
            else
            {
                const Geometry* g = dynamic_cast<const Geometry*>( scope );
                emitEdge( "IS", scope, g->getIntersectionProgram(), ST_INTERSECTION, 0 );
                emitEdge( "BB", scope, g->getBoundingBoxProgram(), ST_BOUNDING_BOX, 0 );
            }
        }
        break;

        case RT_OBJECT_GEOMETRY_INSTANCE:
        {
            const GeometryInstance* gi = dynamic_cast<const GeometryInstance*>( scope );
            emitEdge( "G", scope, gi->getGeometry(), ReverseScoped );
            for( int i = 0; i < gi->getMaterialCount(); ++i )
                emitEdge( "M" + std::to_string( i ), scope, gi->getMaterial( i ), ReverseScoped );
        }
        break;

        case RT_OBJECT_GEOMETRY_GROUP:
        case RT_OBJECT_GROUP:
        {
            const AbstractGroup* grp = dynamic_cast<const AbstractGroup*>( scope );
            emitEdge( "AS", scope, grp->getAcceleration(), ReverseScoped );
            emitEdge( "LW", scope, grp->getVisitProgram(), Scoped );
            emitEdge( "BB", scope, grp->getBoundingBoxProgram(), Scoped );

            int n = grp->getChildCount();
            for( int i = 0; i < n; ++i )
                emitEdge( "c" + std::to_string( i ), scope, grp->getChild( i ), Unscoped );
        }
        break;

        case RT_OBJECT_MATERIAL:
        {
            const Material* m   = dynamic_cast<const Material*>( scope );
            size_t          nrt = scope->getContext()->getRayTypeCount();
            for( size_t i = 0; i < nrt; ++i )
            {
                emitEdge( "CH" + std::to_string( i ), scope, m->getClosestHitProgram( i ), ST_CLOSEST_HIT, i );
                emitEdge( "AH" + std::to_string( i ), scope, m->getAnyHitProgram( i ), ST_ANY_HIT, i );
            }
        }
        break;

        case RT_OBJECT_PROGRAM:
        {
            const Program* p = dynamic_cast<const Program*>( scope );
            for( const CanonicalProgram* cp : p->m_canonicalPrograms )
            {
                emitEdge( p, cp );
            }
        }
        break;

        case RT_OBJECT_SELECTOR:
        {
            const Selector* s = dynamic_cast<const Selector*>( scope );
            emitEdge( "LW", scope, s->getVisitProgram(), Scoped );
            emitEdge( "BB", scope, s->getBoundingBoxProgram(), Scoped );

            int n = s->getChildCount();
            for( int i = 0; i < n; ++i )
                emitEdge( "c" + std::to_string( i ), scope, s->getChild( i ), ST_NODE_VISIT, 0 );
        }
        break;

        case RT_OBJECT_TRANSFORM:
        {
            const Transform* t = dynamic_cast<const Transform*>( scope );
            emitEdge( "LW", scope, t->getVisitProgram(), Scoped );
            emitEdge( "BB", scope, t->getBoundingBoxProgram(), Scoped );
            emitEdge( "c", scope, t->getChild(), Unscoped );
        }
        break;

        case RT_OBJECT_POSTPROCESSINGSTAGE:
        {
        }
        break;

        default:
            RT_ASSERT( !!!"Illegal scope" );
            break;
    }

    for( unsigned int i = 0; i < lwar; ++i )
    {
        Variable*    var = scope->getVariableByIndex( i );
        VariableType vt  = var->getType();
        switch( vt.baseType() )
        {
            case VariableType::GraphNode:
            {
                GraphNode* node = var->getGraphNode();
                if( node )
                    emitEdge( var->getName(), scope, node, Unscoped );
            }
            break;
            case VariableType::Program:
            {
                Program* p = var->getProgram();
                if( p )
                    emitEdge( var->getName(), scope, p, p->isBindless() ? Scoped : Unscoped );
            }
            break;
            default:
            {
                // Ignore other variables
            }
            break;
        }
    }

    out << "}";

    // Wrap it up
    out << "}\"\n";
    out << "];\n";
}

void NodegraphPrinter::emitCanonicalProgram( const CanonicalProgram* cp )
{
    int id = cp->getID();
    out << "cp" << id << " [\n";
    out << "shape = Mrecord\n";
    out << "style = filled\n";
    out << "label = \"{CanonicalProgram " << id << "\\n"
        << cp->getInputFunctionName() << "\\n"
        << "addr = " << cp << "\\n";

    out << "| var refs\\n";
    const CanonicalProgram::VariableReferenceListType& vars = cp->getVariableReferences();
    for( const VariableReference* var : vars )
        out << var->getInfoString() << "\\n";

    out << "| att refs\\n";
    const CanonicalProgram::VariableReferenceListType& atts = cp->getAttributeReferences();
    for( const VariableReference* att : atts )
        out << att->getInfoString() << "\\n";

    out << "| properties\\n";
    emitGraphProperty( "semType", cp->m_usedAsSemanticType );
    emitGraphProperty( "usedByRayType", cp->m_usedByRayType );
    emitGraphProperty( "direct caller", cp->m_directCaller );

    if( cp->tracesUnknownRayType() )
    {
        out << "producesRayTypes: any\\n";
    }
    else
    {
        emitGraphProperty( "producesRayTypes", cp->m_producesRayTypes );
    }

    emitGraphProperty( "usedByDevice", cp->m_usedOnDevice );

    out << "| bindings\\n";
    for( const VariableReference* varref : vars )
    {
        bool                checkBuffer    = true;
        bool                checkGraphNode = true;
        bool                checkProgram   = true;
        bool                checkTexture   = true;
        VariableReferenceID refid          = varref->getReferenceID();
        out << varref->getInfoString();
        emitGraphProperty( " \\-\\> Var", m_bindingManager->getVariableBindingsForReference( refid ) );
        switch( varref->getType().baseType() )
        {
            case VariableType::Buffer:
            case VariableType::DemandBuffer:
                emitGraphProperty( "\\-\\> Buffer IDs", m_bindingManager->getBufferBindingsForReference( refid ) );
                checkBuffer = false;
                break;
            case VariableType::GraphNode:
                emitGraphProperty( "\\-\\> GraphNode IDs", m_bindingManager->getGraphNodeBindingsForReference( refid ) );
                checkGraphNode = false;
                break;
            case VariableType::Program:
                emitGraphProperty( "\\-\\> Program Scope IDs", m_bindingManager->getProgramBindingsForReference( refid ) );
                checkProgram = false;
                break;
            case VariableType::TextureSampler:
                emitGraphProperty( "\\-\\> Texture Sampler IDs", m_bindingManager->getTextureBindingsForReference( refid ) );
                checkTexture = false;
                break;
            default:
                // do nothing
                break;
        }
        if( checkBuffer && !m_bindingManager->getBufferBindingsForReference( refid ).empty() )
            emitGraphProperty( "\\-\\> Buffer IDs SHOULD BE EMPTY!!", m_bindingManager->getBufferBindingsForReference( refid ) );
        if( checkGraphNode && !m_bindingManager->getGraphNodeBindingsForReference( refid ).empty() )
            emitGraphProperty( "\\-\\> GraphNode IDs SHOULD BE EMPTY!!",
                               m_bindingManager->getGraphNodeBindingsForReference( refid ) );
        if( checkProgram && !m_bindingManager->getProgramBindingsForReference( refid ).empty() )
            emitGraphProperty( "\\-\\> Program Scope IDs SHOULD BE EMPTY!!",
                               m_bindingManager->getProgramBindingsForReference( refid ) );
        if( checkTexture && !m_bindingManager->getTextureBindingsForReference( refid ).empty() )
            emitGraphProperty( "\\-\\> Texture Sampler IDs SHOULD BE EMPTY!!",
                               m_bindingManager->getTextureBindingsForReference( refid ) );
    }
    out << "| callsites\\n";
    for( const CallSiteIdentifier* cs : cp->m_ownedCallSites )
    {
        out << cs->getInputName() << " \\-\\> Callees: \\{";
        auto calleeIt = cs->m_potentialCallees.begin();
        if( calleeIt != cs->m_potentialCallees.end() )
        {
            out << *( calleeIt++ );
            for( ; calleeIt != cs->m_potentialCallees.end(); ++calleeIt )
            {
                out << "," << *calleeIt;
            }
        }
        out << "\\}\\n";
    }
    out << "| called from\\n";
    for( const CallSiteIdentifier* cs : cp->m_calledFromCallsites )
    {
        out << cs->getInputName() << " (Owner: " << cs->m_parent->getID() << "\\-\\> Callees: \\{";
        auto calleeIt = cs->m_potentialCallees.begin();
        if( calleeIt != cs->m_potentialCallees.end() )
        {
            out << *( calleeIt++ );
            for( ; calleeIt != cs->m_potentialCallees.end(); ++calleeIt )
            {
                out << "," << *calleeIt;
            }
        }
        out << "\\}\\n";
    }

    // Wrap it up
    out << "}\"\n";
    out << "];\n";
}

void NodegraphPrinter::emitEdge( const std::string& name, const LexicalScope* from, const LexicalScope* to, EdgeKind edgeKind )
{
    int u = m_unique++;
    out << "| <u" << u << "> " << name;
    if( to )
    {
        edges << "s" << from->getScopeID() << ":u" << u << " -> s" << to->getScopeID();
        if( edgeKind == Unscoped )
            edges << " [style = dashed]";
        else if( edgeKind == ReverseScoped )
            edges << " [style = dotted]";
        else if( edgeKind == VirtualChild )
            edges << " [color = blue constraint = false]";
        edges << '\n';
    }
}

void NodegraphPrinter::emitEdge( const std::string& name, const LexicalScope* from, const LexicalScope* to, SemanticType stype, unsigned int index )
{
    emitEdge( name, from, to, Scoped );

    // Print virtual children via a brute force search of program objects
    int u = m_unique - 1;
    for( const auto& program : m_objectManager->getPrograms() )
    {
        for( const auto& vparent : program->m_virtualParents )
        {
            if( vparent.scopeid == from->getScopeID() && vparent.stype == stype && vparent.index == index )
                edges << "s" << from->getScopeID() << ":u" << u << " -> s" << program->getScopeID()
                      << " [color = blue constraint = false]\n";
        }
    }
}

void NodegraphPrinter::emitEdge( const Program* from, const CanonicalProgram* to )
{
    edges << "s" << from->getScopeID() << " -> cp" << to->getID() << '\n';
}

void NodegraphPrinter::emitEdge( const GlobalScope* from, const Program* to )
{
    edges << "s" << from->getScopeID() << " -> s" << to->getScopeID();
    edges << " [style = dashed color = red]" << '\n';
}

template <typename T>
void emitGraphPropertyP( std::ostringstream& out, const T& val )
{
    out << val;
}

template <>
void emitGraphPropertyP( std::ostringstream& out, const ProgramRoot& val )
{
    out << val.toString();
}

template <>
void emitGraphPropertyP( std::ostringstream& out, const SemanticType& val )
{
    out << semanticTypeToAbbreviationString( val );
}

template <>
void emitGraphPropertyP( std::ostringstream& out, const VariableReferenceBinding& val )
{
    out << val.toString();
}

template <>
void emitGraphPropertyP( std::ostringstream& out, const std::pair<VariableReferenceID, VariableReferenceID>& val )
{
    out << "(" << val.first << ", " << val.second << ")";
}

template <typename T>
void NodegraphPrinter::emitGraphProperty( const std::string& name, const T& prop )
{
    out << name << ": \\{";
    for( typename T::const_iterator it = prop.begin(); it != prop.end(); ++it )
    {
        if( it != prop.begin() )
            out << ',';
        emitGraphPropertyP( out, *it );
        int count = prop.count( *it );
        if( count != 1 )
            out << "[" << count << "]";
    }
    out << "\\}\\n";
}

template <typename T>
void NodegraphPrinter::emitGraphPropertySingle( const std::string& name, const T& prop )
{
    out << name << ": \\{";
    emitGraphPropertyP( out, prop.count() );
    out << "\\}\\n";
}

void NodegraphPrinter::emitMap( const std::string& name, const std::map<VariableReferenceID, VariableReferenceID>& map )
{
    out << name << ": \\{";
    for( auto it = map.begin(); it != map.end(); ++it )
    {
        if( it != map.begin() )
            out << ',';
        emitGraphPropertyP<std::pair<VariableReferenceID, VariableReferenceID>>( out, *it );
    }
    out << "\\}\\n";
}

template <typename T>
void NodegraphPrinter::emitGraphPropertyMulti( const std::string& name, const T& propMulti )
{
    out << name << ": ";
    if( propMulti.empty() )
        out << "\\<empty\\>\\n";
    else if( propMulti.m_map.size() > 1 )
        out << "\\n";
    for( typename T::MapType::const_iterator it = propMulti.m_map.begin(); it != propMulti.m_map.end(); ++it )
    {
        emitGraphProperty( std::to_string( it->first ), it->second );
    }
}

void NodegraphPrinter::emitVirtualParents( const std::string& label, const Program* program )
{
    if( program->m_virtualParents.empty() && program->m_rootAnnotations.empty() )
        return;
    out << " | <vparents> ";
    if( !program->m_virtualParents.empty() )
        emitGraphProperty( label, program->m_virtualParents );
    for( auto record : program->m_rootAnnotations )
    {
        out << record.first.toString() << ": ";
        for( auto map : record.second.programReferences )
        {
            out << "r" << map.first << "-\\>"
                << "r" << map.second << " ";
        }
    }
    out << "\\n";
}

void NodegraphPrinter::emitBindingManager()
{
    out << "bindings [\n";
    out << "shape = Mrecord\n";
    out << "label = \"{Binding Manager \\n";
    emitGraphProperty( "xform height", m_bindingManager->m_transformHeight );
    out << " | Bindings\\n";
    if( m_printFullBindingManager )
    {
        emitGraphPropertyMulti( "variable", m_bindingManager->m_variableBindings );
        emitGraphPropertyMulti( "graph node", m_bindingManager->m_graphNodeBindings );
        emitGraphPropertyMulti( "program", m_bindingManager->m_programBindings );
        emitGraphPropertyMulti( "buffer", m_bindingManager->m_bufferBindings );
        emitGraphPropertyMulti( "texture", m_bindingManager->m_textureBindings );
    }
    emitGraphProperty( "attribute", m_bindingManager->m_attributeBindings );

    out << " | Ilwerse Bindings\\n";
    emitGraphPropertyMulti( "graph node", m_bindingManager->m_ilwerseGraphNodeBindings );
    emitGraphPropertyMulti( "program", m_bindingManager->m_ilwerseProgramBindings );
    emitGraphPropertyMulti( "buffer", m_bindingManager->m_ilwerseBufferBindings );
    emitGraphPropertyMulti( "texture", m_bindingManager->m_ilwerseTextureBindings );

    // Wrap it up
    out << "}\"\n";
    out << "];\n";
}
