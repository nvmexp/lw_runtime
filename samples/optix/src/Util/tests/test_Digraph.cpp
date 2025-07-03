#include <srcTests.h>

#include <corelib/adt/Digraph.h>

using namespace corelib;


TEST( Digraph, AddNode )
{
    typedef int   NodeDataTy;
    typedef float EdgeDataTy;
    typedef Digraph<NodeDataTy, EdgeDataTy> DigraphTy;
    DigraphTy        cg;
    NodeDataTy       nd( 1 );
    DigraphTy::Node* n = cg.addNode( nd );
    ASSERT_EQ( n->data(), nd );
}

TEST( Digraph, AddNodeByKey )
{
    typedef int   NodeDataTy;
    typedef float EdgeDataTy;
    typedef short KeyTy;
    typedef Digraph<NodeDataTy, EdgeDataTy, KeyTy> DigraphTy;
    DigraphTy        cg;
    NodeDataTy       nd( 1 );
    KeyTy            key( 3 );
    DigraphTy::Node* n = cg.addNode( nd, key );
    EXPECT_EQ( n->data(), nd );
    ASSERT_EQ( cg.getNode( key ), n );
}

TEST( Digraph, AddEdge )
{
    typedef int   NodeDataTy;
    typedef float EdgeDataTy;
    typedef Digraph<NodeDataTy, EdgeDataTy> DigraphTy;
    DigraphTy        cg;
    NodeDataTy       nd1( 1 );
    NodeDataTy       nd2( 2 );
    DigraphTy::Node* n1 = cg.addNode( nd1 );
    DigraphTy::Node* n2 = cg.addNode( nd2 );
    EXPECT_EQ( n1->data(), nd1 );
    EXPECT_EQ( n2->data(), nd2 );
    EdgeDataTy       ed( 5 );
    DigraphTy::Edge* e = cg.addEdge( n1, n2, ed );
    ASSERT_EQ( e->data(), ed );
    ASSERT_EQ( e->to(), n2 );
}

TEST( Digraph, ModifyNodeData )
{
    typedef int   NodeDataTy;
    typedef float EdgeDataTy;
    typedef Digraph<NodeDataTy, EdgeDataTy> DigraphTy;
    DigraphTy        cg;
    NodeDataTy       nd1( 1 );
    NodeDataTy       nd2( 2 );
    DigraphTy::Node* n = cg.addNode( nd1 );
    ASSERT_EQ( n->data(), nd1 );
    n->data() = nd2;
    ASSERT_EQ( n->data(), nd2 );
}

TEST( Digraph, NodeIteration )
{
    typedef int   NodeDataTy;
    typedef float EdgeDataTy;
    typedef Digraph<NodeDataTy, EdgeDataTy> DigraphTy;
    DigraphTy cg;
    std::map<DigraphTy::Node*, NodeDataTy> inserted;
    for( int i = 0; i < 5; ++i )
    {
        NodeDataTy       nd( i );
        DigraphTy::Node* n = cg.addNode( nd );
        inserted.insert( std::make_pair( n, nd ) );
    }
    for( DigraphTy::iterator I = cg.begin(), IE = cg.end(); I != IE; ++I )
    {
        DigraphTy::Node* n = *I;
        ASSERT_EQ( n->data(), inserted[n] );
        inserted.erase( n );
    }
    ASSERT_TRUE( inserted.empty() );
}

TEST( Digraph, EdgeIteration )
{
    typedef int NodeDataTy;
    typedef int EdgeDataTy;
    typedef Digraph<NodeDataTy, EdgeDataTy> DigraphTy;
    DigraphTy cg;

    NodeDataTy       nd1( 1 );
    DigraphTy::Node* n1 = cg.addNode( nd1 );

    std::map<DigraphTy::Edge*, EdgeDataTy> inserted_edges;
    std::map<EdgeDataTy, DigraphTy::Node*> inserted_nodes;
    for( int i = 0; i < 5; ++i )
    {
        EdgeDataTy       ed( i );
        NodeDataTy       nd2( ed * 10 );
        DigraphTy::Node* n2 = cg.addNode( nd2 );
        DigraphTy::Edge* e  = cg.addEdge( n1, n2, ed );
        inserted_edges.insert( std::make_pair( e, ed ) );
        inserted_nodes.insert( std::make_pair( ed, n2 ) );
    }
    for( DigraphTy::Node::iterator I = n1->begin(), IE = n1->end(); I != IE; ++I )
    {
        DigraphTy::Edge* e = *I;
        // Make sure the data in the edge matches what we put in
        ASSERT_EQ( e->data(), inserted_edges[e] );
        // Make sure the callee matches what we put in
        ASSERT_EQ( e->to(), inserted_nodes[e->data()] );
        // Make sure the callee's data matches what we put in the callee
        ASSERT_EQ( inserted_nodes[e->data()]->data(), e->data() * 10 );
        inserted_edges.erase( e );
        inserted_nodes.erase( e->data() );
    }
    ASSERT_TRUE( inserted_edges.empty() );
    ASSERT_TRUE( inserted_nodes.empty() );
}
