#include <srcTests.h>

#include <corelib/misc/String.h>
#include <src/Util/IDMap.h>
#include <src/Util/IdPool.h>
#include <src/Util/ReusableIDMap.h>

#include <vector>

using namespace optix;
using namespace testing;


///////////////////////////////////////////////////////////////////////////////
//
// IDPool
//
///////////////////////////////////////////////////////////////////////////////

TEST( IdPool, RecyclesIDs )
{
    IdPool<> pool;
    EXPECT_THAT( pool.get(), Eq( 0 ) );
    EXPECT_THAT( pool.get(), Eq( 1 ) );
    EXPECT_THAT( pool.get(), Eq( 2 ) );
    EXPECT_THAT( pool.get(), Eq( 3 ) );

    pool.free( 1 );

    ASSERT_THAT( pool.get(), Eq( 1 ) );
    ASSERT_THAT( pool.get(), Eq( 4 ) );
}

TEST( IdPool, FreeThrowsWithUnallocatedZero )
{
    IdPool<> pool;

    ASSERT_ANY_THROW( pool.free( 0 ) );
}

TEST( IdPool, FreeThrowsWithIlwalidID_000 )
{
    IdPool<> pool;

    ASSERT_ANY_THROW( pool.free( 0 ) );
}

TEST( IdPool, FreeThrowsWithIlwalidID_999 )
{
    IdPool<> pool;

    ASSERT_ANY_THROW( pool.free( 999 ) );
}

TEST( IdPool, FreeThrowsWithIlwalidNegativeID )
{
    IdPool<> pool;

    ASSERT_ANY_THROW( pool.free( -1 ) );
}


///////////////////////////////////////////////////////////////////////////////
//
// ReusableIDMap
//
///////////////////////////////////////////////////////////////////////////////

TEST( ReusableIDMap, Works )
{
    ReusableIDMap<char*> idMap( 1 );
    char                 val = 'a';
    int                  oldId;
    {
        ReusableID id = idMap.insert( &val );
        oldId         = *id;
    }
    ASSERT_ANY_THROW( idMap.get( oldId ) );

    ReusableID id = idMap.insert( &val );
    ASSERT_THAT( *id, Eq( oldId ) );
}

namespace {

// Utility to perform the reservation of the given ids.
template <typename T>
void reserveIds( ReusableIDMap<T>& idMap, const std::vector<ReusableIDValue>& reservedIds )
{
    for( ReusableIDValue v : reservedIds )
        idMap.reserveIdForHint( v );
    idMap.finalizeReservedIds();
}

// Simple utility struct to keep both ids and their indices together. While the former
// is required for keeping the entry alive, the second one is used for correctness check.
struct InsertedValues
{
    std::vector<ReusableID>      ids;
    std::vector<ReusableIDValue> idValues;
};

// Utility to combine insertions. The passed in hints argument allows insertions with hints.
// The only restriction is that hints is either empty of completly in sync with values. Eg,
// if there are four values and only the third value should be inserted with hint X, then
//   hints = {-1, -1, X, -1}
// where ReusableIDMap<T>::NoHint == -1.
template <typename T>
InsertedValues insertValues( ReusableIDMap<T>&            idMap,
                             const std::vector<T>&        values,
                             std::vector<ReusableIDValue> hints = std::vector<ReusableIDValue>() )
{
    InsertedValues res;
    if( hints.size() != values.size() )
        hints.resize( values.size(), ReusableIDMap<T>::NoHint );
    for( size_t i = 0; i < values.size(); ++i )
    {
        res.ids.push_back( idMap.insert( values[i], hints[i] ) );
        res.idValues.push_back( *res.ids.back() );
    }
    return res;
}

// Only to ease typing and readability.
const ReusableIDValue NoHintId = ReusableIDMap<char>::NoHint;
}

TEST( ReusableIDMap, InsertValues )
{
    ReusableIDMap<char>          idMap;
    InsertedValues               res = insertValues( idMap, {'a', 'b'} );
    std::vector<ReusableIDValue> expectedIds{0, 1};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, InsertValuesWithBase )
{
    ReusableIDMap<char>          idMap( 2 );
    InsertedValues               res = insertValues( idMap, {'a', 'b'} );
    std::vector<ReusableIDValue> expectedIds{2, 3};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, InsertValueAfterReservedWithNoHoles )
{
    ReusableIDMap<char> idMap;
    reserveIds( idMap, {0, 1} );
    InsertedValues               res = insertValues( idMap, {'a'} );
    std::vector<ReusableIDValue> expectedIds{2};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, InsertValueAfterReservedWithNoHolesWithBase )
{
    ReusableIDMap<char> idMap( 1 );
    reserveIds( idMap, {1, 2} );
    InsertedValues               res = insertValues( idMap, {'a'} );
    std::vector<ReusableIDValue> expectedIds{3};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, InsertValuesWhileRespectingContinuousReservedIds )
{
    ReusableIDMap<char> idMap( 0 );
    reserveIds( idMap, {1, 2} );
    InsertedValues               res = insertValues( idMap, {'a', 'b'} );
    std::vector<ReusableIDValue> expectedIds{0, 3};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, InsertValuesWithHints )
{
    ReusableIDMap<char> idMap( 0 );
    reserveIds( idMap, {1, 2} );
    InsertedValues               res = insertValues( idMap, {'a', 'b'}, {1, 2} );
    std::vector<ReusableIDValue> expectedIds{1, 2};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, InsertSomeValuesWithHints )
{
    ReusableIDMap<char> idMap( 0 );
    reserveIds( idMap, {1, 2} );
    InsertedValues res = insertValues( idMap, {'a', 'b', 'c', 'd'}, {NoHintId, NoHintId, 1, 2} );
    // the first two values are inserted exactly as in InsertValuesWhileRespectingContinuousReservedIds()
    std::vector<ReusableIDValue> expectedIds{0, 3, 1, 2};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, InsertValuesWhileRespectingReservedIdsWithHoles )
{
    ReusableIDMap<char> idMap( 0 );
    reserveIds( idMap, {1, 4} );
    InsertedValues               res = insertValues( idMap, {'a', 'b', 'c', 'd'} );
    std::vector<ReusableIDValue> expectedIds{0, 2, 3, 5};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, InsertValuesWhileRespectingReservedIdsWithHolesAndBase )
{
    ReusableIDMap<char> idMap( 1 );
    reserveIds( idMap, {1, 4, 5, 6, 7} );
    InsertedValues               res = insertValues( idMap, {'a', 'b', 'c', 'd'} );
    std::vector<ReusableIDValue> expectedIds{2, 3, 8, 9};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );
}

TEST( ReusableIDMap, RecycleFreedIds )
{
    ReusableIDMap<char>          idMap;
    InsertedValues               res = insertValues( idMap, {'a', 'b', 'c'} );
    std::vector<ReusableIDValue> expectedIds{0, 1, 2};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );

    // remove an entry and store id in freeList to check for later recycling
    std::vector<ReusableIDValue> freeList = {*res.ids[1]};
    res.ids[1].reset();

    // now insert again, the first value should recycle the free id from freeList
    res.ids.push_back( idMap.insert( 'd' ) );
    ASSERT_THAT( *res.ids.back(), Eq( freeList[0] ) );
    res.ids.push_back( idMap.insert( 'e' ) );
    ASSERT_THAT( *res.ids.back(), Eq( 3 ) );  // 3 is the first unused id, since 2 is the hightest used one
}

TEST( ReusableIDMap, RecycleFreedIdsWhileRespectingReservedIdsWithHoles )
{
    ReusableIDMap<char> idMap;
    reserveIds( idMap, {1, 4, 5, 6, 7} );
    InsertedValues               res = insertValues( idMap, {'a', 'b', 'c', 'd'} );
    std::vector<ReusableIDValue> expectedIds{0, 2, 3, 8};

    ASSERT_THAT( res.idValues, Eq( expectedIds ) );

    // remove two entries, both res.ids[1] and res.ids[3], and store their ids in freeList to check for later recycling
    std::vector<ReusableIDValue> freeList = {*res.ids[1], *res.ids[3]};
    res.ids[1].reset();
    res.ids[3].reset();

    // now insert again, the first values should recycle the free ids from freeList
    res.ids.push_back( idMap.insert( 'e' ) );
    ASSERT_THAT( *res.ids.back(), Eq( freeList[0] ) );
    res.ids.push_back( idMap.insert( 'f' ) );
    ASSERT_THAT( *res.ids.back(), Eq( freeList[1] ) );
    res.ids.push_back( idMap.insert( 'g' ) );
    ASSERT_THAT( *res.ids.back(), Eq( 9 ) );  // 9 is the first unused id, since 8 is the hightest used one
}

TEST( ReusableIDMapIterator, ReachesEnd )
{
    ReusableIDMap<int*>           idMap;
    int                           val = 1234;
    ReusableID                    id  = idMap.insert( &val );
    ReusableIDMap<int*>::iterator it  = idMap.begin();

    ++it;

    ASSERT_TRUE( it == idMap.end() );
}

TEST( ReusableIDMapIterator, SkipsReservedSlots )
{
    ReusableIDMap<int*> idMap( 1 );
    int                 val = 1234;
    ReusableID          id  = idMap.insert( &val );

    ReusableIDMap<int*>::mapIterator it = idMap.mapBegin();

    ASSERT_THAT( *it->second, Eq( val ) );
}


TEST( ReusableIDMapIterator, SkipsHoles )
{
    ReusableIDMap<char*> idMap;
    char                 vals[] = {'0', '1', '2'};
    ReusableID           id0    = idMap.insert( &vals[0] );
    ReusableID           id1    = idMap.insert( &vals[1] );
    ReusableID           id2    = idMap.insert( &vals[2] );
    id1.reset();  // creates a hole in the map
    ReusableIDMap<char*>::mapIterator it = idMap.mapBegin();
    EXPECT_THAT( *it->second, Eq( vals[0] ) );

    ++it;

    ASSERT_THAT( *it->second, Eq( vals[2] ) );
}


///////////////////////////////////////////////////////////////////////////////
//
// IDMap
//
///////////////////////////////////////////////////////////////////////////////


TEST( IDMap, DoubleInsertionYieldsSingleID )
{
    IDMap<std::string> idMap;
    int                id1 = idMap.insert( "hello" );

    int id2 = idMap.insert( "hello" );

    ASSERT_THAT( id1, Eq( id2 ) );
}

TEST( IDMap, CannotFindValueNotInserted )
{
    IDMap<std::string> idMap;
    idMap.insert( "hello" );

    int id = idMap.getID( "world" );

    ASSERT_THAT( id, Eq( idMap.ILWALID_INDEX ) );
}

TEST( IDMap, ThrowsOnOverflowWithSignedIndexType )
{
    // Explicitly using signed char, because plain char is treated as unsigned
    // by default on aarch64, which causes this test to fail.
    IDMap<int, signed char> idMap;
    for( int i = 0; i < 128; ++i )
        idMap.insert( i );

    ASSERT_ANY_THROW( idMap.insert( 128 ) );
}

TEST( IDMap, ThrowsOnOverflowWithUnsignedIndexType )
{
    IDMap<int, unsigned char> idMap;
    for( int i = 0; i < 255; ++i )
        idMap.insert( i );

    ASSERT_ANY_THROW( idMap.insert( 255 ) );
}

TEST( Stringf, Works )
{
    int i = 1234;

    std::string str = corelib::stringf( "%08d", i );

    ASSERT_THAT( str, StrEq( "00001234" ) );
}
