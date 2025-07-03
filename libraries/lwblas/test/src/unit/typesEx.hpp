#define LWTENSOR_UNIT_TEST // to access ContractionDescriptor::canonicalizeModes

#include "lwtensor.h"
#include "lwtensor/internal/types.h"
#include "lwtensor/internal/contractionDescriptor.h"
#include "lwtensor/internal/cache.h"

namespace lwtensor
{
namespace test
{
}; /* end namespace test */
}; /* end namespace lwtensor */


#ifdef DEBUG
template<typename Cacheline>
void printCache(Cache<Cacheline> &cache)
{
    auto cl = cache.getFront();
    int count = 0;
    while(cl != nullptr && count < 8)
    {
        printf("%s:%d\n", cl->getKey().c_str(), cl->getValue());
        cl = cl->getNext();
        count++;
    }
}
#endif

TEST(ContractionDescriptor, canonicalizeModes1)
{
    using namespace LWTENSOR_NAMESPACE;
    std::vector<mode_type> modeA{11,37};
    std::vector<mode_type> modeB{17,37};
    std::vector<mode_type> modeC{11,17};
    std::vector<mode_type> canonA{0,1};
    std::vector<mode_type> canonB{2,1};
    std::vector<mode_type> canonC{0,2};

    std::vector<mode_type> outA(modeA.size());
    std::vector<mode_type> outB(modeB.size());
    std::vector<mode_type> outC(modeC.size());
    ContractionDescriptorInternal::canonicalizeModes(
            modeA, modeB, modeC,
            outA, outB, outC);
    for(int i=0; i < modeA.size(); ++i){
        EXPECT_EQ(canonA[i], outA[i]);
    }
    for(int i=0; i < modeB.size(); ++i){
        EXPECT_EQ(canonB[i], outB[i]);
    }
    for(int i=0; i < modeC.size(); ++i){
        EXPECT_EQ(canonC[i], outC[i]);
    }
}

TEST(ContractionDescriptor, canonicalizeModes2)
{
    using namespace LWTENSOR_NAMESPACE;
    std::vector<mode_type> modeA{27,11,37};
    std::vector<mode_type> modeB{17,27,37};
    std::vector<mode_type> modeC{11,17,27};
    std::vector<mode_type> canonA{0,1,2};
    std::vector<mode_type> canonB{3,0,2};
    std::vector<mode_type> canonC{1,3,0};

    std::vector<mode_type> outA(modeA.size());
    std::vector<mode_type> outB(modeB.size());
    std::vector<mode_type> outC(modeC.size());
    ContractionDescriptorInternal::canonicalizeModes(
            modeA, modeB, modeC,
            outA, outB, outC);
    for(int i=0; i < modeA.size(); ++i){
        EXPECT_EQ(canonA[i], outA[i]);
    }
    for(int i=0; i < modeB.size(); ++i){
        EXPECT_EQ(canonB[i], outB[i]);
    }
    for(int i=0; i < modeC.size(); ++i){
        EXPECT_EQ(canonC[i], outC[i]);
    }
}

TEST(Cache, test1)
{
    using namespace std;
    using namespace LWTENSOR_NAMESPACE;
    try
    {
        using Key = std::string;
        using Value = int;
        typedef Cache<Key, Value>::Cacheline MyCacheline;
        Cache<Key, Value> cache;
        const int numCachelines = 4;
        MyCacheline cachelines[numCachelines];
        cache.attachCachelines(cachelines, numCachelines);
        cache.put("a", 1);
        EXPECT_EQ( cachelines[0].getValue(), 1);
        cache.put("a", 2);
        // ensure that we hit
        EXPECT_EQ( cache.size(), 1);
        // ensure that value has been updated
        EXPECT_EQ( cachelines[0].getValue(), 2);
        Value value;
        // expect a hit
        EXPECT_EQ( cache.get("a", value), true);
        EXPECT_EQ( value, 2);
        //expect a miss
        EXPECT_EQ( cache.get("b", value), false);
        cache.detachCachelines();
    }
    catch( const exception & e )
    {
        cout << e.what() << endl;
    }
}

TEST(Cache, test2)
{
    using namespace std;
    using namespace LWTENSOR_NAMESPACE;
    try
    {
        using Key = std::string;
        using Value = int;
        typedef Cache<Key, Value>::Cacheline MyCacheline;
        Cache<Key, Value> cache;
        const int numCachelines = 3;
        MyCacheline cachelines[numCachelines];
        cache.attachCachelines(cachelines, numCachelines);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        cache.put("a", 4);
        // cache state: a, c, b
        Value val;
        EXPECT_EQ(cache.get("a", val), true);
        // cache state: a, c, b
        EXPECT_EQ(cache.get("b", val), true);
        // cache state: b, a, c
        EXPECT_EQ(cache.get("c", val), true);
        // cache state: c, b, a
        cache.put("d", 2);
        // cache state: d, c, b
        EXPECT_EQ(cache.get("a", val), false);
        // cache state: d, c, b
        EXPECT_EQ(cache.get("b", val), true);
        // cache state: b, d, c
        EXPECT_EQ(cache.get("c", val), true);
        // cache state: c, b, d
        EXPECT_EQ(cache.get("d", val), true);
        // cache state: d, c, b
        cache.put("b", 7);
        // cache state: b, d, c
        cache.put("a", 3);
        // cache state: a, b, d
        EXPECT_EQ(cache.get("c", val), false);
        // cache state: a, b, d
        EXPECT_EQ(cache.get("b", val), true);
        // cache state: b, a, d
        EXPECT_EQ(val, 7);
        EXPECT_EQ(cache.get("b", val), true);
        // cache state: b, a, d
        cache.put("x", 8);
        // cache state: x, b, a
        EXPECT_EQ(cache.get("b", val), true);
        // cache state: b, x, a
        EXPECT_EQ(cache.get("a", val), true);
        // cache state: a, b, x
        cache.detachCachelines();
#ifdef DEBUG
        printf("line:%d\n",__LINE__); printCache<MyCacheline>(cache);
#endif
    }
    catch( const exception & e )
    {
        cout << e.what() << endl;
    }
}

#undef LWTENSOR_UNIT_TEST
