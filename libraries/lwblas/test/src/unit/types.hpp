#include "lwtensor.h"
#include "lwtensor/internal/context.h"

namespace lwtensor
{
namespace test
{
}; /* end namespace test */
}; /* end namespace lwtensor */


TEST(Context, constructor)
{
  using namespace std;
  try
  {
    Context context;
  }
  catch( const exception & e )
  {
    cout << e.what() << endl;
  }
}

//TEST(Context, allocateReductionPool)
//{
//  using namespace std;
//  try
//  {
//    Context context;
//    EXPECT_EQ( context.allocateReductionPool( 0 ), 
//        nullptr );
//    auto* pool = context.allocateReductionPool( 2 );
//    EXPECT_NE( pool, nullptr ); 
//    EXPECT_EQ( context.allocateReductionPool( 1 ),
//        pool );
//  }
//  catch( const exception & e )
//  {
//    cout << e.what() << endl;
//  }
//}

