//
// Copyright (c) 2020 - 2021, LWPU CORPORATION. All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//


/// @file

#include "consumer.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_common.h"

class consumer_unit_test: public LwSciStreamTest {
public:
   consumer_unit_test( ) {
       // initialization code here
   }

   void SetUp( ) {
       // code here will execute just before the test ensues
   }

   void TearDown( ) {
       // code here will be called just after the test completes
       // ok to through exceptions from here if need be
   }

   ~consumer_unit_test( )  {
       // cleanup any pending stuff, but no exceptions and no gtest
       // ASSERT* allowed.
   }   // put in any custom data members that you need
};


TEST_F (consumer_unit_test, initialization) {
    LwSciStream::Consumer();
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


