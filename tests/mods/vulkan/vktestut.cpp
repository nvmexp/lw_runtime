/*
* LWIDIA_COPYRIGHT_BEGIN
*
* Copyright 2017-2019 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* LWIDIA_COPYRIGHT_END
*/

#define BOOST_TEST_MODULE VulkanModule
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#ifdef _WIN32
#   pragma warning(push)
#   pragma warning(disable: 4265) // class has virtual functions, but destructor is not virtual
#endif
#include <boost/test/unit_test.hpp>
#ifdef _WIN32
#   pragma warning(pop)
#endif

int main(int argc, char* argv[])
{
    return boost::unit_test::unit_test_main(init_unit_test, argc, argv);
}
