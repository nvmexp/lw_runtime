/*
 * Copyright (c) 2009 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __CPPOGTEST_H__
#define __CPPOGTEST_H__

// while well-intentioned, deprecation pragmas don't allow standard Microsoft C++ STL
// headers to be included without tons of warnings
#define LWOG_NO_DEPRECATE

#include "ogtest.h"
#include "cppstring.h"
#include "cppcheck.h"

//
// cppogtest.h
//
// Infrastructure for C++-based lwntests.
//
// This infrastructure allows test writers to stamp out new C++-based
// lwntests by defining a class with methods to implement the various parts
// of an lwntest:  getDescription(), isSupported(), initGraphics(),
// doGraphics(), and exitGraphics().  The class definition may also contain
// data and utility methods.  For classes where common code is used to
// implement multiple variants of a test, data members may be provided by the
// class allowing the test methods to determine which instance of the test is
// running.
//
// To support ten different variants of a "stupid" lwntest, one might create
// a class defined as follows:
//
//     class StupidTest {
//     public:
//         int variant;
//         StupidTest(int _variant) : variant(_variant) {}
//         OGTEST_CppMethods();
//     };
//
// In this example, the member <variant> would be an integer in the range
// [0,9] identifying which version of the test the instance corresponds to.
// The constructor in this example creates a test instance and takes an
// integer identifying the variant.  
//
// The "OGTEST_CppMethods()" line uses a macro defined to specify provides
// prototypes for the methods common to all test classes.  The test file
// obviously also needs to create the implementation of each of these methods:
// 
//     lwString StupidTest::getDescription() { ... }
//     int  StupidTest::isSupported()        { ... }
//     void StupidTest::initGraphics()       { ... }
//     void StupidTest::doGraphics()         { ... }
//     void StupidTest::exitGraphics()       { ... }
//
// Alternately, the test can skip the "OGTEST_CppMethods" approach and provide
// the implementations of the class in the body of the class definition
// itself.
//
// Other data and methods may be added to the class as desired.  For example,
// if the test runs in multiple cells, one might create a utility method such
// as:
//
//         void renderCell(int cellX, int cellY);
//
// to draw just a single cell.  The utility methods have access to everything
// in the class instance, including the variant information.
//
// Since all class methods have access to the class instance, static data
// members may be used to colweniently pass around information that might
// otherwise be communicated by function arguments or static global variables.
// Don't use non-static members for such temporary data, since we have one
// global instance of the test class object for each individual test.
//
// IMPORTANT NOTE:  The core of lwntest, including the main headers and
// important utility modules, is still compiled as C code.  When compiling C++
// test modules, non-C++ headers should be wrapped like:
//
//     extern "C" {
//       #include "c_header_file.h"
//     }
//
// to make sure the C++ code can properly link with C code.
//
#ifndef __cplusplus
#error "cppogtest.h can only be used in C++ code"
#endif


// OGTEST_CppMethods:  Utility macro to define prototypes for the primary
// methods for a test class.
#define OGTEST_CppMethods()         \
    lwString    getDescription();   \
    int         isSupported(void);  \
    void        initGraphics(void); \
    void        doGraphics(void);   \
    void        exitGraphics(void);

// OGTEST_CppPerfMethods:  Utility macro to define prototypes for the primary
// methods for a performance test class.
#define OGTEST_CppPerfMethods()                                             \
    lwString    getDescription();                                           \
    int         isSupported(void);                                          \
    void        initGraphics(void);                                         \
    /* DoGraphics doesn't exist for performance tests. */                   \
    void        exitGraphics(void);                                         \
    void        doBenchmark(const char *testName, int subtest,              \
                             int cycles, int path);                         \
    void        getBenchmark(BENCHMARKSUMMARY *info, const char *testName,  \
                             int subtest, int cycles, int path);            \
    int         getNumSubtests(const char *testName, int path);

// OGTEST_CppFuncAndPerfMethods:  Utility macro to define prototypes for the
// primary methods for a combined test class supporting both functional and
// performance testing.
#define OGTEST_CppFuncAndPerfMethods()  \
    OGTEST_CppPerfMethods()             \
    void        doGraphics(void);


// NOTE:  We redefine the OGTEST_IsSupported() macro here to tag the filename
// variable with 'extern "C"'.  A hack like this needs to be necessary to get
// lwntest to link on Windows (adding "extern" or removing "const" also seems
// to work there).  I'm not sure why adding 'extern "C"' inside an 'extern
// "C"' block matters, but whatever...
#undef  OGTEST_IsSupported
#define OGTEST_IsSupported(a)                                       \
    OGTEST_IsSupportedProto(a);                                     \
    extern "C" OGTEST_FileNameDefine(a) OGTEST_IsSupportedProto(a)


//
// OGTEST_CppTest is a macro used to implement the C linkage for each of the
// individual tests using the test class.  Each individual test variant should
// have an OGTEST_CppTest macro to create its C test functions.  The macro
// takes three arguments:
//
//   _class:  the name of the class implementing the test
//   _name:   the name of the test itself
//   _init:   a parenthesized expression containing a list of arguments
//            to be passed to the class' constructor for that test instance
//
// In the example above, the test file would include the following:
//
//    OGTEST_CppTest(StupidTest, stupid0, (0));
//    ...
//    OGTEST_CppTest(StupidTest, stupid9, (9));
//
// This will create static globals of class StupidTest named "stupid0_test"
// through "stupid9_test", as well as C entry points for each test function.
//
// Obviously, a large number of test variants can be stamped out using macros.
//
#define OGTEST_CppTest(_class, _name, _init)                                \
static _class lwog_##_name _init;                                           \
extern "C" {                                                                \
    OGTEST_InitGraphics(_name)      { lwogtestCheck.initGraphics(); lwog_##_name.initGraphics(); }     \
    OGTEST_DoGraphics(_name)        { if (!lwogtestCheck.hasFailed()) lwog_##_name.doGraphics(); }     \
    OGTEST_ExitGraphics(_name)      { lwog_##_name.exitGraphics(); lwogtestCheck.exitGraphics(); }     \
    OGTEST_GetDescription(_name)    {                                       \
        lwog_snprintf(str, MAX_TEST_DESCRIPTION_LENGTH, "%s",               \
                      lwog_##_name.getDescription().c_str());               \
    }                                                                       \
    OGTEST_IsSupported(_name)       { return lwog_##_name.isSupported(); }  \
}

// OGTEST_CppPerfTest is a similar macro that stamps out class instances and C
// functions for performance tests.
#define OGTEST_CppPerfTest(_class, _name, _init)                            \
static _class lwog_##_name _init;                                           \
extern "C" {                                                                \
    OGTEST_InitGraphics(_name)      { lwogtestCheck.initGraphics(); lwog_##_name.initGraphics(); }     \
    /* DoGraphics doesn't exist for performance tests. */                   \
    OGTEST_ExitGraphics(_name)      { lwog_##_name.exitGraphics(); lwogtestCheck.exitGraphics(); }     \
    OGTEST_GetDescription(_name)    {                                       \
        lwog_snprintf(str, MAX_TEST_DESCRIPTION_LENGTH, "%s",               \
                      lwog_##_name.getDescription().c_str());               \
    }                                                                       \
    OGTEST_IsSupported(_name)       { return lwog_##_name.isSupported(); }  \
    OGTEST_GetBenchmark(_name)      {                                       \
        lwog_##_name.getBenchmark(info, testName, subtest, cycles, path);   \
    }                                                                       \
    OGTEST_DoBenchmark(_name)       {                                       \
        lwog_##_name.doBenchmark(testName, subtest, cycles, path);          \
    }                                                                       \
    OGTEST_GetNumSubtests(_name)    {                                       \
        return lwog_##_name.getNumSubtests(testName, path);                 \
    }                                                                       \
}

// OGTEST_CppFuncAndPerfTest is a similar macro that stamps out class
// instances and C functions for combined functional/performance tests.
#define OGTEST_CppFuncAndPerfTest(_class, _name, _init)                 \
    OGTEST_CppPerfTest(_class, _name, _init)                            \
    extern "C" {                                                        \
        OGTEST_DoGraphicsProto(_name);                                  \
        OGTEST_DoGraphicsProto(_name)   { lwog_##_name.doGraphics(); }  \
    }

#endif  // #ifndef __CPPOGTEST_H__
