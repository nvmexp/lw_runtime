
#ifndef TESTDCGMIMODULE_H
#define TESTDCGMIMODULE_H

#include <set>

class TestDcgmiModule
{
public:
    TestDcgmiModule() {}

    // virtual to satisfy ancient compiler
    virtual ~TestDcgmiModule() {}

    virtual int Run() = 0;

    virtual std::string GetTag() = 0;

};

#endif

