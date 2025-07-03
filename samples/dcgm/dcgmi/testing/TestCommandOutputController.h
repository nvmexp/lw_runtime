#ifndef TESTCOMMANDOUTPUTCONTROLLER_H
#define TESTCOMMANDOUTPUTCONTROLLER_H

#include <string>

#include "TestDcgmiModule.h"

class TestCommandOutputController : public TestDcgmiModule {
public:
    TestCommandOutputController();
    virtual ~TestCommandOutputController();
    
    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Run();
    std::string GetTag();  
    
private:
    /*************************************************************************/
    /*
     * Actual test cases. These should return a status like below
     *
     * Returns 0 on success
     *        <0 on fatal error. Will abort entire framework
     *        >0 on non-fatal error
     *
     **/
    int TestHelperDisplayValue();
};

#endif
