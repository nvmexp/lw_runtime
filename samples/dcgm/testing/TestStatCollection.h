#ifndef TESTSTATCOLLECTION_H
#define TESTSTATCOLLECTION_H

#include "TestLwcmModule.h"

class TestStatCollection : public TestLwcmModule
{
public:
    TestStatCollection();
    ~TestStatCollection();

    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

private:

    /*************************************************************************/
    /* Individual test cases */
    int TestCollectionMerge();
    int TestPerformance();

    /*************************************************************************/

};

#endif //TESTSTATCOLLECTION_H
