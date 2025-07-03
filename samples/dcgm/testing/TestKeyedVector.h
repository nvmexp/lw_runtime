#ifndef TESTKEYEDVECTOR_H
#define TESTKEYEDVECTOR_H

#include "TestLwcmModule.h"
#include "keyedvector.h"

/*****************************************************************************/
class TestKeyedVector : public TestLwcmModule
{
public:
    TestKeyedVector();
    ~TestKeyedVector();

    /*************************************************************************/
    /* Inherited methods from TestLwcmModule */
    int Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

    /*************************************************************************/
private:

    /*************************************************************************/
    /*
     * Sub function of TestFindByKey
     */
    int TestFindByKeyAtSize(int testNelems);

    /*************************************************************************/
    /*
     * Actual test cases. These should return a status like below
     *
     * Returns 0 on success
     *        <0 on fatal error. Will abort entire framework
     *        >0 on non-fatal error
     *
     **/
    int TestTimingLinearInsert();
    int TestTimingRandomInsert();
    int TestLinearRemoveForward();
    int TestLinearRemoveByRangeForward();
    int TestLinearInsertBackward();
    int TestRemoveRangeByLwrsor();
    int TestFindByKey();
    int TestFindByIndex();
    int TestLinearInsertForward();

    /*************************************************************************/
    /*
     * Helper for verifying the elements of the keyed vector are indeed in order
     *
     * Returns 0 if OK
     *        !0 if keyed vector is corrupt
     *
     */
    int HelperVerifyOrder(keyedvector_p kv);

    /*************************************************************************/
    /*
     * Helper for verifying the cached size of the keyed vector against the
     * computed size of the keyed vector
     *
     * Returns 0 if OK
     *        !0 if keyed vector is corrupt
     *
     */
    int HelperVerifySize(keyedvector_p kv);

    /*************************************************************************/
    /* Virtual method inherited from TestLwcmModule */
    bool IncludeInDefaultList(void)
    {
        return false;
    }

    /*************************************************************************/
};


#endif //TESTKEYEDVECTOR_H
