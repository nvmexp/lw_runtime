//
// Copyright (c) 2020, LWPU CORPORATION. All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//
/// @file

#include <limits>
#include <functional>
#include "sciwrap.h"
#include "lwscistream_common.h"
#include "producer.h"
#include "pool.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "test_common.h"
#include "lwscistream_types.h"
#include "lwscibuf_internal.h"
#include "lwscistream_LwSciBufObjRef_mock.h"
#include "lwscistream_LwSciSyncAttrListClone_mock.h"
#include "lwscistream_LwSciSyncObjRef_mock.h"
#include "lwscistream_LwSciSyncFenceDup_mock.h"
#include "lwscistream_LwSciBufAttrListClone_mock.h"
#include "panic.h"

class lwsciwrap_unit_test : public LwSciStreamTest
{
public:
    lwsciwrap_unit_test()
    {
        // initialization code here
    }

    void SetUp()
    {
        // code here will execute just before the test ensues
    }

    void TearDown()
    {
        // code here will be called just after the test completes
        // ok to through exceptions from here if need be
    }

    ~lwsciwrap_unit_test()
    {
        // cleanup any pending stuff, but no exceptions allowed
    }
    // put in any custom data members that you need
};

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_getErr_Success}
 * @testcase{21730414}
 * @verify{19789035}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::getErr()}
 * @testbehavior{
 * Setup:
 * Instantiate an object of type LwSciWrap::SyncAttr.
 *
 * Call to getErr() API on LwSciWrap:SyncAttr object, should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::getErr()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_getErr_Success )
{
    /* initial set up */
    unsigned ret;
    LwSciWrap::SyncAttr attr;
    /*test case*/
    ret=attr.getErr();
    EXPECT_EQ(ret,LwSciError_Success);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_viewVal__Success}
 * @testcase{21730416}
 * @verify{19389000}
 * @verify{19389006}
 * @verify{19389012}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::viewVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncAttrList and initialize it with 1U
 * 2.Instantiate an LwSciWrap::SyncAttr object with the LwSciSyncAttrList declared in step 1.
 *
 * Call to viewVal() API on LwSciWrap:SyncAttr object, should return
 * the same LwSciSyncAttrList (1U) it was initialized with.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::viewVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_viewVal__Success)
{
    /* initial set up */
    LwSciSyncAttrList Synclist(1), ret;
    LwSciWrap::SyncAttr attr(Synclist);

    /* test case */

    ret=attr.viewVal();
    EXPECT_EQ(ret,Synclist);
}





/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_takeVal_Success}
 * @testcase{21730419}
 * @verify{19789044}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U
 * 2.Instantiate an LwSciWrap::BufObj object with LwSciBufObj declared in step 1,
 * and with paramOwn, paramDup arguments as false.
 *
 * Call to takeVal() API on LwSciWrap::BufObj object, should return LwSciError_Success and
 * ilwoke LwSciSyncObjRef() API once}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_takeVal_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    LwSciBufObj obj1(2);
    LwSciWrap::BufObj bufObj(obj1);


    /*test case*/
    EXPECT_CALL(nbrm, LwSciBufObjRef(obj1)).Times(1);
    EXPECT_EQ(LwSciError_Success, bufObj.takeVal(obj1));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nbrm));

}



/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_take_Success}
 * @testcase{21730421}
 * @verify{20140659}
 * @verify{19389030}
 * @verify{19389036}
 * @verify{19389042}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::take()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::BufObj object with LwSciBufObj declared in step 1,
 * and with paramOwn, paramDup arguments as true.
 *
 * Call to take() API on LwSciWrap::BufObj object created in step 2, should return
 * a new LwSciWrap::BufObj instance.
 * Call to viewVal() API on LwSciWrap::BufObj object created in step 2 should return 0U
 * Call to viewVal() API on new LwSciWrap::BufObj instance returned by take() call
 * should return LwSciBufObj as 2U.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::take()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_take_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    LwSciBufObj obj1(2U), ret;
    LwSciWrap::BufObj bufObj(obj1,true,true),bufObj1;

        /*test case*/
    //After moving to new instance the content from current
    // instance should be cleared
    bufObj1=bufObj.take();
    ret=bufObj.viewVal();
    EXPECT_NE(ret,obj1);
    //check whether content is moved to new wrapper instance
    ret=bufObj1.viewVal();
    EXPECT_EQ(ret,obj1);
}



/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncAttr_fCopy_Success}
 * @testcase{21730424}
 * @verify{19789053}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fCopy}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncAttrList and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncAttr
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncAttrList (created in step 1) as parameter should return LwSciError_Success
 * and it should ilwoke LwSciSyncAttrListClone() API once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncAttr_fCopy_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciSyncAttrList attr(2U);
    LwSciSyncAttrListCloneMock nsalcm;

    EXPECT_CALL(nsalcm, LwSciSyncAttrListClone(attr,_)).Times(1);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncAttr> obj;
    LwSciError errorType;


        /*test case*/
    errorType=obj.fCopy(attr);
    EXPECT_EQ(errorType,LwSciError_Success);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nsalcm));
}


/**
 * @testname{lwsciwrap_unit_test.Wrapinfo_WTSyncAttr_fFree_Success}
 * @testcase{21730427}
 * @verify{19789059}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fFree}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwsciSyncAttrlist and initialize it with 1U.
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncAttr
 * template type.
 *
 * Call to fFree() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncAttrList (created in step 1) as parameter should set the LwsciSyncAttrlist
 * to LwSciWrap::WrapInfo::cIlwalid.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fFree()}
 */
TEST_F(lwsciwrap_unit_test, Wrapinfo_WTSyncAttr_fFree_Success)
{
    /* initial set up */
    LwSciSyncAttrList attrlist(1U);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncAttr> obj;

        /*test case*/
    obj.fFree(attrlist);
    EXPECT_EQ(attrlist,obj.cIlwalid);
}




/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncAttr_fValid_true}
 * @testcase{21730430}
 * @verify{19789062}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwsciSyncAttrlist and initialize it with 1U.
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncAttr
 * template type.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncAttrList (created in step1) as parameter should return true}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncAttr_fValid_true)
{
    /* initial set up */
    LwSciSyncAttrList attr1(1U);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncAttr> obj;
    bool isTrue;

        /*test case*/
    isTrue=obj.fValid(attr1);
    EXPECT_TRUE(isTrue);
}



/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncObj_fCopy_Success}
 * @testcase{21730432}
 * @verify{19789071}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fCopy()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize with 1U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncObj
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncObj (created in step1) as parameter should return LwSciError_Success
 * and it should ilwoke LwSciSyncObjRef() API once}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */

TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncObj_fCopy_Success)
{
    /* initial set up */

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncObjRefMock nssorm;

    LwSciSyncObj syncobject(1U);

    EXPECT_CALL(nssorm, LwSciSyncObjRef(syncobject)).Times(1);


    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncObj> obj;
    LwSciError errorType;
        /*test case*/
    errorType=obj.fCopy(syncobject);
    EXPECT_EQ(errorType,LwSciError_Success);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssorm));
}



/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncObj_fFree_Success}
 * @testcase{21730435}
 * @verify{19789074}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fFree()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize it with 1U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncObj
 * template type.
 *
 * Call to fFree() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncObj (created in step1) as parameter, should set the LwSciSyncObj
 * to LwSciWrap::WrapInfo::cIlwalid}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fFree()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncObj_fFree_Success)
{
    /* initial set up */
    LwSciSyncObj syncObj(1U);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncObj> obj;
    unsigned ret;

        /*test case*/
    obj.fFree(syncObj);
    EXPECT_EQ(syncObj,obj.cIlwalid);
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncObj_fValid_true}
 * @testcase{21730438}
 * @verify{19789077}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize it with 1U.
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncObj
 * template type.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncObj (created in step1) as parameter should return true}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncObj_fValid_true)
{
    /* initial set up */
    LwSciSyncObj syncObj(1U);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncObj> obj;
    bool isTrue;

        /*test case*/
    isTrue=obj.fValid(syncObj);
    EXPECT_TRUE(isTrue);
}



/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncFence_fCopy_Success}
 * @testcase{21730441}
 * @verify{19789086}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fCopy()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence and initialize it with 1U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncFence
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncFence (created in step1) as parameter should return LwSciError_Success
 * and it should ilwoke LwSciSyncFenceDup() API once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncFence_fCopy_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncFenceDupMock nssfdm;

    LwSciSyncFence fenceobj1{(1U)};
    LwSciError errorType;

    EXPECT_CALL(nssfdm, LwSciSyncFenceDup(&fenceobj1,_)).Times(1);

    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncFence> obj;

        /*test case*/
    errorType=obj.fCopy(fenceobj1);
    EXPECT_EQ(errorType,LwSciError_Success);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssfdm));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncFence_fFree_Success}
 * @testcase{21730443}
 * @verify{19789089}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fFree()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence and initialize it with 1U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncFence
 * template type.
 *
 * Call to Free() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncFence (created in step1) as parameter should set the LwSciSyncFence
 * to LwSciWrap::WrapInfo::cIlwalid}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fFree()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncFence_fFree_Success)
{
    /* initial set up */
    LwSciSyncFence fenceObj{1U};
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncFence> obj;

        /*test case*/
    EXPECT_TRUE(memcmp(fenceObj.payload, obj.cIlwalid.payload, sizeof(fenceObj)));
    obj.fFree(fenceObj);
    EXPECT_FALSE(memcmp(fenceObj.payload, obj.cIlwalid.payload, sizeof(fenceObj)));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncFence_fValid_true}
 * @testcase{21730446}
 * @verify{19789092}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncFence
 * template type.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncFence (created in step1) as parameter should return true}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncFence_fValid_true)
{
    /* initial set up */
    LwSciSyncFence fenceObj{1U};
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncFence> obj;
    bool isTrue;

        /*test case*/
    isTrue=obj.fValid(fenceObj);
    EXPECT_TRUE(isTrue);
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufAttr_fCopy_Success}
 * @testcase{21730449}
 * @verify{19789104}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fCopy()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufAttr
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufAttrList (created in step1) as parameter should return LwSciError_Success
 * and should ilwoke LwSciBufAttrListClone() API once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufAttr_fCopy_Success)
{
        /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufAttrListCloneMock nsbalcm;
    LwSciBufAttrList attrlist(2U);

    EXPECT_CALL(nsbalcm, LwSciBufAttrListClone(attrlist,_)).Times(1);

    LwSciError errorType;
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufAttr> obj;

        /*test case*/
    errorType=obj.fCopy(attrlist);
    EXPECT_EQ(errorType,LwSciError_Success);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nsbalcm));
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufAttr_fFree_Success}
 * @testcase{21730452}
 * @verify{19789107}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fFree()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufAttr
 * template type.
 *
 * Call to fFree() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufAttrList (created in step1) as parameter, should set the LwSciBufAttrList
 * to LwSciWrap::WrapInfo::cIlwalid.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fFree()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufAttr_fFree_Success)
{
    /* initial set up */
    LwSciBufAttrList attrlist(2U);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufAttr> obj;
        /*test case*/
    obj.fFree(attrlist);
    EXPECT_EQ(attrlist,obj.cIlwalid);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufAttr_fValid_Success}
 * @testcase{21730455}
 * @verify{19789110}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufAttr
 * template type.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufAttrList (created in step1) as parameter should return true}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fFree()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufAttr_fValid_Success)
{
    /* initial set up */
    LwSciBufAttrList attrlist(2U);
    bool isTrue;
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufAttr> obj;

        /*test case*/
    isTrue=obj.fValid(attrlist);
    EXPECT_TRUE(isTrue);
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufObj_fCopy_Success}
 * @testcase{21730457}
 * @verify{19790043}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fCopy()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufObj
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufObj (created in step1) as parameter, should return LwSciError_Success
 * and it should ilwoke LwSciBufObjRef once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufObj_fCopy_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    LwSciBufObj bufObj(2U);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufObj> obj;
    LwSciError errorType;

    EXPECT_CALL(nbrm, LwSciBufObjRef(bufObj)).Times(1);

        /*test case*/
    errorType=obj.fCopy(bufObj);
    EXPECT_EQ(errorType,LwSciError_Success);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nbrm));
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufObj_fFree_Success}
 * @testcase{21730460}
 * @verify{19790046}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fFree()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufObj
 * template type.
 *
 * Call to fFree() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufObj (created in step1) as parameter, should set the LwSciBufObj to
 * LwSciWrap::WrapInfo::cIlwalid}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fFree()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufObj_fFree_Success)
{
    /* initial set up */
    LwSciBufObj lwscibufObj(2U);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufObj> obj;


        /*test case*/
    obj.fFree(lwscibufObj);
    EXPECT_EQ(lwscibufObj,obj.cIlwalid);
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufObj_fValid_true}
 * @testcase{21730463}
 * @verify{19790049}
 * @testpurpose{Test positive scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufObj
 * template type.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufObj (created in step1) as parameter should return true.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufObj_fValid_true)
{
    /* initial set up */
    LwSciBufObj lwscibufObj(2U);
    bool isTrue;
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufObj> obj;

        /*test case*/
    isTrue=obj.fValid(lwscibufObj);
    EXPECT_TRUE(isTrue);
}

/*Negative test specifications*/


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncAttr_fValid_false}
 * @testcase{21730465}
 * @verify{19789062}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwsciSyncAttrlist and initialize it with 1U.
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncAttr
 * template type.
 * 3.Call fFree() API on the LwSciWrap::WrapInfo object.
 * with the LwSciSyncAttrList (created in step 1) as parameter.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncAttrList (created in step1) as parameter should return false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncAttr_fValid_false)
{
    /* initial set up */
    LwSciSyncAttrList attr1(1U);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncAttr> obj;
    bool isTrue;
    obj.fFree(attr1);
        /*test case*/
    isTrue=obj.fValid(attr1);
    EXPECT_FALSE(isTrue);
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncObj_fValid_false}
 * @testcase{21730468}
 * @verify{19789077}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize it with 1U.
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncObj
 * template type.
 * 3.Call fFree() API on the LwSciWrap::WrapInfo object.
 * with the LwSciSyncObj (created in step 1) as parameter.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncObj (created in step1) as parameter should return false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncObj_fValid_false)
{
    /* initial set up */
    LwSciSyncObj syncObj(1);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncObj> obj;
    bool isTrue;
    obj.fFree(syncObj);
        /*test case*/
    isTrue=obj.fValid(syncObj);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncFence_fValid_false}
 * @testcase{21730470}
 * @verify{19789092}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncFence
 * template type.
 * 3.Call Free() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncFence (created in step1) as parameter.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncFence (created in step1) as parameter should return false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncFence_fValid_false)
{
    /* initial set up */
    LwSciSyncFence fenceObj{1U};
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncFence> obj;
    bool isTrue;
    obj.fFree(fenceObj);

        /*test case*/
    isTrue=obj.fValid(fenceObj);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufAttr_fValid_false}
 * @testcase{21730473}
 * @verify{19789110}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufAttr
 * template type.
 * 3.Call fFree() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufAttrList (created in step1) as parameter.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufAttrList (created in step1) as parameter should return false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufAttr_fValid_false)
{
    /* initial set up */
    LwSciBufAttrList attrlist(2U);
    bool isTrue;
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufAttr> obj;

        /*test case*/
    obj.fFree(attrlist);
    isTrue=obj.fValid(attrlist);
    EXPECT_FALSE(isTrue);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufObj_fValid_false}
 * @testcase{21730476}
 * @verify{19790049}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fValid()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufObj
 * template type.
 * 3.Call fFree() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufObj (created in step1) as parameter.
 *
 * Call to fValid() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufObj (created in step1) as parameter should return false}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fValid()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufObj_fValid_false)
{
    /* initial set up */
    LwSciBufObj lwscibufObj(2U);
    bool isTrue;
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufObj> obj;

        /*test case*/
    obj.fFree(lwscibufObj);
    isTrue=obj.fValid(lwscibufObj);
    EXPECT_FALSE(isTrue);
}

/* Additional positive test specifications*/

//For getErrr() API

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncObj_getErr_Success}
 * @testcase{21730479}
 * @verify{19789035}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::getErr()}
 * @testbehavior{
 * Setup:
 * Instantiate an object of type LwSciWrap::SyncObj
 *
 * Call to getErr() API on LwSciWrap:SyncObj object, should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::getErr()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncObj_getErr_Success)
{
    /* initial set up */
    unsigned ret;
    LwSciWrap::SyncObj sync;

    /*test case*/
    ret=sync.getErr();
    EXPECT_EQ(ret,LwSciError_Success);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncFence_getErr_Success}
 * @testcase{21730482}
 * @verify{19789035}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::getErr()}
 * @testbehavior{
 * Setup:
 * Instantiate an object of type LwSciWrap::SyncFence
 *
 * Call to getErr() API on LwSciWrap:SyncFence object, should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::getErr()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncFence_getErr_Success)
{
    /* initial set up */
    unsigned ret;
    LwSciWrap::SyncFence syncfence;

    /*test case*/
    ret=syncfence.getErr();
    EXPECT_EQ(ret,LwSciError_Success);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciBufAttrList_getErr_Success}
 * @testcase{21730486}
 * @verify{19789035}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::getErr()}
 * @testbehavior{
 * Setup:
 * Instantiate an object of type LwSciWrap::BufAttr
 *
 * Call to getErr() API on LwSciWrap:BufAttr object, should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::getErr()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciBufAttrList_getErr_Success)
{
    /* initial set up */
    unsigned ret;
    LwSciWrap::BufAttr bufattr;

    /*test case*/
    ret=bufattr.getErr();
    EXPECT_EQ(ret,LwSciError_Success);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciBufObj_getErr_Success}
 * @testcase{21730490}
 * @verify{19789035}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::getErr()}
 * @testbehavior{
 * Setup:
 * Instantiate an object of type LwSciWrap::BufObj
 *
 * Call to getErr() API on LwSciWrap:BufObj object, should return LwSciError_Success.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::getErr()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciBufObj_getErr_Success)
{
    /* initial set up */
    unsigned ret;
    LwSciWrap::BufObj bufobj;

    /*test case*/
    ret=bufobj.getErr();
    EXPECT_EQ(ret,LwSciError_Success);
}

//For viewVal() API

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncObj_viewVal_Success}
 * @testcase{21730494}
 * @verify{19789038}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::viewVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize it with 1U
 * 2.Instantiate an LwSciWrap::SyncObj object with the LwSciSyncObj declared in step 1.
 *
 * Call to viewVal() API on LwSciWrap:SyncObj object, should return
 * the same LwSciSyncObj (1U) it was initialized with.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::viewVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncObj_viewVal_Success)
{
    /* initial set up */
    LwSciSyncObj syncobj(1U), ret;
    LwSciWrap::SyncObj obj(syncobj);

    /* test case */

    ret=obj.viewVal();
    EXPECT_EQ(ret,syncobj);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncFence_viewVal_Success}
 * @testcase{21730505}
 * @verify{19789038}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::viewVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence and initialize it with 1U
 * 2.Instantiate an LwSciWrap::SyncFence object with the LwSciSyncFence declared in step 1.
 *
 * Call to viewVal() API on LwSciWrap:SyncFence object, should return
 * the same LwSciSyncFence (1U) it was initialized with.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::viewVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncFence_viewVal_Success)
{
    /* initial set up */
    LwSciSyncFence ret;
    LwSciSyncFence fenceobj{1U};
    LwSciWrap::SyncFence obj(fenceobj);

    /* test case */

    ret=obj.viewVal();
    EXPECT_EQ(ret.payload[0],1U);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciBufAttrList_viewVal_Success}
 * @testcase{21730510}
 * @verify{19789038}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::viewVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize it with 1U
 * 2.Instantiate an LwSciWrap::BufAttr object with the LwSciBufAttrList declared in step 1.
 *
 * Call to viewVal() API on LwSciWrap:BufAttr object, should return
 * the same LwSciBufAttrList (1U) it was initialized with.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::viewVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciBufAttrList_viewVal_Success)
{
    /* initial set up */

    LwSciBufAttrList buffList(1U), ret;
    LwSciWrap::BufAttr obj(buffList);

    /* test case */

    ret=obj.viewVal();
    EXPECT_EQ(ret,buffList);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciBufObj_viewVal_Success}
 * @testcase{21730515}
 * @verify{19789038}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::viewVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 1U
 * 2.Instantiate an LwSciWrap::BufObj object with the LwSciBufObj declared in step 1.
 *
 * Call to viewVal() API on LwSciWrap:BufObj object, should return
 * the same LwSciBufObj (1U) it was initialized with.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::viewVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciBufObj_viewVal_Success)
{
    /* initial set up */
    LwSciBufObj bufObj(1U), ret;
    LwSciWrap::BufObj obj(bufObj);

    /* test case */

    ret=obj.viewVal();
    EXPECT_EQ(ret,bufObj);
}

// For takeVal() API


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncObj_takeVal_Success}
 * @testcase{21730518}
 * @verify{19789044}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize it with 2U
 * 2.Instantiate an LwSciWrap::SyncObj object with LwSciSyncObj declared in step 1,
 * and with paramOwn, paramDup arguments as false.
 *
 * Call to takeVal() API on LwSciWrap::BufObj object, should return LwSciError_Success and
 * ilwoke LwSciSyncObjRef() API once}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncObj_takeVal_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncObjRefMock nssorm;

    LwSciSyncObj obj1(2U);
    LwSciWrap::SyncObj syncobj(obj1);


    /*test case*/
    EXPECT_CALL(nssorm, LwSciSyncObjRef(obj1)).Times(1);
    EXPECT_EQ(LwSciError_Success, syncobj.takeVal(obj1));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssorm));

}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncFence_takeVal_Success}
 * @testcase{21730523}
 * @verify{19789044}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence and initialize it with 2U
 * 2.Instantiate an LwSciWrap::SyncFence object with LwSciSyncFence declared in step 1,
 * and with paramOwn, paramDup arguments as true.
 *
 * Call to takeVal() API on LwSciWrap::BufObj object, should return LwSciError_Success and
 * should not ilwoke LwSciSyncFenceDup() API}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncFence_takeVal_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncFenceDupMock nssfdm;

    LwSciSyncFence obj1{(2U)};
    LwSciWrap::SyncFence syncfence(obj1,true,true,LwSciError_Success);


    /*test case*/
    EXPECT_CALL(nssfdm, LwSciSyncFenceDup(&obj1,_)).Times(0);
    EXPECT_EQ(LwSciError_Success, syncfence.takeVal(obj1));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssfdm));

}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciBufAttrList_takeVal_Success}
 * @testcase{21730528}
 * @verify{19789044}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize it with 2U
 * 2.Instantiate an LwSciWrap::BufAttr object with LwSciBufAttrList declared in step 1,
 * and with paramOwn, paramDup arguments as false.
 *
 * Call to takeVal() API on LwSciWrap::BufObj object, should return LwSciError_Success and
 * ilwoke LwSciSyncAttrListClone() API once}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciBufAttrList_takeVal_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufAttrListCloneMock nsbalcm;

    LwSciBufAttrList obj1{(2U)};
    LwSciWrap::BufAttr bufattr(obj1);


    /*test case*/
    EXPECT_CALL(nsbalcm, LwSciBufAttrListClone(obj1,_)).Times(1);
    EXPECT_EQ(LwSciError_Success, bufattr.takeVal(obj1));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nsbalcm));

}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncAttrList_takeVal_Success}
 * @testcase{21730533}
 * @verify{19789044}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncAttrList and initialize it with 2U
 * 2.Instantiate an LwSciWrap::SyncAttr object with LwSciSyncAttrList declared in step 1,
 * and with paramOwn, paramDup arguments as false.
 *
 * Call to takeVal() API on LwSciWrap::BufObj object, should return LwSciError_Success and
 * ilwoke LwSciSyncAttrListClone() API once}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncAttrList_takeVal_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncAttrListCloneMock nssalcm;

    LwSciSyncAttrList obj1(2U);
    LwSciWrap::SyncAttr syncattr(obj1);


    /*test case*/
    EXPECT_CALL(nssalcm, LwSciSyncAttrListClone(obj1,_)).Times(1);
    EXPECT_EQ(LwSciError_Success, syncattr.takeVal(obj1));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssalcm));

}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncObj_take_Success}
 * @testcase{21730538}
 * @verify{20140659}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::take()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::SyncObj object with LwSciSyncObj declared in step 1,
 * and with paramOwn, paramDup arguments as true.
 *
 * Call to take() API on LwSciWrap::SyncObj object created in step 2, should return
 * a new LwSciWrap::SyncObj instance.
 * Call to viewVal() API on LwSciWrap::SyncObj object created in step 2 should return 0U
 * Call to viewVal() API on new LwSciWrap::SyncObj instance returned by take() call
 * should return LwSciSyncObj as 2U.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::take()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncObj_take_Success)
{
    /* initial set up */
     using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncObjRefMock nssorm;

    LwSciSyncObj obj1(2U), ret;
    LwSciWrap::SyncObj syncobj(obj1,true,true),syncobj1;

        /*test case*/
    //After moving to new instance the content from current
    // instance should be cleared

    syncobj1=syncobj.take();
    ret=syncobj.viewVal();
    EXPECT_NE(ret,obj1);
    //check whether content is moved to new wrapper instance
    ret=syncobj1.viewVal();
    EXPECT_EQ(ret,obj1);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncFence_take_Success}
 * @testcase{21730543}
 * @verify{20140659}
 * @verify{19788867}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::take()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::SyncFence object with SyncFence declared in step 1,
 * and with paramOwn, paramDup arguments as true.
 *
 * Call to take() API on LwSciWrap::SyncFence object created in step 2, should return
 * a new LwSciWrap::SyncFence instance.
 * Call to viewVal() API on LwSciWrap::SyncFence object created in step 2 should return payload
 * equal to cIlwalid.
 * Call to viewVal() API on new LwSciWrap::SyncFence instance returned by take() call should return
 * payload equal to the payload of object obj1 created in step 1.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::take()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncFence_take_Success)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncFenceDupMock nssfdm;

//   unsigned ret;
    LwSciSyncFence obj1{(2U)},ret;
    LwSciWrap::SyncFence syncfence(obj1,true,true),syncfence1;
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncFence> obj;
        /*test case*/
    //After moving to new instance the content from current
    // instance should be cleared


    syncfence1=syncfence.take();
    ret=syncfence.viewVal();
    EXPECT_FALSE(memcmp(ret.payload, obj.cIlwalid.payload, sizeof(ret)));
   //check whether content is moved to new wrapper instance
    ret=syncfence1.viewVal();
    EXPECT_FALSE(memcmp(ret.payload, obj1.payload, sizeof(obj1)));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciBufAttrList_take_Success}
 * @testcase{21730548}
 * @verify{20140659}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::take()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::BufAttr object with LwSciBufAttrList declared in step 1,
 * and with paramOwn, paramDup arguments as true.
 *
 * Call to take() API on LwSciWrap::BufAttr object created in step 2, should return
 * a new LwSciWrap::BufAttr instance.
 * Call to viewVal() API on LwSciWrap::BufAttr object created in step 2 should return 0U
 * Call to viewVal() API on new LwSciWrap::BufAttr instance returned by take() call
 * should return LwSciBufAttrList as 2U.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::take()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciBufAttrList_take_Success)
{
    /* initial set up */
    LwSciBufAttrListCloneMock nsbalcm;
    LwSciBufAttrList obj1(2U), ret;
    LwSciWrap::BufAttr bufattr(obj1,true,true),bufattr1;

        /*test case*/
    //After moving to new instance the content from current
    // instance should be cleared
    ret=bufattr.viewVal();
    EXPECT_EQ(ret,obj1);

    bufattr1=bufattr.take();

    ret=bufattr.viewVal();
    EXPECT_NE(ret,obj1);
    //check whether content is moved to new wrapper instance
    ret=bufattr1.viewVal();
    EXPECT_EQ(ret,obj1);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciSyncAttrList_take_Success}
 * @testcase{21730555}
 * @verify{20140659}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::take()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncAttrList and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::SyncAttr object with LwSciSyncAttrList declared in step 1,
 * and with paramOwn, paramDup arguments as true.
 *
 * Call to take() API on LwSciWrap::SyncAttr object created in step 2, should return
 * a new LwSciWrap::SyncAttr instance.
 * Call to viewVal() API on LwSciWrap::SyncAttr object created in step 2 should return 0U
 * Call to viewVal() API on new LwSciWrap::SyncAttr instance returned by take() call
 * should return LwSciSyncAttrList as 2U.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::take()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciSyncAttrList_take_Success)
{
    /* initial set up */
    LwSciSyncAttrList obj1(2U), ret;
    LwSciWrap::SyncAttr syncattr(obj1,true,false),syncattr1;

        /*test case*/
    //After moving to new instance the content from current
    // instance should be cleared
    ret=syncattr.viewVal();
    EXPECT_EQ(ret,obj1);

    syncattr1=syncattr.take();

    ret=syncattr.viewVal();
    EXPECT_NE(ret,obj1);
    //check whether content is moved to new wrapper instance
    ret=syncattr1.viewVal();
    EXPECT_EQ(ret,obj1);
}

/* additional coverage +ve cases */


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciBufOb_takeVal_Success1}
 * @testcase{21730558}
 * @verify{19789044}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U
 * 2.Instantiate an LwSciWrap::BufObj object with LwSciBufObj declared in step 1
 * and with paramOwn, paramDup arguments as true.
 *
 * Call to takeVal() API on LwSciWrap::BufObj object, should return LwSciError_Success and
 * should not ilwoke LwSciBufObjRef() API}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciBufOb_takeVal_Success1)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    LwSciBufObj obj1(2U);
    LwSciWrap::BufObj bufObj(obj1,true,true);


    /*test case*/
    EXPECT_CALL(nbrm, LwSciBufObjRef(obj1)).Times(0);
    EXPECT_EQ(LwSciError_Success, bufObj.takeVal(obj1));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nbrm));

}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_LwSciBufOb_takeVal_Success2}
 * @testcase{21730567}
 * @verify{19789044}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U
 * 2.Instantiate an LwSciWrap::BufObj object with LwSciBufObj declared in step 1
 * and with paramOwn, paramDup arguments as false(with default values).
 *
 * Call to takeVal() API on LwSciWrap::BufObj object, should return LwSciError_Success
 * and ilwoke LwSciBufObjRef() API once}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_LwSciBufOb_takeVal_Success2)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    LwSciBufObj obj1(2U);
    LwSciWrap::BufObj bufObj(obj1,false,false);


    /*test case*/
    EXPECT_CALL(nbrm, LwSciBufObjRef(obj1)).Times(1);
    EXPECT_EQ(LwSciError_Success, bufObj.takeVal(obj1));
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nbrm));

}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_getErr_BadParameter}
 * @testcase{21730571}
 * @verify{19789035}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::getErr()}
 * @testbehavior{
 * Setup:
 * 1.Declare LwSciSyncAttrList and initialize it with 2U
 * 2.Instantiate an object of type LwSciWrap::SyncAttr object with LwSciSyncAttrList declared in
 * step 1 and with paramOwn(as true),paramDup(as true),LwSciError_BadParameter arguments.
 *
 * Call to getErr() API on LwSciWrap:SyncAttr object, should return LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::getErr()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_getErr_BadParameter)
{
    /* initial set up */
    unsigned ret;
    LwSciSyncAttrList Synclist(2U);
    LwSciWrap::SyncAttr attr(Synclist,true,true,LwSciError_BadParameter);

    /*test case*/
    ret=attr.getErr();
    EXPECT_EQ(ret,LwSciError_BadParameter);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_take_Success1}
 * @testcase{21730576}
 * @verify{20140659}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::take()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::BufObj object with LwSciBufObj declared in step 1,
 * and with paramOwn(as a true), paramDup(as a false) arguments.
 *
 * Call to take() API on LwSciWrap::BufObj object created in step 2, should return
 * a new LwSciWrap::BufObj instance.
 * Call to viewVal() API on LwSciWrap::BufObj object created in step 2 should return 0U
 * Call to viewVal() API on new LwSciWrap::BufObj instance returned by take() call should return
 * LwSciBufObj as 2U.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::take()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_take_Success1)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    LwSciBufObj obj1(2U), ret;
    LwSciWrap::BufObj bufObj(obj1,true,false),bufObj1;

        /*test case*/
    //After moving to new instance the content from current
    // instance should be cleared

    bufObj1=bufObj.take();
    ret=bufObj.viewVal();
    EXPECT_NE(ret,obj1);
    //check whether content is moved to new wrapper instance
    ret=bufObj1.viewVal();
    EXPECT_EQ(ret,obj1);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_take_Success2}
 * @testcase{21822428}
 * @verify{20140659}
 * @verify{19788864}
 * @testpurpose{Test positive scenario of LwSciWrap::Wrapper::take()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::BufObj object with LwSciBufObj declared in step 1,
 * and with paramOwn(as a false), paramDup(as a false) arguments.
 *
 * Call to take() API on LwSciWrap::BufObj object created in step 2, should return
 * a new LwSciWrap::BufObj instance.
 * Call to viewVal() API on LwSciWrap::BufObj object created in step 2 should return 2U
 * Call to viewVal() API on new LwSciWrap::BufObj instance returned by take() call should return
 * LwSciBufObj as 2U.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper::take()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_take_Success2)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    LwSciBufObj obj1(2U), ret;
    LwSciWrap::BufObj bufObj(obj1),bufObj1;

        /*test case*/
    //  content from current
    // instance should be copied new instance

    bufObj1=bufObj.take();
    ret=bufObj.viewVal();
    EXPECT_EQ(ret,obj1);
    //check whether content is copied to new wrapper instance
    ret=bufObj1.viewVal();
    EXPECT_EQ(ret,obj1);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncAttr_fCopy_BadParameter}
 * @testcase{21822429}
 * @verify{19789053}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fCopy}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncAttrList and initialize it with NULL.
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncAttr
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncAttrList (created in step 1) as parameter should return LwSciError_BadParameter
 * and it should ilwoke LwSciSyncAttrListClone() API once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncAttr_fCopy_BadParameter)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;

    LwSciSyncAttrList attr(NULL);
    LwSciSyncAttrListCloneMock nsalcm;

    EXPECT_CALL(nsalcm, LwSciSyncAttrListClone(attr,_)).Times(1)
      .WillRepeatedly(Return(LwSciError_BadParameter));;
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncAttr> obj;
    LwSciError errorType;


        /*test case*/
   // obj.fFree();
    errorType=obj.fCopy(attr);
    EXPECT_EQ(errorType,LwSciError_BadParameter);

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nsalcm));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncFence_fCopy_BadParameter}
 * @testcase{21822430}
 * @verify{19789086}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fCopy()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence and initialize it with NULL.
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncFence
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncFence (created in step1) as parameter should return LwSciError_BadParameter
 * and it should ilwoke LwSciSyncFenceDup() API once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncFence_fCopy_BadParameter)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncFenceDupMock nssfdm;

    LwSciSyncFence fenceobj1{(NULL)};
    LwSciError errorType;

    EXPECT_CALL(nssfdm, LwSciSyncFenceDup(&fenceobj1,_)).Times(1)
      .WillRepeatedly(Return(LwSciError_BadParameter));;

    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncFence> obj;

        /*test case*/
    errorType=obj.fCopy(fenceobj1);
    EXPECT_EQ(errorType,LwSciError_BadParameter);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssfdm));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufAttr_fCopy_BadParameter}
 * @testcase{21822431}
 * @verify{19789104}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fCopy()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize it with NULL.
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufAttr
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufAttrList (created in step1) as parameter should return LwSciError_BadParameter
 * and should ilwoke LwSciBufAttrListClone() API once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufAttr_fCopy_BadParameter)
{
        /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufAttrListCloneMock nsbalcm;
    LwSciBufAttrList attrlist(NULL);

    EXPECT_CALL(nsbalcm, LwSciBufAttrListClone(attrlist,_)).Times(1)
      .WillRepeatedly(Return(LwSciError_BadParameter));;

    LwSciError errorType;
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufAttr> obj;

        /*test case*/
    errorType=obj.fCopy(attrlist);
    EXPECT_EQ(errorType,LwSciError_BadParameter);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nsbalcm));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTBufObj_fCopy_BadParameter}
 * @testcase{21822432}
 * @verify{19790043}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fCopy()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with NULL
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTBufObj
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciBufObj (created in step1) as parameter, should return LwSciError_BadParameter
 * and it should ilwoke LwSciBufObjRef once.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTBufObj_fCopy_BadParameter)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    LwSciBufObj bufObj(NULL);
    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTBufObj> obj;
    LwSciError errorType;

    EXPECT_CALL(nbrm, LwSciBufObjRef(bufObj)).Times(1)
    .WillRepeatedly(Return(LwSciError_BadParameter));;

        /*test case*/
    errorType=obj.fCopy(bufObj);
    EXPECT_EQ(errorType,LwSciError_BadParameter);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nbrm));
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapinfo_WTSyncObj_fCopy_BadParameter}
 * @testcase{21822433}
 * @verify{19789071}
 * @testpurpose{Test negative scenario of LwSciWrap::WrapInfo::fCopy()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize with NULL
 * 2.Instantiate an object of type LwSciWrap::WrapInfo with LwSciWrap::WrapType::WTSyncObj
 * template type.
 *
 * Call to fCopy() API on the LwSciWrap::WrapInfo object
 * with the LwSciSyncObj (created in step1) as parameter should return LwSciError_BadParameter
 * and it should ilwoke LwSciSyncObjRef() API once}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::WrapInfo::fCopy()}
 */

TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapinfo_WTSyncObj_fCopy_BadParameter)
{
    /* initial set up */

    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncObjRefMock nssorm;

    LwSciSyncObj syncobject(NULL);

    EXPECT_CALL(nssorm, LwSciSyncObjRef(syncobject)).Times(1).
      WillRepeatedly(Return(LwSciError_BadParameter));;

    LwSciWrap::WrapInfo<LwSciWrap::WrapType::WTSyncObj> obj;
    LwSciError errorType;
        /*test case*/
    errorType=obj.fCopy(syncobject);
    EXPECT_EQ(errorType,LwSciError_BadParameter);
    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssorm));
}


/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_object_creation}
 * @testcase{21822434}
 * @verify{19788867}
 * @verify{19788861}
 * @testpurpose{Test object creation of LwSciWrap::Wrapper}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::BufObj object with LwSciBufObj declared in step 1,
 * and with paramOwn(as a false), paramDup(as a true) arguments.
 * }
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.}
 * @verifyFunction{LwSciWrap::Wrapper}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;

    /*test case*/
    LwSciBufObj obj1(2U);
    LwSciWrap::BufObj bufObj(obj1,false,true);
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_takeVal_BadParameter}
 * @testcase{22810068}
 * @verify{19789044}
 * @testpurpose{Test negative scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufObj and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::BufObj object with LwSciBufObj declared in step 1,
 * and with paramOwn(as a true), paramDup(as a true) arguments.
 *
 * Call to getErr() API and takeVal() API on LwSciWrap:BufObj object, should return
 * LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - LwSciBufObjRef() returns LwSciError_BadParameter.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_takeVal_BadParameter)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufObjRefMock nbrm;
    unsigned ret;

    LwSciBufObj obj1(2U);

    EXPECT_CALL(nbrm, LwSciBufObjRef(obj1)).Times(1)
        .WillRepeatedly(Return(LwSciError_BadParameter));

    LwSciWrap::BufObj bufObj(obj1,true,true);

    ret=bufObj.getErr();
    EXPECT_EQ(ret,LwSciError_BadParameter);

    EXPECT_EQ(LwSciError_BadParameter, bufObj.takeVal(obj1));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nbrm));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_takeVal_BadParameter2}
 * @testcase{22059869}
 * @verify{19789044}
 * @testpurpose{Test negative scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciBufAttrList and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::BufAttr object with LwSciBufAttrList declared in step 1,
 * and with paramOwn(as a true), paramDup(as a true) arguments.
 *
 * Call to getErr() API and takeVal() API on LwSciWrap:BufAttr object, should return
 * LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - LwSciBufAttrListClone() returns LwSciError_BadParameter.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_takeVal_BadParameter2)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciBufAttrListCloneMock nsbalcm;
    unsigned ret;

    LwSciBufAttrList attrlist(2U);

    EXPECT_CALL(nsbalcm, LwSciBufAttrListClone(attrlist,_)).Times(1)
        .WillRepeatedly(Return(LwSciError_BadParameter));

    LwSciWrap::BufAttr bufAttr(attrlist,true,true);

    ret=bufAttr.getErr();
    EXPECT_EQ(ret,LwSciError_BadParameter);

    EXPECT_EQ(LwSciError_BadParameter, bufAttr.takeVal(attrlist));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nsbalcm));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_takeVal_BadParameter3}
 * @testcase{22059872}
 * @verify{19789044}
 * @testpurpose{Test negative scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncFence and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::SyncFence object with LwSciSyncFence declared in step 1,
 * and with paramOwn(as a true), paramDup(as a true) arguments.
 *
 * Call to getErr() API and takeVal() API on LwSciWrap:SyncFence object, should return
 * LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - LwSciSyncFenceDup() returns LwSciError_BadParameter.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_takeVal_BadParameter3)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncFenceDupMock nssfdm;
    uint32_t ret;

    LwSciSyncFence fenceobj1 {(2U)};

    EXPECT_CALL(nssfdm, LwSciSyncFenceDup(_,_)).Times(1)
        .WillRepeatedly(Return(LwSciError_BadParameter));

    LwSciWrap::SyncFence syncFence(fenceobj1,true,true);

    ret=syncFence.getErr();
    EXPECT_EQ(ret,LwSciError_BadParameter);

    EXPECT_EQ(LwSciError_BadParameter, syncFence.takeVal(fenceobj1));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssfdm));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_takeVal_BadParameter4}
 * @testcase{22060007}
 * @verify{19789044}
 * @testpurpose{Test negative scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncObj and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::SyncObj object with LwSciSyncObj declared in step 1,
 * and with paramOwn(as a true), paramDup(as a true) arguments.
 *
 * Call to getErr() API and takeVal() API on LwSciWrap:SyncObj object, should return
 * LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - LwSciSyncObjRef() returns LwSciError_BadParameter.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_takeVal_BadParameter4)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncObjRefMock nssorm;
    uint32_t ret;

    LwSciSyncObj syncobject(1U);

    EXPECT_CALL(nssorm, LwSciSyncObjRef(syncobject)).Times(1)
        .WillRepeatedly(Return(LwSciError_BadParameter));

    LwSciWrap::SyncObj syncObj(syncobject,true,true);

    ret=syncObj.getErr();
    EXPECT_EQ(ret,LwSciError_BadParameter);

    EXPECT_EQ(LwSciError_BadParameter, syncObj.takeVal(syncobject));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nssorm));
}

/**
 * @testname{lwsciwrap_unit_test.LwSciWrap_Wrapper_takeVal_BadParameter5}
 * @testcase{22060010}
 * @verify{19789044}
 * @testpurpose{Test negative scenario of LwSciWrap::Wrapper::takeVal()}
 * @testbehavior{
 * Setup:
 * 1.Declare an LwSciSyncAttrList and initialize it with 2U.
 * 2.Instantiate an LwSciWrap::SyncAttr object with LwSciSyncAttrList declared in step 1,
 * and with paramOwn(as a true), paramDup(as a true) arguments.
 *
 * Call to getErr() API and takeVal() API on LwSciWrap:SyncAttr object, should return
 * LwSciError_BadParameter.}
 * @testmethod{Requirements Based}
 * @casederiv{Analysis of Requirement}
 * @testelw{t194}
 * @testconfig{No special test configuration required. Test binary is usable directly.}
 * @testinput{No input external to the test application is required.
 * Precondition:
 *   Following LwSciStream API calls are replaced with mocks:
 *      - LwSciSyncAttrListClone() returns LwSciError_BadParameter.}
 * @verifyFunction{LwSciWrap::Wrapper::takeVal()}
 */
TEST_F(lwsciwrap_unit_test, LwSciWrap_Wrapper_takeVal_BadParameter5)
{
    /* initial set up */
    using ::testing::_;
    using ::testing::Mock;
    using ::testing::Return;
    LwSciSyncAttrListCloneMock nsalcm;
    uint32_t ret;

    LwSciSyncAttrList attr(2U);

    EXPECT_CALL(nsalcm, LwSciSyncAttrListClone(attr,_)).Times(1)
        .WillRepeatedly(Return(LwSciError_BadParameter));

    LwSciWrap::SyncAttr syncAttr(attr,true,true);

    ret=syncAttr.getErr();
    EXPECT_EQ(ret,LwSciError_BadParameter);

    EXPECT_EQ(LwSciError_BadParameter, syncAttr.takeVal(attr));

    EXPECT_TRUE(Mock::VerifyAndClearExpectations(&nsalcm));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


