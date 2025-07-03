

#include <atomic>

#ifdef __linux__

#include <sys/eventfd.h>

#include <unistd.h>

#endif

#include "lwscistream_common.h"

#include "common_includes.h"

#include "glob_test_vars.h"



using namespace LwSciStream;



LwSciError LwSciSyncAttrListClone(

     LwSciSyncAttrList origAttrList,

     LwSciSyncAttrList* newAttrList)

{

    if(test_lwscisync.LwSciSyncAttrListClone_fail == true)

    {

        test_lwscisync.LwSciSyncAttrListClone_fail = false;

      return LwSciError_BadParameter;

    }

    *newAttrList = origAttrList;

    return LwSciError_Success;

}





LwSciError LwSciSyncObjRef(

    LwSciSyncObj syncObj)

{

    return LwSciError_Success;

}





LwSciError LwSciBufAttrListClone(

  LwSciBufAttrList origAttrList,

  LwSciBufAttrList* newAttrList)

{

    *newAttrList = origAttrList;

    return LwSciError_Success;

}



LwSciError LwSciSyncFenceDup(

     const LwSciSyncFence* srcSyncFence,

     LwSciSyncFence* dstSyncFence)

{

    return LwSciError_Success;

}



LwSciError LwSciBufObjRef(LwSciBufObj bufObj)

{

    return LwSciError_Success;

}