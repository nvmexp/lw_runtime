/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2009-2009 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#define DEFINE_OBJGPU
#include "regops.h"
#include "utility.h"

//dummy callback function used in tests
LwU32 readCallbackFun(void *params)
{
    int *param = (int*)params;
    //return the i/p arg as read value
    return *param;
}

//dummy callback function used in tests
LwU32 writeCallbackFun(void *params, LwU32 value)
{
    int *param = (int*)params;

    //write the multiplication of i/p param & value to be written
    return value*(*param);
}

void install_will_return_always_with_readRegOpHead_null()
{
    int addr = 0;
    int value = 0xabc;
    int status;
    //excercise
    status = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr, value);

    //verify
    UNIT_ASSERT(status == REGOP_WILL_RETURN_ALWAYS_APLLIED);

    //teardown
    destroyRegopLists();

}

void install_will_return_always_node_absent_head_present()
{
    int addr1 = 0, addr2 = 0xabd;
    int value1 = 0xabc, value2 = 0xabf;
    int status1, status2;
    //excercise
    status1 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr1, value1);
    status2 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr2, value2);

    //verify
    UNIT_ASSERT(status1 == REGOP_WILL_RETURN_ALWAYS_APLLIED);
    UNIT_ASSERT(status2 == REGOP_WILL_RETURN_ALWAYS_APLLIED);

    //teardown
    destroyRegopLists();
}

void install_will_return_always_more_than_once()
{
    int addr = 1;
    int value1 = 0xabc;
    int value2 = 0xabd;
    int status1, status2;
    //exercise
    status1 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr, value1);
    status2 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr, value2);

    //verify
    UNIT_ASSERT(status1 == REGOP_WILL_RETURN_ALWAYS_APLLIED);
    UNIT_ASSERT(status2 == REGOP_WILL_RETURN_ALWAYS_EROOR);

    //teardown
    destroyRegopLists();
}

//******************************************************************************************
void install_regopRead_callback_with_readRegOpHead_null()
{
    int addr = 2;
    int value = 0xabc;
    int status;

    //exercise
    status = UTAPI_INSTALL_READ_CALLBACK(addr, (void *)value, (void *)value);

    //verify
    UNIT_ASSERT(status == REGOP_CALLBACK_APLLIED);

    //teardown
    destroyRegopLists();

}

void install_regopRead_callback_with_observerNode_present()
{

    int addr = 3;
    int value = 0xabc;
    int status1,status2;

    //exercise
    status1 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr, value);
    status2 = UTAPI_INSTALL_READ_CALLBACK(addr, (void *)value, (void *)value);

    //verify
    UNIT_ASSERT(status1 == REGOP_WILL_RETURN_ALWAYS_APLLIED);
    UNIT_ASSERT(status2 == REGOP_CALLBACK_ERROR);

    //teardown
    destroyRegopLists();

}

void install_regopRead_callback_with_readRegOpHead_not_null()
{

    int addr1 = 4, addr2 = 5;
    int value = 0xabc;
    int status1,status2;

    //exercise
    status1 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr1, value);
    status2 = UTAPI_INSTALL_READ_CALLBACK(addr2, (void *)value, (void *)value);

    //verify
    UNIT_ASSERT(status1 == REGOP_WILL_RETURN_ALWAYS_APLLIED);
    UNIT_ASSERT(status2 == REGOP_CALLBACK_APLLIED);

    //teardown
    destroyRegopLists();

}
//************************************************************************************************
void install_regopRead_willreturn_count_with_readRegOpHead_null()
{
    int addr = 6;
    int value = 0xabc;
    int count = 2;
    int status;

    //exercise
    status = UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr, value, count);

    //verify
    UNIT_ASSERT(status == REGOP_WILL_RETURN_FOR_COUNT_APLLIED);

    //teardown
    destroyRegopLists();

}

void install_regopRead_willreturn_count_with_observerNode_present()
{

    int addr = 7;
    int value1 = 0xabc, value2 = 0xabd;
    int status1,status2;
    int count1=4, count2=3;

    //exercise
    status1 = UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr, value1, count1);
    status2 = UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr, value2, count2);
    //verify
    UNIT_ASSERT(status1 == REGOP_WILL_RETURN_FOR_COUNT_APLLIED);
    UNIT_ASSERT(status2 == REGOP_WILL_RETURN_FOR_COUNT_APLLIED);

    //teardown
    destroyRegopLists();
}

void install_regopRead_willreturn_count_with_readRegOpHead_not_null()
{

    int addr1 = 8, addr2 = 9;
    int value1 = 0xabc, value2 = 0xabd;
    int status1,status2;
    int count =4;

    //exercise
    status1 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr1, value1);
    status2 = UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr2, value2, count);

    //verify
    UNIT_ASSERT(status1 == REGOP_WILL_RETURN_ALWAYS_APLLIED);
    UNIT_ASSERT(status2 == REGOP_WILL_RETURN_FOR_COUNT_APLLIED);

    //teardown
    destroyRegopLists();

}

void install_regopRead_willreturn_count_with_willreturn_always()
{

    int addr = 10;
    int value1 = 0xabc, value2 = 0xabd;
    int status1,status2;
    int count =4;

    //exercise
    status1 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr, value1);
    status2 = UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr, value2, count);

    //verify
    UNIT_ASSERT(status1 == REGOP_WILL_RETURN_ALWAYS_APLLIED);
    UNIT_ASSERT(status2 == REGOP_WILL_RETURN_FOR_COUNT_ERROR);

    //teardown
    destroyRegopLists();

}

void install_regopRead_willreturn_count_with_callback()
{

    int addr = 11;
    int value = 0xabc;
    int status1,status2;
    int count = 4;

    //exercise
    status1 = UTAPI_INSTALL_READ_CALLBACK(addr, (void *)value, (void *)value);
    status2 = UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr, value, count);

    //verify
    UNIT_ASSERT(status1 == REGOP_CALLBACK_APLLIED);
    UNIT_ASSERT(status2 == REGOP_WILL_RETURN_FOR_COUNT_ERROR);

    //teardown
    destroyRegopLists();

}
//******************************************************************************************
void intall_write_mirror_with_head_null()
{
    int addr = 0xabc;
    int value = 0xcde;
    int status;

    //exercise
    status = UTAPI_INSTALL_WRITE_MIRROR(addr, value);

    //verify
    UNIT_ASSERT(status == REGOP_WRITE_MIRROR_APPLIED);

    //teardown
    destroyRegopLists();
}

void install_write_mirror_head_not_null_regopnode_not_present()
{
    int addr1 = 0xabc, addr2 = 0xcde;
    int value = 0x123;
    int status1, status2;

    //exercise
    status1 = UTAPI_INSTALL_WRITE_MIRROR(addr1, value);
    status2 = UTAPI_INSTALL_WRITE_MIRROR(addr2, value);

    //verify
    UNIT_ASSERT(status1 == REGOP_WRITE_MIRROR_APPLIED);
    UNIT_ASSERT(status2 == REGOP_WRITE_MIRROR_APPLIED);

    //teardown
    destroyRegopLists();
}

void install_write_mirror_with_no_other_mirror_present()
{
    int addr = 0xabc;
    int value = 0xcde;
    int status1, status2;

    //exercise
    status1 = UTAPI_INSTALL_WRITE_CALLBACK(addr, writeCallbackFun, (void*)addr);
    status2 = UTAPI_INSTALL_WRITE_MIRROR(addr, value);

    //verify
    UNIT_ASSERT(status1 == REGOP_WRITE_CALLBACK_APPLIED);
    UNIT_ASSERT(status2 == REGOP_WRITE_MIRROR_APPLIED);

    //teardown
    destroyRegopLists();
}

void install_write_mirror_with_another_mirror_present()
{

    int addr = 0xabc;
    int value1 = 0xcde, value2 = 0xadf;
    int status1, status2;

    //exercise
    status1 = UTAPI_INSTALL_WRITE_MIRROR(addr, value1);
    status2 = UTAPI_INSTALL_WRITE_MIRROR(addr, value2);

    //verify
    UNIT_ASSERT(status1 == REGOP_WRITE_MIRROR_APPLIED);
    UNIT_ASSERT(status2 == REGOP_WRITE_MIRROR_APPLIED);

    //teardown
    destroyRegopLists();
}

//*********************************************************************************************8
void install_write_callback_wiht_no_head()
{
    int addr = 0xabc;
    int value =0x123;
    int status;

    //exercise
    status = UTAPI_INSTALL_WRITE_CALLBACK(addr, writeCallbackFun, (void*)&value);

    //verify
    UNIT_ASSERT(status == REGOP_WRITE_CALLBACK_APPLIED);

    //teardown
    destroyRegopLists();
}

void install_write_callback_with_head_and_regopnode_not_present()
{
    int addr1 = 0xabc, addr2 = 0xcde;
    int value = 0x123;
    int status1, status2;

    //exercise
    status1 = UTAPI_INSTALL_WRITE_MIRROR(addr1, value);
    status2 = UTAPI_INSTALL_WRITE_CALLBACK(addr2, writeCallbackFun, (void*)&value);

    //verify
    UNIT_ASSERT(status1 == REGOP_WRITE_MIRROR_APPLIED);
    UNIT_ASSERT(status2 == REGOP_WRITE_CALLBACK_APPLIED);

    //teardown
    destroyRegopLists();
}

void install_write_callback_with_regponode_present()
{
    int addr1 = 0xabc, addr2 = addr1;
    int value = 0x123;
    int status1, status2;

    //exercise
    status1 = UTAPI_INSTALL_WRITE_MIRROR(addr1, value);
    status2 = UTAPI_INSTALL_WRITE_CALLBACK(addr2, writeCallbackFun, (void*)&value);

    //verify
    UNIT_ASSERT(status1 == REGOP_WRITE_MIRROR_APPLIED);
    UNIT_ASSERT(status2 == REGOP_WRITE_CALLBACK_APPLIED);

    //teardown
    destroyRegopLists();
}

void install_write_callback_with_callback_already_present()
{
    int addr1 = 0xabc, addr2 = addr1;
    int value = 0x123;
    int status1, status2;

    //exercise
    status1 = UTAPI_INSTALL_WRITE_CALLBACK(addr2, writeCallbackFun, (void*)&value);
    status2 = UTAPI_INSTALL_WRITE_CALLBACK(addr2, writeCallbackFun, (void*)&value);

    //verify
    UNIT_ASSERT(status1 == REGOP_WRITE_CALLBACK_APPLIED);
    UNIT_ASSERT(status2 == REGOP_WRITE_CALLBACK_ERROR);

    //teardown
    destroyRegopLists();
}

//*********************************************************************************************
void read_register_never_written_no_associated_node()
{
    int addr = 0xabc;
    int value;
    int status1, status2;

    //exercise
    value = UTAPI_READ_REGISTER(addr);

    //verify
    //SUT should hit an assert
    UNIT_ASSERT(value == 0xACDCACDC);

    //teardown
    destroyRegopLists();
}

void read_register_never_written_associated_node()
{
    int addr1 = 0xabc, addr2 = addr1;
    int value = 0x123, retVal;
    int status;

    //exercise
    status = UTAPI_INSTALL_WRITE_MIRROR(addr1, value);
    retVal = UTAPI_READ_REGISTER(addr2);

    //verify
    //SUT should hit an assert
    UNIT_ASSERT(retVal == 0xACDCACDC);

    //teardown
    destroyRegopLists();
}

void read_register_previously_written()
{
    int addr1 = 0xabc, addr2 = addr1;
    int value = 0x123;
    int retVal;
    int status1, status2;

    //exercise
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr1, value);
    retVal = UTAPI_READ_REGISTER(addr2);

    //verify
    //SUT should hit an assert
    UNIT_ASSERT(value == retVal);

    //teardown
    destroyRegopLists();
}

//*************************************************************************

void read_reg_nth_never_written_no_head()
{
    int addr = 0xabc;
    int value;
    int status1, status2;
    LwBool wasWritten;

    //exercise
    value = UTAPI_READ_VALUE_ON_NTH_WRITE(addr, 3, &wasWritten);

    //verify
    UNIT_ASSERT(wasWritten == FALSE);
    UNIT_ASSERT(value == 0xACDCACDC);

    //teardown
    destroyRegopLists();
}

void read_reg_nth_never_written_head_present()
{
    int addr1 = 0xabc, addr2 = 0xcde;
    int value = 0x123, retVal;
    int status1, status2;
    LwBool wasWritten;

    //exercise
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr1, value);
    retVal = UTAPI_READ_VALUE_ON_NTH_WRITE(addr2, 3, &wasWritten);

    //verify
    UNIT_ASSERT(wasWritten == FALSE);
    UNIT_ASSERT(retVal == 0xACDCACDC);

    //teardown
    destroyRegopLists();
}

void read_reg_nth_previously_written()
{
    int addr1 = 0xabc, addr2 = addr1;
    int retVal;
    int status1, status2;
    LwBool wasWritten;
    int i;
    int n = (rand()%10) + 1;

    //exercise
    for (i=1;i<=10;i++)
        unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr1, i);
    retVal = UTAPI_READ_VALUE_ON_NTH_WRITE(addr2, n, &wasWritten);

    //verify
    UNIT_ASSERT(wasWritten == TRUE);
    UNIT_ASSERT(n == retVal);

    //teardown
    destroyRegopLists();
}

void read_reg_nth_previously_written_n_greater_than_times_written()
{
    int addr1 = 0xabc, addr2 = addr1;
    int retVal;
    int status1, status2;
    LwBool wasWritten;
    int i;
    int n = (rand() + 11);

    //exercise
    for (i=1;i<=10;i++)
        unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr1, i);
    retVal = UTAPI_READ_VALUE_ON_NTH_WRITE(addr2, n, &wasWritten);

    //verify
    UNIT_ASSERT(wasWritten == FALSE);
    UNIT_ASSERT(retVal == 0xACDCACDC);

    //teardown
    destroyRegopLists();
}

//*****************************************************************************************

void gpu_write_no_head_present()
{
    int addr = 0xabc;
    int value = 0xdef;
    int verifValue;
    int status;

    //execise
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr, value);
    verifValue = UTAPI_READ_REGISTER(addr);

    //verify
    UNIT_ASSERT(verifValue == value);

    //teardown
    destroyRegopLists();
}

void gpu_write_head_present_corresponding_node_absent()
{
    int addr1 = 0xabc, addr2 = 0xabd;
    int value = 0xdef;
    int verifVal;
    int status1, status2;

    //exercise
    status1 = UTAPI_INSTALL_WRITE_CALLBACK(addr1, writeCallbackFun, (void*)&value);
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr2, value);
    verifVal = UTAPI_READ_REGISTER(addr2);

    //verify
    UNIT_ASSERT(verifVal == value);

    //teardown
    destroyRegopLists();
}

void gpu_mirrored_write()
{
    int addr1 = 0xabc, addr2 = 0xabd;
    int value = 0xdef;
    int verifVal1, verifVal2;
    int status1, status2;

    //exercise
    status1 = UTAPI_INSTALL_WRITE_MIRROR(addr1, addr2);
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr1, value);
    verifVal1 = UTAPI_READ_REGISTER(addr1);
    verifVal2 = UTAPI_READ_REGISTER(addr2);

    //verify
    UNIT_ASSERT(status1 == REGOP_WRITE_MIRROR_APPLIED);
    UNIT_ASSERT(verifVal1 == value);
    UNIT_ASSERT(verifVal2 == value);

    //teardown
    destroyRegopLists();
}

void gpu_write_with_callback()
{
    int addr = 0xabc;
    int value = 0x123;
    int params = 2;//write the double of the value
    int verifValue;
    int status;

    //exercise
    status = UTAPI_INSTALL_WRITE_CALLBACK(addr, writeCallbackFun, (void*)&params);
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr, value);
    verifValue = UTAPI_READ_REGISTER(addr);

    //verify
    UNIT_ASSERT(status == REGOP_WRITE_CALLBACK_APPLIED);
    UNIT_ASSERT(verifValue == (value*params));

    //teardown
    destroyRegopLists();
}

void gpu_write_node_present_no_mirror_no_callback()
{
    int addr = 0xabc;
    int value1 = 0xdef, value2 = 0xabd;
    int verifVal1, verifVal2;
    int status1, status2;

    //exercise

    //node created at this instant
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr, value1);
    verifVal1 = UTAPI_READ_REGISTER(addr);

    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr, value2);
    verifVal2 = UTAPI_READ_REGISTER(addr);

    //verify
    UNIT_ASSERT(verifVal1 == value1);
    UNIT_ASSERT(verifVal2 == value2);

    //teardown
    destroyRegopLists();
}

//******************************************************************************************

void gpu_read_head_absent_never_written()
{
    int addr = 0xabc;
    int retVal;

    //exercise
    retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr);

    //verfiy

    //should hit an assert in SUT
    UNIT_ASSERT(retVal == 0xACDCACDC);

    //teardown
    destroyRegopLists();
}

void gpu_read_head_absent_previously_written()
{
    int addr = 0xabc;
    int value = 0xdef;
    int retVal;

    //exercise
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr, value);
    retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr);

    //verfiy
    UNIT_ASSERT(value == retVal);

    //teardown
    destroyRegopLists();
}

void gpu_read_node_absent_never_written()
{
    int addr1 = 0xabc, addr2 = 0xafe;
    int value = 0xdef;
    int retVal;
    int status;

    //exercise
    status = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr1, value);
    retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr2);

    //verify
    UNIT_ASSERT(status == REGOP_WILL_RETURN_ALWAYS_APLLIED);
    //should hit an assert in SUT
    UNIT_ASSERT(retVal == 0xACDCACDC);

    //teardown
    destroyRegopLists();
}

void gpu_read_node_absent_previously_written()
{
    int addr1 = 0xabc, addr2 = 0xafe;
    int value = 0xdef;
    int retVal;
    int status;

    //exercise
    status = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr1, value);
    unitGpuWriteRegister032(NULL, DEVICE_INDEX_GPU, addr2, value);
    retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr2);

    //verify
    UNIT_ASSERT(status == REGOP_WILL_RETURN_ALWAYS_APLLIED);
    UNIT_ASSERT(retVal == value);

    //teardown
    destroyRegopLists();
}

void gpu_read_callback_present()
{
    int addr = 0xabc;
    int param = 0xdef;
    int status;
    int retVal;

    //exercise
    status = UTAPI_INSTALL_READ_CALLBACK(addr, readCallbackFun, (void *)&param);
    retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr);

    //verify
    UNIT_ASSERT(retVal == param);

    //teardown
    destroyRegopLists();
}

void gpu_read_will_return()
{
    int addr = 0xabc;
    int willRetVal = 0xdef;
    int retVal;
    int status;
    int i;
    int n = (rand()%11) + 1;

    //exercise
    status = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr, willRetVal);

    //verify
    for( i=0; i<n; i++)
    {
        retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr);
        UNIT_ASSERT(retVal == willRetVal);
    }

    //teardown
    destroyRegopLists();
}

void gpu_read_will_return_for_count()
{
    int addr = 0xabc;
    int willRetVal = 0xdef;
    int retVal;
    int value1 = 0xafb, value2 = 0xdfe;
    int count1 = 4, count2 = 2;
    int status1, status2, status3;
    int i;
    int n = (rand()%10) + 1;

    //exercise

    //
    //wiil return value1 count1 times, then value2 count2 times
    //and thereafter willRetVal always
    //
    status1 =UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr, value1, count1);
    status1 =UTAPI_INSTALL_READ_RETURN_UNTIL_COUNT(addr, value2, count2);
    status3 = UTAPI_INSTALL_READ_RETURN_ALWAYS(addr, willRetVal);

    //verify
    for (i=1; i<=count1; i++)
    {
        retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr);
        UNIT_ASSERT(retVal == value1);
    }

    for (i; i<=(count1+count2); i++)
    {
        retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr);
        UNIT_ASSERT(retVal == value2);
    }

    for( i; i<=(n+count1+count2); i++)
    {
        retVal = unitGpuReadRegister032(NULL, DEVICE_INDEX_GPU, addr);
        UNIT_ASSERT(retVal == willRetVal);
    }

    //teardown
    destroyRegopLists();
}

