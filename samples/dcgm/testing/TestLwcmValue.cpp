
#include "TestLwcmValue.h"
#include "lwcmvalue.h"

/*****************************************************************************/
TestLwcmValue::TestLwcmValue()
{

}

/*****************************************************************************/
TestLwcmValue::~TestLwcmValue()
{

}

/*************************************************************************/
std::string TestLwcmValue::GetTag()
{
    return std::string("lwcmvalue");
}

/*****************************************************************************/
int TestLwcmValue::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    return 0;
}

/*****************************************************************************/
int TestLwcmValue::Cleanup()
{
    return 0;
}

/*****************************************************************************/
int TestLwcmValue::TestColwersions()
{
    int i;
    int Nerrors = 0;
    int valueInt32;
    long long valueInt64;
    double valueDouble;

    /* All of the values in this list should match at each array index */
    int Ncolwersions = 6;
    int int32Values[6] =       {0,   25,   DCGM_INT32_BLANK, DCGM_INT32_NOT_FOUND, DCGM_INT32_NOT_SUPPORTED, DCGM_INT32_NOT_PERMISSIONED};
    long long int64Values[6] = {0,   25,   DCGM_INT64_BLANK, DCGM_INT64_NOT_FOUND, DCGM_INT64_NOT_SUPPORTED, DCGM_INT64_NOT_PERMISSIONED};
    double fp64Values[6] =     {0.0, 25.0, DCGM_FP64_BLANK,  DCGM_FP64_NOT_FOUND,  DCGM_FP64_NOT_SUPPORTED,  DCGM_FP64_NOT_PERMISSIONED};

    for(i = 0; i < Ncolwersions; i++)
    {
        valueInt32 = lwcmvalue_int64_to_int32(int64Values[i]);
        if(valueInt32 != int32Values[i])
        {
            fprintf(stderr, "i64 -> i32 failed for index %d\n", i);
            Nerrors++;
        }

        valueInt64 = lwcmvalue_int32_to_int64(int32Values[i]);
        if(valueInt64 != int64Values[i])
        {
            fprintf(stderr, "i32 -> i64 failed for index %d\n", i);
            Nerrors++;
        }

        valueDouble = lwcmvalue_int64_to_double(int64Values[i]);
        if(valueDouble != fp64Values[i])
        {
            fprintf(stderr, "i64 -> fp64 failed for index %d\n", i);
            Nerrors++;
        }

        valueInt64 = lwcmvalue_double_to_int64(fp64Values[i]);
        if(valueInt64 != int64Values[i])
        {
            fprintf(stderr, "fp64 -> int64 failed for index %d\n", i);
            Nerrors++;
        }

        valueDouble = lwcmvalue_int32_to_double(int32Values[i]);
        if(valueDouble != fp64Values[i])
        {
            fprintf(stderr, "i32 -> fp64 failed for index %d\n", i);
            Nerrors++;
        }

        valueInt32 = lwcmvalue_double_to_int32(fp64Values[i]);
        if(valueInt32 != int32Values[i])
        {
            fprintf(stderr, "i32 -> fp64 failed for index %d\n", i);
            Nerrors++;
        }
    }

    if(Nerrors > 0)
    {
        fprintf(stderr, "TestLwcmValue::TestColwersions %d colwersions failed.\n", Nerrors);
        return 1;
    }

    return 0;
}

/*****************************************************************************/
int TestLwcmValue::Run()
{

    int st;
    int Nfailed = 0;

    st = TestColwersions();
    if(st)
    {
        Nfailed++;
        fprintf(stderr, "TestLwcmValue::TestColwersions FAILED with %d\n", st);
        if(st < 0)
            return -1;
    }
    else
        printf("TestLwcmValue::TestColwersions PASSED\n");

    if(Nfailed < 1)
        printf("All TestLwcmValue tests PASSED\n");

    return 0;
}

/*****************************************************************************/
