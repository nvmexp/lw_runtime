#include <iostream>

#include "TestCommandOutputController.h"
#include "CommandOutputController.h"
#include "dcgm_structs.h"

TestCommandOutputController::TestCommandOutputController()
{
}

TestCommandOutputController::~TestCommandOutputController()
{
}

std::string TestCommandOutputController::GetTag()
{
    return "outputcontroller";
}

int TestCommandOutputController::Run()
{
    int st;
    int numFailed = 0;

    st = TestHelperDisplayValue();
    if (st)
    {
        numFailed++;
        fprintf(stderr, "TestCommandOutputContoller::TestHelperDisplayValue FAILED with %d.\n", st);
    }
    else
        printf("TestCommandOutputContoller::TestHelperDisplayValue PASSED\n");

    return numFailed;
}

int TestCommandOutputController::TestHelperDisplayValue()
{
    CommandOutputController coc;
    int ret = 0;
    char buf[1024];

    snprintf(buf, sizeof(buf), "%s", DCGM_STR_BLANK);
    std::string retd = coc.HelperDisplayValue(buf);
    if (retd != "Not Specified")
    {
        ret = -1;
        fprintf(stderr, "'%s' should have been transformed into 'Not Specified', but was '%s'.\n",
                DCGM_STR_BLANK, retd.c_str());
    }

    snprintf(buf, sizeof(buf), "%s", DCGM_STR_NOT_FOUND);
    retd = coc.HelperDisplayValue(buf);
    if (retd != "Not Found")
    {
        ret = -1;
        fprintf(stderr, "'%s' should have been transformed into 'Not Found', but was '%s'.\n",
                DCGM_STR_NOT_FOUND, retd.c_str());
    }

    snprintf(buf, sizeof(buf), "%s", DCGM_STR_NOT_SUPPORTED);
    retd = coc.HelperDisplayValue(buf);
    if (retd != "Not Supported")
    {
        ret = -1;
        fprintf(stderr, "'%s' should have been transformed into 'Not Supported', but was '%s'.\n",
                DCGM_STR_NOT_SUPPORTED, retd.c_str());
    }

    snprintf(buf, sizeof(buf), "%s", DCGM_STR_NOT_PERMISSIONED);
    retd = coc.HelperDisplayValue(buf);
    if (retd != "Insf. Permission")
    {
        ret = -1;
        fprintf(stderr, "'%s' should have been transformed into 'Insf. Permission', but was '%s'.\n",
                DCGM_STR_NOT_PERMISSIONED, retd.c_str());
    }

    snprintf(buf, sizeof(buf), "Brando\tSando");
    retd = coc.HelperDisplayValue(buf);
    if (retd != "Brando Sando")
    {
        ret = -1;
        fprintf(stderr, "'%s' should have been transformed into 'Brando Sando', but was '%s'.\n",
                buf, retd.c_str());
    }

    snprintf(buf, sizeof(buf), "There's weird\nspacing\n\there.");
    retd = coc.HelperDisplayValue(buf);
    if (retd != "There's weird spacing  here.")
    {
        ret = -1;
        fprintf(stderr, "'%s' should have been transformed into 'There's weird spacing  here', but was '%s'.\n",
                buf, retd.c_str());
    }

    return ret;
}

