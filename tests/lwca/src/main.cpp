#include "gtest/gtest.h"
#include "utils/utils.h"
#include "lwda_utils/lwda_utils.h"
#include "dvs_utils/dvs_utils.h"
#include <lwos.h>

int main(int argc, char **argv)
{
    lwosInit();

    ::utils::UtilsElwironment *utils_elw = new ::utils::UtilsElwironment(argv[0]);
    ::lwca::LwdaElwironment *lwda_elw = new ::lwca::LwdaElwironment();

    utils_elw->parseArguments(argc, argv);
    lwda_elw->parseArguments(argc, argv);

    (void)::testing::AddGlobalTestElwironment(utils_elw);
    (void)::testing::AddGlobalTestElwironment(lwda_elw);

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    if (!utils_elw->isChildProcess()) {
        // DVS specific result output
        char elwvar[1024];
        if (!lwosGetElw("DVS_ACTIVE", elwvar, sizeof(elwvar))
            && !strncmp(elwvar, "yes", 4)) {
            delete listeners.Release(listeners.default_result_printer());
            listeners.Append(new ::dvs::DvsLwdaAppsUnitTestResultPrinter(std::cout));
        }
    }
    else {
        delete listeners.Release(listeners.default_result_printer());
        listeners.Append(new ::utils::QuietUnitTestResultPrinter(std::cout));
    }

    listeners.Append(new ::lwca::LwdaCleanupListener());

    return RUN_ALL_TESTS();
}
