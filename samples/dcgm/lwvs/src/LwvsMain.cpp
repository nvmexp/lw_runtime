#include <string>
#include "LwidiaValidationSuite.h"
#include "Plugin.h"
#include "er.h"
#include "common.h"
#include "JsonOutput.h"

/* Linker stub for dynamic logging. We can remove this if we change er to something else */
int erLogLevelB = ER_ERR; /* Default logging level. Change this or *erLogLevelP to dynamically
                             change the level that things are logged at with ER macros */
int *erLogLevelP = &erLogLevelB;

namespace
{
    const size_t SUCCESS = 0;
    const size_t ERROR_IN_COMMAND_LINE = 1;
    const size_t ERROR_UNHANDLED_EXCEPTION = 2;
}

int main_should_stop = 0; /* Global boolean to say whether we should be exiting or not. This
                             is set by the signal handler if we receive a CTRL-C or other terminating
                             signal */

/*****************************************************************************/
static void main_sig_handler(int signo)
{
    switch(signo)
    {
        case SIGINT:
        case SIGQUIT:
        case SIGKILL:
        case SIGTERM:
            PRINT_ERROR("%d", "Received signal %d. Requesting stop.", signo);
            main_should_stop = 1;
            lwvsCommon.mainReturnCode = MAIN_RET_ERROR; /* Still counts as an error */
            break;


        case SIGUSR1:
        case SIGUSR2:
            PRINT_ERROR("%d", "Ignoring SIGUSRn (%d)", signo);
            break;

        case SIGHUP:
            /* This one is usually used to tell a process to reread its config.
             * Treating this as a no-op for now
             */
            PRINT_ERROR("", "Ignoring SIGHUP");
            break;

        default:
            PRINT_ERROR("%d", "Received unknown signal %d. Ignoring", signo);
            break;
    }
}

/*****************************************************************************/
int main(int argc, char **argv)
{

    LwidiaValidationSuite *lwvs = NULL;

    struct sigaction sigHandler;
    sigHandler.sa_handler = main_sig_handler;
    sigemptyset(&sigHandler.sa_mask);
    sigHandler.sa_flags = 0;
    lwvsCommon.mainReturnCode = MAIN_RET_OK; /* Gets set by LwidiaValidationSuite constructor, but not until later */

    /* Install signal handlers */
    sigaction(SIGINT, &sigHandler, NULL);
    sigaction(SIGTERM, &sigHandler, NULL);

  
    try
    {
        // declare new LWVS object
        lwvs = new LwidiaValidationSuite();
        lwvs->go(argc,argv);

    }
    catch (std::runtime_error &e)
    {
        if (lwvsCommon.jsonOutput == false) {
            std::cerr << e.what() << std::endl;
        } else {
            Json::Value jv;
            jv[LWVS_NAME][LWVS_VERSION_STR] = DRIVER_MAJOR_VERSION;
            jv[LWVS_NAME][LWVS_RUNTIME_ERROR] = e.what();
            std::cerr << jv.toStyledString() << std::endl;
        }

        PRINT_ERROR("%s", "Got runtime_error: %s", e.what());
        PRINT_ERROR("%llx", "Global error mask is: 0x%064llx", lwvsCommon.errorMask);
        if(lwvs)
            delete(lwvs);
        lwvsCommon.mainReturnCode = MAIN_RET_ERROR;
        return lwvsCommon.mainReturnCode;
    }
    catch (std::exception &e)
    {
        //std::cerr << "unhandled Exception reached the top of main: "
        //          << e.what() << ", application will now exit" << std::endl;
        //std::cerr << e.what() << std::endl;
        if (lwvsCommon.jsonOutput == false) {
            std::cerr << e.what() << std::endl;
        } else {
            Json::Value jv;
            jv[LWVS_NAME][LWVS_VERSION_STR] = DRIVER_MAJOR_VERSION;
            jv[LWVS_NAME][LWVS_RUNTIME_ERROR] = e.what();
            std::cerr << jv.toStyledString() << std::endl;
        }

        PRINT_ERROR("%llx", "Global error mask is: 0x%064llx", lwvsCommon.errorMask);
        if (lwvs)
            delete lwvs; /* This deletes the logger, so no more PRINT_ macros after this */
        lwvsCommon.mainReturnCode = MAIN_RET_ERROR;
        return lwvsCommon.mainReturnCode; //ERROR_UNHANDLED_EXCEPTION would cause a core dump
    }

    delete lwvs; /* This deletes the logger, so no more PRINT_ macros after this */
    return lwvsCommon.mainReturnCode;
}
