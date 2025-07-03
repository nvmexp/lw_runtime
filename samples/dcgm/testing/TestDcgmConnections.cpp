#include "TestDcgmConnections.h"
#include <string>
#include <iostream>
#include <stdexcept>
#include "dcgm_agent.h"
#include "LwcmThread.h"
#include "timelib.h"

/*****************************************************************************/
TestDcgmConnections::TestDcgmConnections()
{

}

/*****************************************************************************/
TestDcgmConnections::~TestDcgmConnections()
{

}

/*************************************************************************/
std::string TestDcgmConnections::GetTag()
{
    return std::string("connections");
}

/*****************************************************************************/
int TestDcgmConnections::Cleanup(void)
{
    return 0;
}

/*****************************************************************************/
int TestDcgmConnections::Init(std::vector<std::string>argv, std::vector<test_lwcm_gpu_t>gpus)
{
    m_gpus = gpus;
    return 0;
}

/*****************************************************************************/
void TestDcgmConnections::CompleteTest(std::string testName, int testReturn, int &Nfailed)
{
    if (testReturn)
    {
        Nfailed++;
        std::cerr << "TestDcgmConnections::" << testName << " FAILED with " << testReturn << std::endl;

        // fatal test failure
        if (testReturn < 0)
            throw std::runtime_error("fatal test failure");
    } else
    {
        std::cout << "TestDcgmConnections::" << testName << " PASSED" << std::endl;
    }
}

/*****************************************************************************/
int TestDcgmConnections::TestDeadlockSingle(void)
{
    dcgmReturn_t dcgmReturn;
    int i;
    dcgmHandle_t dcgmHandle;
    char *badHostname = (char *)"127.0.0.1:61000"; /* Shouldn't be be able to connect */
    int startTime = timelib_secSince1970();
    int testDurationSecs = 30;

    printf("TestDeadlockSingle running for %d seconds.\n", testDurationSecs);

    for(i = 0; (int)timelib_secSince1970() - startTime < testDurationSecs; i++)
    {
        dcgmReturn = dcgmConnect(badHostname, &dcgmHandle);
        if(dcgmReturn == DCGM_ST_OK)
        {
            dcgmDisconnect(dcgmHandle);
            std::cerr << "TestDeadlockSingle skipping due to actually being able to connect to " << badHostname << ::std::endl;
            return 0;
        }
        else if(dcgmReturn != DCGM_ST_CONNECTION_NOT_VALID)
        {
            std::cerr << "Unexpected return " << dcgmReturn << " from dcgmConnect() iteration" << i << std::endl;
            return 100;
        }
    }

    printf("TestDeadlockSingle finished without deadlocking.\n");
    return 0;
}

/*****************************************************************************/
class TestDeadlockMultiThread : public LwcmThread
{
public:
    int m_threadIndex;

    TestDeadlockMultiThread(int threadIndex)
    {
        m_threadIndex = threadIndex;    
    }

    ~TestDeadlockMultiThread()
    {

    }

    void run(void)
    {
        dcgmReturn_t dcgmReturn;
        int Ntimes = 100;
        int i;
        dcgmHandle_t dcgmHandle;
        char *badHostname = (char *)"127.0.0.1:61000"; /* Shouldn't be be able to connect */

        for(i = 0; Ntimes && !ShouldStop(); i++)
        {
            dcgmReturn = dcgmConnect(badHostname, &dcgmHandle);
            if(dcgmReturn == DCGM_ST_OK)
            {
                dcgmDisconnect(dcgmHandle);
                std::cerr << "TestDeadlockMulti thread " << m_threadIndex << " aborting due to actually being able to connect to " << badHostname << ::std::endl;
                break;
            }
            else if(dcgmReturn != DCGM_ST_CONNECTION_NOT_VALID)
            {
                std::cerr << "TestDeadlockMulti thread " << m_threadIndex << " got unexpected return " << dcgmReturn << " from dcgmConnect() iteration" << i << std::endl;
                break;
            }

            if(i % (Ntimes / 10) == 0) //Log every 10%
            {
                std::cout << "TestDeadlockMulti at iteration " << (i+1) << "/" << Ntimes << std::endl;
            }
        }

        std::cout << "Thread " << m_threadIndex << " exiting." << std::endl;
    }
};

/*****************************************************************************/
#define TDC_NUM_WORKERS 4

int TestDcgmConnections::TestDeadlockMulti(void)
{
    dcgmReturn_t dcgmReturn;    
    int i;
    TestDeadlockMultiThread *workers[TDC_NUM_WORKERS] = {0};
    dcgmHandle_t dcgmHandle;
    char *badHostname = (char *)"127.0.0.1:61000"; /* Shouldn't be be able to connect */
    int workersLeft = TDC_NUM_WORKERS;
    int startTime = timelib_secSince1970();

    /* Make a single connection to make sure our threads will work */
    dcgmReturn = dcgmConnect(badHostname, &dcgmHandle);
    if(dcgmReturn == DCGM_ST_OK)
    {
        std::cerr << "TestDeadlockMulti skipping due to actually being able to connect to " << badHostname << ::std::endl;
        dcgmDisconnect(dcgmHandle);
        return 0;
    }
    else if(dcgmReturn != DCGM_ST_CONNECTION_NOT_VALID)
    {
        std::cerr << "Unexpected return " << dcgmReturn << " from dcgmConnect() " << std::endl;
        return 100;
    }

    std::cout << "Starting " << (int)TDC_NUM_WORKERS << " workers." << std::endl;

    for(i = 0; i < TDC_NUM_WORKERS; i++)
    {
        workers[i] = new TestDeadlockMultiThread(i);
        workers[i]->Start();
    }

    std::cout << "Waiting for " << (int)TDC_NUM_WORKERS << " workers." << std::endl;
    
    while(workersLeft > 0)
    {
        for(i = 0; i < TDC_NUM_WORKERS; i++)
        {
            if(!workers[i])
                continue;
            
            if(timelib_secSince1970() - startTime > 30)
            {
                std::cout << "Requesting stop of worker " << i << " after 30 seconds." << std::endl;
                workers[i]->Stop();
            }
            
            /* Wait() returns 0 if the thread is gone */
            if(!workers[i]->Wait(1000))
            {
                delete workers[i];
                workers[i] = 0;
                workersLeft--;
            }
        }
    }

    std::cout << "All workers exited." << std::endl;
    return 0;
}

/*****************************************************************************/
int TestDcgmConnections::Run()
{
    int Nfailed = 0;

    try
    {
        CompleteTest("TestDeadlockSingle", TestDeadlockSingle(), Nfailed);
        CompleteTest("TestDeadlockMultiThread", TestDeadlockMulti(), Nfailed);
    }
    // fatal test return olwrred
    catch(const std::runtime_error &e)
    {
        return -1;
    }

    if(Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    printf("All tests passed\n");

    return 0;
}

/*****************************************************************************/

