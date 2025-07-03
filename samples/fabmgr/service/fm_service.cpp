#include <stdio.h>

#include <windows.h>
#include <iostream>
#include <winsvc.h>
#include <string>
#include <strsafe.h>
#include <aclapi.h>
#include <cstdio>

#include <stdio.h>

#include <stdlib.h>
#include <signal.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <limits.h>
#include <cstring>
#include <stdexcept>
/*
#include "GlobalFabricManager.h"           
#include "LocalFabricManager.h"

#include "errno.h"
#include "fm_cmd_parser.h"
#include "fm_config_options.h"
#include "fm_log.h"

#include "FMVersion.h"
#include "fm_helper.h"
*/
typedef struct {
    int argc;
    char **argv;
} ARGS;

#define SERVICE_NAME "lw-fabricmanager"
int ifservice = 0;
void fmServiceStop();

HANDLE fmEvent = NULL;
SERVICE_STATUS_HANDLE handle = NULL;
_SERVICE_STATUS status;

char* windowsErrorString(DWORD error) {
    char buf[256];
    FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
               NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), 
               buf, (sizeof(buf) / sizeof(wchar_t)), NULL);
    return buf;
}
/*
static int 
fmCleanup(int status) 
{
    // first destroy local FM instance
    if (gFmGlobalCtx.pLocalFM) {
        delete gFmGlobalCtx.pLocalFM;
        gFmGlobalCtx.pLocalFM = NULL;
    }

    // delete global FM instance at the end
    if (gFmGlobalCtx.pGlobalFM) {
        delete gFmGlobalCtx.pGlobalFM;
        gFmGlobalCtx.pGlobalFM = NULL;
    }

    if (gFmGlobalCtx.pCmdParser) {
        fabricManagerCmdParserDestroy(gFmGlobalCtx.pCmdParser);
        gFmGlobalCtx.pCmdParser = NULL;
    }

    // status = 0 signifies failure during fm initialization
    if (status == 0 && gFMConfigOptions.fmDaemonize && ifservice) {
        fmServiceStop();
    }

    return ERROR_SUCCESS;
}
*/
/*
static int 
enableLocalFM() 
{
    LocalFmArgs_t *lfm = new LocalFmArgs_t;
    lfm->sharedFabric = gFMConfigOptions.sharedFabricMode == 0 ? false : true;
    lfm->bindInterfaceIp = strdup(gFMConfigOptions.bindInterfaceIp);
    lfm->fmStartingTcpPort = gFMConfigOptions.fmStartingTcpPort;
    lfm->domainSocketPath = strdup(gFMConfigOptions.fmUnixSockPath);
    // create the local FM instance
    try {
        gFmGlobalCtx.pLocalFM = new LocalFabricManagerControl(lfm);
    }
    catch (const std::runtime_error &e) {
        fprintf(stderr, "%s\n", e.what());
        return -1;
    }

    return 0;
}

static int 
enableGlobalFM() 
{
    GlobalFmArgs_t *gfm = new GlobalFmArgs_t;
    gfm->sharedFabric = gFMConfigOptions.sharedFabricMode == 0 ? false : true;
    gfm->fmStartingTcpPort = gFMConfigOptions.fmStartingTcpPort;
    gfm->restart = gFmGlobalCtx.pCmdParser->restart;
    gfm->domainSocketPath = strdup(gFMConfigOptions.fmUnixSockPath);
    gfm->stateFileName = strdup(gFMConfigOptions.fmStateFileName);
    gfm->fmLibCmdBindInterface = strdup(gFMConfigOptions.fmLibCmdBindInterface);
    gfm->fmLibCmdSockPath = strdup(gFMConfigOptions.fmLibCmdUnixSockPath);
    gfm->fmLibPortNumber = gFMConfigOptions.fmLibPortNumber;
    gfm->fmBindInterfaceIp = gFMConfigOptions.bindInterfaceIp;
    // create the global FM instance
    try {
        gFmGlobalCtx.pGlobalFM = new GlobalFabricManager(gfm);
    }
    catch (const std::runtime_error &e) {
        fprintf(stderr, "%s\n", e.what());
        return -1;
    }

    return 0;
}
*/

DWORD WINAPI fmServiceWorkerThread(LPVOID lpParam) 
{
    //fabricManagerCmdParseReturn_t cmdParseRet;
    ARGS *args = (ARGS*) lpParam;
    int i = 0;
/*    
    // gFmGlobalCtx.parentPid = getpid();
    gFmGlobalCtx.pCmdParser = fabricManagerCmdParserInit(args->argc, args->argv, fabricManagerArgs, FM_CMD_COUNT, 
                                                         displayFabricManagerUsage, displayFabricManagerHelpMsg);
    if (NULL == gFmGlobalCtx.pCmdParser) {
        return -1;
    }

    // parse all the command lines
    cmdParseRet = fabricManagerCmdProcessing(gFmGlobalCtx.pCmdParser);
    if (CMD_PARSE_ST_OK != cmdParseRet) {
        if (cmdParseRet == CMD_PARSE_ST_BADPARAM) {
            fprintf(stderr, "Unable to start lw-fabricmanager: bad command line parameter. \n");
        } else {
            fprintf(stderr, "Unable to start lw-fabricmanager: generic error. \n");
        }
        fabricManagerCmdParserDestroy(gFmGlobalCtx.pCmdParser);
        return -1;
    }

    // // Print out Fabric Manager and quit (when -v option is used)
    if (gFmGlobalCtx.pCmdParser->printVersion) {
        displayFabricManagerVersionInfo();
        fabricManagerCmdParserDestroy(gFmGlobalCtx.pCmdParser);
        return 0;
    }

     if (fabricManagerLoadConfigOptions(gFmGlobalCtx.pCmdParser->configFilename) < 0) {
        // error already logged
        return -1;
    }
    
    lwosInit();
    // set logging options/config
    fabricManagerInitLog(gFMConfigOptions.logLevel, gFMConfigOptions.logFileName,
                         gFMConfigOptions.appendToLogFile, gFMConfigOptions.maxLogFileSize,
                         gFMConfigOptions.useSysLog);

    // dump the current configuration options
    dumpLwrrentConfigOptions();

    if (gFMConfigOptions.enableLocalFM) {
        if (enableLocalFM() < 0) {
            // we can't continue, error is already logged
            fmCleanup(0);
            return -1;
        }
    }

    // start global fm instance
    if (gFMConfigOptions.enableGlobalFM) {
        if (enableGlobalFM() < 0) {
            // we can't continue, error is already logged
            fmCleanup(0);
            return -1;
        }
    }

    while (WaitForSingleObject(fmEvent, 0) != WAIT_OBJECT_0) {
        Sleep(TIMETOSLEEP);
    }

    //FM Initialization is a success
    return fmCleanup(1);
*/
    return ERROR_SUCCESS;
}

void fmServiceInstall()
{
    char *fileNameBuf = new char[120];
    SC_HANDLE scManagerHandle;
    SC_HANDLE scServiceHandle;
    DWORD ret = GetModuleFileNameA(NULL, (LPSTR)fileNameBuf, 120);
    if (ret == 0) {
        fprintf(stdout, "Error at getting fully qualified path for the specified FM Module."
                " Failed with error: %s\n", windowsErrorString(GetLastError()));
        return;
    }

    scManagerHandle =  OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);

    if (scManagerHandle == NULL) {
        fprintf(stdout, "Unable to establish connection with Service Control Manager." 
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    scServiceHandle = CreateService(scManagerHandle, SERVICE_NAME, SERVICE_NAME, SC_MANAGER_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS, 
                                    SERVICE_DEMAND_START, SERVICE_ERROR_NORMAL, fileNameBuf, NULL, NULL, NULL, NULL, NULL);


    if (scServiceHandle == NULL)  {
        fprintf(stdout, "Unable to create Fabricmanager service and add it to SCM database."
                " Failed with error: %s\n", windowsErrorString(GetLastError()));
        return;
    }

    CloseServiceHandle(scServiceHandle);
    CloseServiceHandle(scManagerHandle);
}

void fmServiceDelete()
{
    SC_HANDLE scManagerHandle;
    SC_HANDLE scServiceHandle;
    bool ret;

    scManagerHandle =  OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);
    if (scManagerHandle == NULL) {
        fprintf(stdout, "Unable to establish connection with Service Control Manager." 
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    scServiceHandle = OpenService(scManagerHandle, SERVICE_NAME, SC_MANAGER_ALL_ACCESS);

    if (scServiceHandle == NULL)  {
        fprintf(stdout, "Unable to open the Fabricmanager service. " 
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    ret = DeleteService(scServiceHandle);

    if (ret == 0)  {
        fprintf(stdout, "Unable to mark the Fabricmanager service for deletion from SCM. " 
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    CloseServiceHandle(scServiceHandle);
    CloseServiceHandle(scManagerHandle);
}

void fmServiceStart(int argc, char **argv)
{
    SC_HANDLE scManagerHandle;
    SC_HANDLE scServiceHandle;
    _SERVICE_STATUS_PROCESS lpBuf;
    DWORD numBytesNeeded;
    bool ret;   

    scManagerHandle =  OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);
    if (scManagerHandle == NULL) {
        fprintf(stdout, "Unable to establish connection with Service Control Manager." 
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    scServiceHandle = OpenService(scManagerHandle, SERVICE_NAME, SC_MANAGER_ALL_ACCESS);

    if (scServiceHandle == NULL)  {
        fprintf(stdout, "Unable to open the Fabricmanager service. " 
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    ret = QueryServiceStatusEx(scServiceHandle, SC_STATUS_PROCESS_INFO, (LPBYTE)&lpBuf, sizeof(_SERVICE_STATUS_PROCESS), &numBytesNeeded);
    if (ret == 0) {
        fprintf(stdout, "Unable to Query the current status of fabricmanager service based on specified information level. "
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    ret =  StartService(scServiceHandle, argc, (LPCSTR*)argv);
    if (ret == 0) {
        fprintf(stdout, "Unable to start Fabricmanager service. "
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    CloseServiceHandle(scServiceHandle);
    CloseServiceHandle(scManagerHandle);
}

void fmServiceStop()
{
    SC_HANDLE scManagerHandle;
    SC_HANDLE scServiceHandle;
    SERVICE_STATUS_PROCESS lpBuf;
    DWORD numBytesNeeded;
    bool ret;

    scManagerHandle =  OpenSCManager(NULL, NULL, SC_MANAGER_ALL_ACCESS);
    if (scManagerHandle == NULL) {
        fprintf(stdout, "Unable to establish connection with Service Control Manager." 
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    scServiceHandle = OpenService(scManagerHandle, SERVICE_NAME, SC_MANAGER_ALL_ACCESS);
    if (scServiceHandle == NULL)  {
        fprintf(stdout, "Unable to open the Fabricmanager service. " 
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    ret = QueryServiceStatusEx(scServiceHandle, SC_STATUS_PROCESS_INFO, (LPBYTE)&lpBuf, sizeof(_SERVICE_STATUS_PROCESS), &numBytesNeeded);
    if (ret == 0) {
        fprintf(stdout, "Unable to Query the current status of fabricmanager service based on specified information level. "
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
        return;
    }

    if (lpBuf.dwLwrrentState == SERVICE_RUNNING) {
        ret = ControlService(scServiceHandle, SERVICE_CONTROL_STOP, (LPSERVICE_STATUS)&status);
    }

    while (lpBuf.dwLwrrentState != SERVICE_STOP_PENDING && lpBuf.dwLwrrentState != SERVICE_STOPPED) {
        ret = QueryServiceStatusEx(scServiceHandle, SC_STATUS_PROCESS_INFO, (LPBYTE)&lpBuf, sizeof(_SERVICE_STATUS_PROCESS), &numBytesNeeded);
    }

    CloseServiceHandle(scServiceHandle);
    CloseServiceHandle(scManagerHandle);
}

void WINAPI fmServiceHandler(DWORD ctr) 
{
	switch(ctr) {
		case SERVICE_CONTROL_STOP:
			{
				bool ret;
				if (status.dwLwrrentState != SERVICE_RUNNING)
					break;

				status.dwLwrrentState = SERVICE_STOP_PENDING;
				status.dwControlsAccepted = 0;
				status.dwWin32ExitCode = 0;
				ret = SetServiceStatus(handle, &status);
				if (ret == 0) {
					fprintf(stdout, "Unable to Set status of fabricmanager service. "
                    " Failed with error : %s\n", windowsErrorString(GetLastError()));
				}
				SetEvent(fmEvent);
			}
			break;
		case SERVICE_CONTROL_CONTINUE:
			break;
		case SERVICE_CONTROL_SHUTDOWN:
			break;
	}
}



void
WINAPI fmServiceMain(DWORD args, LPTSTR *argv)
{
	handle = RegisterServiceCtrlHandler(SERVICE_NAME, fmServiceHandler);
	if (handle == NULL) {
		fprintf(stdout, "Unable to register a function to handle service control requests. "
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
	}

	ZeroMemory (&status, sizeof (status));
	status.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
	status.dwLwrrentState = SERVICE_START_PENDING; 
	status.dwWin32ExitCode = 0;

	bool ret = SetServiceStatus(handle, &status);
	if (ret == 0) {
		fprintf(stdout, "Unable to Set status of fabricmanager service. "
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
	}

	fmEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
	if (fmEvent == NULL) {
		fprintf(stdout, "Unable to create an event object. "
                    " Failed with error : %s\n", windowsErrorString(GetLastError()));
	}

	status.dwControlsAccepted = SERVICE_ACCEPT_STOP;
	status.dwLwrrentState = SERVICE_RUNNING;
	ret = SetServiceStatus(handle, &status);
	if (ret == 0) {
		fprintf(stdout, "Unable to Set status of fabricmanager service. "
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
	}

	ARGS cmdargs = {(int) args-1, (char**)&argv[1]};
	HANDLE hThread = CreateThread (NULL, 0, fmServiceWorkerThread, &cmdargs, 0, NULL);
	WaitForSingleObject(hThread, INFINITE);

	CloseHandle(fmEvent);

	status.dwControlsAccepted = 0;
	status.dwLwrrentState = SERVICE_STOPPED;

	ret = SetServiceStatus(handle, &status);
	if (ret == 0) {
		fprintf(stdout, "Unable to Set status of fabricmanager service. "
                " Failed with error : %s\n", windowsErrorString(GetLastError()));
	}

}

int main(int argc, char**argv)
{
    if (strcmp(argv[argc-1], "create") == 0) {
        fmServiceInstall(); 
    }

    else if(strcmp(argv[argc-1], "start") == 0) {
        fmServiceStart(argc-1, argv);
    }

    else if(strcmp(argv[argc-1], "stop") == 0) {
        fmServiceStop();
    }

    else if(strcmp(argv[argc-1], "delete") == 0) {
        fmServiceDelete();
    }
    
    SERVICE_TABLE_ENTRY serviceTable[] = {
        {SERVICE_NAME, (LPSERVICE_MAIN_FUNCTION)fmServiceMain},
        {NULL, NULL}
        };

    int ret = StartServiceCtrlDispatcher(serviceTable); 

    return 0;
}