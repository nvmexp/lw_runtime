#include "wrapper.h"
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cstring>
#include <stdlib.h>
#include <iostream>
#include <zlib.h>
#include <libtar.h>
#include <fcntl.h>
#include <dirent.h>
#include "PluginStrings.h"
#include "lwml.h"
#include "errno.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>

#define DIR_NAME_LENGTH 12
#define FILE_NAME_LENGTH 19
#define EUD_PATH_LENGTH 17
#define DEVICE_OPTION_LENGTH 10
#define BUFF_SIZE 262144 //256k recommended zlib size for buffers

/* TBD Define EUD launch options and Logfile behavior*/

EUDWrapper::EUDWrapper()
{
    m_infoStruct.name = "Diagnostic";
    m_infoStruct.shortDescription = "Plugin launching hardware diagnostic and logging results";
    m_infoStruct.testGroups = "Hardware";
    m_infoStruct.selfParallel = true;
    m_infoStruct.logFileTag = EUD_PLUGIN_LF_NAME;

    defaultTp = new TestParameters();
    defaultTp->AddString(PS_RUN_IF_GOM_ENABLED, "True");
    defaultTp->AddString(EUD_PLUGIN_RUN_MODE, "long");
    defaultTp->AddString(EUD_STR_IS_ALLOWED, "False");
    defaultTp->AddString(EUD_LOG_FILE_PATH,"");
    
    m_infoStruct.defaultTestParameters = defaultTp;

    m_StatCollection = new StatCollection();
    m_EudDir = NULL;
    eud_bytes = NULL;
    eud_size = 0;
}

EUDWrapper::~EUDWrapper()
{
    if (NULL != m_EudDir)
    {
        cleanDirectory(m_EudDir);
        delete[] m_EudDir;
    }
    delete defaultTp;
}

void EUDWrapper::cleanDirectory(char *tarDir)
{
    DIR *dir =  opendir(tarDir);
    struct dirent *dirEntry;
    char *filename = new char[DIR_NAME_LENGTH + 257]; // '/' + dirEntry.d_name  <= 257

    while ((dirEntry = readdir(dir)) != NULL)
    {
        if (dirEntry->d_type == DT_REG)
        {
            sprintf(filename, "%s/%s", tarDir, dirEntry->d_name);
            unlink((const char*)filename);
        }
    }

    delete[] filename;

    closedir(dir);
    rmdir(tarDir);
}

int EUDWrapper::createTempFile(FILE **file, char *path)
{
    int fd = mkstemp(path);

    if (fd == -1)
        return 1;

    *file = fdopen(fd, "w+");

    if (*file == NULL)
        return 1;

    return 0;
}

int EUDWrapper::deflate(FILE *src, FILE *dest)
{
    gzFile srcFile = gzdopen(fileno(src), "r");
    unsigned char *buff = new unsigned char[BUFF_SIZE];
    int read;

    if (srcFile == NULL)
    {
        delete[] buff;
        return 1;
    }

    do
    {
        read = gzread(srcFile, buff, BUFF_SIZE);
        if (fwrite(buff, read, 1, dest) == 0 && !gzeof(srcFile))
        {
            delete[] buff;
            return 1;
        }

    } while (read == BUFF_SIZE);

    if (!gzeof(srcFile))
    {
        delete[] buff;
        return 1;
    }

    gzclose(srcFile);
    fflush(dest);
    delete[] buff;

    return 0;
}

int EUDWrapper::checkForGraphicsProcesses(const std::vector<unsigned int> &gpuList)
{
    unsigned int lwmlGpuIndex;
    lwmlDevice_t lwmlDevice;
    unsigned int graphicsProcesses = 0;
    std::vector<unsigned int>::iterator gpuIt;

    if (lwmlInit() != LWML_SUCCESS)
        return 1;

    for (gpuIt = gpuList.begin(); gpuIt != gpuList.end(); gpuIt++)
    {
        lwmlGpuIndex = *gpuIt;
        if (lwmlDeviceGetHandleByIndex(lwmlGpuIndex, &lwmlDevice) != LWML_SUCCESS)
            return 1;

        if (lwmlDeviceGetGraphicsRunningProcesses(lwmlDevice, &graphicsProcesses, NULL) != LWML_SUCCESS)
            return 1;

        if (graphicsProcesses > 0)
            return 1;
    }

    return 0;
}

int EUDWrapper::openEudSymbols(std::string symbolsPath)
{
    void * lib_handle = NULL;

    lib_handle = dlopen (symbolsPath.c_str(), RTLD_LAZY);
    if (!lib_handle)
    {
        return -1;
    }

    eud_bytes = (unsigned int *)dlsym(lib_handle, "eud_bytes");
    if (eud_bytes == NULL)
    {
        PRINT_ERROR("", "Unable to find the hardware diagnostics \"bytes\" symbol");
        return -1;
    }

    eud_size = *((unsigned long *)dlsym(lib_handle, "EUD_SIZE"));
    if (eud_size == 0)
    {
        PRINT_ERROR("", "Unable to find the hardware diagnostics \"size\" symbol");
        return -1;
    }

    return 0;
}

void EUDWrapper::go(TestParameters *testParameters, const std::vector<unsigned int> &gpuList)
{
    char const *argv [6];
    char *eudFile = new char[FILE_NAME_LENGTH]();
    char *eudFileDeflated = new char[FILE_NAME_LENGTH]();
    char *eudPath = new char[EUD_PATH_LENGTH]();
    FILE *eudTgz;
    FILE *eudTar;
    int bytesToWrite = BUFF_SIZE;
    int bytesLeft;
    int head;
    unsigned int returnCode = 0;
    std::stringstream args, ss;
    
    clearWarnings();

    if(!testParameters->GetBoolFromString(EUD_STR_IS_ALLOWED))
    {
        addWarning("The hardware diagnostic is skipped for this GPU.");
        setResult(LWVS_RESULT_SKIP);
        delete[] eudFile;
        delete[] eudFileDeflated;
        delete[] eudPath;
        return;
    }

    if (openEudSymbols(LWVS_EUD_SYMBOLS_PATH) != 0)
    {
        addWarning("Unable to find hardware diagnostic, skipping this test.");
        setResult(LWVS_RESULT_SKIP);
        delete[] eudFile;
        delete[] eudFileDeflated;
        delete[] eudPath;
        return;
    }

    if (getuid() != 0) // check if we are root
    {
        addWarning("Hardware diagnostic tests must be run as root");
        setResult(LWVS_RESULT_SKIP);
        delete[] eudFile;
        delete[] eudFileDeflated;
        delete[] eudPath;
        return;
    }

    if (checkForGraphicsProcesses(gpuList))
    {
        addWarning("Already running graphic processes, skipping");
        setResult(LWVS_RESULT_SKIP);
        delete[] eudFile;
        delete[] eudFileDeflated;
        delete[] eudPath;
        return;
    }

    bytesLeft = eud_size; // now that eud_size is set
    argv[0] = "mods";
    argv[1] = new char[DEVICE_OPTION_LENGTH];
    argv[2] = NULL;

    m_EudDir = new char[DIR_NAME_LENGTH]();
    strcpy(m_EudDir, "/tmp/XXXXXX");

    if (mkdtemp(m_EudDir) == NULL)
    {
        addWarning("Failed to create hardware diagnostic temp directory");
        setResult(LWVS_RESULT_FAIL);
        goto cleanup;
    }

    sprintf(eudFile, "%s/XXXXXX", m_EudDir);
    sprintf(eudFileDeflated, "%s/XXXXXX", m_EudDir);

    //Creating EUD.tgz temp file and EUD.tar destination file
    if (createTempFile(&eudTgz, eudFile) != 0 || createTempFile(&eudTar, eudFileDeflated) != 0)
    {
        addWarning("Failed to create hardware diagnostic temp file");
        setResult(LWVS_RESULT_FAIL);
        fclose(eudTar);
        goto cleanup;
    }

    do
    {
        head = eud_size - bytesLeft;

        if (bytesLeft < bytesToWrite)
        {
            bytesToWrite = bytesLeft;
            bytesLeft = 0;
        }
        else
            bytesLeft -= BUFF_SIZE;

        if (fwrite((void*) (eud_bytes + head), sizeof(unsigned int), bytesToWrite, eudTgz) == 0 || ferror(eudTgz))
        {
            addWarning("Unable to write to hardware diagnostic temp file");
            fclose(eudTar);
            goto cleanup;
        }
    } while (bytesLeft != 0);

    fflush(eudTgz);
    rewind(eudTgz);

    if (deflate(eudTgz, eudTar) != 0)
    {
        addWarning("Unable to deflate hardware diagnostic tarball");
        setResult(LWVS_RESULT_FAIL);
        goto cleanup;
    }

    //We have to close the file to reopen it with libtar
    fclose(eudTar);

    if (untar(eudFileDeflated, m_EudDir) != 0)
    {
        setResult(LWVS_RESULT_FAIL);
        goto cleanup;
    }

    sprintf(eudPath, "%s/mods", m_EudDir);

    args.str("");
    if (gpuList.size() > 0)
        args << "pci_devices=";

    for (std::vector<unsigned int>::iterator gpuIt = gpuList.begin(); gpuIt != gpuList.end();)
    {
        unsigned int gpuId = *gpuIt;
        lwmlReturn_t ret;
        lwmlDevice_t device;
        lwmlPciInfo_t pciInfo;

        ret = lwmlDeviceGetHandleByIndex(*gpuIt, &device);
        if (ret != LWML_SUCCESS)
        {
            setResult(LWVS_RESULT_FAIL);
            addWarning("Unable to get device handle through LWML");
            goto cleanup;
        }
        ret = lwmlDeviceGetPciInfo(device, &pciInfo);
        if (ret != LWML_SUCCESS)
        {
            setResult(LWVS_RESULT_FAIL);
            addWarning("Unable to get PCI info from LWML");
            goto cleanup;
        }

        PRINT_DEBUG("%s","PCI Info returned %s", pciInfo.busId);

        args << pciInfo.busId;
        gpuIt++;
        if (gpuIt != gpuList.end())
            args << ",";
    }
 
    if (testParameters->GetString(EUD_PLUGIN_RUN_MODE) == "medium")
        args << " lwvs_mode=medium";

    args << " disable_progress_bar";

    /* Pass the EUD log file path to mods if the log file path is specified*/
    if(testParameters->GetString(EUD_LOG_FILE_PATH) != "")
        args << " logfilename="<<testParameters->GetString(EUD_LOG_FILE_PATH);
    
    PRINT_DEBUG("%s", "args to mods are: %s", args.str().c_str());

    try {
        returnCode = launch(args.str(), eudPath);
    } catch (std::exception &e)
    {
        setResult(LWVS_RESULT_FAIL);
        goto cleanup;
    }
    if (returnCode == 0)
        setResult(LWVS_RESULT_PASS);
    else 
        setResult(LWVS_RESULT_FAIL);

    for (std::vector<unsigned int>::iterator gpuIt = gpuList.begin(); gpuIt != gpuList.end(); gpuIt++)
        m_StatCollection->SetGpuStat(*gpuIt, "test_return_code", (long long int) returnCode);

    ss << "Return code from diagnostic test = " << std::setfill('0') << std::setw(12) << returnCode;
    addInfoVerbose(ss.str());
    
    WriteLog(testParameters->GetString(PS_LOGFILE), LWVS_LOGFILE_TYPE_JSON, m_StatCollection);

 cleanup:
    fclose(eudTgz);
    delete[] argv[1];
    delete[] eudFile;
    delete[] eudFileDeflated;
    delete[] eudPath;
}

int EUDWrapper::launch(std::string args, std::string eudPath)
{
    // use popen instead so we can capture the output
    FILE *fd;
    char c_output[1028]; // C-style input buffer for fgets
    std::string command_str = eudPath + " " + args;
    std::stringstream EudOutputss;
    std::string EudOutput;
    int retVal;

    fd = popen(command_str.c_str(), "r");
    if (!fd)
    {
        throw std::runtime_error("Unable to execute EUD()");
    }

    while(fgets(c_output, 1024, fd) != NULL)
    {
        EudOutputss << c_output;
    }

    // only grab the error code line
    EudOutput = EudOutputss.str();
    std::size_t pos = EudOutput.find("Error Code = ");
    if (pos == std::string::npos)
        throw std::runtime_error("Internal error in parsing the hardware diagnostic output");
        
    EudOutput = EudOutput.substr(pos);

    // find the return code
    EudOutput = EudOutput.substr(EudOutput.find_first_of("0123456789"), 12);

    std::istringstream(EudOutput) >> retVal;

    pclose(fd);

    return retVal;
}

int EUDWrapper::untar(char *tarPath, char *tarDir)
{
    TAR *handle = NULL;

    if (tar_open(&handle, tarPath, NULL, O_RDONLY, 0644, TAR_GNU) != 0)
    {
        addWarning("Unable to open the tar archive");
        return 1;
    }

    if (tar_extract_all(handle, tarDir) != 0)
    {
        addWarning("Unable to extract the tar archive");
        return 1;
    }

    if (tar_close(handle) != 0)
    {
        addWarning("Unable to close the tar archive");
        return 1;
    }

    return 0;
}

int EUDWrapper::WriteLog(std::string logFileName, int logFileType, StatCollection *statCollection)
{
    std::string outString;

    switch(logFileType)
    {
        case LWVS_LOGFILE_TYPE_BINARY:
            /* Binary is handled on statCollection creation */
            return 0;

        case LWVS_LOGFILE_TYPE_JSON:
            outString = statCollection->ToJson();
            break;

        case LWVS_LOGFILE_TYPE_TEXT:
            outString = statCollection->ToString();
            break;

        default:
            fprintf(stderr, "Unknown logFileType %d\n", logFileType);
            return -1;
    }

    FILE *fp = fopen(logFileName.c_str(), "wt");
    if(!fp)
    {
        fprintf(stderr, "Unable to open log file %s\n", logFileName.c_str());
        return -1;
    }

    int st = fputs(outString.c_str(), fp);
    if(st < 0)
    {
        fprintf(stderr, "Errno %s (%d) while writing to log file %s\n",
                strerror(errno), errno, logFileName.c_str());
        fclose(fp);
        return -1;
    }

    fclose(fp);
    fp = 0;
    return 0;
}

/*****************************************************************************/
extern "C" {
    Plugin *maker() {
        return new EUDWrapper;
    }
    class proxy {
    public:
        proxy()
        {
            factory["EUDWrapper"] = maker;
        }
    };
    proxy p;
}

