#include "utils/utils.h"
#include <lwos_platform.h>

#if lwosOsIsWindows()
#include <windows.h>
#else
#include <spawn.h>
#include <sys/wait.h>
#include <sys/types.h>
#if lwosOsIsApple()
#include <signal.h>
#endif
#endif

namespace utils
{

static UtilsElwironment *lwrrent_elw = nullptr;

/* Parses a string as a command line flag.  The string should have
 * the format "--flag=value".  When def_optional is true, the "=value"
 * part can be omitted.
 *
 * Returns the value of the flag, or NULL if the parsing failed.
 * *** Copied from google test *** */
const char *UtilsElwironment::parseFlagValue(const char *str, const char *flag, bool def_optional)
{
    /* str and flag must not be NULL. */
    if (str == nullptr || flag == nullptr) {
        return nullptr;
    }

    /* The flag must start with "--" followed by GTEST_FLAG_PREFIX_. */
    const std::string flag_str = std::string("--") + flag;
    const size_t flag_len = flag_str.length();
    if (strncmp(str, flag_str.c_str(), flag_len) != 0) {
        return nullptr;
    }

    /* Skips the flag name. */
    const char *flag_end = str + flag_len;

    /* When def_optional is true, it's OK to not have a "=value" part. */
    if (def_optional && (flag_end[0] == '\0')) {
        return flag_end;
    }

    /* If def_optional is true and there are more characters after the
     * flag name, or if def_optional is false, there must be a '=' after
     * the flag name. */
    if (flag_end[0] != '=') {
        return nullptr;
    }

    /* Returns the string after "=". */
    return flag_end + 1;
}

UtilsElwironment::UtilsElwironment(const char *_programName) : m_isStressMode(false), m_isChildProcess(false),
    m_parentProcessId(), m_extra(), m_processIdx(0), programName(_programName)
{
    lwrrent_elw = this;
}

UtilsElwironment::~UtilsElwironment()
{
}

void UtilsElwironment::SetUp()
{
}

void UtilsElwironment::TearDown()
{
}

void UtilsElwironment::parseArguments(int argc, char **argv)
{
    for (int i = 0; i < argc; i++) {
        const char *value = NULL;
        if ((value = parseFlagValue(argv[i], "stress", true)) != nullptr) {
            m_isStressMode = !(*value == '0' || *value == 'f' || *value == 'F');
        }
        else if ((value = parseFlagValue(argv[i], "child", true)) != nullptr) {
            m_isChildProcess = !(*value == '0' || *value == 'f' || *value == 'F');
        }
        else if ((value = parseFlagValue(argv[i], "parent", false)) != nullptr) {
            std::stringstream ss = std::stringstream(value);
            unsigned x;
            ss >> x;
            m_parentProcessId = static_cast<LWOSPid>(x);
        }
        else if ((value = parseFlagValue(argv[i], "idx", false)) != nullptr) {
            std::stringstream ss = std::stringstream(value);
            unsigned x;
            ss >> x;
            m_processIdx = x;
        }
        else if ((value = parseFlagValue(argv[i], "extra", false)) != nullptr) {
            m_extra = value;
        }
        else if ((value = parseFlagValue(argv[i], "help", true)) != nullptr) {
            std::cout << "Utils-specific options:\n"
                      << "--stress\tEnable stress mode" << std::endl
                      << "--child \tThis process is a spawned child of another process" << std::endl
                      << "--parent=PID\tPid of the parent process" << std::endl
                      << "--extra=STR\tExtra argument passed, usually by the parent process" << std::endl
                      << std::endl;
            break;
        }
    }

#if !lwosOsIsWindows()
    if (!m_isChildProcess) {
        // On POSIX, we need to ensure that any processes we spawn are properly
        // cleaned up if the main application dies for whatever reason.  To do
        // this, we create a new process group for the first process with it as
        // the group leader.  When the process group becomes orphaned, the group
        // will be sent an unhandled signal, which will kill all processes in
        // the group.  All children of the first process will inherit this
        // process group.  This *may* fail if the root process was made session
        // leader of the process group (vscode does this), but those are corner
        // cases we don't partilwlarly care about.
        // On Windows, we manage this via Jobs.  See Process::start for more
        // information.
        (void)setpgid(0, 0);
    }
#endif
}

bool UtilsElwironment::createProcess(Process& process, size_t id, std::string extra) const
{
    std::string pid_str, test_case_str, id_str;

    {
        std::stringstream ss;
        ss << "--parent=" << lwosProcessId();
        pid_str = ss.str();
    }

    {
        const ::testing::TestInfo *info = ::testing::UnitTest::GetInstance()->lwrrent_test_info();
        std::stringstream ss;
        ss << "--gtest_filter=" << info->test_case_name() << '.' << info->name();
        test_case_str = ss.str();
    }

    {
        std::stringstream ss;
        ss << "--idx=" << id;
        id_str = ss.str();
    }

    std::vector<const char *> args = { test_case_str.c_str(), "--child", id_str.c_str(), pid_str.c_str() };
    if (m_isStressMode) {
        args.push_back("--stress");
    }
    if (!extra.empty()) {
        extra = "--extra=" + extra;
        args.push_back(extra.c_str());
    }

    return process.start(programName.c_str(), args) == 0;
}

const UtilsElwironment *UtilsElwironment::getElw()
{
    return lwrrent_elw;
}

/***************************************************
 * QuietUnitTestResultPrinter class implementation *
 ***************************************************/

QuietUnitTestResultPrinter::QuietUnitTestResultPrinter(std::ostream& stream) : m_stream(stream)
{

}

QuietUnitTestResultPrinter::~QuietUnitTestResultPrinter()
{

}

void QuietUnitTestResultPrinter::OnTestPartResult(const ::testing::TestPartResult& result)
{
    if (result.passed() /* || result.skipped() */) {
        return;
    }

    m_stream << "FAILURE at " << result.file_name() << ':' << result.line_number() << std::endl
             << '\t' << result.message() << std::endl;
}


/********************************
 * Process class implementation *
 ********************************/

Process::Process() : processHandle(0)
{

}
Process::~Process()
{
    EXPECT_EQ(terminate(), 0);
    EXPECT_EQ(wait(), 0);
}

#if lwosOsIsWindows()

static HANDLE globalJob;

struct pipeHandlerArgs {
    HANDLE std_out_pipe[2];
    pipeHandlerArgs(HANDLE rd1, HANDLE wr1) : std_out_pipe {rd1, wr1} {}
};

static void jobInit(void)
{
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info = { 0 };
    globalJob = CreateJobObject(NULL, NULL);

    if ((globalJob == NULL) || (globalJob == ILWALID_HANDLE_VALUE)) {
        return;
    }
    // This is the magic that kills child processes if the parent dies
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(globalJob, JobObjectExtendedLimitInformation, &info, sizeof(info));
    // Make sure this is the only reference to this job group
    SetHandleInformation(globalJob, HANDLE_FLAG_INHERIT, 0);
}

// The thread that manages this function will forward everything to the parent's
// stdout.  This is the preferred way to continuously pipe a child's output to
// the parent's console
static int pipeHandlerFunc(void *userArgs)
{
    struct pipeHandlerArgs *args = (struct pipeHandlerArgs *)userArgs;
    while (1) {
        char buffer;
        DWORD nbytes = 0;
        if (!ReadFile(args->std_out_pipe[0], &buffer, sizeof(buffer), &nbytes, NULL) || nbytes == 0) {
            break;
        }
        WriteFile(args->std_out_pipe[1], &buffer, nbytes, &nbytes, NULL);
    }

    // Clean up the read handle, which should free the pipe, and clean the arguments passed to us
    CloseHandle(args->std_out_pipe[0]);
    delete args;

    return 0;
}

int Process::start(const char *progname, std::vector<const char *> args)
{
    static lwosOnceControl jobInitOnce = LWOS_ONCE_INIT;
    HANDLE childStdout_rd;
    STARTUPINFO info = { sizeof(info) };
    PROCESS_INFORMATION processInfo;
    SELWRITY_ATTRIBUTES saAttr;

    // Make sure the job object is initialized
    lwosOnce(&jobInitOnce, jobInit);

    if (processHandle != 0) {
        return -1;
    }

    std::string cmdLine(progname);
    // Join all the arguments into one string
    for (size_t i = 0; i < args.size(); i++) {
        cmdLine += " ";
        cmdLine += args[i];
    }

    saAttr.nLength = sizeof(SELWRITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSelwrityDescriptor = NULL;

    // Make a pipe to be used for the child's stdout/stderr
    if (!CreatePipe(&childStdout_rd, &info.hStdOutput, &saAttr, 0)) {
        return GetLastError();
    }
    // Mark the read handle as not inheritable, so the child doesn't get a
    // reference
    if (!SetHandleInformation(childStdout_rd, HANDLE_FLAG_INHERIT, 0)) {
        return GetLastError();
    }

    // Pipe stderr to stdout and drop stdin
    info.hStdError = info.hStdOutput;
    info.hStdInput = ILWALID_HANDLE_VALUE;
    info.dwFlags = STARTF_USESTDHANDLES;

    pipeHandlerArgs *p = new pipeHandlerArgs(childStdout_rd, GetStdHandle(STD_OUTPUT_HANDLE));
    if (lwosThreadCreate(&thread, pipeHandlerFunc, p)) {
        return -1;
    }

    // Create the process as suspended so we can modify some of it's run state attributes later
    if (!CreateProcess(NULL, const_cast<char *>(cmdLine.c_str()), NULL, NULL, TRUE, CREATE_SUSPENDED, NULL, NULL,
                       &info, &processInfo)) {
        return GetLastError();
    }

    processHandle = (unsigned long long)processInfo.hProcess;

    // Assign the child to this job object such that, if the job owner dies,
    // all the processes assigned to the job object also die
    // Note: this *may* fail on win7 systems when processes spawn processes that
    // spawn processes as win7 doesn't support nested jobs.  We ignore this since
    // the failure still lives the nested processes part of the parent job,
    // which will properly terminate any orphaned processes
    if ((globalJob != NULL) && (globalJob != ILWALID_HANDLE_VALUE)) {
        AssignProcessToJobObject(globalJob, processInfo.hProcess);
    }

    // Close our side of the stdout pipe
    CloseHandle(info.hStdOutput);

    // All done modifying the process's run state, resume it!
    ResumeThread(processInfo.hProcess);

    return 0;
}

int Process::wait()
{
    DWORD exit_code = 0;
    int retcode = 0;

    if (processHandle == 0) {
        return 0;
    }

    // Wait for the process to complete
    if (WaitForSingleObject((HANDLE)processHandle, INFINITE) == WAIT_FAILED) {
        return GetLastError();
    }

    // Wait for the thread to finish it's work (which means the pipe will EOF
    // and complete the thread's loop)
    lwosThreadJoin(thread, &retcode);
    if (retcode != 0) {
        return retcode;
    }

    (void)GetExitCodeProcess((HANDLE)processHandle, &exit_code);

    CloseHandle((HANDLE)processHandle);
    processHandle = 0;

    return static_cast<int>(exit_code);
}

int Process::terminate()
{
    if (processHandle == 0) {
        return 0;
    }

    if (TerminateProcess((HANDLE)processHandle, ~0U)) {
        return GetLastError();
    }

    return 0;
}

LWOSPid Process::getProcessId() const
{
    return (LWOSPid)GetProcessId((HANDLE)processHandle);
}

int Process::waitAny(ProcessList& processes, size_t& idx)
{
    DWORD exit_code = 0;
    std::vector<HANDLE> handles(processes.size(), ILWALID_HANDLE_VALUE);

    for (size_t i = 0; i < processes.size(); i++) {
        handles[i] = (HANDLE)processes[i]->processHandle;
    }

    if ((idx = WaitForMultipleObjects((DWORD)handles.size(), &handles[0], FALSE,
                                      INFINITE)) == WAIT_FAILED) {
        return GetLastError();
    }

    if (idx >= WAIT_ABANDONED_0 && idx < (WAIT_ABANDONED_0 + handles.size())) {
        idx -= WAIT_ABANDONED_0;
    }
    else if (idx < (WAIT_OBJECT_0 + handles.size())) {
        idx -= WAIT_OBJECT_0;
    }

    (void)GetExitCodeProcess(handles[idx], &exit_code);
    CloseHandle(handles[idx]);
    processes[idx]->processHandle = 0;

    return (int)exit_code;
}

void Process::waitAll(ProcessList& processes, std::vector<int>& exitCodes)
{
    std::vector<HANDLE> handles(processes.size(), ILWALID_HANDLE_VALUE);
    exitCodes.resize(processes.size());

    for (size_t i = 0; i < processes.size(); i++) {
        handles[i] = (HANDLE)processes[i]->processHandle;
    }

    if (WaitForMultipleObjects((DWORD)handles.size(), &handles[0], TRUE, INFINITE) == WAIT_FAILED) {
        for (size_t i = 0; i < exitCodes.size(); i++) {
            exitCodes[i] = -1;
        }
        return;
    }

    for (size_t i = 0; i < handles.size(); i++) {
        DWORD exit_code;
        (void)GetExitCodeProcess(handles[i], &exit_code);
        CloseHandle(handles[i]);
        processes[i]->processHandle = 0;
        exitCodes[i] = exit_code;
    }
}

#else
// **************************
// * The calls within this section must remain POSIX standard for portability
// * reasons.  Do not attempt to use non-POSIX compliant functions unless
// * prefaced with a proper OS-specific define.
// **************************
int Process::start(const char *progname, std::vector<const char *> args)
{
    int result = 0;
    posix_spawnattr_t attr;

    if (processHandle != 0) {
        return -1;
    }

    // Last argument needs to be a null pointer
    args.push_back(nullptr);
    // First argument needs to be argv[0]
    args.insert(args.begin(), progname);

    result = posix_spawnattr_init(&attr);
    if (result != 0) {
        return result;
    }

#ifdef POSIX_SPAWN_USEVFORK
    // We're going to fork+exec, but we don't know when or where, so don't
    // bother copying page tables, especially with LWCA which reserves large
    // swaths of VA space, allowing for quicker forking
    posix_spawnattr_setflags(&attr, POSIX_SPAWN_USEVFORK);
    if (result != 0) {
        goto done;
    }
#endif

    // Yes, we're using fork+exec via spawn here in order to maintain
    // consistency between platforms.  File descriptors (including std*) are
    // still inherited, but you cannot continue in the child where you left off
    // -- this is intentional.
    result = posix_spawnp((pid_t *)&processHandle, progname, NULL, &attr,
                          const_cast<char *const *>(&args[0]), NULL);

done:
    (void)posix_spawnattr_destroy(&attr);

    return result;
}

int Process::wait()
{
    int status = 0;

    if (processHandle == 0) {
        return 0;
    }

    while (waitpid((pid_t)processHandle, &status, 0) > 0) {
        if (WIFEXITED(status)) {
            processHandle = 0;
            return WEXITSTATUS(status);
        }
    }

    return errno;
}

int Process::terminate()
{
    if (processHandle == 0) {
        return 0;
    }

    return kill((pid_t)processHandle, SIGTERM);
}

LWOSPid Process::getProcessId() const
{
    return (LWOSPid)processHandle;
}

int Process::waitAny(ProcessList& processes, size_t& idx)
{
    pid_t pid;
    int status = 0;

    while ((pid = waitpid(-1, &status, 0)) > 0) {
        if (!WIFEXITED(status)) {
            continue;
        }
        for (size_t i = 0; i < processes.size(); i++) {
            if ((pid_t)processes[i]->processHandle != pid) {
                continue;
            }

            idx = i;
            processes[i]->processHandle = 0;
            return WEXITSTATUS(status);
        }
        // Nope, not one of these, keep going...
    }
    return errno;
}

void Process::waitAll(ProcessList& processes, std::vector<int>& exitCodes)
{
    size_t procsLeft = processes.size();
    pid_t pid;
    int status = 0;

    while (procsLeft > 0 && (pid = waitpid(-1, &status, 0)) > 0) {
        if (!WIFEXITED(status)) {
            continue;
        }
        for (size_t i = 0; i < processes.size(); i++) {
            if ((pid_t)processes[i]->processHandle != pid) {
                continue;
            }
            exitCodes[i] = WEXITSTATUS(status);
            processes[i]->processHandle = 0;
            procsLeft--;
            break;
        }
    }
}

#endif

}
