#include <errno.h>
#include <string>
#include <unistd.h>

#include "DcgmUtilities.h"
#include "logging.h"

/***********************************************************************************************/
/* 
Execute a process where args is a vector of arguments for the process and args[0] is the name of the 
exelwtable to run.

infp, outfp, errfp will point to stdin, stdout, and stderr of the forked process.
If stderrToStdout is true, errfp is ignored, and stdout of child is redirected to outfp.

Returns the pid of the forked process. The returned pid is < 0 if there was an error when forking or creating the pipes
*/
pid_t DcgmUtilForkAndExecCommand(std::vector<std::string> &args, int *infp, int *outfp, int *errfp, bool stderrToStdout) 
{    
    pid_t pid;
    const int READ = 0, WRITE = 1, ERROR = 2;
    int p_stdin[2], p_stdout[2], p_stderr[2];

    // Ensure outfp is not null
    if (outfp == NULL)
    {
        PRINT_ERROR("", "Output file descriptor cannot be NULL.");
        return -1;
    }

    // Create pipes for stdin, out, and err
    if (pipe(p_stdin) != 0)
    {
        PRINT_ERROR("%s", "Could not create stdin pipe for external command. %s", strerror(errno));
        return -1;
    }
    if (pipe(p_stdout) != 0)
    {
        PRINT_ERROR("%s", "Could not create stdout pipe for external command. %s", strerror(errno));
        // Close pipes before returning
        close(p_stdin[READ]);
        close(p_stdin[WRITE]);
        return -1;
    }
    if (pipe(p_stderr) != 0)
    {
        PRINT_ERROR("%s", "Could not create stderr pipe for external command. %s", strerror(errno));
        // Close pipes before returning
        close(p_stdin[READ]);
        close(p_stdin[WRITE]);
        close(p_stdout[READ]);
        close(p_stdout[WRITE]);
        return -1;
    }


    pid = fork();
    if (pid < 0) // Failed to create child
    {
        PRINT_ERROR("%s %s", "Could not fork to run the external command '%s': %s", args[0].c_str(), strerror(errno));
        // Close pipes before returning
        close(p_stdin[READ]);
        close(p_stdin[WRITE]);
        close(p_stdout[READ]);
        close(p_stdout[WRITE]);
        close(p_stderr[READ]);
        close(p_stderr[WRITE]);
        return pid;
    }
    else if (pid == 0) // Child
    {
        // Connect std streams to appropriate pipes
        if (dup2(p_stdin[READ], READ) == -1)
        {
            PRINT_ERROR("%s %s", "Could not connect pipe to stdin of external command '%s': %s", 
                        args[0].c_str(), strerror(errno));
            exit(1);
        }
        if (dup2(p_stdout[WRITE], WRITE) == -1)
        {
            PRINT_ERROR("%s %s", "Could not connect pipe to stdout of external command '%s': %s", 
                        args[0].c_str(), strerror(errno));
            exit(1);
        }
        
        // Handle stderr
        if (stderrToStdout)
        {
            if (dup2(p_stdout[WRITE], ERROR) == -1)
            {
                PRINT_ERROR("%s %s", "Could not redirect stderr to stdout pipe of external command '%s': %s", 
                            args[0].c_str(), strerror(errno));
                exit(1);
            }
        }
        else 
        {
            if (dup2(p_stderr[WRITE], ERROR) == -1)
            {
                PRINT_ERROR("%s %s", "Could not connect pipe to stderr of external command '%s': %s", 
                            args[0].c_str(), strerror(errno));
                exit(1);
            }
        }
        // Close pipes - only the parent will use them
        close(p_stdin[READ]);
        close(p_stdin[WRITE]);
        close(p_stdout[READ]);
        close(p_stdout[WRITE]);
        close(p_stderr[READ]);
        close(p_stderr[WRITE]);

        // Colwert args to argv style char** for execvp
        std::vector<const char*> argv(args.size()+1);
        for (unsigned int i = 0; i < args.size(); i++)
        {
            argv[i] = args[i].c_str();
        }
        argv[args.size()] = NULL;

        execvp(argv[0], const_cast<char**>(argv.data()));
        PRINT_ERROR("%s %s", "Could not exec '%s': %s", argv[0], strerror(errno));
        exit(1);
    }
    // Parent
    // Close unused ends of pipes
    close(p_stdin[READ]);
    close(p_stdout[WRITE]);
    close(p_stderr[WRITE]);
    if (infp == NULL)
    {
        close(p_stdin[WRITE]);
    }
    else
    {    
        *infp = p_stdin[WRITE];
    }

    *outfp = p_stdout[READ];
    
    if (errfp == NULL || stderrToStdout)
    {    
        close(p_stderr[READ]);
    }
    else
    {    
        *errfp = p_stderr[READ];
    }

    return pid;
}
