/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2012 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#define _GNU_SOURCE
#define __STDC_FORMAT_MACROS
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>

#include "os.h"
#include "exts.h"

/*
 * Run the command specified by 'cmd', piping STDOUT from this process
 * to the command.
 *
 * Note that while the child process is running, we want STDOUT of
 * lwwatch redirected to it; this is done using dup2(2) to make
 * STDOUT_FILENO be a copy of the file descriptor underlying the
 * stream returned by popen(2).  To restore STDOUT afterwards we use
 * the following: dup STDOUT's fd before redirecting STDOUT, and then
 * restore STDOUT using that saved fd.
 *
 * References:
 *  http://c-faq.com/stdio/undofreopen.html
 *  http://c-faq.com/stdio/rd.kirby.c
 *  http://stackoverflow.com/questions/584868/rerouting-stdin-and-stdout-from-c
 */

struct command_data_t {
    FILE *stream;
    struct sigaction old_signal_action;
    int saved_stdout_fd;
};

void *start_command(const char *cmd)
{
    struct sigaction action;
    struct command_data_t *command_data;
    int ret;

    command_data = calloc(1, sizeof(struct command_data_t));

    if (!command_data) {
        return NULL;
    }

    /* we may receive SIGPIPEs if the child process exits abnormally */
    memset(&action, 0, sizeof(action));
    action.sa_handler = SIG_IGN;
    if (sigaction(SIGPIPE, &action, &command_data->old_signal_action) != 0) {
        fprintf(stderr, "Failed to reconfigure SIGPIPE handler: %s\n",
                strerror(errno));
        free(command_data);
        return NULL;
    }

    if (fflush(stdout) != 0) {
        fprintf(stderr, "Failed to fflush(3) stdout: %s\n", strerror(errno));
        goto fail;
    }

    command_data->saved_stdout_fd = dup(STDOUT_FILENO);
    if (command_data->saved_stdout_fd == -1) {
        fprintf(stderr, "Failed to dup(2) stdout: %s\n", strerror(errno));
        goto fail;
    }

    command_data->stream = popen(cmd, "w");
    if (!command_data->stream) {
        fprintf(stderr, "Failed to run `%s`: %s\n", cmd, strerror(errno));
        goto fail;
    }

    if (dup2(fileno(command_data->stream), STDOUT_FILENO) != STDOUT_FILENO) {
        fprintf(stderr, "Failed to dup2(2) lwwatch's STDOUT to the "
                "STDIN of `%s`: %s\n", cmd, strerror(errno));
        goto fail;
    }

    return command_data;

 fail:

    if (command_data->stream) {
        pclose(command_data->stream);
    }

    /* restore the original signal handler for SIGPIPE */
    sigaction(SIGPIPE, &command_data->old_signal_action, NULL);

    free(command_data);
    return NULL;
}

void complete_command(void *data)
{
    struct command_data_t *command_data = data;

    fflush(command_data->stream);

    dup2(command_data->saved_stdout_fd, fileno(stdout));
    close(command_data->saved_stdout_fd);
    clearerr(stdout);

    pclose(command_data->stream);

    /* restore the original signal handler for SIGPIPE */
    sigaction(SIGPIPE, &command_data->old_signal_action, NULL);

    free(command_data);
}
