/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2007-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include <stdio.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>
#include <sys/stat.h>

#include <readline/readline.h>
#include <readline/history.h>

#include "os.h"
#include "exts.h"

LwU32 osCheckControlC(void)
{
    return 0;
}

LwU32 osGetInputLine(LwU8 *prompt, LwU8 *buffer, LwU32 bytes)
{
    char *line;

    line = readline((char *)prompt);
    add_history(line);

    strncpy((char *)buffer, line, bytes);
    buffer[bytes-1] = '\0';

    return 0;
}

LwU64 GetExpression(const char *args)
{
    LwU64 value = 0;
    char *endp;

    endp = NULL;
    (void)GetExpressionEx(args, &value, &endp);

    return value;
}

BOOL GetExpressionEx(const char *args, LwU64 *value, char **endp)
{
    if (!args || !args[0] || !endp)
        return FALSE;

    *value = strtoull(args, endp, 0);

    return ((*endp) != args);
}

static struct {
    char *name;
    void (*dispatch)(char *args);
} api_ops[] = {
#define LWWATCH_API(x) { #x, (void (*)(char *args))x },
#include "exts.h"
    { NULL, NULL }
};
#undef LWWATCH_API

void main_loop(int (*callback)(char *name, char *args))
{
    const char *home = getelw("HOME");
    const char *history_file_name = "/.lw/lwwatch_history.txt";
    char *history_path = NULL;
    char *line, *name, *cmd, *args, *pipe;
    int i, quit = 0;
    unsigned int seconds, ms;
    void *data = NULL;

    if (home) {
        history_path = malloc(strlen(home) + strlen(history_file_name) + 1);
        if (history_path) {
            strcpy(history_path, home);
            strcat(history_path, history_file_name);
            read_history(history_path);
        }
    }

    while (!quit) {
        line = readline("lw> ");
        if (!line) break;

        if (strlen(line) == 0) {
            free(line);
            continue;
        }

        cmd = name = strdup(line);
        if (!cmd) {
            fprintf(stderr, "strdup() failed (%d)!\n", errno);
            free(line);
            continue;
        }
        while (*name == ' ')
            name++;

        pipe = strchr(name, '|');
        if (pipe) {
            *pipe++ = '\0';
            data = start_command(pipe);
        }

        args = strchr(name, ' ');
        if (args)
            *args++ = '\0';
        else
            args = "";

        if ((strcasecmp(name, "quit") == 0) ||
            (strcasecmp(name, "q") == 0) ||
            (strcasecmp(name, "exit") == 0)) {
            quit = 1;
        } else if (strcasecmp(name, "delay") == 0) {
            if (!args || (strlen(args) == 0))
                fprintf(stderr, "%s: expects an argument!\n", name);
            else {
                ms = atoi(args);
                usleep(ms * 1000);
            }
        } else if (strcasecmp(name, "wait") == 0) {
            if (!args || (strlen(args) == 0))
                fprintf(stderr, "%s: expects an argument!\n", name);
            else {
                seconds = atoi(args);
                sleep(seconds);
            }
        } else if (!callback || callback(name, args)) {
            if (*name == '!')
                name++;
            for (i = 0; api_ops[i].name; i++) {
                if (strcasecmp(api_ops[i].name, name) == 0) {
                    api_ops[i].dispatch(args);
                    break;
                }
            }
            if (!api_ops[i].name)
                fprintf(stderr, "%s: not a valid/supported command!\n", name);
        }

        if (data) {
            complete_command(data);
            data = NULL;
        }

        free(cmd);

        add_history(line);
        free(line);
    }

    if (history_path) {
        char *path_copy = strdup(history_path);
        char *dir_path = path_copy ? dirname(path_copy) : NULL;

        if (dir_path) {
            mkdir(dir_path, 0700);
        }

        free(path_copy);

        /* Don't allow unbounded history growth */
        stifle_history(10000);
        write_history(history_path);
        free(history_path);
    }
}
