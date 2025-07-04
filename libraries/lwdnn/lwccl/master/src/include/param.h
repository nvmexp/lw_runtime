/*************************************************************************
 * Copyright (c) 2017-2019, LWPU CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

static const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

static void setElwFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char elwVar[1024];
  char elwValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Elw Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(elwVar, line, std::min(1024,s));
    elwVar[s] = '\0';
    s++;
    strncpy(elwValue, line+s, 1024);
    setelw(elwVar, elwValue, 0);
  }
  if (line) free(line);
  fclose(file);
}

static void initElw() {
  char confFilePath[1024];
  const char * userDir = userHomeDir();
  if (userDir) {
    sprintf(confFilePath, "%s/.lwcl.conf", userDir);
    setElwFile(confFilePath);
  }
  sprintf(confFilePath, "/etc/lwcl.conf");
  setElwFile(confFilePath);
}


#define NCCL_PARAM(name, elw, default_value) \
pthread_mutex_t ncclParamMutex##name = PTHREAD_MUTEX_INITIALIZER; \
int64_t ncclParam##name() { \
  static_assert(default_value != -1LL, "default value cannot be -1"); \
  static int64_t value = -1LL; \
  pthread_mutex_lock(&ncclParamMutex##name); \
  if (value == -1LL) { \
    value = default_value; \
    char* str = getelw("NCCL_" elw); \
    if (str && strlen(str) > 0) { \
      errno = 0; \
      int64_t v = strtoll(str, NULL, 0); \
      if (errno) { \
        INFO(NCCL_ALL,"Invalid value %s for %s, using default %lu.", str, "NCCL_" elw, value); \
      } else { \
        value = v; \
        INFO(NCCL_ALL,"%s set by environment to %lu.", "NCCL_" elw, value);  \
      } \
    } \
  } \
  pthread_mutex_unlock(&ncclParamMutex##name); \
  return value; \
}

#endif
