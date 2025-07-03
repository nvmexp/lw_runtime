#include "Config.h"

#if IPC_ENABLED == 1
#pragma warning(push)
#pragma warning(disable:4267 4244)
#include "ipc.pb.cc"
#pragma warning(pop)
#endif