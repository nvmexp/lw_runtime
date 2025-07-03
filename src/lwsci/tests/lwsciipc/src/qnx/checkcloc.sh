#/bin/sh
#
# safety build

MODULE=$TEGRA_TOP/qnx/src/tools/lwsciipc_init
cloc $MODULE/*.c $MODULE/*.h

MODULE=$TEGRA_TOP/qnx/src/resmgrs/lwsciipc
cloc $MODULE/*.c $MODULE/*.h

MODULE=$TEGRA_TOP/qnx/src/resmgrs/lwivc
cloc $MODULE/*.c $MODULE/*.h

MODULE=$TEGRA_TOP/gpu/drv/drivers/lwsci/lwsciipc
HEAD=$TEGRA_TOP/gpu/drv/drivers/lwsci/inc
cloc $MODULE/src/lwsciipc.c $MODULE/src/lwsciipc_ipc.c $MODULE/src/lwsciipc_ivc.c $MODULE/src/lwsciipc_os_error.c $MODULE/src/lwsciipc_os_qnx.c $MODULE/inc/lwsciipc_common.h $MODULE/inc/lwsciipc_i*.h $MODULE/inc/lwsciipc_os*.h $MODULE/inc/lwsciipc_log.h $MODULE/inc/lwsciipc_static*.h $MODULE/inc/cheetah-ivc-dev.h $HEAD/internal/lwsciipc_internal.h $HEAD/public/lwsciipc.h

MODULE=$TEGRA_TOP/gpu/drv/drivers/lwsci/lwscievent
cloc $MODULE/lwscievent_qnx.c $HEAD/internal/lwscievent_internal.h $HEAD/internal/lwscievent_qnx.h $HEAD/public/lwscievent.h

