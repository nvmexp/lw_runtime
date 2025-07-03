#!/bin/csh -e
if ( -f commands.tmp ) then
    rm "commands.tmp"
endif

# use host config if building for host
if ( $1 == "host" ) then
    echo 'options C_COMPILER_CFG_SOURCE PY_CONFIGURATOR' >> commands.tmp
    echo 'options C_COMPILER_FAMILY_NAME GNU_Native' >> commands.tmp
    echo 'options C_COMPILER_HIERARCHY_STRING GNU Native_6.3_C' >> commands.tmp
    echo 'options C_COMPILER_PY_ARGS --lang c --version 6.3' >> commands.tmp
    echo 'options C_COMPILER_TAG GNU_C_63' >> commands.tmp
    echo 'options C_COMPILER_VERSION_CMD gcc --version' >> commands.tmp
    echo 'options C_COMPILE_CMD gcc -c -g -U__x86_64__ -std=gnu++98 -DNDEBUG -DGL_EXPERT -DLWPMAPI -DLWCONFIG_PROFILE=tegragpu_unix_arm_embedded_external_profile -D_LANGUAGE_C -D__NO_CTYPE -DLW_BUILD_CONFIGURATION_EXPOSING_T18X=1 -DLW_BUILD_CONFIGURATION_EXPOSING_T19X=1 -DLW_BUILD_CONFIGURATION_EXPOSING_T23X=0 -DLW_DEBUG=0 -DLW_IS_SAFETY=1 -DLWCFG_ENABLED -DLW_BUILD_EMBEDDED=1 -DPIC  -DLW_TEGRA_MIRROR_INCLUDES -DLW_SCI_DESKTOP_COMPATIBLE_HEADERS -D_QNX_SOURCE -DLW_QNX' >> commands.tmp
    echo 'options C_DEFINE_LIST LW_QNX LW_IS_SAFETY=1 LW_TEGRA_MIRROR_INCLUDES LW_SCI_DESKTOP_COMPATIBLE_HEADERS' >> commands.tmp
    echo 'options C_EDG_FLAGS -w --gcc --gnu_version 60300' >> commands.tmp
    echo 'options C_LINKER_VERSION_CMD ld --version' >> commands.tmp
    echo 'options C_LINK_CMD gcc -g' >> commands.tmp
    echo 'options C_PREPROCESS_CMD gcc -E -C -U__x86_64__ -std=gnu++98 -DNDEBUG -DGL_EXPERT -DLWPMAPI -DLWCONFIG_PROFILE=tegragpu_unix_arm_embedded_external_profile -D_LANGUAGE_C -D__NO_CTYPE -DLW_BUILD_CONFIGURATION_EXPOSING_T18X=1 -DLW_BUILD_CONFIGURATION_EXPOSING_T19X=1 -DLW_BUILD_CONFIGURATION_EXPOSING_T23X=0 -DLW_DEBUG=0 -DLW_IS_SAFETY=1 -DLWCFG_ENABLED -DLW_BUILD_EMBEDDED=1 -DPIC  -DLW_TEGRA_MIRROR_INCLUDES -DLW_SCI_DESKTOP_COMPATIBLE_HEADERS -D_QNX_SOURCE -DLW_QNX' >> commands.tmp
    echo 'options VCAST_PREPROCESS_PREINCLUDE $(VECTORCAST_DIR)/DATA/gnu_native/intrinsics.h' >> commands.tmp
    echo 'options VCAST_TYPEOF_OPERATOR TRUE' >> commands.tmp
# else use target config
else
    echo 'options C_COMPILER_CFG_SOURCE BUILT_IN_TAG' >> commands.tmp
    echo 'options C_COMPILER_HIERARCHY_STRING GNU Target_Linux_ARM_C' >> commands.tmp
    echo 'options C_COMPILER_TAG GNUARM_LINUX_C' >> commands.tmp
    echo 'options C_COMPILE_CMD qcc -Vgcc_ntoaarch64le -c -g  -DNDEBUG -DGL_EXPERT -DLWPMAPI -DLWCONFIG_PROFILE=tegragpu_unix_arm_embedded_external_profile -D_LANGUAGE_C -D__NO_CTYPE -DLW_BUILD_CONFIGURATION_EXPOSING_T18X=1 -DLW_BUILD_CONFIGURATION_EXPOSING_T19X=1 -DLW_BUILD_CONFIGURATION_EXPOSING_T23X=0 -DLW_DEBUG=0 -DLW_IS_SAFETY=1 -DLWCFG_ENABLED -DLW_BUILD_EMBEDDED=1 -DPIC  -DLW_TEGRA_MIRROR_INCLUDES -D_QNX_SOURCE -DLW_QNX' >> commands.tmp
    echo 'options C_DEFINE_LIST ' >> commands.tmp
    echo 'options C_EDG_FLAGS -w --gcc --gnu_version 40200 --c99' >> commands.tmp
    echo "options C_EXELWTE_CMD $PWD/execute.sh $TARGET" >> commands.tmp
    echo 'options C_LINK_CMD qcc -Vgcc_ntoaarch64le -g' >> commands.tmp
    echo 'options C_PREPROCESS_CMD qcc -E -Vgcc_ntoaarch64le -C -DNDEBUG -DGL_EXPERT -DLWPMAPI -DLWCONFIG_PROFILE=tegragpu_unix_arm_embedded_external_profile -D_LANGUAGE_C -D__NO_CTYPE -DLW_BUILD_CONFIGURATION_EXPOSING_T18X=1 -DLW_BUILD_CONFIGURATION_EXPOSING_T19X=1 -DLW_BUILD_CONFIGURATION_EXPOSING_T23X=0 -DLW_DEBUG=0 -DLW_IS_SAFETY=1 -DLWCFG_ENABLED -DLW_BUILD_EMBEDDED=1 -DPIC  -DLW_TEGRA_MIRROR_INCLUDES -D_QNX_SOURCE -DLW_QNX' >> commands.tmp
    echo 'options VCAST_BUFFER_OUTPUT TRUE' >> commands.tmp
    echo 'options VCAST_DISPLAY_UNINST_EXPR FALSE' >> commands.tmp
    echo 'options VCAST_DUMP_BUFFER TRUE' >> commands.tmp
    echo 'options VCAST_EXELWTE_WITH_STDIO TRUE' >> commands.tmp
    echo 'options VCAST_EXELWTE_WITH_STDOUT TRUE' >> commands.tmp
    echo 'options VCAST_FILE_INDEX TRUE' >> commands.tmp
    echo 'options VCAST_NO_STDIN TRUE' >> commands.tmp
    echo 'options VCAST_STDIO TRUE' >> commands.tmp
endif
# common config
echo 'options VCAST_ENABLE_FUNCTION_CALL_COVERAGE TRUE' >> commands.tmp
echo 'options C_COMPILER_OUTPUT_FLAG -o' >> commands.tmp
echo 'options C_DEBUG_CMD gdb' >> commands.tmp
echo 'options VCAST_ASSEMBLY_FILE_EXTENSIONS asm s' >> commands.tmp
echo 'options VCAST_COLLAPSE_STD_HEADERS COLLAPSE_SYSTEM_HEADERS' >> commands.tmp
echo 'options VCAST_COMMAND_LINE_DEBUGGER TRUE' >> commands.tmp
echo 'options VCAST_DISABLE_STD_WSTRING_DETECTION TRUE' >> commands.tmp
echo 'options VCAST_DISPLAY_FUNCTION_COVERAGE TRUE' >> commands.tmp
echo 'options VCAST_ELWIRONMENT_FILES ' >> commands.tmp
echo 'options VCAST_HAS_LONGLONG TRUE' >> commands.tmp
echo 'options WHITEBOX YES' >> commands.tmp
echo 'clear_default_source_dirs ' >> commands.tmp
# testable dirs
echo 'options TESTABLE_SOURCE_DIR ../../../../lwscicommon/inc/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ../../../../lwscicommon/src/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ../../../../inc/internal/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ../../../../inc/public/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ../../../../../unix/rmapi_tegra/lwrminclude/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ../../../../../unix/rmapi_tegra/mirror/mirror/tegra_top/core/include/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ../../../../../unix/rmapi_tegra/mirror/mirror/tegra_top/core-private/include/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ../../../../../unix/rmapi_tegra/mirror/mirror/tegra_top_hidden/core/include/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ./include/' >> commands.tmp
echo 'options TESTABLE_SOURCE_DIR ../../../../../../sdk/lwpu/inc/' >> commands.tmp


# execute commands
echo "environment build $UNIT.elw" >> commands.tmp
echo "-e$UNIT tools script run $UNIT.tst" >> commands.tmp
echo "-e$UNIT execute batch" >> commands.tmp
echo "-e$UNIT tools import_coverage $UNIT.cvr" >> commands.tmp
echo "-e$UNIT reports custom full ${UNIT}_full_report.html" >> commands.tmp
$VECTORCAST_DIR/clicast -lC tools execute commands.tmp false
# Keep blank line, otherwise clicast won't be execited
