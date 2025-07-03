-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCISYNC_IPC_TABLE
-- Unit(s) Under Test: lwscisync_ipc_table
--
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:REMOVED_CL_PREFIX
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Subprogram: LwSciSyncCoreCopyIpcTable

-- Test Case: TC_001.LwSciSyncCoreCopyIpcTable.NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreCopyIpcTable.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreCopyIpcTable.NormalOperation}
 *
 * @verifyFunction{This test case checks all entries of input LwSciSyncCoreIpcTable copied to new LwSciSyncCoreIpcTable Sucessfully}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory
 * ipcTable[0].ipcPerm[0].ipcRoute[0] set to 55
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].needCpuAccess set to true and ipcTable[0].ipcPerm[0].requiredPerm set to LwSciSyncAccessPerm_WaitOnly}
 *
 * @testbehavior{newIpcTable[0].ipcRouteEntries set to 1
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The contents of ipcTable[0].ipcRoute copied to newIpcTable[0].ipcRoute
 * newIpcTable[0].ipcPermEntries set to 1
 * LwSciCommonCalloc() receives correct arguments(second event)
 * newIpcTable[0].ipcPerm[0].ipcRouteEntries to 1 
 * newIpcTable[0].ipcPerm[0].needCpuAccess to true and newIpcTable[0].ipcPerm[0].requiredPerm to LwSciSyncAccessPerm_WaitOnly
 * LwSciCommonCalloc() receives correct arguments(third event)
 * LwSciCommonMemcpyS() receives correct arguments
 * The contents of ipcTable[0].ipcPerm[0].ipcRoute[0] copied to newIpcTable[0].ipcPerm[0].ipcRoute[0] by memcpy() function
 * returned LwSciError_Success}
 *
 * @testcase{18852207}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute[0]:25
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].ipcRoute[0]:55
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRoute[0]:25
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm[0].ipcRoute[0]:55
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm[0].needCpuAccess:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(3)1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
else if(cnt==2){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrIpcPerm) ) }}
}
else if(cnt==3){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciIpcEndpoint) ) }}
}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcRoute ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcPerm[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable>>[0].ipcRoute ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreCopyIpcTable.ipcPerm[0].ipcRoute_LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreCopyIpcTable.ipcPerm[0].ipcRoute_LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreCopyIpcTable.ipcPerm[0].ipcRoute_LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test case checks the error path when LwSciCommonCalloc() fails to allocate memory for ipcPerm[0].ipcRoute of newIpcTable}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to validmemory
 * ipcTable[0].ipcPerm[0].needCpuAccess set to true and ipcTable[0].ipcPerm[0].requiredPerm set to LwSciSyncAccessPerm_WaitOnly}
 *
 * @testbehavior{newIpcTable[0].ipcRouteEntries set to 1
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The contents of ipcTable[0].ipcRoute copied to newIpcTable[0].ipcRoute by memcpy() function.
 * newIpcTable[0].ipcPermEntries set to 1
 * LwSciCommonCalloc() receives correct arguments(second event)
 * newIpcTable[0].ipcPerm[0].ipcRouteEntries to 1 
 * newIpcTable[0].ipcPerm[0].needCpuAccess to true and newIpcTable[0].ipcPerm[0].requiredPerm to LwSciSyncAccessPerm_WaitOnly
 * LwSciCommonCalloc() receives correct arguments(third event)
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852210}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute[0]:25
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(3)1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if (cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = NULL;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
else if(cnt==2){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrIpcPerm) ) }}
}
else if(cnt==3){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciIpcEndpoint) ) }}
}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1){
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcPerm[0].ipcRoute ) }}
}
else if(cnt==2){
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcPerm ) }}
}
else if(cnt==3)
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcRoute ) }}
}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreCopyIpcTable.ipcPerm[].ipcRouteEntries_ZERO
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreCopyIpcTable.ipcPerm[].ipcRouteEntries_ZERO
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreCopyIpcTable.ipcPerm[].ipcRouteEntries_ZERO}
 *
 * @verifyFunction{This test case checks the error path when ipcPerm[0].ipcRouteEntries is ZERO}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 0
 * ipcTable[0].ipcPerm[0].needCpuAccess set to true and ipcTable[0].ipcPerm[0].requiredPerm set to LwSciSyncAccessPerm_WaitOnly}
 *
 * @testbehavior{newIpcTable[0].ipcRouteEntries set to 1
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The contents of ipcTable[0].ipcRoute copied to newIpcTable[0].ipcRoute by memcpy() function.
 * newIpcTable[0].ipcPermEntries set to 1
 * LwSciCommonCalloc() receives correct arguments(second event)
 * newIpcTable[0].ipcPerm[0].ipcRouteEntries to 0
 * newIpcTable[0].ipcPerm[0].needCpuAccess to true and newIpcTable[0].ipcPerm[0].requiredPerm to LwSciSyncAccessPerm_WaitOnly
 * returned LwSciError_Success}
 *
 * @testcase{18852213}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute[0]:25
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRoute[0]:25
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm[0].needCpuAccess:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if (cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
else if(cnt==2){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrIpcPerm) ) }}
}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreCopyIpcTable.ipcPermEntries_LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreCopyIpcTable.ipcPermEntries_LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreCopyIpcTable.ipcPermEntries_LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test case checks the error path when LwSciCommonCalloc() fails to allocate memory for ipcPerm of newIpcTable}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{newIpcTable[0].ipcRouteEntries set to 1
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The contents of ipcTable[0].ipcRoute copied to newIpcTable[0].ipcRoute by memcpy() function.
 * newIpcTable[0].ipcPermEntries set to 1
 * LwSciCommonCalloc() receives correct arguments(second event)
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852216}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute[0]:25
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if (cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
else if(cnt==2)

<<uut_prototype_stubs.LwSciCommonCalloc.return>> = NULL;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
else if(cnt==2){
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrIpcPerm) ) }}
}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreCopyIpcTable.ipcPermEntries_ZERO_Success
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreCopyIpcTable.ipcPermEntries_ZERO_Success
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreCopyIpcTable.ipcPermEntries_ZERO_Success}
 *
 * @verifyFunction{This test case checks the error path when input ipcPermEntries is ZERO}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcPermEntries set to 0}
 *
 * @testbehavior{newIpcTable[0].ipcRouteEntries set to 1
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The contents of ipcTable[0].ipcRoute copied to newIpcTable[0].ipcRoute by memcpy() function.
 * newIpcTable[0].ipcPermEntries set to 0
 * returned LwSciError_Success}
 *
 * @testcase{18852219}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute[0]:5
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRoute[0]:5
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPermEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciIpcEndpoint) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(size_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreCopyIpcTable.ipcRouteEntries_LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreCopyIpcTable.ipcRouteEntries_LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreCopyIpcTable.ipcRouteEntries_LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test case checks the error path when LwSciCommonCalloc() fails to allocate memory for ipcRoute  of  newIpcTable.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcRoute set to valid memory}
 *
 * @testbehavior{newIpcTable[0].ipcRouteEntries set to 1
 * LwSciCommonCalloc() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852222}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPermEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciIpcEndpoint) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreCopyIpcTable.ipcRouteEntries_NULL_Success
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreCopyIpcTable.ipcRouteEntries_NULL_Success
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreCopyIpcTable.ipcRouteEntries_NULL_Success}
 *
 * @verifyFunction{This test case checks the error path when input ipcRouteEntries not found.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 0}
 *
 * @testbehavior{newIpcTable[0].ipcRouteEntries set to 0
 * returned LwSciError_Success}
 *
 * @testcase{18852225}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable[0].ipcPermEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.return:LwSciError_Success
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciSyncCoreCopyIpcTable.newIpcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreCopyIpcTable.newIpcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreCopyIpcTable.newIpcTable_NULL}
 *
 * @verifyFunction{This testcase checks panic condition when newIpcTable is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to valid memory
 * newIpcTable set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18852228}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciSyncCoreCopyIpcTable.ipcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreCopyIpcTable
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCoreCopyIpcTable.ipcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCoreCopyIpcTable.ipcTable_NULL}
 *
 * @verifyFunction{This testcase checks panic condition when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to NULL
 * newIpcTable set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18852231}
 *
 * @verify{18844554}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.ipcTable:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreCopyIpcTable.newIpcTable:<<malloc 1>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreCopyIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreExportIpcTable

-- Test Case: TC_001.LwSciSyncCoreExportIpcTable.NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreExportIpcTable.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreExportIpcTable.NormalOperation}
 *
 * @verifyFunction{This test case checks ipc table export to a descriptor successfully. }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success(IpcPermEntry)
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_IpcEndpoints)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcPerm)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory 
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(second event)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcPerm)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments 
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * LwSciCommonFree() receives correct arguments(third event)
 * returned LwSciError_Success}
 *
 * @testcase{18852234}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT3:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize[0]:6
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:(2)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25,33
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0,1,0,3,2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(6)8,9
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER1>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( VECTORCAST_BUFFER1 ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( VECTORCAST_BUFFER1) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute[0] ) }}
else if(cnt==7)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPermEntries ) }}
else if(cnt==8)
{{ (<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0] ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr.txbufPtr[0]
{{ <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr>>[0] == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries_ZERO_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries_ZERO_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries_ZERO_NormalOperation}
 *
 * @verifyFunction{This test case checks ipc table export to a descriptor successfully when ipcTable[0].ipcPerm[0].ipcRouteEntries set to ZERO}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success(IpcPermEntry)
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_IpcEndpoints)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcPerm)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory 
 * ipcTable[0].ipcPermEntries set to 2
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 0
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(second event)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcPerm)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments 
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * LwSciCommonFree() receives correct arguments(third event)
 * returned LwSciError_Success}
 *
 * @testcase{18852237}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT3:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize[0]:6
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:(2)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:281,33
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0,1,0,3,2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(2)8,264,(3)8,9
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER1>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( VECTORCAST_BUFFER1 ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( VECTORCAST_BUFFER1) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr.txbufPtr[0]
{{ <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr>>[0] == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreExportIpcTable.Key_IpcPermEntry_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreExportIpcTable.Key_IpcPermEntry_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreExportIpcTable.Key_IpcPermEntry_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where Key is IpcPermEntry and LwSciCommonTransportAppendKeyValuePair() returns error
 * when no space is left in transport buffer to append the key-value pair}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success(IpcPermEntry)
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcPerm)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory 
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 0}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcPerm)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * LwSciCommonFree() receives correct arguments(third event)
 * returned LwSciError_NoSpace}
 *
 * @testcase{18852240}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4,2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25,17
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0,3,2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(4)8,9
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
else if(cnt==6)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>>) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0]  ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreExportIpcTable.Key_NumIpcPerm_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreExportIpcTable.Key_NumIpcPerm_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreExportIpcTable.Key_NumIpcPerm_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where Key is NumIpcPerm and LwSciCommonTransportAppendKeyValuePair() returns error
 * when no space is left in transport buffer to append the key-value pair}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success(IpcPermEntry)
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_NumIpcPerm)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory 
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 0}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments (IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcPerm)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * LwSciCommonFree() receives correct arguments(third event)
 * returned LwSciError_NoSpace}
 *
 * @testcase{18852243}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4,2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25,17
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0,3
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(4)8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
else if(cnt==5)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>>) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries_is_ZERO_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries_is_ZERO_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries_is_ZERO_NormalOperation}
 *
 * @verifyFunction{This test case checks error path when ipcTable[0].ipcRouteEntries is ZERO}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success(IpcPermEntry)
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcEndpoints)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to null
 * ipcTable[0].ipcRouteEntries set to 0}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments (IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments (IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852246}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4,2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25,17
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0,3,2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(4)8,9
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER1>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( VECTORCAST_BUFFER1 ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( VECTORCAST_BUFFER1) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == (<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries_is_ZERO_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries_is_ZERO_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries_is_ZERO_NormalOperation}
 *
 * @verifyFunction{This test case checks error path when ipcTable[0].ipcPermEntries is ZERO}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportAllocTxBufferForKeys()returns to LwSciError_Success
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)
 * LwSciCommonTransportPrepareBufferForTx() returns to LwSciError_Success}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory
 * ipcTable[0].ipcPermEntries set to 0
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * returned LwSciError_Success}
 *
 * @testcase{18852249}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:16
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:1,0
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreExportIpcTable.Key_IpcEndpoints_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreExportIpcTable.Key_IpcEndpoints_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreExportIpcTable.Key_IpcEndpoints_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where Key is IpcEndpoints and LwSciCommonTransportAppendKeyValuePair() returns error
 * when no space is left in transport buffer to append the key-value pair}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcEndpoints)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory 
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to null
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * LwSciCommonFree() receives correct arguments(third event)
 * returned LwSciError_NoSpace}
 *
 * @testcase{18852252}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:(2)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25,33
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0,1,0
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(5)8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
else if(cnt==6)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>>) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCoreExportIpcTable.Key_NumIpcEndpoint_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreExportIpcTable.Key_NumIpcEndpoint_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreExportIpcTable.Key_NumIpcEndpoint_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where Key is NumIpcEndpoint and LwSciCommonTransportAppendKeyValuePair() returns error
 * when no space is left in transport buffer to append the key-value pair}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success(IpcPermEntry)
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory 
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * LwSciCommonFree() receives correct arguments(third event)
 * returned LwSciError_NoSpace}
 *
 * @testcase{18852255}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:(2)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25,33
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0,1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(4)8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
else if(cnt==5)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>>) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncCoreExportIpcTable.LwSciCommonTransportAllocTxBufferForKeys_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCoreExportIpcTable.LwSciCommonTransportAllocTxBufferForKeys_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCoreExportIpcTable.LwSciCommonTransportAllocTxBufferForKeys_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where LwSciCommonTransportAllocTxBufferForKeys()returns error when memory allocation failed}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success(IpcPermEntry)
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to NULL
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_InsufficientMemory}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() receives correct arguments (IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852258}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:(2)4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25,33
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(3)8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf.txbuf[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER>> );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_Success );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return>> = ( LwSciError_InsufficientMemory );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>>) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries_ZERO
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_010.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries_ZERO
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries_ZERO}
 *
 * @verifyFunction{This test case checks success case when ipcTable[0].ipcPermEntries and ipcTable[0].ipcRouteEntries are set to ZERO}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory  
 * ipcTable[0].ipcPermEntries set to 0
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 0}
 *
 * @testbehavior{LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * returned LwSciError_Success}
 *
 * @testcase{18852261}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Success
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries_set_MAX_OverflowCase
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_011.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries_set_MAX_OverflowCase
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries_set_MAX_OverflowCase}
 *
 * @verifyFunction{This test case checks overflow case when ipcTable[0].ipcRouteEntries set to MAX}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportPrepareBufferForTx() return to LwSciError_Success(IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory  
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to null
 * ipcTable[0].ipcRouteEntries set to MAX}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * LwSciCommonFree() receives correct arguments(third event)
 * returned LwSciError_Overflow}
 *
 * @testcase{18852264}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:<<MAX>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(3)8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0]) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> ) }}
else if(cnt==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>>) }}




TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_IpcEndpoints_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_012.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_IpcEndpoints_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_IpcEndpoints_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where Key is IpcEndpoints and LwSciCommonTransportAppendKeyValuePair() returns error((IpcPermEntry)
 * when no space is left in transport buffer to append the key-value pair}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory  
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to null
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_IpcEndpoints)(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * returned LwSciError_NoSpace}
 *
 * @testcase{18852267}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:9,6
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1,0
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,(3)8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9999999999999999);
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9595959595959595 );


TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)


{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9595959595959595 ) }}
else if(cnt==2)

{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9999999999999999 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
else if(cnt==3)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==4)
{{ ((int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(cnt==5)
{{ (*(int*)<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>>) == ( 1 ) }}
else if(cnt==6)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( <<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_NumIpcEndpoint_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_013.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_NumIpcEndpoint_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_NumIpcEndpoint_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where Key is NumIpcEndpoint and LwSciCommonTransportAppendKeyValuePair() returns error(IpcPermEntry)
 * when no space is left in transport buffer to append the key-value pair}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory 
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to null
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NumIpcEndpoint)(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * returned LwSciError_NoSpace}
 *
 * @testcase{18852270}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5,1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9999999999999999);
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9595959595959595 );


TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)


{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9595959595959595 ) }}
else if(cnt==2)

{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9999999999999999 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_RequiredPerm_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_014.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_RequiredPerm_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_RequiredPerm_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where Key is RequiredPerm and LwSciCommonTransportAppendKeyValuePair() returns error(IpcPermEntry)
 * when no space is left in transport buffer to append the key-value pair}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_Success(LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace(LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to null
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments (LwSciSyncCoreIpcTableKey_RequiredPerm)(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * returned LwSciError_NoSpace}
 *
 * @testcase{18852273}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4,5
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9999999999999999);
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9595959595959595 );


TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_Success );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return>> = ( LwSciError_NoSpace );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)


{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9595959595959595 ) }}
else if(cnt==2)

{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9999999999999999 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].requiredPerm ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_NeedCpuAccess_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_015.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_NeedCpuAccess_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncCoreExportIpcTable.IpcPermEntry_Key_NeedCpuAccess_LwSciCommonTransportAppendKeyValuePair_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where Key is NeedCpuAccess and LwSciCommonTransportAppendKeyValuePair() returns error(IpcPermEntry)
 * when no space is left in transport buffer to append the key-value pair}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_Success(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() return to LwSciError_NoSpace (LwSciSyncCoreIpcTableKey_NeedCpuAccess)(IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to null
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportAppendKeyValuePair() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * returned LwSciError_NoSpace}
 *
 * @testcase{18852276}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_NoSpace
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9999999999999999);
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9595959595959595 );


TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)


{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9595959595959595 ) }}
else if(cnt==2)

{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9999999999999999 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable>>[0].ipcPerm[0].needCpuAccess ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciSyncCoreExportIpcTable.IpcPermEntry_LwSciCommonTransportAllocTxBufferForKeys_InsufficientMemory
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_016.LwSciSyncCoreExportIpcTable.IpcPermEntry_LwSciCommonTransportAllocTxBufferForKeys_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_016.LwSciSyncCoreExportIpcTable.IpcPermEntry_LwSciCommonTransportAllocTxBufferForKeys_InsufficientMemory}
 *
 * @verifyFunction{This test case checks where LwSciCommonTransportAllocTxBufferForKeys() returns error(IpcPermEntry) when memory allocation failed}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory
 * 'txbuf' parameter of LwSciCommonTransportAllocTxBufferForKeys() set to valid memory(IpcPermEntry)
 * LwSciCommonTransportAllocTxBufferForKeys() return to LwSciError_InsufficientMemory (IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to null
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportAllocTxBufferForKeys() receives correct arguments(IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments(IpcPermEntry)
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852279}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:25
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9999999999999999);
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9595959595959595 );



TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)


{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9595959595959595 ) }}
else if(cnt==2)

{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9999999999999999 ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_017.LwSciSyncCoreExportIpcTable.ipcPerm.ipcRouteEntries_MAX_OverflowCase
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_017.LwSciSyncCoreExportIpcTable.ipcPerm.ipcRouteEntries_MAX_OverflowCase
TEST.NOTES:
/**
 * @testname{TC_017.LwSciSyncCoreExportIpcTable.ipcPerm.ipcRouteEntries_MAX_OverflowCase}
 *
 * @verifyFunction{This test case checks overdlow case when ipcPerm[0].ipcRouteEntries set to MAX}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory
 * LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to MAX
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to NULL
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments (second event)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * returned LwSciError_Overflow}
 *
 * @testcase{18852282}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:<<MAX>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9999999999999999);
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0x9595959595959595 );



TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9595959595959595 ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_018.LwSciSyncCoreExportIpcTable.LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_018.LwSciSyncCoreExportIpcTable.LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_018.LwSciSyncCoreExportIpcTable.LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test case checks error path when LwSciCommonCalloc() fails}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to valid memory
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments(second event)
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonFree() receives correct arguments(second event)
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852285}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (0xFFFFFFFFFFFFFFFF );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( 0xFFFFFFFFFFFFFFFF ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_019.LwSciSyncCoreExportIpcTable.txbufSize_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_019.LwSciSyncCoreExportIpcTable.txbufSize_NULL
TEST.NOTES:
/**
 * @testname{TC_019.LwSciSyncCoreExportIpcTable.txbufSize_NULL}
 *
 * @verifyFunction{This testcase checks panic condition when txbufSize is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to valid memory
 * txbufSize set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18852288}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufSize:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_020.LwSciSyncCoreExportIpcTable.txbufPtr_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_020.LwSciSyncCoreExportIpcTable.txbufPtr_NULL
TEST.NOTES:
/**
 * @testname{TC_020.LwSciSyncCoreExportIpcTable.txbufPtr_NULL}
 *
 * @verifyFunction{This testcase checks panic condition when txbufPtr is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to valid memory
 * txbufPtr set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18852291}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.txbufPtr:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_021.LwSciSyncCoreExportIpcTable.ipcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreExportIpcTable
TEST.NEW
TEST.NAME:TC_021.LwSciSyncCoreExportIpcTable.ipcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_021.LwSciSyncCoreExportIpcTable.ipcTable_NULL}
 *
 * @verifyFunction{This testcase checks panic condition when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18852294}
 *
 * @verify{18844572}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreExportIpcTable.ipcTable:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreExportIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreImportIpcTable

-- Test Case: TC_001.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation}
 *
 * @verifyFunction{This test case checks imported LwSciSyncCoreIpcTable from a descriptor successfully}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcEndpoints
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcEndpoints)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 15
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments(second event)
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcEndpoints)
 * LwSciCommonMemcpyS() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852297}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:15
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:15,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:(4)4,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:(4)4,8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcEndpoints );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable.ipcTable[0].ipcPerm.ipcPerm[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute != ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_LBV
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_LBV
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_LBV}
 *
 * @verifyFunction{This test case checks error path when size set to lower boundary value}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcEndpoints
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcEndpoints)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 1
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments(second event)
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcEndpoints)
 * LwSciCommonMemcpyS() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852300}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:1,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:(4)4,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:(4)4,8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcEndpoints );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable.ipcTable[0].ipcPerm.ipcPerm[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute != ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_LW
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_LW
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_LW}
 *
 * @verifyFunction{This test case checks error path when size set to nominal value}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcEndpoints
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcEndpoints)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 55
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 8
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments(second event)
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcEndpoints)
 * LwSciCommonMemcpyS() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852303}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:55
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:55,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:(4)4,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:(4)4,8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcEndpoints );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable.ipcTable[0].ipcPerm.ipcPerm[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute != ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_UBV
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_UBV
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_IpcEndpoints_NormalOperation_UBV}
 *
 * @verifyFunction{This test case checks error path when size set to upper boundary value}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcEndpoints
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcEndpoints)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to MAX
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments(second event)
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcEndpoints)
 * LwSciCommonMemcpyS() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852306}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:<<MAX>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:<<MAX>>,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:(4)4,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:(4)4,8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcEndpoints );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable.ipcTable[0].ipcPerm.ipcPerm[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute != ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreImportIpcTable.ipcPermEntry_IpcEndpoints_size_different_than_promised
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreImportIpcTable.ipcPermEntry_IpcEndpoints_size_different_than_promised
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreImportIpcTable.ipcPermEntry_IpcEndpoints_size_different_than_promised}
 *
 * @verifyFunction{This test case checks error path when IpcEndpoints size different than promised in ipcPerm}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 9
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcEndpoints
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcEndpoints)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcEndpoints)
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory address returned by LwSciCommonCalloc()
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPermEntries set to 1
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852309}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:(3)8,9
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcEndpoints );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_NumIpcEndpoint_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_NumIpcEndpoint_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreImportIpcTable.ipcPermEntry_key_NumIpcEndpoint_NormalOperation}
 *
 * @verifyFunction{This test case checks imported LwSciSyncCoreIpcTable from a descriptor successfully }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 8
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() returns to valid memory(second event)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments
 * ipcTable[0].ipcPerm[0].ipcRoute set to valid memory address returned by LwSciCommonCalloc()
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPermEntries set to 1
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852312}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:(3)8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreImportIpcTable.IpcPermEntry_key_NumIpcEndpoint_LwSciCommonCalloc()_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreImportIpcTable.IpcPermEntry_key_NumIpcEndpoint_LwSciCommonCalloc()_fails
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreImportIpcTable.IpcPermEntry_key_NumIpcEndpoint_LwSciCommonCalloc()_fails}
 *
 * @verifyFunction{This test case checks error path when LwSciCommonCalloc() fails and NumIpcEndpoint key in IpcPerm}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success 
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 8
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852315}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32,8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm[0].ipcRoute == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCoreImportIpcTable.Ilwalid_size_of_NumIpcEndpoint_key_in_IpcPerm
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreImportIpcTable.Ilwalid_size_of_NumIpcEndpoint_key_in_IpcPerm
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreImportIpcTable.Ilwalid_size_of_NumIpcEndpoint_key_in_IpcPerm}
 *
 * @verifyFunction{This test case checks error path when Invalid size for NumIpcEndpoint key in IpcPerm found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 9
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852318}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:(2)8,9
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_RequiredPerm_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_RequiredPerm_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_RequiredPerm_NormalOperation}
 *
 * @verifyFunction{This test case checks imported LwSciSyncCoreIpcTable from a descriptor successfully }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to vSciSyncCoreIpcTableKey_RequiredPerm
 *       length set to 8
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_RequiredPerm)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_RequiredPerm)
 * ipcTable[0].ipcPerm set to valid memory address returned by LwSciCommonCalloc()
 * ipcTable[0].ipcPermEntries set to 1
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852321}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_RequiredPerm );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncCoreImportIpcTable.Ilwalid_size_for_RequiredPerm_key_in_IpcPerm
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_010.LwSciSyncCoreImportIpcTable.Ilwalid_size_for_RequiredPerm_key_in_IpcPerm
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncCoreImportIpcTable.Ilwalid_size_for_RequiredPerm_key_in_IpcPerm}
 *
 * @verifyFunction{This test case checks error path when Invalid size for RequiredPerm key in IpcPerm found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to vSciSyncCoreIpcTableKey_RequiredPerm
 *       length set to 9
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_RequiredPerm)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_RequiredPerm)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852324}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:(2)8,9
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_RequiredPerm );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_NeedCpuAccess_Tag_is_not_allowed_here_in_IpcTable
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_011.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_NeedCpuAccess_Tag_is_not_allowed_here_in_IpcTable
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_NeedCpuAccess_Tag_is_not_allowed_here_in_IpcTable}
 *
 * @verifyFunction{This test case checks error path when Tag NeedCpuAccess is not allowed here in IpcTable}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NeedCpuAccess
 *       length set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NeedCpuAccess)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NeedCpuAccess)
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * ipcTable[0].ipcPerm[0].needCpuAccess set to true
 * ipcTable[0].ipcPermEntries set to 1
 * LwSciCommonFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852327}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NeedCpuAccess );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>>[0] = ( 8 );

else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>>[0] = ( 1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==4)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_NeedCpuAccess_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_012.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_NeedCpuAccess_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncCoreImportIpcTable.IpcPermEntry_Key_NeedCpuAccess_NormalOperation}
 *
 * @verifyFunction{This test case checks imported LwSciSyncCoreIpcTable from a descriptor successfully }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NeedCpuAccess
 *       length set to 1 
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NeedCpuAccess)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NeedCpuAccess)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852330}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NeedCpuAccess );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>>[0] = ( 8 );

else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>>[0] = ( 1 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncCoreImportIpcTable.Ilwalid_size_for_NeedCpuAccess_key_in_IpcPerm
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_013.LwSciSyncCoreImportIpcTable.Ilwalid_size_for_NeedCpuAccess_key_in_IpcPerm
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncCoreImportIpcTable.Ilwalid_size_for_NeedCpuAccess_key_in_IpcPerm}
 *
 * @verifyFunction{This test case checks error path when Invalid size for NeedCpuAccess key in IpcPerm found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry) 
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NeedCpuAccess 
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NeedCpuAccess)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry) 
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NeedCpuAccess)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852333}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NeedCpuAccess );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_importReconciled_true_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_014.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_importReconciled_true_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_importReconciled_true_NormalOperation}
 *
 * @verifyFunction{This test case checks imported LwSciSyncCoreIpcTable from a descriptor successfully }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm) 
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success (second event)
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm) 
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments((Key_IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852336}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_LwSciCommonTransportGetNextKeyValuePair_Overflow
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_015.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_LwSciCommonTransportGetNextKeyValuePair_Overflow
TEST.NOTES:
/**
 * @testname{TC_015.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_LwSciCommonTransportGetNextKeyValuePair_Overflow}
 *
 * @verifyFunction{This test case checks where Key is IpcPermEntry LwSciCommonTransportGetNextKeyValuePair() returns error when internal arithmetic overflow oclwrs}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Overflow(Key_IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcPermEntry)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Overflow}
 *
 * @testcase{18852339}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return>> = ( LwSciError_Success );
else if(cnt==3)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return>> = ( LwSciError_Overflow );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_LwSciCommonTransportGetRxBufferAndParams_InsufficientResource
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_016.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_LwSciCommonTransportGetRxBufferAndParams_InsufficientResource
TEST.NOTES:
/**
 * @testname{TC_016.LwSciSyncCoreImportIpcTable.key_IpcPermEntry_LwSciCommonTransportGetRxBufferAndParams_InsufficientResource}
 *
 * @verifyFunction{This test case checks where key is IpcPermEntry LwSciCommonTransportGetRxBufferAndParams() returns error when memory allocation failed
 * when memory allocation failed}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm) 
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcPermEntry
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_IpcPermEntry)
 * LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_InsufficientResource(key IpcPermEntry)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true
 * ipcTable[0].ipcRoute set to valid memory
 * ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm set to valid memory}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(second event)
 * LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments(second event)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_InsufficientResource}
 *
 * @testcase{18852342}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return>> = ( LwSciError_Success );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return>> = ( LwSciError_InsufficientMemory);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcPerm );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcPermEntry );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>>[0] ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_017.LwSciSyncCoreImportIpcTable.Key_NumIpcPerm_IpcTable_descriptor_is_missing_tag
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_017.LwSciSyncCoreImportIpcTable.Key_NumIpcPerm_IpcTable_descriptor_is_missing_tag
TEST.NOTES:
/**
 * @testname{TC_017.LwSciSyncCoreImportIpcTable.Key_NumIpcPerm_IpcTable_descriptor_is_missing_tag}
 *
 * @verifyFunction{This test case checks error path when IpcTable descriptor has missing tag}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm
 *       length set to 8
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852345}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcPerm
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_018.LwSciSyncCoreImportIpcTable.Key_NumIpcPerm_LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_018.LwSciSyncCoreImportIpcTable.Key_NumIpcPerm_LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_018.LwSciSyncCoreImportIpcTable.Key_NumIpcPerm_LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test case checks error path when LwSciCommonCalloc() fails}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm
 *       length set to 8
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)
 * LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852348}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcPerm
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:32
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcPerm == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_019.LwSciSyncCoreImportIpcTable.ipcperm_entries_zero
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_019.LwSciSyncCoreImportIpcTable.ipcperm_entries_zero
TEST.NOTES:
/**
 * @testname{TC_019.LwSciSyncCoreImportIpcTable.ipcperm_entries_zero}
 *
 * @verifyFunction{This test case checks error path when ipc perm entries is zero}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm
 *       length set to 8
 *       value set to 0
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852351}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcPerm
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcPermEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_020.LwSciSyncCoreImportIpcTable.Ilwalid_length_of_NumIpcPerm
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_020.LwSciSyncCoreImportIpcTable.Ilwalid_length_of_NumIpcPerm
TEST.NOTES:
/**
 * @testname{TC_020.LwSciSyncCoreImportIpcTable.Ilwalid_length_of_NumIpcPerm}
 *
 * @verifyFunction{This test case checks error path when Invalid length of Key 'NumIpcPerm' found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcPerm
 *       length set to 4
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcPerm)}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcPerm)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852354}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcPerm
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:4
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_021.LwSciSyncCoreImportIpcTable.key_IpcEndpoints_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_021.LwSciSyncCoreImportIpcTable.key_IpcEndpoints_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_021.LwSciSyncCoreImportIpcTable.key_IpcEndpoints_NormalOperation}
 *
 * @verifyFunction{This test case checks imported LwSciSyncCoreIpcTable from a descriptor successfully }
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows: (Key_NumIpcEndpoint)
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success
 * LwSciCommonCalloc() returns to valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows: (Key_IpcEndpoints )
 *       key set to LwSciSyncCoreIpcTableKey_IpcEndpoints 
 *       length set to 8
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success
 * }
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to false
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcEndpoints) 
 * LwSciCommonMemcpyS() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852357}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:(2)4,16
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:(2)4,8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcEndpoints );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &VECTORCAST_INT3 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcRoute != ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_022.LwSciSyncCoreImportIpcTable.Sizeof_ipcRoute_different_from_promised
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_022.LwSciSyncCoreImportIpcTable.Sizeof_ipcRoute_different_from_promised
TEST.NOTES:
/**
 * @testname{TC_022.LwSciSyncCoreImportIpcTable.Sizeof_ipcRoute_different_from_promised}
 *
 * @verifyFunction{This test case checks error path when Size of ipcRoute different from promised}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 8
 *       value set to 0
 *       rdFinish set to false
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success(Key_NumIpcEndpoint)
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_IpcEndpoints 
 *       length set to 8
 *       value set to 0
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success (Key_IpcEndpoints)
 * LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to false
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_NumIpcEndpoint)
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments(Key_IpcEndpoints)
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852360}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_NumIpcEndpoint );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>>[0] = ( LwSciSyncCoreIpcTableKey_IpcEndpoints );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( false );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>>[0] = ( true );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcRoute == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_023.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_IpcTable_descriptor_is_missing_tag
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_023.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_IpcTable_descriptor_is_missing_tag
TEST.NOTES:
/**
 * @testname{TC_023.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_IpcTable_descriptor_is_missing_tag}
 *
 * @verifyFunction{This test case checks error path when IpcTable descriptor has missing tag}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 8
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success
 * LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to false
 * ipcTable[0].ipcRouteEntries set to 1}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852363}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcEndpoint
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRoute
{{ <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable>>[0].ipcRoute == ( <<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_024.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_024.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_024.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test case checks error path when LwSciCommonCalloc() fails}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 8
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success
 * LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to false}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852366}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcEndpoint
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_025.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_value_set_toMAX
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_025.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_value_set_toMAX
TEST.NOTES:
/**
 * @testname{TC_025.LwSciSyncCoreImportIpcTable.Key_NumIpcEndpoint_value_set_toMAX}
 *
 * @verifyFunction{This test case checks overflow condition when value set to MAX}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 8
 *       value set to MAX
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to false}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Overflow}
 *
 * @testcase{18852369}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:<<MAX>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcEndpoint
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_026.LwSciSyncCoreImportIpcTable.Ilwalid_length_of_NumIpcEndpoints
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_026.LwSciSyncCoreImportIpcTable.Ilwalid_length_of_NumIpcEndpoints
TEST.NOTES:
/**
 * @testname{TC_026.LwSciSyncCoreImportIpcTable.Ilwalid_length_of_NumIpcEndpoints}
 *
 * @verifyFunction{This test case checks error path when Invalid length of Key 'NumIpcEndpoints' found}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 4
 *       value set to 1
 *       rdFinish set to true
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to false}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852372}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcEndpoint
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:4
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_027.LwSciSyncCoreImportIpcTable.Tag_is_not_allowed_here_in_IpcTable
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_027.LwSciSyncCoreImportIpcTable.Tag_is_not_allowed_here_in_IpcTable
TEST.NOTES:
/**
 * @testname{TC_027.LwSciSyncCoreImportIpcTable.Tag_is_not_allowed_here_in_IpcTable}
 *
 * @verifyFunction{This test case checks error path when Tag is not allowed here in IpcTable}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NumIpcEndpoint
 *       length set to 8
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8
 * importReconciled set to true}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852375}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NumIpcEndpoint
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_028.LwSciSyncCoreImportIpcTable.Unrecognized_tag_in_IpcTable
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_028.LwSciSyncCoreImportIpcTable.Unrecognized_tag_in_IpcTable
TEST.NOTES:
/**
 * @testname{TC_028.LwSciSyncCoreImportIpcTable.Unrecognized_tag_in_IpcTable}
 *
 * @verifyFunction{This test case checks error path when importReconciled is false and Unrecognized tag in IpcTable found}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NeedCpuAccess
 *       length set to 8
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to MAX
 * importReconciled set to False}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852378}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:<<MAX>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NeedCpuAccess
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:<<MAX>>
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_029.LwSciSyncCoreImportIpcTable.IpcTable_descriptor_is_missing_tag
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_029.LwSciSyncCoreImportIpcTable.IpcTable_descriptor_is_missing_tag
TEST.NOTES:
/**
 * @testname{TC_029.LwSciSyncCoreImportIpcTable.IpcTable_descriptor_is_missing_tag}
 *
 * @verifyFunction{This test case checks error path when IpcTable descriptor has missing tag and importReconciled is true}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NeedCpuAccess
 *       length set to 8
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Success}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 1
 * importReconciled set to True}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_BadParameter}
 *
 * @testcase{18852381}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NeedCpuAccess
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:4
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:4
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_030.LwSciSyncCoreImportIpcTable.LwSciCommonTransportGetNextKeyValuePair_Overflow
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_030.LwSciSyncCoreImportIpcTable.LwSciCommonTransportGetNextKeyValuePair_Overflow
TEST.NOTES:
/**
 * @testname{TC_030.LwSciSyncCoreImportIpcTable.LwSciCommonTransportGetNextKeyValuePair_Overflow}
 *
 * @verifyFunction{This test case checks where LwSciCommonTransportGetNextKeyValuePair() returns error when internal arithmetic overflow oclwrs}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_Success
 * 'rxbuf' parameters of  LwSciCommonTransportGetRxBufferAndParams() set to a valid memory
 * the following parameters of LwSciCommonTransportGetNextKeyValuePair() set as follows:
 *       key set to LwSciSyncCoreIpcTableKey_NeedCpuAccess
 *       length set to 8
 *       value set to 1
 *       rdFinish set to True
 * LwSciCommonTransportGetNextKeyValuePair() returns LwSciError_Overflow}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 1
 * importReconciled set to True}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() recieves correct arguments
 * LwSciCommonTransportGetNextKeyValuePair() receives correct arguments
 * LwSciCommonTransportBufferFree() receives correct arguments
 * returned LwSciError_Overflow}
 *
 * @testcase{18852384}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.importReconciled:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCoreIpcTableKey_NeedCpuAccess
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value[0]:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Overflow
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_031.LwSciSyncCoreImportIpcTable.LwSciCommonTransportGetRxBufferAndParams_InsufficientResource
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_031.LwSciSyncCoreImportIpcTable.LwSciCommonTransportGetRxBufferAndParams_InsufficientResource
TEST.NOTES:
/**
 * @testname{TC_031.LwSciSyncCoreImportIpcTable.LwSciCommonTransportGetRxBufferAndParams_InsufficientResource}
 *
 * @verifyFunction{This test case checks where LwSciCommonTransportGetRxBufferAndParams() returns error when memory allocation failed}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonTransportGetRxBufferAndParams() returns to LwSciError_InsufficientResource}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 8}
 *
 * @testbehavior{LwSciCommonTransportGetRxBufferAndParams() receives correct arguments
 * returned LwSciError_InsufficientResource}
 *
 * @testcase{18852387}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:8
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = (VECTORCAST_INT2 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc
<<lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc>> = ( &VECTORCAST_INT1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_032.LwSciSyncCoreImportIpcTable.size_ZERO_below_LBV
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_032.LwSciSyncCoreImportIpcTable.size_ZERO_below_LBV
TEST.NOTES:
/**
 * @testname{TC_032.LwSciSyncCoreImportIpcTable.size_ZERO_below_LBV}
 *
 * @verifyFunction{This testcase checks error path when size is zero(just below lower boundary value)}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to valid memory
 * size set to 0}
 *
 * @testbehavior{LwSciError_BadParameter returned}
 *
 * @testcase{18852390}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc:VECTORCAST_INT1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_Success
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.return:LwSciError_BadParameter
TEST.END

-- Test Case: TC_033.LwSciSyncCoreImportIpcTable.desc_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_033.LwSciSyncCoreImportIpcTable.desc_NULL
TEST.NOTES:
/**
 * @testname{TC_033.LwSciSyncCoreImportIpcTable.desc_NULL}
 *
 * @verifyFunction{This testcase checks LwSciSyncCoreImportIpcTable() panics when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to valid memory
 * desc set to NULL
 * size set to 1}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18852393}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_034.LwSciSyncCoreImportIpcTable.ipcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreImportIpcTable
TEST.NEW
TEST.NAME:TC_034.LwSciSyncCoreImportIpcTable.ipcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_034.LwSciSyncCoreImportIpcTable.ipcTable_NULL}
 *
 * @verifyFunction{This testcase checks LwSciSyncCoreImportIpcTable() panics when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to NULL
 * desc set to valid memory
 * size set to 1}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreImportIpcTable() panics}
 *
 * @testcase{18852396}
 *
 * @verify{18844569}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.ipcTable:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.desc:VECTORCAST_INT1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreImportIpcTable.size:1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreImportIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreIpcTableAddBranch

-- Test Case: TC_001.LwSciSyncCoreIpcTableAddBranch.NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreIpcTableAddBranch.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreIpcTableAddBranch.NormalOperation}
 *
 * @verifyFunction{This testcase checks new branch(ipcroute path) added to the ipc perm tree successfully.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{slot set to 0
 * ipctable.ipcPermEntries set to 1
 * ipcTable.ipcPerm[slot].ipcRoute set to NULL
 * ipcTableWithRoute[slot].ipcRoute[] set to a valid memory
 * ipcTableWithRoute[slot].ipcRouteEntries set to 1
 * needCpuAccess set to true
 * requiredperm set to "LwSciSyncAccessPerm_SignalOnly"}
 *
 * @testbehavior{ipcTable.ipcPerm[slot].ipcRouteEntries set to 1
 * ipcTable.ipcPerm[slot].ipcRoute[] to a valid memory
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The content of ipcTableWithRoute.ipcRoute copied to ipcTable.ipcPerm[slot].ipcRoute
 * ipcTable.ipcPerm[slot].needCpuAccess set to true
 * ipcTable.ipcPerm[slot].requiredperm set to "LwSciSyncAccessPerm_SignalOnly"
 * returned LwSciError_Success}
 *
 * @testcase{18852399}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute[0]:55
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRouteEntries:0x1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute[0]:55
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreIpcTableAddBranch.slot_aboveUBV
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreIpcTableAddBranch.slot_aboveUBV
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreIpcTableAddBranch.slot_aboveUBV}
 *
 * @verifyFunction{This testcase checks error path when slot set to just above upper boundary value with respect to ipcPermEntries.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{}
 *
 * @testinput{slot set to 10
 * ipctable.ipcPermEntries set to 10
 * ipctable.ipcTableWithRoute set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableAddBranch() panics}
 *
 * @testcase{18852405}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:10
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:10
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciSyncCoreIpcTableAddBranch.slot_LBV
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreIpcTableAddBranch.slot_LBV
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreIpcTableAddBranch.slot_LBV}
 *
 * @verifyFunction{This testcase checks error path when slot set to lower boundary value with respect to ipcPermEntries.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{slot set to 0
 * ipctable.ipcPermEntries set to 10
 * ipcTable.ipcPerm[slot].ipcRoute set to NULL
 * ipcTableWithRoute[slot].ipcRoute[] set to a valid memory
 * ipcTableWithRoute[slot].ipcRouteEntries set to 1
 * needCpuAccess set to true
 * requiredperm set to "LwSciSyncAccessPerm_SignalOnly"}
 *
 * @testbehavior{ipcTable.ipcPerm[slot].ipcRouteEntries set to 1
 * ipcTable.ipcPerm[slot].ipcRoute[] to a valid memory
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The content of ipcTableWithRoute.ipcRoute copied to ipcTable.ipcPerm[slot].ipcRoute
 * ipcTable.ipcPerm[slot].needCpuAccess set to true
 * ipcTable.ipcPerm[slot].requiredperm set to "LwSciSyncAccessPerm_SignalOnly"
 * returned LwSciError_Success}
 *
 * @testcase{18852408}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm:<<malloc 10>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:10
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute[0]:55
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRouteEntries:0x1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute[0]:55
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreIpcTableAddBranch.slot_LW_valid
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreIpcTableAddBranch.slot_LW_valid
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreIpcTableAddBranch.slot_LW_valid}
 *
 * @verifyFunction{This testcase checks error path when slot set to valid nominal value with respect to ipcPermEntries.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{slot set to 4
 * ipctable.ipcPermEntries set to 10
 * ipcTable.ipcPerm[slot].ipcRoute set to NULL
 * ipcTableWithRoute[slot].ipcRoute set to a valid memory
 * ipcTableWithRoute[slot].ipcRouteEntries set to 1
 * needCpuAccess set to true
 * requiredperm set to "LwSciSyncAccessPerm_SignalOnly"}
 *
 * @testbehavior{ipcTable.ipcPerm[slot].ipcRouteEntries set to 1
 * ipcTable.ipcPerm[slot].ipcRoute[] to a valid memory
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The content of ipcTableWithRoute.ipcRoute copied to ipcTable.ipcPerm[slot].ipcRoute
 * ipcTable.ipcPerm[slot].needCpuAccess set to true
 * ipcTable.ipcPerm[slot].requiredperm set to "LwSciSyncAccessPerm_SignalOnly"
 * returned LwSciError_Success}
 *
 * @testcase{18852411}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm:<<malloc 10>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[4].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:10
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:4
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute[0]:55
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRouteEntries:0x1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[4].ipcRoute[0]:55
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[4].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[4].needCpuAccess:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[4].requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreIpcTableAddBranch.slot_LW_Ilwalid
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreIpcTableAddBranch.slot_LW_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreIpcTableAddBranch.slot_LW_Ilwalid}
 *
 * @verifyFunction{This testcase checks error path when slot set to invalid nominal value with respect to ipcPermEntries.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{}
 *
 * @testinput{slot set to 25
 * ipctable.ipcPermEntries set to 10
 * ipctable.ipcTableWithRoute set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableAddBranch() panics}
 *
 * @testcase{18852414}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:10
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:25
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciSyncCoreIpcTableAddBranch.slot_UBV
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreIpcTableAddBranch.slot_UBV
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreIpcTableAddBranch.slot_UBV}
 *
 * @verifyFunction{This testcase checks error path when slot set to upper boundary value with respect to ipcPermEntries.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{slot set to 9
 * ipctable.ipcPermEntries set to 10
 * ipcTable.ipcPerm[slot].ipcRoute set to NULL
 * ipcTableWithRoute[slot].ipcRoute[] set to a valid memory
 * ipcTableWithRoute[slot].ipcRouteEntries set to 1
 * needCpuAccess set to true
 * requiredperm set to "LwSciSyncAccessPerm_SignalOnly"}
 *
 * @testbehavior{ipcTable.ipcPerm[slot].ipcRouteEntries set to 1
 * ipcTable.ipcPerm[slot].ipcRoute[] to a valid memory
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonMemcpyS() receives correct arguments
 * The content of ipcTableWithRoute.ipcRoute copied to ipcTable.ipcPerm[slot].ipcRoute
 * ipcTable.ipcPerm[slot].needCpuAccess set to true
 * ipcTable.ipcPerm[slot].requiredperm set to "LwSciSyncAccessPerm_SignalOnly"
 * returned LwSciError_Success}
 *
 * @testcase{18852417}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm:<<malloc 10>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[9].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:10
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:9
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute[0]:55
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRouteEntries:0x1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[9].ipcRoute[0]:55
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[9].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[9].needCpuAccess:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[9].requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute>>[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCoreIpcTableAddBranch.routelenOf_ipcRouteEntries_greater_than_halfOf_typeLwSciIpcEndpoint
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreIpcTableAddBranch.routelenOf_ipcRouteEntries_greater_than_halfOf_typeLwSciIpcEndpoint
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreIpcTableAddBranch.routelenOf_ipcRouteEntries_greater_than_halfOf_typeLwSciIpcEndpoint}
 *
 * @verifyFunction{This testcase checks panic condition when routelength Of ipcRouteEntries are greater than half Of type LwSciIpcEndpoint}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{slot set to 0
 * ipctable.ipcPermEntries set to 1
 * ipcTable.ipcPerm[slot].ipcRoute set to NULL
 * ipcTableWithRoute[slot].ipcRouteEntries set to 0x2000000000000000
 * needCpuAccess set to false
 * requiredperm set to "LwSciSyncAccessPerm_WaitOnly"}
 *
 * @testbehavior{ipcTable.ipcPerm[slot].ipcRouteEntries set to 0x2000000000000000
 * LwSciCommonCalloc() receives correct arguments
 * LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18852420}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRouteEntries:0x2000000000000000
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:VECTORCAST_BUFFER
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:0x2000000000000000
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciSyncCoreIpcTableAddBranch.LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCoreIpcTableAddBranch.LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCoreIpcTableAddBranch.LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This testcase checks error path when LwSciCommonCalloc() fails}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{slot set to 0
 * ipctable.ipcPermEntries  set to 1
 * ipcTable.ipcPerm[slot].ipcRoute set to NULL
 * ipcTableWithRoute[slot].ipcRouteEntries set to 1
 * needCpuAccess set to false
 * requiredperm set to "LwSciSyncAccessPerm_WaitOnly"}
 *
 * @testbehavior{ipcTable.ipcPerm[slot].ipcRouteEntries set to 1
 * LwSciCommonCalloc() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852423}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRouteEntries:0x1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.requiredPerm:LwSciSyncAccessPerm_WaitOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_WaitOnly
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
TEST.END_FLOW
TEST.END

-- Test Case: TC_010.LwSciSyncCoreIpcTableAddBranch.ipcPerm[0].ipcRoute_set_non-NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_010.LwSciSyncCoreIpcTableAddBranch.ipcPerm[0].ipcRoute_set_non-NULL
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncCoreIpcTableAddBranch.ipcPerm[0].ipcRoute_set_non-NULL}
 *
 * @verifyFunction{This test case checks panic behaviour when ipcPerm[0].ipcRoute set to valid memory}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable[slot].ipcPerm[slot].ipcRoute set to valid memory
 * ipcTable[slot].ipcPermEntries set to 1
 * slot set to 0
 * ipcTableWithRoute set to valid memory
 * needCpuAccess set to false
 * requiredperm set to "LwSciSyncAccessPerm_WaitOnly"}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program}
 *
 * @testcase{18852426}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_011.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_011.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute_NULL
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute_NULL}
 *
 * @verifyFunction{This testcase checks panic condition when ipcTableWithRoute is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to valid memory
 * slot set to 1
 * ipcTableWithRoute set to NULL
 * needCpuAccess set to false
 * requiredperm set to "LwSciSyncAccessPerm_WaitOnly"}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableAddBranch() panics}
 *
 * @testcase{18852429}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_012.LwSciSyncCoreIpcTableAddBranch.ipcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_012.LwSciSyncCoreIpcTableAddBranch.ipcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncCoreIpcTableAddBranch.ipcTable_NULL}
 *
 * @verifyFunction{This testcase checks LwSciSyncCoreIpcTableAddBranch() panics when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to NULL
 * slot set to 1
 * ipcTableWithRoute set to valid memory
 * needCpuAccess set to false
 * requiredperm set to "LwSciSyncAccessPerm_WaitOnly"}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableAddBranch() panics}
 *
 * @testcase{18852432}
 *
 * @verify{18844551}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_013.LwSciSyncCoreIpcTableAddBranch.Ilwalid_requiredPerm
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAddBranch
TEST.NEW
TEST.NAME:TC_013.LwSciSyncCoreIpcTableAddBranch.Ilwalid_requiredPerm
TEST.NOTES:
/**
* @testname{TC_013.LwSciSyncCoreIpcTableAddBranch.Ilwalid_requiredPerm}
*
* @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableAddBranch for invalid requiredPerm.}
*
* @testpurpose{Unit testing of LwSciSyncCoreIpcTableAddBranch.}
*
* @casederiv{Analysis of Requirements.}
*
* @testsetup{The following stub function(s) are simulated to panic as per respective SWUD:
* - None
* All other stub functions are simulated to return success as per respective SWUD.}
*
* @testinput{- slot set to 0
* - ipctable.ipcPermEntries set to 1
* - ipcTable.ipcPerm[slot].ipcRoute set to NULL
* - needCpuAccess set to true
* - requiredperm set to "LwSciSyncAccessPerm_Auto"}
*
* @testbehavior{- LwSciSyncCoreIpcTableAddBranch panics.
* - Control flow is verified by ensuring function call sequence as expected by sequence diagram/code to realize the functionality.
* - Data flow is verified by ensuring stub function(s) receive correct arguments as per their respective SWUD.}
*
* @teststeps{- Simulate stub functions (using the test infra/tool of choice) as per "TEST SETUP".
* - Set input arguments (using the test infra/tool of choice) to the function under test as per "INPUTS".
* - Set assertions (using the test infra/tool of choice) as per "EXPECTED BEHAVIOR".
* - Execute the test and check all assertions pass.}
*
* @testplatform{DRIVEOS_QNX_Safety, DRIVEOS_QNX_Standard, DRIVEOS_Linux}
*
* @testcase{}
*
* @verify{18844551}
*/
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPerm[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.slot:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.requiredPerm:LwSciSyncAccessPerm_Auto
TEST.VALUE:uut_prototype_stubs.LwSciSyncCorePermLEq.return:false
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permA:LwSciSyncAccessPerm_Auto
TEST.EXPECTED:uut_prototype_stubs.LwSciSyncCorePermLEq.permB:LwSciSyncAccessPerm_WaitSignal
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAddBranch
  uut_prototype_stubs.LwSciSyncCorePermLEq
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreIpcTableAppend

-- Test Case: TC_001.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries_set_to_zero_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAppend
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries_set_to_zero_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries_set_to_zero_NormalOperation}
 *
 * @verifyFunction{This test case checks appending of new ipc route path to ipc route successfully  when ipcroute set to valid memory}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 0
 * ipcEndpoint set to 5}
 *
 * @testbehavior{ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcRoute[0] to 5
 * LwSciCommonCalloc() receives correct argumens
 * returned LwSciError_Success}
 *
 * @testcase{18852435}
 *
 * @verify{18844548}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRoute[0]:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcEndpoint:5
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRoute[0]:5
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:0x1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.ATTRIBUTES:lwscisync_ipc_table.LwSciSyncCoreIpcTableAddBranch.ipcTableWithRoute[0].ipcRouteEntries:INPUT_BASE=16
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAppend
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAppend
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>,<<uut_prototype_stubs.LwSciCommonCalloc.size>>));
TEST.END_STUB_VAL_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries_set_to_non-zero_NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAppend
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries_set_to_non-zero_NormalOperation
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries_set_to_non-zero_NormalOperation}
 *
 * @verifyFunction{This test case checks appending of new ipc route path to ipc route successfully  when ipcroute set to valid memory}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable[0].ipcRouteEntries set to 1
 * ipcEndpoint set to 2}
 *
 * @testbehavior{ipcTable[0].ipcRoute set to 1
 * returned LwSciError_Success }
 *
 * @testcase{18852438}
 *
 * @verify{18844548}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcEndpoint:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRoute[0]:2
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.return:LwSciError_Success
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAppend
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAppend
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciSyncCoreIpcTableAppend.LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAppend
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreIpcTableAppend.LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreIpcTableAppend.LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test case checks the error path when LwSciCommonCalloc() fails.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{ipcTable set to valid memory
 * ipcTable[0].ipcRouteEntries set to 0
 * ipcEndpoint set to 1}
 *
 * @testbehavior{ipcTable[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcRoute set to NULL
 * LwSciCommonCalloc() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852441}
 *
 * @verify{18844548}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcEndpoint:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAppend
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAppend
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciSyncCoreIpcTableAppend.ipcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableAppend
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreIpcTableAppend.ipcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreIpcTableAppend.ipcTable_NULL}
 *
 * @verifyFunction{This testcase checks LwSciSyncCoreIpcTableAppend() panics when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to NULL
 * ipcEndpoint set to 1}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableAppend() panics}
 *
 * @testcase{18852444}
 *
 * @verify{18844548}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcTable:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableAppend.ipcEndpoint:1
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableAppend
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreIpcTableLwtSubTree

-- Test Case: TC_001.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_matches_ipcRoute_entry_with_needCpuAccess_true
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableLwtSubTree
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_matches_ipcRoute_entry_with_needCpuAccess_true
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_matches_ipcRoute_entry_with_needCpuAccess_true}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableLwtSubTree() for success use case
 * input ipcEndpoint matches last entry in ipcRoute in ipcTable's ipcPerm and needCpuAccess set to true in ipcTable's ipcPerm}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipcEndpoint set to 3
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 2
 * ipcTable[0].ipcPerm[0].ipcRoute[1] set to 3
 * ipcTable[0].ipcPerm[0].needCpuAccess set to true
 * ipcTable[0].ipcPerm[0].requiredperm set to LwSciSyncAccessPerm_WaitSignal
 * needCpuAccess points to memory holding sum of cpu access permission
 * requiredPerm points to memory holding sum of required permissions}
 *
 * @testbehavior{ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * needCpuAccess set to true
 * requiredperm set to LwSciSyncAccessPerm_WaitSignal}
 *
 * @testcase{18852447}
 *
 * @verify{18844560}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 2>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRoute[1]:3
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRouteEntries:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].needCpuAccess:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcEndpoint:3
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm:<<malloc 1>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess[0]:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm[0]:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_matches_ipcRoute_entry_with_needCpuAccess_false
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableLwtSubTree
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_matches_ipcRoute_entry_with_needCpuAccess_false
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_matches_ipcRoute_entry_with_needCpuAccess_false}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableLwtSubTree() for success use case
 * input ipcEndpoint matches last entry in ipcRoute in ipcTable's ipcPerm and needCpuAccess set to false in ipcTable's ipcPerm}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipcEndpoint set to 3
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 2
 * ipcTable[0].ipcPerm[0].ipcRoute[1] set to 3
 * ipcTable[0].ipcPerm[0].needCpuAccess set to false
 * ipcTable[0].ipcPerm[0].requiredperm set to LwSciSyncAccessPerm_WaitSignal
 * needCpuAccess points to memory holding sum of cpu access permission
 * requiredPerm points to memory holding sum of required permissions}
 *
 * @testbehavior{ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * needCpuAccess set to false
 * requiredperm set to LwSciSyncAccessPerm_WaitSignal}
 *
 * @testcase{18852450}
 *
 * @verify{18844560}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 2>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRoute[1]:3
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRouteEntries:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].needCpuAccess:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcEndpoint:3
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess[0]:true
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess[0]:false
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm[0]:LwSciSyncAccessPerm_WaitSignal
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_doesnt_match_ipcRoute_entry
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableLwtSubTree
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_doesnt_match_ipcRoute_entry
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreIpcTableLwtSubTree.input_ipcEndpoint_doesnt_match_ipcRoute_entry}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableLwtSubTree() for success use case
 * when last entry of ipcRoute in ipcTable's ipcPerm doesn't match to input ipcEndpoint}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipcTable[0].ipcPermEntries set to 1
 * ipcEndpoint set to 2
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 2
 * ipcTable[0].ipcPerm[0].ipcRoute[1] set to 1
 * needCpuAccess points to memory holding sum of cpu access permission
 * requiredPerm points to memory holding sum of required permissions}
 *
 * @testbehavior{ipcTable[0].ipcPerm[0].ipcPermEntries set to 0}
 *
 * @testcase{18852453}
 *
 * @verify{18844560}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRoute[0]:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcEndpoint:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess[0]:false
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPermEntries:0
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_NULL_requiredPerm
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableLwtSubTree
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_NULL_requiredPerm
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_NULL_requiredPerm}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableLwtSubTree() for panic use case
 * where LwSciSyncCoreIpcTableLwtSubTree() panics when requiredPerm is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipcTable set to valid memory address
 * ipcEndpoint set to 1
 * needCpuAccess set to valid memory address
 * requiredPerm set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableLwtSubTree() panics}
 *
 * @testcase{18852456}
 *
 * @verify{18844560}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcEndpoint:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_NULL_needCpuAccess
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableLwtSubTree
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_NULL_needCpuAccess
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_NULL_needCpuAccess}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableLwtSubTree() for panic use case
 * where LwSciSyncCoreIpcTableLwtSubTree() panics when needCpuAccess is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipcTable set to valid memory address
 * ipcEndpoint set to 1
 * needCpuAccess set to NULL
 * requiredPerm set to valid memory address}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableLwtSubTree() panics}
 *
 * @testcase{18852459}
 *
 * @verify{18844560}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcEndpoint:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm:<<malloc 1>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_NULL_ipcTable
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableLwtSubTree
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_NULL_ipcTable
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_ipcTable_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableLwtSubTree() for panic use case
 * where LwSciSyncCoreIpcTableLwtSubTree() panics when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipcTable set to NULL
 * ipcEndpoint set to 1
 * needCpuAccess set to valid memory address
 * requiredPerm set to valid memory address}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableLwtSubTree() panics}
 *
 * @testcase{18852462}
 *
 * @verify{18844560}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcEndpoint:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm:<<malloc 1>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_zero_ipcRouteEntries_in_ipcPerm
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableLwtSubTree
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_zero_ipcRouteEntries_in_ipcPerm
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreIpcTableLwtSubTree.Panic_due_to_zero_ipcRouteEntries_in_ipcPerm}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableLwtSubTree() for panic use case
 * where LwSciSyncCoreIpcTableLwtSubTree() panics when zero ipcRouteEntries in ipcTable's ipcPerm}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipcTable set to valid memory address
 * ipcTable[0].ipcPermEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 0
 * needCpuAccess set to valid memory address
 * requiredPerm set to valid memory address}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableLwtSubTree() panics}
 *
 * @testcase{}
 *
 * @verify{18844560}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPerm[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.needCpuAccess:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableLwtSubTree.requiredPerm:<<malloc 1>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableLwtSubTree
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreIpcTableFree

-- Test Case: TC_001.LwSciSyncCoreIpcTableFree.Success
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableFree
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreIpcTableFree.Success
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreIpcTableFree.Success}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableFree() for success use case
 * when freeing of memory resources in ipcTable is successful.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipctable[0].ipcRoute set to valid memory address
 * ipctable[0].ipcRouteEntries set to 1
 * ipctable[0].ipcPerm[0].ipcRoute set to valid memory address
 * ipctable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipctable[0].ipcPerm set to valid memory address
 * ipctable[0].ipcPermEntries set to 1}
 *
 * @testbehavior{ipctable[0].ipcRoute being freed by LwSciCommonFree()
 * ipctable[0].ipcRouteEntries set to 0
 * ipctable[0].ipcPerm[0].ipcRoute memory being freed by LwSciCommonFree()
 * ipctable[0].ipcPerm being freed by LwSciCommonFree()
 * ipctable[0].ipcPermEntries set to 0}
 *
 * @testcase{18852465}
 *
 * @verify{18844557}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPermEntries:1
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPermEntries:0
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count=0;
count++;
if(count==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable>>[0].ipcRoute ) }}
else if(count==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable>>[0].ipcPerm[0].ipcRoute ) }}
else if(count==3)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable>>[0].ipcPerm ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreIpcTableFree.Free_ipcTable_with_null_ipcPerm
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableFree
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreIpcTableFree.Free_ipcTable_with_null_ipcPerm
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreIpcTableFree.Free_ipcTable_with_null_ipcPerm}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableFree() for success use case
 * when null ipcPerm in ipcTable}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipctable[0].ipcRoute set to valid memory address
 * ipctable[0].ipcRouteEntries set to 1
 * ipctable[0].ipcPerm set to NULL
 * ipctable[0].ipcPermEntries set to 0}
 *
 * @testbehavior{ipctable[0].ipcRoute being freed by LwSciCommonFree()
 * ipctable[0].ipcPerm being freed by LwSciCommonFree()
 * ipctable[0].ipcRouteEntries set to 0
 * ipctable[0].ipcPermEntries set to 0}
 *
 * @testcase{18852468}
 *
 * @verify{18844557}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPermEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable[0].ipcPermEntries:0
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int count=0;
count++;
if(count==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable>>[0].ipcRoute ) }}
else if(count==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable>>[0].ipcPerm ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreIpcTableFree.Failure_due_to_NULL_ipcTable
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableFree
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreIpcTableFree.Failure_due_to_NULL_ipcTable
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreIpcTableFree.Failure_due_to_NULL_ipcTable}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciSyncCoreIpcTableFree() for failure use case
 * when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{ipcTable returns to NULL}
 *
 * @testbehavior{Nothing}
 *
 * @testcase{18852471}
 *
 * @verify{18844557}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableFree.ipcTable:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableFree
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableFree
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreIpcTableGetPermAtSubTree

-- Test Case: TC_001.LwSciSyncCoreIpcTableGetPermAtSubTree.NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableGetPermAtSubTree
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreIpcTableGetPermAtSubTree.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreIpcTableGetPermAtSubTree.NormalOperation}
 *
 * @verifyFunction{This test case checks error path when ipcEndpoint equals to ipcPerm[0].ipcRoute[0] and Get sum of permissions of a subtree}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable[0].ipcPermEntries set to 1
 * ipcEndpoint set to 2
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute[0] set to 2
 * ipcTable[0].ipcPerm[0].requiredPerm set to LwSciSyncAccessPerm_SignalOnly}
 *
 * @testbehavior{perm set to LwSciSyncAccessPerm_SignalOnly
 * returned LwSciError_Success}
 *
 * @testcase{18852474}
 *
 * @verify{18844563}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm[0].ipcRoute[0]:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm[0].requiredPerm:LwSciSyncAccessPerm_SignalOnly
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcEndpoint:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.perm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.perm[0]:LwSciSyncAccessPerm_WaitSignal
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.perm[0]:LwSciSyncAccessPerm_SignalOnly
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:(2)8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:(2)8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableGetPermAtSubTree
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableGetPermAtSubTree
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcEndpoint_isNotEquals_to_ipcPerm[0].ipcRoute[0]
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableGetPermAtSubTree
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcEndpoint_isNotEquals_to_ipcPerm[0].ipcRoute[0]
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcEndpoint_isNotEquals_to_ipcPerm[0].ipcRoute[0]}
 *
 * @verifyFunction{This test case checks the error path when the ipcEndpoint is not equal to ipcPerm[0].ipcRoute[0]}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable[0].ipcPermEntries set to 1
 * ipcEndpoint set to 3
 * ipcTable[0].ipcPerm[0].ipcRouteEntries set to 1
 * ipcTable[0].ipcPerm[0].ipcRoute[0] set to 2}
 *
 * @testbehavior{returned LwSciError_BadParameter}
 *
 * @testcase{18852477}
 *
 * @verify{18844563}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm[0].ipcRoute[0]:2
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPerm[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable[0].ipcPermEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcEndpoint:3
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.perm:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:8
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:8
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableGetPermAtSubTree
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableGetPermAtSubTree
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>,<<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreIpcTableGetPermAtSubTree.perm_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableGetPermAtSubTree
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreIpcTableGetPermAtSubTree.perm_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreIpcTableGetPermAtSubTree.perm_NULL}
 *
 * @verifyFunction{This testcase checks LwSciSyncCoreIpcTableGetPermAtSubTree() panics when perm is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to valid memory
 * ipcEndpoint set to 1
 * perm set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableGetPermAtSubTree() panics}
 *
 * @testcase{18852480}
 *
 * @verify{18844563}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcEndpoint:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.perm:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableGetPermAtSubTree
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableGetPermAtSubTree
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable_NULL}
 *
 * @verifyFunction{This testcase checks LwSciSyncCoreIpcTableGetPermAtSubTree() panics when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{ipcTable set to NULL
 * ipcEndpoint set to 1
 * perm set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableGetPermAtSubTree() panics}
 *
 * @testcase{18852483}
 *
 * @verify{18844563}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcTable:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.ipcEndpoint:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableGetPermAtSubTree.perm:<<malloc 1>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableGetPermAtSubTree
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreIpcTableRouteIsEmpty

-- Test Case: TC_001.LwSciSyncCoreIpcTableRouteIsEmpty.route_isempty
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableRouteIsEmpty
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreIpcTableRouteIsEmpty.route_isempty
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreIpcTableRouteIsEmpty.route_isempty}
 *
 * @verifyFunction{This test-case checks that the IPC Route in the LwSciSyncCoreIpcTable is empty}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{}
 *
 * @testinput{'ipcTable[0].ipcRoute' set to NULL
 * 'ipcTable[0].ipcRouteEntries' set to 0}
 *
 * @testbehavior{returns true}
 *
 * @testcase{18852486}
 *
 * @verify{18844566}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.return:false
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.return:true
TEST.ATTRIBUTES:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable[0].ipcRoute[0]::INPUT_BASE=16
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableRouteIsEmpty
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableRouteIsEmpty
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciSyncCoreIpcTableRouteIsEmpty.ipcroute_isNotempty
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableRouteIsEmpty
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreIpcTableRouteIsEmpty.ipcroute_isNotempty
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreIpcTableRouteIsEmpty.ipcroute_isNotempty}
 *
 * @verifyFunction{This test-case verifies that the IPC Route in the LwSciSyncCoreIpcTable is not empty}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{'ipcTable[0].ipcRoute' pointer points to a valid memory
 * 'ipcTable[0].ipcRouteEntries' set to 1}
 *
 * @testbehavior{returns false}
 *
 * @testcase{18852489}
 *
 * @verify{18844566}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable[0].ipcRoute:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable[0].ipcRoute[0]:0x5555555555555
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable[0].ipcRouteEntries:1
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.return:true
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.return:false
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableRouteIsEmpty
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableRouteIsEmpty
TEST.END_FLOW
TEST.END

-- Test Case: TC_003_LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableRouteIsEmpty
TEST.NEW
TEST.NAME:TC_003_LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_003_LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable_NULL}
 *
 * @verifyFunction{This testcase checks LwSciSyncCoreIpcTableRouteIsEmpty() panics when ipcTable is NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{"ipctable" pointer points to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program
 * LwSciSyncCoreIpcTableRouteIsEmpty() panics}
 *
 * @testcase{18852492}
 *
 * @verify{18844566}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableRouteIsEmpty.ipcTable:<<null>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableRouteIsEmpty
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCoreIpcTableTreeAlloc

-- Test Case: TC_001.LwSciSyncCoreIpcTableTreeAlloc.NormalOperation
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableTreeAlloc
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreIpcTableTreeAlloc.NormalOperation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreIpcTableTreeAlloc.NormalOperation}
 *
 * @verifyFunction{This test case checks the requested number of branches in LwSciSyncCoreAttrIpcPerm array are allocated successfully }
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{'ipcTable[0].ipcPerm' pointer points to a valid memory
 * 'size' set to 50}
 *
 * @testbehavior{ipcTable.ipcPermEntries set to 50
 * LwSciCommonCalloc() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852495}
 *
 * @verify{18844545}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.size:50
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:50
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:50
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrIpcPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable>>[0].ipcPerm == ( (LwSciSyncCoreAttrIpcPerm*)<<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreIpcTableTreeAlloc.size_LW
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableTreeAlloc
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreIpcTableTreeAlloc.size_LW
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreIpcTableTreeAlloc.size_LW}
 *
 * @verifyFunction{This test case checks the success scenario when 'size' is set to Nominal value}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{'ipcPerm' pointer points to a valid memory
 * 'size' set to 1500}
 *
 * @testbehavior{ipcTable.ipcPermEntries set to 1500
 * LwSciCommonCalloc() receives correct arguments
 * returned LwSciError_Success}
 *
 * @testcase{18852498}
 *
 * @verify{18844545}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.size:1500
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:1500
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1500
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrIpcPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm
{{ <<lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable>>[0].ipcPerm == ( (LwSciSyncCoreAttrIpcPerm*)<<uut_prototype_stubs.LwSciCommonCalloc.return>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreIpcTableTreeAlloc.size_LBV
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableTreeAlloc
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreIpcTableTreeAlloc.size_LBV
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreIpcTableTreeAlloc.size_LBV}
 *
 * @verifyFunction{This test-case checks the success scenario when 'size' is set to lower boundary value}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{}
 *
 * @testinput{'ipcTable' set to valid memory
 * 'size' set 0}
 *
 * @testbehavior{returned LwSciError_Success}
 *
 * @testcase{18852501}
 *
 * @verify{18844545}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.size:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_Success
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciSyncCoreIpcTableTreeAlloc.LwSciCommonCalloc_fails
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableTreeAlloc
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreIpcTableTreeAlloc.LwSciCommonCalloc_fails
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreIpcTableTreeAlloc.LwSciCommonCalloc_fails}
 *
 * @verifyFunction{This test-case checks error path when LwSciCommonCalloc() fails}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{LwSciCommonCalloc() returns to NULL}
 *
 * @testinput{'ipcTable[0].ipcPerm' set to valid memory
 * 'size' set to 1}
 *
 * @testbehavior{ipcTable.ipcPermEntries set to 1
 * LwSciCommonCalloc() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852504}
 *
 * @verify{18844545}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.size:50
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:50
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:50
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrIpcPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreIpcTableTreeAlloc.size_UBV
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableTreeAlloc
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreIpcTableTreeAlloc.size_UBV
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreIpcTableTreeAlloc.size_UBV}
 *
 * @verifyFunction{This test case checks the checks error path size is set to upper boundary value}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns to valid memory}
 *
 * @testinput{'ipcPerm' pointer points to a valid memory
 * 'size' set to SIZE_MAX}
 *
 * @testbehavior{ipcTable.ipcPermEntries set to MAX
 * LwSciCommonCalloc() receives correct arguments
 * returned LwSciError_InsufficientMemory}
 *
 * @testcase{18852507}
 *
 * @verify{18844545}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable:<<malloc 1>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:0
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.size:<<MAX>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_Unknown
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRoute:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcRouteEntries:0
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPerm:<<null>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable[0].ipcPermEntries:<<MAX>>
TEST.EXPECTED:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:<<MAX>>
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( calloc(<<uut_prototype_stubs.LwSciCommonCalloc.numItems>>, <<uut_prototype_stubs.LwSciCommonCalloc.size>>) );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreAttrIpcPerm) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreIpcTableTreeAlloc.ipcTable_NULL
TEST.UNIT:lwscisync_ipc_table
TEST.SUBPROGRAM:LwSciSyncCoreIpcTableTreeAlloc
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreIpcTableTreeAlloc.ipcTable_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreIpcTableTreeAlloc.ipcTable_NULL}
 *
 * @verifyFunction{This test case checks error path when ipcTable set to NULL}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{}
 *
 * @testinput{'ipcTable' set to NULL
 * 'size' set to 1}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18852510}
 *
 * @verify{18844545}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.ipcTable:<<null>>
TEST.VALUE:lwscisync_ipc_table.LwSciSyncCoreIpcTableTreeAlloc.size:50
TEST.FLOW
  lwscisync_ipc_table.c.LwSciSyncCoreIpcTableTreeAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END
