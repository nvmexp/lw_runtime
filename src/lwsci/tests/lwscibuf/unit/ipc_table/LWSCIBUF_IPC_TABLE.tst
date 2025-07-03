-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCIBUF_IPC_TABLE
-- Unit(s) Under Test: lwscibuf_ipc_table
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

-- Subprogram: LwSciBufAddIpcTableEntry

-- Test Case: TC_001.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_first_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_001.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_first_ilwocation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_first_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when LwSciCommonCalloc() return NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- LwSciCommonCalloc() returns NULL on  first invocation.
 * - Set global ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] to 1
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 8.
 * - value set to allocated memory
 * - allowDups set to true.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciBufAddIpcTableEntry() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856116}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:true
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:16
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (NULL);
TEST.END_STUB_VAL_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_002.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when LwSciCommonCalloc() return NULL on 2nd invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on first, second invocation and NULL on third invocation.
 * - Set global ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] to 2
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len 8.
 * - LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.}
 *
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 *  - ipcRoute reference ipcRoute global instance.
 *  - attrKey 17.
 *  - len set to 8.
 *  - value set to allocated memory
 *  - allowDups set to true.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer to another.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufAddIpcTableEntry() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856119}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_INT1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:true
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList  );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv)  ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciIpcEndpoint) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_003.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when LwSciCommonCalloc() return NULL on 3rd invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on first, second, third invocation and NULL on fourth invocation.
 * - Set global ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len 8.
 * - LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - attrKey set to 17.
 * - len set to 8.
 * - value set to allocated memory
 * - allowDups set to true.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to get no of elements to allocate and size to get size of each element
 * LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer to another.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufAddIpcTableEntry() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856122}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj.ipcEndpointList:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_INT1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:true
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList  );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv)  ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 8 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciIpcEndpoint) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_004.LwSciBufAddIpcTableEntry.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAddIpcTableEntry.Error.InsufficientMemory5}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when LwSciCommonCalloc() return NULL on 2rd invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL on second invocation
 * - Set global ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len 8.}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 8.
 * - value set to allocated memory
 * - allowDups set to true.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to get no of elements to allocate and size to get size of each element
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufAddIpcTableEntry should return LwSciEroror_InsufficientMemory.}
 *
 * @testcase{18856128}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:true
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv)  ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAddIpcTableEntry.Failure_when_validEntryCount_greaterthan_allocEntryCount
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_006.LwSciBufAddIpcTableEntry.Failure_when_validEntryCount_greaterthan_allocEntryCount
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAddIpcTableEntry.Failure_when_validEntryCount_greaterthan_allocEntryCount}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when global variable validEntryCount is greater than allocEntryCount.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,3
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 8.
 * - value set to allocated memory
 * - allowDups set to true.}
 *
 * @testbehavior{LwSciBufAddIpcTableEntry() should return LwSciError_InsufficientResource.}
 *
 * @testcase{18856131}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:3
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:true
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_InsufficientResource
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufAddIpcTableEntry.Failure_due_to_len_is_out_of_range
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_007.LwSciBufAddIpcTableEntry.Failure_due_to_len_is_out_of_range
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufAddIpcTableEntry.Failure_due_to_len_is_out_of_range}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when length is out of range}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTableAttrData, ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len 8.
 * - LwSciCommonMemcmp() returns 0}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to sizeof(uint64_t) + 1.
 * - value set to allocated memory
 * - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciBufAddIpcTableEntry() should return LwSciError_IlwalidState.}
 *
 * @testcase{18856134}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_IlwalidState
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].endpointCount) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = ( sizeof(uint64_t) + 1 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufAddIpcTableEntry.Panic_when_global_ipcRoute_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_008.LwSciBufAddIpcTableEntry.Panic_when_global_ipcRoute_is_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufAddIpcTableEntry.Panic_when_global_ipcRoute_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when ipcTableEntry->ipcRoute is NULL(Source route has zero endpoints.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcRoute->ipcRoute[0].endpointCount to 0
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len 8.}
 * - ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute set to NULL
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 8.
 * - value set to allocated memory
 * - allowDups set to true.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion.}
 *
 * @testcase{18856137}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:true
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufAddIpcTableEntry.Panic_when_srcRoute_Has_Zero_Endpoints
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_009.LwSciBufAddIpcTableEntry.Panic_when_srcRoute_Has_Zero_Endpoints
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufAddIpcTableEntry.Failure_when_srcRoute_Has_Zero_Endpoints}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when Source route has zero endpoints.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1,0
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len 8.}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 8.
 * - value set to allocated memory
 * - allowDups set to true.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion.}
 *
 * @testcase{18856140}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:true
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufAddIpcTableEntry.Success_when_Match_found_for_IpcTableEntry
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_010.LwSciBufAddIpcTableEntry.Success_when_Match_found_for_IpcTableEntry
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufAddIpcTableEntry.Success_when_Match_found_for_IpcTableEntry}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when Duplicate entries are disabled and matching entry is found in the table.}
 *
 * @casederiv{Analysis of Requirements.}
 *.
 * @testsetup{- Set global ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] to 19.21
 * - Set ipcRoute->ipcRoute[0].ipcEndpointList[0] and ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] to 17,22
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[2].ipcEndpointList[0] to 17
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 2,1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len sizeof(puint64DestBuf[0]).
 * - LwSciCommonMemcmp() returns 0
 * - Initialize puint64DestBuf array with random values
 * - LwSciBufIsMaxValue returns LwSciError_Success}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to global sizeof(puint64DestBuf[0]).
 * - value set to address of global puint64DestBuf[4]
 * - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciBufIsMaxValue() receives correct input arguments
 * LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer to another.
 * LwSciBufAddIpcTableEntry() should return LwSciError_Success.}
 *
 * @testcase{18856143}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:19
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:21
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:22
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:0x1111111111111111
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:0x2222222222222222
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:0x3333333333333333
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:0x4444444444444444
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:0x5555555555555555
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.isBigger[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.return:LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  uut_prototype_stubs.LwSciBufIsMaxValue
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src1
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src2
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src2>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.len
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.len>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.isBigger
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.isBigger>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciBufAddIpcTableEntry.Success_when_Match_Not_Found_for_IpcTableEntry
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_011.LwSciBufAddIpcTableEntry.Success_when_Match_Not_Found_for_IpcTableEntry
TEST.NOTES:
/**
 * @testname{TC_011.LwSciBufAddIpcTableEntry.Success_when_Match_Not_Found_for_IpcTableEntry}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when Duplicate entries are disabled and matching entry is not found in the table.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set ipcRoute->ipcRoute[0].ipcEndpointList[0] to 17
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcEndpointList[0] to 17
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 13 and len 8.
 * - LwSciCommonCalloc() returns valid memory
 * - LwSciCommonMemcmp() returns 0
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 8.
 * - value set to allocated memory
 * - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciCommonCalloc() is called to get no of elements to allocate and size to get size of each element
 * LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer to another.
 * LwSciBufAddIpcTableEntry() should return LwSciError_Success.}
 *
 * @testcase{18856146}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value:VECTORCAST_BUFFER
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_INT2
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.key:17
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.len:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> );
}
if( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.value );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 8 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount ) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciBufAddIpcTableEntry.Success_when_recMaxOrMin_set_to_true
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_012.LwSciBufAddIpcTableEntry.Success_when_recMaxOrMin_set_to_true
TEST.NOTES:
/**
 * @testname{TC_012.LwSciBufAddIpcTableEntry.Success_when_recMaxOrMin_set_to_true}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when Duplicate entries are disabled and matching entry is found in the table.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] to 19.21
 * - Set ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0] and ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] and  to 17,22
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 2,1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17, len set to sizeof(puint64DestBuf),
 *   value set to address of puint64DestBuf[4].
 * - puint64DestBuf arrary is initilaized
 * - LwSciCommonMemcmp() returns 0
 * - LwSciBufIsMaxValue() returns LwSciError_Success
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to global sizeof(puint64DestBuf[0]).
 * - value set to allocated memory
 * - allowDups set to false.
 * - recMaxOrMin set to true.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciBufIsMaxValue() receives correct input arguments
 * LwSciBufAddIpcTableEntry() should return LwSciError_Success.}
 *
 * @testcase{18856149}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:19
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:21
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:22
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:0x1111111111111111
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:0x2222222222222222
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:0x3333333333333333
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:0x4444444444444444
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:0x5555555555555555
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.recMaxOrMin:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.isBigger[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  uut_prototype_stubs.LwSciBufIsMaxValue
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src1
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src2
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src2>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.len
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.len>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciBufAddIpcTableEntry.Success_due_to_global_pointer_value_is_null
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_013.LwSciBufAddIpcTableEntry.Success_due_to_global_pointer_value_is_null
TEST.NOTES:
/**
 * @testname{TC_013.LwSciBufAddIpcTableEntry.Success_due_to_global_pointer_value_is_null}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when Entry in table contain NULL value.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] to 19.21
 * - Set ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0] and ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] and  to 17,22
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 2,1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17, len set to sizeof(puint64DestBuf),
 *   value set to address of puint64DestBuf[4].
 * - puint64DestBuf arrary is initilaized
 * - ipcTableAttrData[0].value set to NULL
 * - LwSciCommonMemcmp() returns 0
 * - LwSciBufIsMaxValue() returns LwSciError_Success
 *
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 4.
 * - value set to allocated memory
 * - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciBufIsMaxValue() receives correct input arguments
 * LwSciBufAddIpcTableEntry() should return LwSciError_Success.}
 *
 * @testcase{18856152}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:19
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:21
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:22
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:0x1111111111111111
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:0x2222222222222222
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:0x3333333333333333
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:0x4444444444444444
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:0x5555555555555555
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.isBigger[0]:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  uut_prototype_stubs.LwSciBufIsMaxValue
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src1
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src2
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src2>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.len
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.len>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = (4);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = (4);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciBufAddIpcTableEntry.Panic_due_to_ipcRoute_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_014.LwSciBufAddIpcTableEntry.Panic_due_to_ipcRoute_is_NULL
TEST.NOTES:
/**
 * @testname{TC_014.LwSciBufAddIpcTableEntry.Panic_due_to_ipcRoute_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when ipcRoute is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{- ipcTable set to global ipcTable instance.
 * - ipcRoute set to NULL.
 * - len set to 10
 * - value set to allocated memory}
 *
 * @testbehavior{LwSciBufAddIpcTableEntry() should terminate by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856155}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:10
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciBufAddIpcTableEntry.Panic_due_to_ipcTable_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_015.LwSciBufAddIpcTableEntry.Panic_due_to_ipcTable_is_NULL
TEST.NOTES:
/**
 * @testname{TC_015.LwSciBufAddIpcTableEntry.Panic_due_to_ipcTable_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when ipcTable is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{- ipcTable set to null.
 * - ipcRoute set to valid instance.
 * - len set to 10
 * - value set to allocated memory}
 *
 * @testbehavior{LwSciBufAddIpcTableEntry() should terminate by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856158}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:10
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_016.LwSciBufAddIpcTableEntry.Panic_due_to_len_is_0
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_016.LwSciBufAddIpcTableEntry.Panic_due_to_len_is_0
TEST.NOTES:
/**
 * @testname{TC_016.LwSciBufAddIpcTableEntry.Panic_due_to_len_is_0}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when input length of value is 0.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{None}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 0.
 * - value reference address of global puint64DestBuf instance
 * - allowDups set to false.}
 *
 * @testbehavior{LwSciBufAddIpcTableEntry() should terminate by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856161}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = (0);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_017.LwSciBufAddIpcTableEntry.Success_due_to_len_is_1
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_017.LwSciBufAddIpcTableEntry.Success_due_to_len_is_1
TEST.NOTES:
/**
 * @testname{TC_017.LwSciBufAddIpcTableEntry.Success_due_to_len_is_1}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when Duplicate entries are disabled and matching entry is found in the table.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] to 19.21
 * - Set ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0] and ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] and  to 17,22
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 2,1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17, len set to 1,
 *   value set to address of puint64DestBuf[4].
 * - puint64DestBuf arrary is initilaized
 * - LwSciCommonMemcmp() returns 0
 * - LwSciBufIsMaxValue() returns LwSciError_Success }
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 *  - ipcRoute reference ipcRoute global instance.
 *  - attrKey set to 17.
 *  - len set to 1.
 *  - value reference global puint64DestBuf instance
 *  - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciBufIsMaxValue() receives correct input arguments
 * LwSciBufAddIpcTableEntry() should return LwSciError_Success.}
 *
 * @testcase{18856164}
 *
 * @verify{18843051}
 */



TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:19
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:21
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:22
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:0x1111111111111111
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:0x2222222222222222
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:0x3333333333333333
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:0x4444444444444444
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:0x5555555555555555
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.isBigger[0]:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  uut_prototype_stubs.LwSciBufIsMaxValue
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src1
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src2
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src2>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.len
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.len>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = (1);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = (1);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_018.LwSciBufAddIpcTableEntry.len_is_2
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_018.LwSciBufAddIpcTableEntry.len_is_2
TEST.NOTES:
/**
 * @testname{TC_025.LwSciBufAddIpcTableEntry.len_is_2}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when len is 2}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] to 19.21
 * - Set ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0] and ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] and  to 17,22
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 2,1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17, len set to 9,
 *   value set to address of puint64DestBuf[4].
 * - puint64DestBuf arrary is initilaized
 * - LwSciCommonMemcmp() returns 0
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 *  - ipcRoute reference ipcRoute global instance.
 *  - attrKey set to 17.
 *  - len set to 2.
 *  - value reference global puint64DestBuf instance
 *  - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciBufAddIpcTableEntry() should return LwSciError_IlwalidState.}
 *
 * @testcase{18856167}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:19
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:21
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:22
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:0x1111111111111111
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:0x2222222222222222
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:0x3333333333333333
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:0x4444444444444444
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:0x5555555555555555
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:2
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_IlwalidState
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = (9);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_019.LwSciBufAddIpcTableEntry.Success_due_to_len_is_4
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_019.LwSciBufAddIpcTableEntry.Success_due_to_len_is_4
TEST.NOTES:
/**
 * @testname{TC_019.LwSciBufAddIpcTableEntry.Success_due_to_len_is_4}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when len is set to 4}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set ipcRoute->ipcRoute[0].ipcEndpointList[0] to 17
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcEndpointList[0] to 17
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 2,1
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 13 and len 8.
 * - LwSciCommonCalloc() returns valid memory
 * - LwSciCommonMemcmp() returns 1
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 4.
 * - value set to allocated memory
 * - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciCommonCalloc() is called to get no of elements to allocate and size to get size of each element
 * LwSciCommonMemcpyS() is called to copy memory of specified size in bytes from one pointer to another.
 * LwSciBufAddIpcTableEntry() should return LwSciError_Success.}
 *
 * @testcase{18856170}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value:VECTORCAST_BUFFER
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.value:VECTORCAST_BUFFER
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj.ipcEndpointList:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:4
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.key:17
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.len:4
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList  );
}
if( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> );
}
if( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.value );
}

i++;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv)  ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 4 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.value ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList  ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount ) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_022.LwSciBufAddIpcTableEntry.len_is_9
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_022.LwSciBufAddIpcTableEntry.len_is_9
TEST.NOTES:
/**
 * @testname{TC_026.LwSciBufAddIpcTableEntry.Failure_due_to_len_mismatches_with_the_value_in_the_node.}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when len is 9}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] to 19.21
 * - Set ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0] and ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] and  to 17,22
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 2,1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17, len set to 9,
 *   value set to address of puint64DestBuf[4].
 * - puint64DestBuf arrary is initilaized
 * - LwSciCommonMemcmp() returns 0
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 *  - ipcRoute reference ipcRoute global instance.
 *  - attrKey set to 17.
 *  - len set to 9.
 *  - value reference address of global puint64DestBuf instance
 *  - allowDups set to false.}
 *
 * @testbehavior{Function should return LwSciError_IlwalidState.}
 *
 * @testcase{18856179}
 *
 * @verify{18843051}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:19
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:21
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:22
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:0x1111111111111111
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:0x2222222222222222
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:0x3333333333333333
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:0x4444444444444444
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:0x5555555555555555
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_IlwalidState
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = (9);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = (9);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_023.LwSciBufAddIpcTableEntry.value_is_null
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_023.LwSciBufAddIpcTableEntry.value_is_null
TEST.NOTES:
/**
 * @testname{TC_023.LwSciBufAddIpcTableEntry.value_is_null}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when value is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Initialize global ipcTable and ipcRoute instance and set them to valid memory}
 *
 * @testinput{-ipcTable reference global ipcTable Instance.
 *  - ipcRoute set to global ipcRoute instance.
 *  - len set to 10
 *  - value set to NULL.
 *
 * @testbehavior{LwSciBufAddIpcTableEntry() Panics}
 *
 * @testcase{18856182}
 *
 * @verify{18843051}
 */



TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:10
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:<<null>>
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_024.LwSciBufAddIpcTableEntry.BadParameter
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_024.LwSciBufAddIpcTableEntry.BadParameter
TEST.NOTES:
/**
 * @testname{TC_024.LwSciBufAddIpcTableEntry.BadParameter}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when Duplicate entries are disabled and matching entry is found in the table.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{- Set global ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] to 19.21
 * - Set ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0] and ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] and  to 17,22
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 2,1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17, len set to sizeof(puint64DestBuf),
 *   value set to address of puint64DestBuf[4].
 * - puint64DestBuf arrary is initilaized
 * - LwSciCommonMemcmp() returns 0
 * - LwSciBufIsMaxValue() returns LwSciError_BadParameter }
 *
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to global sizeof(puint64DestBuf[0]).
 * - value reference global puint64DestBuf instance
 * - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciBufIsMaxValue() receives correct input arguments
 * LwSciBufAddIpcTableEntry() should return LwSciError_BadParameter.}
 *
 * @testcase{22060464}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:19
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:21
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:22
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:0x1111111111111111
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:0x2222222222222222
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:0x3333333333333333
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:0x4444444444444444
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:0x5555555555555555
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.isBigger:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.return:LwSciError_BadParameter
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  uut_prototype_stubs.LwSciBufIsMaxValue
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src1
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src2
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src2>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.len
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.len>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[0]));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_026.LwSciBufAddIpcTableEntry.Failure_due_to_len_mismatches_with_the_value_in_the_node.
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_026.LwSciBufAddIpcTableEntry.Failure_due_to_len_mismatches_with_the_value_in_the_node.
TEST.NOTES:
/**
 * @testname{TC_020.LwSciBufAddIpcTableEntry.Failure_due_to_len_mismatches_with_the_value_in_the_node.}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when len mismatches with the value in the node.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{- Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17, len set to 8
 * - LwSciCommonMemcmp() returns 0
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcRoute reference ipcRoute global instance.
 * - attrKey set to 17.
 * - len set to 20.
 * - value set to allocated memory
 * - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonMemcmp() compares size bytes of memory beginning at ptr1 against the size bytes of memory beginning at ptr2
 * LwSciBufAddIpcTableEntry() should return LwSciError_IlwalidState.}
 *
 * @testcase{22060465}
 *
 * @verify{18843051}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len:20
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_IlwalidState
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonMemcmp
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_027.LwSciBufAddIpcTableEntry.Failure_due_to_global_validEntryCount_is_0
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufAddIpcTableEntry
TEST.NEW
TEST.NAME:TC_027.LwSciBufAddIpcTableEntry.Failure_due_to_global_validEntryCount_is_0
TEST.NOTES:
/**
 * @testname{TC_027.LwSciBufAddIpcTableEntry.Failure_due_to_global_validEntryCount_is_0}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufAddIpcTableEntry() when global validEntryCount is 0.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * - Set global ipcTableEntry, ipcTableAttrData,ipcTable, ipcTable->ipcTableEntryArr,ipcTable->ipcRoute,ipcRoute->ipcEndpointList and ipcRoute to valid memory
 * - Set global ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] to 19.21
 * - Set ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0] and ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] and  to 17,22
 * - Set global endpointCount of ipcTableEntryArr[0] of ipcTable and ipcRoute->ipcRoute[0].endpointCount to 2,1
 * - Set ipcTable->ipcTable[0].ipcTableEntryArr[0].allocEntryCount and ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount to 3,0
 * - Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17, len set to sizeof(puint64DestBuf),
 *   value set to address of puint64DestBuf[4].
 * - puint64DestBuf arrary is initilaized}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 *  - ipcRoute reference ipcRoute global instance.
 *  - attrKey set to 17.
 *  - len set to 1.
 *  - value reference global puint64DestBuf instance
 *  - allowDups set to false.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to get no of elements to allocate and size to get size of each element
 * LwSciBufAddIpcTableEntry() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{22060470}
 *
 * @verify{18843051}
 */



TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:19
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:21
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:22
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 5>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:0x1111111111111111
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:0x2222222222222222
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:0x3333333333333333
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:0x4444444444444444
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:0x5555555555555555
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.allowDups:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcmp.return:0
TEST.VALUE:uut_prototype_stubs.LwSciBufIsMaxValue.isBigger[0]:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufAddIpcTableEntry
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr1
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.ptr2
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.ptr2>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcmp.size
{{ <<uut_prototype_stubs.LwSciCommonMemcmp.size>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src1
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src1>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.src2
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.src2>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufIsMaxValue.len
{{ <<uut_prototype_stubs.LwSciBufIsMaxValue.len>> == ( <<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead  );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[1].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[2].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[1].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = (1);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[2].ipcAttrEntryHead);

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.len>> = (1);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value
<<lwscibuf_ipc_table.LwSciBufAddIpcTableEntry.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>[4]);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufCreateIpcTable

-- Test Case: TC_001.LwSciBufCreateIpcTable.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_first_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufCreateIpcTable
TEST.NEW
TEST.NAME:TC_001.LwSciBufCreateIpcTable.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_first_ilwocation
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufCreateIpcTable.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_first_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufCreateIpcTable() when LwSciCommonCalloc returns NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcTable instance to valid memory
 * LwSciCommonCalloc returns NULL.}
 *
 * @testinput{- entryCount set to 1.
 * - outIpcTable set to global ipcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct input arguments
 * LwSciBufCreateIpcTable() should return LwSciError_InsufficientMemory if there is insufficient memory to complete the operation.}
 *
 * @testcase{18856185}
 *
 * @verify{18843048}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.entryCount:1
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufCreateIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable
<<lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufCreateIpcTable.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufCreateIpcTable
TEST.NEW
TEST.NAME:TC_002.LwSciBufCreateIpcTable.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufCreateIpcTable.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufCreateIpcTable() when LwSciCommonCalloc returns NULL in second invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcTable,ipcTableEntry  and dstIpcTable instances to valid memory.
 * LwSciCommonCalloc returns valid memory pointer on first invocation and NULL on second invocation.}
 *
 * @testinput{- entryCount set to 1.
 * - outIpcTable set to global ipcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct input arguments
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufCreateIpcTable() should return LwSciError_InsufficientMemory if there is insufficient memory to complete the operation.}
 *
 * @testcase{18856188}
 *
 * @verify{18843048}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.entryCount:1
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufCreateIpcTable.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable
<<lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufCreateIpcTable.Success_entryCount_is_17_and_outIpcTable_is_not_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufCreateIpcTable
TEST.NEW
TEST.NAME:TC_003.LwSciBufCreateIpcTable.Success_entryCount_is_17_and_outIpcTable_is_not_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufCreateIpcTable.Success_entryCount_is_17_and_outIpcTable_is_not_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufCreateIpcTable() when
 * - entryCount is 17 and
 * - outIpcTable is not null}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{ipcTableObj.allocEntryCount set to 18
 * ipcTableObj.allocEntryCount set to 19
 * LwSciCommonCalloc returns valid memory}
 *
 * @testinput{- entryCount set to 17.
 * - outIpcTable set to global ipcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct input arguments
 * LwSciBufCreateIpcTable() should return LwSciError_Success.}
 *
 * @testcase{18856191}
 *
 * @verify{18843048}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].allocEntryCount:18
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].validEntryCount:19
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.entryCount:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.return:LwSciError_Unknown
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].allocEntryCount:17
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].validEntryCount:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufCreateIpcTable.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0]>>) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 17 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable
{{ *<<lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufCreateIpcTable.Panic_due_to_entryCount_is_zero
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufCreateIpcTable
TEST.NEW
TEST.NAME:TC_004.LwSciBufCreateIpcTable.Panic_due_to_entryCount_is_zero
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufCreateIpcTable.Panic_due_to_entryCount_is_zero}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufCreateIpcTable() when entryCount is zero.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- entryCount set to 0.
 * - outIpcTable set to valid meory.}
 *
 * @testbehavior{LwSciBufCreateIpcTable() should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{18856194}
 *
 * @verify{18843048}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.entryCount:0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable:<<malloc 1>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciBufCreateIpcTable.Panic_due_to_outputIpcTable_is_null
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufCreateIpcTable
TEST.NEW
TEST.NAME:TC_005.LwSciBufCreateIpcTable.Panic_due_to_outputIpcTable_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufCreateIpcTable.Panic_due_to_outputIpcTable_is_null}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufCreateIpcTable() when outIpcTable is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- entryCount set to 1.
 * - outIpcTable set to NULL.}
 *
 * @testbehavior{LwSciBufCreateIpcTable() should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{18856197}
 *
 * @verify{18843048}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.entryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciBufCreateIpcTable.Success_EntryCount_is_1_and_outIpcTable_is_not_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufCreateIpcTable
TEST.NEW
TEST.NAME:TC_006.LwSciBufCreateIpcTable.Success_EntryCount_is_1_and_outIpcTable_is_not_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufCreateIpcTable.Success_EntryCount_is_1_and_outIpcTable_is_not_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufCreateIpcTable() when
 * - entryCount is 1 and
 * - outIpcTable is not null}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{ipcTableObj.allocEntryCount set to 6
 * ipcTableObj.allocEntryCount set to 7
 * LwSciCommonCalloc returns valid memory}
 *
 * @testinput{- entryCount set to 1
 * - outIpcTable set to global ipcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct input arguments
 * LwSciBufCreateIpcTable() should return LwSciError_Success.}
 *
 * @testcase{22060472}
 *
 * @verify{18843048}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].allocEntryCount:6
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].validEntryCount:7
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.entryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].allocEntryCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].validEntryCount:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufCreateIpcTable.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable
{{ *<<lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufCreateIpcTable.Success_entryCount_is_4_and_outIpcTable_is_not_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufCreateIpcTable
TEST.NEW
TEST.NAME:TC_007.LwSciBufCreateIpcTable.Success_entryCount_is_4_and_outIpcTable_is_not_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufCreateIpcTable.Success_entryCount_is_4_and_outIpcTable_is_not_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufCreateIpcTable() when
 * - entryCount is 4 and
 * - outIpcTable is not null}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{ipcTableObj.allocEntryCount set to 8
 * ipcTableObj.allocEntryCount set to 9
 * LwSciCommonCalloc returns valid memory}
 *
 * @testinput{- entryCount set to 4
 * - outIpcTable set to global ipcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() receives correct input arguments
 * LwSciBufCreateIpcTable() should return LwSciError_Success.}
 *
 * @testcase{22060475}
 *
 * @verify{18843048}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].allocEntryCount:8
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].validEntryCount:9
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.entryCount:4
TEST.VALUE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].allocEntryCount:4
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].validEntryCount:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufCreateIpcTable.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufCreateIpcTable
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i2 = 0;
if( i2 == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 4 ) }}
}
if( i2 == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
i2++;
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable
{{ *<<lwscibuf_ipc_table.LwSciBufCreateIpcTable.outIpcTable>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciBufFreeIpcIter

-- Test Case: TC_001.LwSciBufFreeIpcIter.ipcIter_set_to_allocated_memory
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufFreeIpcIter
TEST.NEW
TEST.NAME:TC_001.LwSciBufFreeIpcIter.ipcIter_set_to_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufFreeIpcIter.ipcIter_set_to_allocated_memory}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufFreeIpcIter() when ipcIter set to allocated memory}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Global ipcIter set to valid memory}
 *
 * @testinput{- ipcIter set to allocated memory.}
 *
 * @testbehavior{LwSciCommonFree() receives correct input arguments}
 *
 * @testcase{18856200}
 *
 * @verify{18843054}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufFreeIpcIter.ipcIter:<<malloc 1>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufFreeIpcIter
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufFreeIpcIter
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscibuf_ipc_table.LwSciBufFreeIpcIter.ipcIter>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciBufFreeIpcRoute

-- Test Case: TC_001.LwSciBufFreeIpcRoute.valPtr_set_to_allocated_memory
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufFreeIpcRoute
TEST.NEW
TEST.NAME:TC_001.LwSciBufFreeIpcRoute.valPtr_set_to_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufFreeIpcRoute.valPtr_set_to_allocated_memory}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufFreeIpcRoute() when input valPtr set to allocated memory}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcRoute instance and ipcRoute->ipcendpointList to valid memory
 * ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 13
 * ipcRoute->ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{valPtr reference global ipcRoute instance}
 *
 * @testbehavior{LwSciCommonFree() receives correct input arguments}
 *
 * @testcase{18856203}
 *
 * @verify{18843057}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:13
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:uut_prototype_stubs.LwSciCommonFree.ptr:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufFreeIpcRoute
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufFreeIpcRoute
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static bool LwSciCommonFree_called_once = false;

if (!LwSciCommonFree_called_once) {
  {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList) }}
  LwSciCommonFree_called_once = true;
} else {
  {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>) }}
}

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufFreeIpcRoute.valPtr
<<lwscibuf_ipc_table.LwSciBufFreeIpcRoute.valPtr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufFreeIpcRoute.valPtr_set_to_global_NULL_pointer
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufFreeIpcRoute
TEST.NEW
TEST.NAME:TC_002.LwSciBufFreeIpcRoute.valPtr_set_to_global_NULL_pointer
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufFreeIpcRoute.valPtr_set_to_global_NULL_pointer}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufFreeIpcRoute() when input valPtr set to global NULL pointer}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create ipcRoute global instance and set it to NULL}
 *
 * @testinput{valPtr reference global ipcRoute instance}
 *
 * @testbehavior{None}
 *
 * @testcase{18856206}
 *
 * @verify{18843057}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufFreeIpcRoute
  lwscibuf_ipc_table.c.LwSciBufFreeIpcRoute
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufFreeIpcRoute.valPtr
<<lwscibuf_ipc_table.LwSciBufFreeIpcRoute.valPtr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufFreeIpcRoute.vallPtr_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufFreeIpcRoute
TEST.NEW
TEST.NAME:TC_003.LwSciBufFreeIpcRoute.vallPtr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufFreeIpcRoute.vallPtr_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufFreeIpcRoute() when input valPtr set to NULL}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{valPtr set to NULL}
 *
 * @testbehavior{None}
 *
 * @testcase{18856209}
 *
 * @verify{18843057}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufFreeIpcRoute.valPtr:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufFreeIpcRoute
  lwscibuf_ipc_table.c.LwSciBufFreeIpcRoute
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufFreeIpcTable

-- Test Case: TC_001.LwSciBufFreeIpcTable.valPtr_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufFreeIpcTable
TEST.NEW
TEST.NAME:TC_001.LwSciBufFreeIpcTable.valPtr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufFreeIpcTable.valPtr_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufFreeIpcTable() when 'valPtr' set to NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{valPtr set to NULL}
 *
 * @testbehavior{None}
 *
 * @testcase{18856212}
 *
 * @verify{18843060}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufFreeIpcTable.valPtr:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufFreeIpcTable
  lwscibuf_ipc_table.c.LwSciBufFreeIpcTable
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufFreeIpcTable.ipcTable_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufFreeIpcTable
TEST.NEW
TEST.NAME:TC_002.LwSciBufFreeIpcTable.ipcTable_is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufFreeIpcTable.ipcTable_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufFreeIpcTable() when global ipcTable pointer set to NULL}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Global ipcTable set to NULL}
 *
 * @testinput{valPtr reference global ipcTable instance}
 *
 * @testbehavior{None}
 *
 * @testcase{18856215}
 *
 * @verify{18843060}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufFreeIpcTable
  lwscibuf_ipc_table.c.LwSciBufFreeIpcTable
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufFreeIpcTable.valPtr
<<lwscibuf_ipc_table.LwSciBufFreeIpcTable.valPtr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufFreeIpcTable.entries_exist_in_ipcTable
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufFreeIpcTable
TEST.NEW
TEST.NAME:TC_003.LwSciBufFreeIpcTable.entries_exist_in_ipcTable
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufFreeIpcTable.entries_exist_in_ipcTable}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufFreeIpcTable() when there are entries in ipcTable}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable, ipcTableEntry and ipcTableAttrData instances and set them to valid memory
 * ipcTable->ipcTableEntryArr set to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * puint64DestBuf[0] set to 17}
 *
 * @testinput{valPtr reference global ipcTable instance}
 *
 * @testbehavior{LwSciCommonFree() receives correct input arguments}
 *
 * @testcase{18856218}
 *
 * @verify{18843060}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:17
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufFreeIpcTable
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufFreeIpcTable
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufFreeIpcTable.valPtr
<<lwscibuf_ipc_table.LwSciBufFreeIpcTable.valPtr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufInitIpcTableIter

-- Test Case: TC_001.LwSciBufInitIpcTableIter.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufInitIpcTableIter
TEST.NEW
TEST.NAME:TC_001.LwSciBufInitIpcTableIter.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufInitIpcTableIter.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufInitIpcTableIter() when LwSciCommonCalloc returns NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create ipcTable and outIpcIter global instances where outIpcIter set to NULL
 * ipcTable set to valid memory
 * LwSciCommonCalloc returns NULL.}
 *
 * @testinput{- inputIpcTable reference global ipcTable
 * - outIpcIter set to allocated memory}
 *
 * @testbehavior{LwSciBufInitIpcTableIter() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856221}
 *
 * @verify{18843093}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>)) ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter>> = ( (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufInitIpcTableIter.Success_due_to_ipcEndPoint_set_to_1
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufInitIpcTableIter
TEST.NEW
TEST.NAME:TC_002.LwSciBufInitIpcTableIter.Success_due_to_ipcEndPoint_set_to_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufInitIpcTableIter.Success_due_to_ipcEndPoint_set_to_1}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufInitIpcTableIter() when ipcEndpoint set to 1}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary values}
 *
 * @testsetup{Create ipcTable and outIpcIter global instances where outIpcIter set to NULL
 * ipcTable set to valid memory
 * LwSciCommonCalloc returns allocated memory}
 *
 * @testinput{- inputIpcTable reference global ipcTable
 * - ipcEndpoint set to 1
 * - outIpcIter set to allocated memory}
 *
 * @testbehavior{LwSciCommonCalloc() is called to get no of elements to allocate and size to get size of each element
 * LwSciBufInitIpcTableIter() should return LwSciError_Success.}
 *
 * @testcase{18856224}
 *
 * @verify{18843093}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.ipcEndpoint:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj[0].ipcEndpoint:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj[0].routeIsEndpointOnly:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj[0].lwrrMatchEntryIdx:MACRO=LWSCIBUF_ILWALID_IPCTABLE_IDX
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>)) ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter
{{ *<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->ipcTable == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->ipcEndpoint == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.ipcEndpoint>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->routeIsEndpointOnly == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.routeIsEndpointOnly>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->lwrrMatchEntryIdx == (0xFFFFFFFFU) }}

TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufInitIpcTableIter.inputIpcTable_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufInitIpcTableIter
TEST.NEW
TEST.NAME:TC_003.LwSciBufInitIpcTableIter.inputIpcTable_is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufInitIpcTableIter.inputIpcTable_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufInitIpcTableIter() when inputIpcTable is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- inputIpcTable set to NULL.
 * - outIpcIter set to allocated memory}
 *
 * @testbehavior{LwSciBufInitIpcTableIter() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{18856227}
 *
 * @verify{18843093}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter:<<malloc 1>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciBufInitIpcTableIter.outIpcIter_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufInitIpcTableIter
TEST.NEW
TEST.NAME:TC_004.LwSciBufInitIpcTableIter.outIpcIter_is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufInitIpcTableIter.outIpcIter_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufInitIpcTableIter() when outIpcIter is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- inputIpcTable set to global ipcTable instance.
 * - outIpcIter set to NULL.}
 *
 * @testbehavior{LwSciBufInitIpcTableIter() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{18856230}
 *
 * @verify{18843093}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufInitIpcTableIter.Success_ipcEndpoint_set_to_Max
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufInitIpcTableIter
TEST.NEW
TEST.NAME:TC_005.LwSciBufInitIpcTableIter.Success_ipcEndpoint_set_to_Max
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufInitIpcTableIter.Success_ipcEndpoint_set_to_Max}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufInitIpcTableIter() when ipcEndpoint set to Max value.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Create ipcTable and outIpcIter global instances where outIpcIter set to NULL
 * ipcTable set to valid memory
 * LwSciCommonCalloc() returns allocated memory}
 *
 * @testinput{ - inputIpcTable reference global ipcTable
 * - ipcEndpoint set to UINT64_MAX
 * - outIpcIter set to allocated memory}
 *
 * @testbehavior{LwSciCommonCalloc() is called to get no of elements to allocate and size to get size of each element
 * LwSciBufInitIpcTableIter() should return LwSciError_Success.}

 * @testcase{22060479}
 *
 * @verify{18843093}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter:<<null>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj[0].routeIsEndpointOnly:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj[0].lwrrMatchEntryIdx:MACRO=LWSCIBUF_ILWALID_IPCTABLE_IDX
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>)) ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.ipcEndpoint
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.ipcEndpoint>> = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter
{{ *<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->ipcTable == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->ipcEndpoint == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.ipcEndpoint>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->routeIsEndpointOnly == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.routeIsEndpointOnly>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->lwrrMatchEntryIdx == (0xFFFFFFFFU) }}

TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufInitIpcTableIter.Success_due_to_ipcEndPoint_set_to_Min
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufInitIpcTableIter
TEST.NEW
TEST.NAME:TC_006.LwSciBufInitIpcTableIter.Success_due_to_ipcEndPoint_set_to_Min
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufInitIpcTableIter.Success_due_to_ipcEndPoint_set_to_Min}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufInitIpcTableIter() when ipcEndpoint set to Min value.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Create ipcTable and outIpcIter global instances where outIpcIter set to NULL
 * ipcTable set to valid memory
 * LwSciCommonCalloc() returns allocated memory}
 *
 * @testinput{- inputIpcTable reference global ipcTable
 * - ipcEndpoint set to 0
 * - outIpcIter set to allocated memory}
 *
 * @testbehavior{LwSciCommonCalloc() is called to get no of elements to allocate and size to get size of each element
 * LwSciBufInitIpcTableIter() should return LwSciError_Success.}
 *
 * @testcase{22060482}
 *
 * @verify{18843093}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.ipcEndpoint:0
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj[0].ipcEndpoint:0
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj[0].routeIsEndpointOnly:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj[0].lwrrMatchEntryIdx:MACRO=LWSCIBUF_ILWALID_IPCTABLE_IDX
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufInitIpcTableIter
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>)) ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter
<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter
{{ *<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.outIpcIter>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIterObj>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->ipcTable == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.inputIpcTable>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->ipcEndpoint == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.ipcEndpoint>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->routeIsEndpointOnly == (<<lwscibuf_ipc_table.LwSciBufInitIpcTableIter.routeIsEndpointOnly>>) }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.outIpcIter>>->lwrrMatchEntryIdx == (0xFFFFFFFFU) }}

TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcIterLwrrGetAttrKey

-- Test Case: TC_001.LwSciBufIpcIterLwrrGetAttrKey.Failure_due_to_IpcIter_is_at_IlwalidIndex
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcIterLwrrGetAttrKey
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcIterLwrrGetAttrKey.Failure_due_to_IpcIter_is_at_IlwalidIndex
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcIterLwrrGetAttrKey.Failure_due_to_IpcIter_is_at_IlwalidIndex}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcIterLwrrGetAttrKey() when global variable ipcIter->ipcIter[0].lwrrMatchEntryIdx is equal to 0xFFFFFFFF.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create ipcIter and ipcTableAttrData global instances  and set them to valid memory
 * ipcTableAttrData->ipcTableAttrData[0].value set to NULL
 * ipcIter->ipcIter[0].lwrrMatchEntryIdx set to UINT32_MAX}
 *
 * @testinput{- ipcIter reference global ipcIter instance having lwrrMatchEntryIdx equal to UINT32_MAX.
 * - attrKey set to 0
 * - len set to allocated memory
 * - value reference global ipcTableAttrData->ipcTableAttrData[0].value}
 *
 * @testbehavior{LwSciBufIpcIterLwrrGetAttrKey() should return LwSciError_IlwalidOperation.}
 *
 * @testcase{18856233}
 *
 * @verify{18843090}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:0x0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.return:LwSciError_IlwalidOperation
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].lwrrMatchEntryIdx
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].lwrrMatchEntryIdx = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Mid_value
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcIterLwrrGetAttrKey
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Mid_value
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Mid_value}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcIterLwrrGetAttrKey() when 'attrKey' input parameter set to Mid value}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Create gobal ipcTableAttrData, ipcTable and ipcIter instances and set them to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * ipcTableAttrData->ipcTableAttrData[0].key set to 17
 * ipcTableAttrData->ipcTableAttrData[0].len set to sizeof global pointer 'puint64SrcBuf'
 * ipcTableAttrData->ipcTableAttrData[0].listEntry.next reference global next pointer of ipcIter instance
 * ipcTableAttrData->ipcTableAttrData[0].listEntry.prev reference global prev pointer of ipcIter instance
 * ipcIter->ipcIter[0].ipcAttrEntryHead.next reference global next pointer of ipcTableAttrData instance
 * ipcIter->ipcIter[0].ipcAttrEntryHead.prev reference global prev pointer of ipcTableAttrData instance
 * puint64SrcBuf->puint64SrcBuf[0] set to 17
 * puint64DestBuf set to NULL
 * uint64DestBufSize set to 0}
 *
 * @testinput{ - ipcIter reference global ipcIter instance
 * - attrKey set to 17
 * - len set to address of global uint64DestBufSize
 * - value reference global ipcTableAttrData->ipcTableAttrData[0].value}
 *
 * @testbehavior{LwSciBufIpcIterLwrrGetAttrKey() should return LwSciError_Success.}
 *
 * @testcase{18856236}
 *
 * @verify{18843090}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].lwrrMatchEntryIdx:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf[0]:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:17
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].len
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len = ( sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf>>) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len.len[0]
{{ <<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len>>[0] == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value.value[0]
{{ <<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value>>[0] == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> == sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf>>)}}
{{ memcmp(<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>, <<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf>>, sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf>>)) == 0 }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0] ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Max_value
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcIterLwrrGetAttrKey
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Max_value
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Max_value}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcIterLwrrGetAttrKey() when 'attrKey' input parameter set to Max value}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Create gobal ipcTableAttrData and ipcIter instances and set them to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * ipcTableAttrData->ipcTableAttrData[0].key set to 17
 * ipcTableAttrData->ipcTableAttrData[0].len set to 4
 * ipcTableAttrData->ipcTableAttrData[0].value set to allocated memory
 * ipcTableAttrData->ipcTableAttrData[0].listEntry.next reference global next pointer of ipcIter instance
 * ipcTableAttrData->ipcTableAttrData[0].listEntry.prev reference global prev pointer of ipcIter instance
 * ipcIter->ipcIter[0].ipcAttrEntryHead.next reference global next pointer of ipcTableAttrData instance
 * ipcIter->ipcIter[0].ipcAttrEntryHead.prev reference global prev pointer of ipcTableAttrData instance
 * puint64DestBuf set to NULL
 * uint64DestBufSize set to 0}
 *
 * @testinput{- ipcIter reference global ipcIter instance
 * - attrKey set to UINT32_MAX
 * - len set to address of global uint64DestBufSize
 * - value reference global ipcTableAttrData->ipcTableAttrData[0].value}
 *
 * @testbehavior{LwSciBufIpcIterLwrrGetAttrKey() should return LwSciError_Success.}
 *
 * @testcase{18856239}
 *
 * @verify{18843090}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:4
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len[0]:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = (malloc(4));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey>> = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value.value[0]
{{ <<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value>>[0] == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>> == NULL}}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> == 0 }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcIterLwrrGetAttrKey.ipcIter_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcIterLwrrGetAttrKey
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcIterLwrrGetAttrKey.ipcIter_is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcIterLwrrGetAttrKey.ipcIter_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcIterLwrrGetAttrKey() when  ipcIter is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- ipcIter set to  NULL.
 * - attrKey set to 0
 * - len set to allocated memory
 * - value set to allocated memory}
 *
 * @testbehavior{LwSciBufIpcIterLwrrGetAttrKey() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{18856242}
 *
 * @verify{18843090}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:0x0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value:<<malloc 1>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciBufIpcIterLwrrGetAttrKey.len_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcIterLwrrGetAttrKey
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcIterLwrrGetAttrKey.len_is_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcIterLwrrGetAttrKey.len_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcIterLwrrGetAttrKey() when value is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- ipcIter set to allocated memory.
 * - attrKey set to 17
 * - len set to NULL
 * - value set to allocated memory.}
 *
 * @testbehavior{LwSciBufIpcIterLwrrGetAttrKey() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{22060485}
 *
 * @verify{18843090}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value:<<malloc 1>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value.value[0]
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value>>[0] = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_BUFFER>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIpcIterLwrrGetAttrKey.value_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcIterLwrrGetAttrKey
TEST.NEW
TEST.NAME:TC_006.LwSciBufIpcIterLwrrGetAttrKey.value_is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIpcIterLwrrGetAttrKey.value_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcIterLwrrGetAttrKey() when len is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- ipcIter set to allocated memory.
 * - attrKey value set to 17.
 * - len set to allocated memory.
 * - value set to NULL}
 *
 * @testbehavior{LwSciBufIpcIterLwrrGetAttrKey() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{22060486}
 *
 * @verify{18843090}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:17
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Min_value
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcIterLwrrGetAttrKey
TEST.NEW
TEST.NAME:TC_007.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Min_value
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufIpcIterLwrrGetAttrKey.Success_due_to_attrKey_setTo_Min_value}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcIterLwrrGetAttrKey() when attrKey set to Min value.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Create gobal ipcTableAttrData and ipcIter instances and set them to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * ipcTableAttrData->ipcTableAttrData[0].key set to 17
 * ipcTableAttrData->ipcTableAttrData[0].value set to allocated memory
 * ipcTableAttrData->ipcTableAttrData[0].len set to 4
 * ipcTableAttrData->ipcTableAttrData[0].listEntry.next reference global next pointer of ipcIter instance
 * ipcTableAttrData->ipcTableAttrData[0].listEntry.prev reference global prev pointer of ipcIter instance
 * ipcIter->ipcIter[0].ipcAttrEntryHead.next reference global next pointer of ipcTableAttrData instance
 * ipcIter->ipcIter[0].ipcAttrEntryHead.prev reference global prev pointer of ipcTableAttrData instance
 * puint64DestBuf set to NULL
 * uint64DestBufSize set to 0}
 *
 * @testinput{- ipcIter reference global ipcIter instance
 * - attrKey set to 0.
 * - len set to address of global uint64DestBufSize
 * - value reference global ipcTableAttrData->ipcTableAttrData[0].value}
 *
 * @testbehavior{LwSciBufIpcIterLwrrGetAttrKey() should return LwSciError_Success.}
 *
 * @testcase{22060490}
 *
 * @verify{18843090}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:4
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize:0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:0x0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len[0]:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
  lwscibuf_ipc_table.c.LwSciBufIpcIterLwrrGetAttrKey
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].value
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = (malloc(4));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value
<<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value.value[0]
{{ <<lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.value>>[0] == ( NULL ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>> == NULL}}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> == 0 }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcRouteClone

-- Test Case: TC_001.LwSciBufIpcRouteClone.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteClone
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcRouteClone.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcRouteClone.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteClone() when LwSciCommonCalloc return NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns NULL
 * Create global ipcRoute and dstIpcRoute instances and set them to valid memory
 * ipcRoute->ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcRoute instance.
 * - dstIpcTableAddr reference global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciBufIpcRouteClone() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856245}
 *
 * @verify{18843066}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteClone.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcRouteClone.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteClone
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcRouteClone.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcRouteClone.Success_due_to_LwSciCommonCalloc_returns_allocated_memory}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteClone() when LwSciCommonCalloc return allocated memory }
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns allocated memory
 * Create global ipcRoute and dstIpcRoute instances and set them to valid memory
 * ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 17
 * ipcRoute->ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcRoute instance.
 * - dstIpcTableAddr reference global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcRouteClone() should return LwSciError_Success.}
 *
 * @testcase{18856248}
 *
 * @verify{18843066}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRouteObj.ipcEndpointList:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj.endpointCount:1
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteClone.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0; 
if( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>  );
}
if( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList );
}
i++;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint)   ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciIpcEndpoint) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcRouteClone.Success_due_to_zero_endpoint_count
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteClone
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcRouteClone.Success_due_to_zero_endpoint_count
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcRouteClone.Success_due_to_zero_endpoint_count}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteClone() when global endpointCount set to 0}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns allocated memory
 * Create global ipcRoute and dstIpcRoute instances and set them to valid memory
 * ipcRoute->ipcRoute[0].endpointCount set to 0}
 *
 * @testinput{- srcIpcTableAddr reference global ipcRoute instance.
 * - dstIpcTableAddr reference global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcRouteClone() should return LwSciError_Success.}
 *
 * @testcase{18856251}
 *
 * @verify{18843066}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute:<<malloc 1>>
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj.endpointCount:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteClone.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcRouteClone.dstIpcRouteAddr_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteClone
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcRouteClone.dstIpcRouteAddr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcRouteClone.dstIpcRouteAddr_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteClone() when dstIpcRouteAddr is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcRoute and dstIpcRoute instances to valid memory}
 *
 * @testinput{- srcIpcTableAddr reference global ipcRoute instance.
 * - dstIpcTableAddr set to NULL}
 *
 * @testbehavior{LwSciBufIpcRouteImport() should terminate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856254}
 *
 * @verify{18843066}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufIpcRouteClone.srcIpcRouteAddr_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteClone
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcRouteClone.srcIpcRouteAddr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcRouteClone.srcIpcRouteAddr_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteClone() when srcIpcRouteAddr is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcRoute and dstIpcRoute instances to valid memory}
 *
 * @testinput{- srcIpcRouteAddr set to NULL.
 * - dstIpcTableAddr reference global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciBufIpcRouteImport() should terminate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856257}
 *
 * @verify{18843066}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIpcRouteClone.Success_due_to_global_ipcRoute_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteClone
TEST.NEW
TEST.NAME:TC_006.LwSciBufIpcRouteClone.Success_due_to_global_ipcRoute_is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIpcRouteClone.Success_due_to_global_ipcRoute_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteClone() when srcIpcRouteAddr points to global instance which is initialized to NULL pointer.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcRoute instance and set it to NULL pointer.
 * Create global dstIpcRoute instance and set it to valid memory}
 *
 * @testinput{- srcIpcRouteAddr pointer set to global ipcRoute instance.
 * - dstIpcRouteAddr set to global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciBufIpcRouteImport() should return LwSciError_Success.}
 *
 * @testcase{18856260}
 *
 * @verify{18843066}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteClone.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>>  ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufIpcRouteClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteClone
TEST.NEW
TEST.NAME:TC_007.LwSciBufIpcRouteClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcRouteClone.Success_due_to_LwSciCommonCalloc_returns_allocated_memory}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteClone() when LwSciCommonCalloc return allocated memory }
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns NULL on second invocation
 * Create global ipcRoute and dstIpcRoute instance and set them to valid memory
 * ipcRoute->ipcEndpointList set to valid memory
 * dstIpcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 17
 * ipcRoute->ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcRoute instance.
 * - dstIpcTableAddr reference global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcRouteClone() should return LwSciError_Success.}
 *
 * @testcase{22060492}
 *
 * @verify{18843066}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute[0].endpointCount:1
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteClone.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcRouteClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint)   ) }}
}
i++;



TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.srcIpcRouteAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr
<<lwscibuf_ipc_table.LwSciBufIpcRouteClone.dstIpcRouteAddr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>[0].ipcEndpointList[0] == 17 }}
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>[0].endpointCount == 1 }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcRouteExport

-- Test Case: TC_001.LwSciBufIpcRouteExport.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExport
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcRouteExport.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcRouteExport.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExport() when LwSciCommonCalloc returns NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns NULL.
 * Set global ipcRoute instance to valid memory}
 *
 * @testinput{- ipcRoute reference global ipcRoute instance
 * - desc set to allocated memory
 * - len set to allocated memory}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciBufIpcRouteExport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856263}
 *
 * @verify{18843081}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.len:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteExport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExport
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcRouteExport.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExport
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcRouteExport.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcRouteExport.Success}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExport() when LwSciCommonCalloc returns allocated memory.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns allocated memory
 * Set global ipcRoute instance to valid memory
 * ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 17
 * ipcRoute->ipcRoute[0].endpointCount set to 1
 * ipcRoute->ipcEndpointList set to valid memory
 * Global pDestBuf set to allocated memory
 * Global puint64DestBuf set to NULL}
 *
 * @testinput{- ipcRoute reference global ipcRoute instance
 * - desc reference global pDestBuf instance
 * - len reference global puint64DestBuf instance}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcRouteExport() should return LwSciError_Success.}
 *
 * @testcase{18856266}
 *
 * @verify{18843081}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj.ipcEndpointList:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteExport.len[0]:16
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(uint64_t) + sizeof(LwSciIpcEndpoint)) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int count=0;
if(count ==0)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> ) }}
}
if(count ==1)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( ( (uint8_t *)(&<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>>) + sizeof(uint64_t) ))}}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].ipcEndpointList[0] ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf
int var = 88;
<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.len
<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcRouteExport.Panic_due_to_desc_is_null
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExport
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcRouteExport.Panic_due_to_desc_is_null
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcRouteExport.Panic_due_to_desc_is_null}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExport() when desc set to NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcRoute instance to valid memory}
 *
 * @testinput{- ipcRoute reference global ipcRoute instance.
 * - desc set to NULL.
 * - len set to allocated memory}
 *
 * @testbehavior{LwSciBufIpcRouteExport() should terminate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856269}
 *
 * @verify{18843081}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.len:<<malloc 1>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcRouteExport.Success_due_to_ipcRoute_Is_Null
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExport
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcRouteExport.Success_due_to_ipcRoute_Is_Null
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcRouteExport.Success_due_to_ipcRoute_Is_Null}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExport() when ipcRoute is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns allocated memory.
 * Set global ipcRoute instance to valid memory}
 *
 * @testinput{- ipcRoute set to NULL
 * - desc reference global pDestBuf instance
 * - len set to allocated memory}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcRouteExport() should return LwSciError_Success.}
 *
 * @testcase{18856272}
 *
 * @verify{18843081}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.len:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteExport.len[0]:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == (  sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufIpcRouteExport.Panic_due_to_len_is_null
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExport
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcRouteExport.Panic_due_to_len_is_null
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcRouteExport.Panic_due_to_len_is_null}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExport() when len is NULL}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcRoute instance and set it to valid memory
 * ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 17
 * ipcRoute->ipcRoute[0].endpointCount set to 1
 * Global puint64DestBuf set to NULL}
 *
 * @testinput{- ipcRoute set to global ipcRoute instance
 * - desc set to allocated memory.
 * - len set to NULL}
 *
 * @testbehavior{LwSciBufIpcRouteExport() should terminate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856275}
 *
 * @verify{18843081}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.desc:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.len:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteExport.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcRouteExportSize

-- Test Case: TC_001.LwSciBufIpcRouteExportSize.discardOuterEndpoint_set_to_false
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExportSize
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcRouteExportSize.discardOuterEndpoint_set_to_false
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcRouteExportSize.discardOuterEndpoint_set_to_false}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExportSize() when ipcRoute set to valid memory and discartOuterEndpoint set to false}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcRoute instance with endpointCount being equal to 10.}
 *
 * @testinput{- ipcRoute reference global ipcRoute instance.
 * - discartOuterEndpoint set to false.}
 *
 * @testbehavior{LwSciBufIpcRouteExportSize() API should return sizeof(LwSciIpcEndpoint) * 10 + sizeof(uint64_t)).}
 *
 * @testcase{18856278}
 *
 * @verify{18843084}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:10
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.discardOuterEndpoint:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.ipcRoute>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].endpointCount + sizeof(uint64_t)) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcRouteExportSize.discardOuterEndpoint_set_to_true
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExportSize
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcRouteExportSize.discardOuterEndpoint_set_to_true
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcRouteExportSize.discardOuterEndpoint_set_to_true}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExportSize() when ipcRoute set to valid memory and discardOuterEndpoint set to true}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcRoute instance with endpointCount being equal to 10.}
 *
 * @testinput{- ipcRoute reference global ipcRoute instance.
 * - discartOuterEndpoint set to true.}
 *
 * @testbehavior{LwSciBufIpcRouteExportSize() API should return sizeof(LwSciIpcEndpoint) * 9 + sizeof(uint64_t).}
 *
 * @testcase{18856281}
 *
 * @verify{18843084}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:10
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.discardOuterEndpoint:true
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.ipcRoute>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].endpointCount + sizeof(uint64_t)) - sizeof(LwSciIpcEndpoint)}}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcRouteExportSize.endpointCount_is_equal_to_MaxValue
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExportSize
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcRouteExportSize.endpointCount_is_equal_to_MaxValue
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcRouteExportSize.endpointCount_is_equal_to_MaxValue}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExportSize() when endpointCount of ipcRoute is equal to MAX value.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcRoute instance with endpointCount being equal to UINT64_MAX value.}
 *
 * @testinput{- ipcRoute reference global ipcRoute instance.
 * - discartOuterEndpoint set to false.}
 *
 * @testbehavior{LwSciBufIpcRouteExportSize() API should return sizeof(LwSciIpcEndpoint) * 9 + sizeof(uint64_t).}
 *
 * @testcase{18856284}
 *
 * @verify{18843084}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.discardOuterEndpoint:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute.ipcRoute[0].endpointCount
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0].endpointCount = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.ipcRoute>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return>> == ( sizeof(LwSciIpcEndpoint) ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcRouteExportSize.ipcRoute_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteExportSize
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcRouteExportSize.ipcRoute_is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcRouteExportSize.ipcRoute_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteExportSize() when ipcRoute parameter is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- ipcRoute set to NULL.
 * - discartOuterEndpoint set to false.}
 *
 * @testbehavior{LwSciBufIpcRouteExportSize() API should return sizeof(LwSciIpcEndpoint).}
 *
 * @testcase{18856287}
 *
 * @verify{18843084}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.ipcRoute:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.discardOuterEndpoint:false
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcRouteExportSize
TEST.END_FLOW
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return>> == ( sizeof(LwSciIpcEndpoint) ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:<<testcase>>
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteExportSize.return>> == sizeof(uint64_t) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcRouteImport

-- Test Case: TC_001.LwSciBufIpcRouteImport.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteImport
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcRouteImport.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcRouteImport.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteImport() when LwSciCommonCalloc return NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns NULL
 * Create global ipcRoute instance and set it to valid memory
 * ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 1
 * ipcRoute->ipcRoute[0].endpointCount set to 1
 * Global puint64DestBuf set to allocated memory
 * puint64DestBuf[0] set to 1
 * Global uint64DestBufSize set to allocated memory}
 *
 * @testinput{- ipcEndpoint set to 1.
 * - desc reference global puint64DestBuf instance
 * - len set to global sizeof(puint64DestBuf) * 2
 * - ipcRoute set to global ipcRoute instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciBufIpcRouteImport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856290}
 *
 * @verify{18843072}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcEndpoint:1
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteImport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (NULL);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) + 8 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = (sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.len
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.len>> = (sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>) * 2);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcRouteImport.Failure_due_to_len_less_than_8
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteImport
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcRouteImport.Failure_due_to_len_less_than_8
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcRouteImport.Failure_due_to_len_less_than_8}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteImport() when len is 7.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcRoute instance and set it to valid memory
 * ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 1
 * ipcRoute->ipcRoute[0].endpointCount set to 1
 * Global puint64DestBuf set to NULL
 *
 * @testinput{- ipcEndpoint set to 1.
 * - desc reference global puint64DestBuf instance
 * - len set to 7
 * - ipcRoute set to global ipcRoute instance.}
 *
 * @testbehavior{LwSciBufIpcRouteImport() returns LwSciError_BadParameter.}
 *
 * @testcase{18856293}
 *
 * @verify{18843072}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf:<<malloc 2>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcEndpoint:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.len:7
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcRouteImport.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteImport
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcRouteImport.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcRouteImport.Success_due_to_LwSciCommonCalloc_returns_allocated_memory}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteImport() when LwSciCommonCalloc returns allocated memory}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return allocated memory
 * Set ipcRoute, ipcRoute->ipcEndpointList to valid memory
 * Create global dstIpcRoute instance and set it to NULL
 * Global puint64DestBuf set to allocated memory with puint64SrcBuf[0] and puint64SrcBuf[1] set to 7,17
 * Global test_desc[0] set to 7
 * Global uint64SrcBufSize set to sizeof(puint64SrcBuf) * 2}
 *
 * @testinput{- ipcEndpoint set to 1.
 * - desc set to valid memory
 * - len set to 64
 * - ipcRoute set to global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcRouteImport() should return LwSciError_Success.}
 *
 * @testcase{18856296}
 *
 * @verify{18843072}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf:<<malloc 7>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf[0]:7
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf[1]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_desc[0]:7
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj.ipcEndpointList:<<malloc 8>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcEndpoint:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc:test_desc
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.len:64
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteImport.return:LwSciError_Success
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList[0] );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}

i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 8 ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList[0] ) }}
}
if(i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>.ipcEndpointList[7] ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
i++;
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc>> ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( ((const uint8_t *)<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc>>) + sizeof(uint64_t)) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 7 *  sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64SrcBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64SrcBufSize>> = ( sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf>>) * 2);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcRouteImport.Failure_due_to_ipcRoute_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteImport
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcRouteImport.Failure_due_to_ipcRoute_is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcRouteImport.Failure_due_to_ipcRoute_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteImport() when ipcRoute is NULL}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcRoute instance and set ipcRoute->ipcEndpointList to valid memory
 * ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 1
 * ipcRoute->ipcRoute[0].endpointCount set to 1
 * Global puint64DestBuf[0] set to 1
 * Global uint64SrcBufSize set to sizeof(puint64SrcBuf)
 *
 * @testinput{- ipcEndpoint set to 1.
 * - desc reference global puint64DestBuf instance
 * - len set to global sizeof(puint64DestBuf) * 2
 * - ipcRoute set to NULL}
 *
 * @testbehavior{LwSciBufIpcRouteImport() returns LwSciError_BadParameter.}
 *
 * @testcase{18856299}
 *
 * @verify{18843072}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcEndpoint:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = (sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.len
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.len>> = (sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>) * 2);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufIpcRouteImport.Failure_due_to_desc_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteImport
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcRouteImport.Failure_due_to_desc_is_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcRouteImport.Failure_due_to_desc_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteImport() when ipcRoute is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcRoute instance and set ipcRoute->ipcEndpointList to valid memory
 * ipcRoute->ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 1
 * ipcRoute->ipcRoute[0].endpointCount set to 1
 * Global puint64DestBuf set to allocated memory
 * puint64DestBuf[0] set to 1}
 *
 * @testinput{- ipcEndpoint set to 1.
 * - desc set to NULL
 * - len set to global sizeof(puint64DestBuf) * 2
 * - ipcRoute set to global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciBufIpcRouteImport() returns LwSciError_BadParameter.}
 *
 * @testcase{22060496}
 *
 * @verify{18843072}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcEndpoint:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = (sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.len
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.len>> = (sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf>>) * 2);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIpcRouteImport.BadParameter_due_to_len_set_to_8
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcRouteImport
TEST.NEW
TEST.NAME:TC_006.LwSciBufIpcRouteImport.BadParameter_due_to_len_set_to_8
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIpcRouteImport.BadParameter_due_to_len_set_to_8}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcRouteImport() when LwSciCommonCalloc returns allocated memory}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return allocated memory
 * Set ipcRoute, ipcRoute->ipcEndpointList, ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList to valid memory
 * Set ipcTableEntry->ipcroute and ipcTableEntry->ipcroute->ipcEndpointList set to valid memory
 * Create global dstIpcRoute instance and set it to NULL
 * Global puint64DestBuf set to allocated memory with puint64SrcBuf[0] and puint64SrcBuf[1] set to 7,17
 * Global test_desc[0] set to 7
 * Global uint64SrcBufSize set to sizeof(puint64SrcBuf) * 2}
 *
 * @testinput{- ipcEndpoint set to 1.
 * - desc set to valid memory
 * - len set to 8
 * - ipcRoute set to global dstIpcRoute instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcRouteImport() should return LwSciError_BadParameter.}
 *
 * @testcase{22060499}
 *
 * @verify{18843072}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute:<<null>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf:<<malloc 7>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf[0]:7
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf[1]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_desc[0]:7
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj.ipcEndpointList:<<malloc 8>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcEndpoint:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc:test_desc
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.len:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcRouteImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcRouteImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRouteRecPriv) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}

i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteImport.desc>> ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64SrcBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64SrcBufSize>> = ( sizeof(*<<USER_GLOBALS_VCAST.<<GLOBAL>>.puint64SrcBuf>>) * 2);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute
<<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute
{{ <<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> == ( <<lwscibuf_ipc_table.LwSciBufIpcRouteImport.ipcRoute>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcTableClone

-- Test Case: TC_001.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when LwSciCommonCalloc returns NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc returns NULL.
 * Create global ipcTable, ipcTableAttrData and dstIpcTable instances and point them to valid memory
 * ipcTable->ipcTableEntryArr,  dstIpcTable->ipcTableEntryArr points to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcTable instance.
 * - dstIpcTableAddr reference global dstIpcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciBufIpcTableClone() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856302}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableClone.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (NULL);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when LwSciCommonCalloc returns NULL on second invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return allocated memory on first invocation and NULL on second invocation.
 * Create global ipcTable, ipcTableEntry , ipcTableAttrData and dstIpcTable instances and point them to valid memory
 * ipcTable->ipcTableEntryArr,  dstIpcTable->ipcTableEntryArr points to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcTable instance.
 * - dstIpcTableAddr reference global dstIpcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableClone() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856305}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableClone.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0]>>) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}


TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when LwSciCommonCalloc returns NULL on third invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return allocated memory on first, second invocation and NULL on third invocation.
 * Create global ipcTable, ipcRoute ,ipcTableAttrData and dstIpcTable instances and point them to valid memory
 * ipcTable->ipcTableEntryArr, ipcRoute->ipcEndpointList and dstIpcTable->ipcTableEntryArr points to valid memory
 * ipcTable->ipcTableEntryArr->ipcRoute and ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcTable instance.
 * - dstIpcTableAddr reference global dstIpcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableClone() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856308}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableClone.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:(4)1
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRoute) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
}
if ( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when LwSciCommonCalloc return NULL on third invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return allocated memory on first, second invocation and NULL on third invocation.
 * Create global ipcTable,  ipcTableAttrData and dstIpcTable instances and point them to valid memory
 * ipcTable->ipcTableEntryArr,  dstIpcTable->ipcTableEntryArr points to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcTable instance.
 * - dstIpcTableAddr reference global dstIpcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableClone() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856311}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableClone.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fifth_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fifth_ilwocation
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fifth_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when LwSciCommonCalloc return NULL on fifth invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return allocated memory on first, second invocation and NULL on third invocation.
 * Create global ipcTable, ipcRoute , dstIpcRoute , ipcTableAttrData and dstIpcTable instances and point them to valid memory
 * ipcTable->ipcTableEntryArr, ipcRoute->ipcEndpointList points to valid memory
 * ipcTable->ipcTableEntryArr->ipcRoute and ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{ - srcIpcTableAddr reference global ipcTable instance.
 * - dstIpcTableAddr reference global dstIpcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableClone() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856314}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRouteObj.ipcEndpointList:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableClone.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute[0].ipcEndpointList );
}
if ( i == 4 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIpcTableClone.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_006.LwSciBufIpcTableClone.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIpcTableClone.Success_due_to_LwSciCommonCalloc_returns_allocated_memory}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when LwSciCommonCalloc return allocated memory for all ilwocations}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return allocated memory on all ilwocations
 * Create global ipcTable , ipcTableAttrData and dstIpcTable instances and point them to valid memory
 * ipcTable->ipcTableEntryArr, ipcTableEntry->ipcRoute points to valid memory
 * ipcTable->ipcTableEntryArr->ipcRoute and ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList->ipcEndpointList[0] set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcTable instance.
 * - dstIpcTableAddr reference global dstIpcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcTableClone() should return LwSciError_Success.}
 *
 * @testcase{18856317}
 *
 * @verify{18843063}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRouteObj.ipcEndpointList:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableClone.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRouteObj>> );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRouteObj>>.ipcEndpointList );
}
if ( i == 4 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> );
}
if ( i == 5 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.value );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
if( i == 5 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.len ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcRouteObj>>.ipcEndpointList ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.value ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 0 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value
int var = 8;
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufIpcTableClone.Panic_due_to_dstIpcTableAddr_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_007.LwSciBufIpcTableClone.Panic_due_to_dstIpcTableAddr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufIpcTableClone.dstOpcTable_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when dstIpcTableAddr is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable, ipcTableAttrData and dstIpcTable instances and set them to valid memory
 * ipcTable->ipcTableEntryArr and dstIpcTable->ipcTableEntryArr set to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcTable instance.
 * - dstIpcTableAddr set to NULL}
 *
 * @testbehavior{LwSciBufIpcTableImport() should termeinate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856320}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufIpcTableClone.Panic_due_to_srcIpcTableAddr_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_008.LwSciBufIpcTableClone.Panic_due_to_srcIpcTableAddr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufIpcTableClone.Panic_due_to_srcIpcTableAddr_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when srcIpcTableAddr is NULL}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable, ipcTableAttrData and dstIpcTable instances and set them to valid memory
 * ipcTable->ipcTableEntryArr and dstIpcTable->ipcTableEntryArr set to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1}
 *
 * @testinput{- srcIpcTableAddr set to NULL
 * - dstIpcTableAddr reference global dstIpcTable instance.}
 *
 * @testbehavior{LwSciBufIpcTableImport() should termeinate exelwtion by ilwoking LwSciCommonPanic().}
 *
 * @testcase{18856323}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufIpcTableClone.Success_due_to_srcIpcTableAddr_and_dstIpcTableAddr_set_to_allocated_memory
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_009.LwSciBufIpcTableClone.Success_due_to_srcIpcTableAddr_and_dstIpcTableAddr_set_to_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufIpcTableClone.Success_due_to_srcIpcTableAddr_and_dstIpcTableAddr_set_to_allocated_memory}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when srcIpcTableAddr points to global instance which is initialized with null pointer}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable and dstIpcTable instances
 * dstIpcTable set to valid meory
 * dstIpcTable->ipcTableEntryArr set to valid memory
 * ipcTable set to NULL
 *
 * @testinput{- srcIpcTableAddr pointer set to global ipcTable instance
 * - dstIpcTableAddr reference global dstIpcTable}
 *
 * @testbehavior{LwSciBufIpcTableImport() should return LwSciError_Success.}
 *
 * @testcase{18856326}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableClone.return:LwSciError_Success
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_Sixth_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableClone
TEST.NEW
TEST.NAME:TC_010.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_Sixth_ilwocation
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufIpcTableClone.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_Sixth_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableClone() when LwSciCommonCalloc return NULL on sixth invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc return allocated memory on first, second invocation and NULL on third invocation.
 * Create global ipcTable , ipcRoute , ipcTableEntry,ipcTableAttrData and dstIpcTable instances and set them to valid memory
 * dstIpcTable->ipcTableEntryArr , ipcTable->ipcTableEntryArr->ipcRoute and  ipcTable->ipcTableEntryArr->ipcRoute->ipCEndpointList set to valid memory
 * ipcTableEntry->ipcRoute set to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTableAttrData set to valid memory
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{- srcIpcTableAddr reference global ipcTable instance.
 * - dstIpcTableAddr reference global dstIpcTable instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableClone() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{22060503}
 *
 * @verify{18843063}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj.ipcRoute:<<malloc 1>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableClone.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableClone
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute[0].ipcEndpointList );
}
if ( i == 4 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> );
}
if ( i == 5 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciIpcEndpoint) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
if( i == 5 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 0 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciIpcEndpoint) * <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );

TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.srcIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr
<<lwscibuf_ipc_table.LwSciBufIpcTableClone.dstIpcTableAddr>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcTableExport

-- Test Case: TC_001.LwSciBufIpcTableExport.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcTableExport.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcTableExport.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when LwSciCommonCalloc returns NULL}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL.
 * Create ipcTable, ipcTableEntry and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciBufIpcTableExport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856329}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:64
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = (NULL);
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t) + 64) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t))  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcTableExport.Failure_due_to_global_variable_endpointCount_set_to_100
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcTableExport.Failure_due_to_global_variable_endpointCount_set_to_100
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcTableExport.Failure_due_to_global_variable_endpointCount_set_to_100}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when global variable endpointCount set to 100}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory.
 * Create ipcTable and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 100}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to false.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableExport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856332}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 100>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:100
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0x40
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardNullIpcRoutes:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.len[0]:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t) + 64) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t))  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcTableExport.Success_due_to_entryExportSize_set_to_0xFFFFFFFFFFFFFFE1
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcTableExport.Success_due_to_entryExportSize_set_to_0xFFFFFFFFFFFFFFE1
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcTableExport.Success_due_to_entryExportSize_set_to_0xFFFFFFFFFFFFFFE1}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when global variable 'entryExportSize' set to 0xFFFFFFFFFFFFFFE1}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory.
 * Create ipcTable and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 * Set ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize to 0xFFFFFFFFFFFFFFE1 }
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to true.
 * - discardNullIpcRoutes set to false.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable
 *
 * @testbehavior{LwSciBufIpcTableExport() should return LwSciError_Success.}
 *
 * @testcase{18856335}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0xFFFFFFFFFFFFFFE1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:0x0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardOuterEndpoint:true
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardNullIpcRoutes:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcTableExport.Failure_due_to_discardOuterEndpoint_set_to_true
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcTableExport.Failure_due_to_discardOuterEndpoint_set_to_true
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcTableExport.Failure_due_to_discardOuterEndpoint_set_to_true}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when discardOuterEndpoint set to true.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory.
 * Create ipcTable, ipcRoute  and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 2
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList set to allocated memory
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount set to 0
 * ipcTableAttrData[0]->ipcTableAttrData[0].value set to allocated memory}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to true.
 * - discardNullIpcRoutes set to false.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableExport() should return LwSciError_Overflow.}
 *
 * @testcase{18856338}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0x40
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:0x0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardOuterEndpoint:true
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardNullIpcRoutes:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t) + 64 - sizeof(LwSciIpcEndpoint)) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t))  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int count=0;
if(count == 0)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc ) }}
}
if(count == 1)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( ( (uint8_t *)(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc) + sizeof(uint64_t)) ) }}
}
if(count == 2)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t*)((uintptr_t)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 3)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 4)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
}
i++;



TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 0 ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 32 ) }}
}
i++;



TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value
int x = 9;
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &x );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufIpcTableExport.Failure_due_to_discardOuterEndpoint_set_to_false
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcTableExport.Failure_due_to_discardOuterEndpoint_set_to_false
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcTableExport.Failure_due_to_discardOuterEndpoint_set_to_false}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when discardOuterEndpoint set to false.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory.
 * Create ipcTable and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 2
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList set to allocated memory
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount set to 0
 * ipcTableAttrData[0]->ipcTableAttrData[0].value set to allocated memory}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to false.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableExport() should return LwSciError_Overflow.}
 *
 * @testcase{18856341}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0x40
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:0x0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardNullIpcRoutes:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t) + 64) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t))  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int count=0;
if(count == 0)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc ) }}
}
if(count == 1)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( ( (uint8_t *)(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc) + sizeof(uint64_t)) ) }}
}
if(count == 2)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t*)((uintptr_t)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 3)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 4)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
}
i++;



TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( (sizeof(uint64_t) + sizeof(LwSciIpcEndpoint)) - sizeof(uint64_t) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 32 ) }}
}
i++;



TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value
int c = 90;
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &c );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIpcTableExport.Success_due_to_ipcEndpoint_set_to_Mid_value
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_006.LwSciBufIpcTableExport.Success_due_to_ipcEndpoint_set_to_Mid_value
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIpcTableExport.Success_due_to_ipcEndpoint_set_to_Mid_value}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when ipcEndpoint is not 0}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory.
 * Create ipcTable, ipctableEntry and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 2
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 17
 * ipcTable->ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTable->ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount set to 1
 * ipcTableAttrData[0]->ipcTableAttrData[0].value set to allocated memory}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcEndpoint set to 17.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to false.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcTableExport() should return LwSciError_Success.}
 *
 * @testcase{18856344}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0x40
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value:VECTORCAST_BUFFER
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:0x11
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardNullIpcRoutes:false
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_var.entryCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_var.totalSize:120
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_var.entryStart[0].entrySize:60
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_var.entryStart[0].keyCount:1
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t) + 64) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t))  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int count=0;
if(count == 0)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc ) }}
}
if(count == 1)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( ( (uint8_t *)(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc) + sizeof(uint64_t)) ) }}
}
if(count == 2)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t*)((uintptr_t)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 3)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 4)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( (sizeof(uint64_t) + sizeof(LwSciIpcEndpoint)) - sizeof(uint64_t) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 32 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry>>[0].ipcRoute ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_var.ipcEndpointSize
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.ipcEndpointSize == ( sizeof(LwSciIpcEndpoint) ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufIpcTableExport.Success.due_to_discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_007.LwSciBufIpcTableExport.Success.due_to_discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufIpcTableExport.Success.due_to_discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when discardOuterEndpoint and discardNullIpcRoutes are set to true.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory.
 * Create ipcTable and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to true.
 * - discardNullIpcRoutes set to true.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciBufIpcTableExport() should return LwSciError_Success.}
 *
 * @testcase{18856347}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0x40
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:0x0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardOuterEndpoint:true
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardNullIpcRoutes:true
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_Success
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufIpcTableExport.Success_due_to_discardNullIpcRoutes_set_to_true_and_ipcEndpoint_set_to_Min_value
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_008.LwSciBufIpcTableExport.Success_due_to_discardNullIpcRoutes_set_to_true_and_ipcEndpoint_set_to_Min_value
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufIpcTableExport.Success_due_to_discardNullIpcRoutes_set_to_true_and_ipcEndpoint_set_to_Min_value}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when discardNullIpcRoutes set to true.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary values}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory.
 * Create ipcTable and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTableAttrData[0]->ipcTableAttrData[0].value set to allocated memory}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to true.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcTableExport() should return LwSciError_Success.}
 *
 * @testcase{18856350}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:4
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0x40
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardNullIpcRoutes:true
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t) + 64) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t))  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int count=0;
if(count == 0)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc ) }}
}
if(count == 1)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( ( (uint8_t *)(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc) + sizeof(uint64_t)) ) }}
}
if(count == 2)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t*)((uintptr_t)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 3)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 4)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( (sizeof(uint64_t) + sizeof(LwSciIpcEndpoint)) - sizeof(uint64_t) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 32 ) }}
}
i++;



TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value
int w = 999;
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &w );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufIpcTableExport.Success_due_to_discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true_and_false
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_009.LwSciBufIpcTableExport.Success_due_to_discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true_and_false
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufIpcTableExport.Success_due_to_discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true_and_false}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when discardOuterEndpoint set to true and discardNullIpcRoutes set to false.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory.
 * Create ipcTable and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTableAttrData[0]->ipcTableAttrData[0].value set to allocated memory}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - discardOuterEndpoint set to true.
 * - discardNullIpcRoutes set to false.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcTableExport() should return LwSciError_Success.}
 *
 * @testcase{18856353}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT2:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0x40
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardOuterEndpoint:true
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.discardNullIpcRoutes:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t) + 64 - sizeof(LwSciIpcEndpoint)) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t))  ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int count=0;
if(count == 0)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc ) }}
}
if(count == 1)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( ( (uint8_t *)(&<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_var>>.entryStart[0].desc) + sizeof(uint64_t)) ) }}
}
if(count == 2)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t*)((uintptr_t)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 3)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) ) }}
}
if(count == 4)
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( (uint8_t *)<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>) }}
}
count++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value ) }}
}
i++;



TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 0 ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 32 ) }}
}
i++;



TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].value
int q = 333;
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].value = ( &q );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufIpcTableExport.Panic_due_to_desc_Is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_010.LwSciBufIpcTableExport.Panic_due_to_desc_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufIpcTableExport.Panic_due_to_desc_Is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when desc is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- ipcTable set to allocated memory
 * - desc set to NULL.
 * - len set to allocated memory}
 *
 * @testbehavior{LwSciBufIpcTableExport() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{18856356}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len:<<malloc 1>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_011.LwSciBufIpcTableExport.Success_due_to_ipcEndpoint_set_to_Max_value
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_011.LwSciBufIpcTableExport.Success_due_to_ipcEndpoint_set_to_Max_value
TEST.NOTES:
/**
 * @testname{TC_011.LwSciBufIpcTableExport.Success_due_to_ipcEndpoint_set_to_Max_value}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when ipcEndpoint set to 2}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Create ipcTable and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - ipcEndpoint set to UINT64_MAX
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{Function should return LwSciError_Success.}
 *
 * @testcase{18856359}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:0x40
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExport.return:LwSciError_Success
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint>> = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> == ( <<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciBufIpcTableExport.Panic_due_to_ipcTable_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_012.LwSciBufIpcTableExport.Panic_due_to_ipcTable_is_NULL
TEST.NOTES:
/**
 * @testname{TC_012.LwSciBufIpcTableExport.Panic_due_to_ipcTable_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when ipcTable is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create ipcTable, ipcTableEntry and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64}
 *
 * @testinput{- ipcTable set to NULL
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciBufIpcTableExport() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{18856362}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:64
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciBufIpcTableExport.Panic_due_to_entryCount_set_to_Zero
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_013.LwSciBufIpcTableExport.Panic_due_to_entryCount_set_to_Zero
TEST.NOTES:
/**
 * @testname{TC_013.LwSciBufIpcTableExport.Panic_due_to_entryCount_set_to_Zero}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() if no valid entries in table.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create ipcTable, ipcTableEntry and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 0
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1}
 *
 * @testinput{- ipcTable reference ipcTable global instance.
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciBufIpcTableExport() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{18856365}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:64
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.len>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.destBufSize>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciBufIpcTableExport.Panic_due_to_len_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExport
TEST.NEW
TEST.NAME:TC_014.LwSciBufIpcTableExport.Panic_due_to_len_is_NULL
TEST.NOTES:
/**
 * @testname{TC_012.LwSciBufIpcTableExport.Panic_due_to_ipcTable_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExport() when ipcTable is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create ipcTable, ipcTableEntry and ipcTableAttrData global instances and set them to valid memory.
 * Set ipcTable->ipcTableEntry and ipcTable->ipcTableEntry->ipcEndpointList to valid memory
 * Create cyclic list by binding ipcAttrEntryHead with ipcTableAttrData global variable having a single entry with key equal to 17 and len equal to 32.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].validEntryCount set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0] set to 1
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount set to 1.
 * ipcTable->ipcTable[0].ipcTableEntryArr[0].entryExportSize set to 64}
 *
 * @testinput{- ipcTable set to NULL
 * - desc reference pDestBuf global instance
 * - len reference uint64DestBufSize global variable}
 *
 * @testbehavior{LwSciBufIpcTableExport() API should terminate by calling LwSciCommonPanic().}
 *
 * @testcase{22060505}
 *
 * @verify{18843075}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:64
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].key:17
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData[0].len:32
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcTable:<<malloc 1>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExport.len:<<null>>
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].ipcAttrEntryHead.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.next
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.next = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData.ipcTableAttrData[0].listEntry.prev
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].listEntry.prev = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].ipcAttrEntryHead);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableExport.desc>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.pDestBuf>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcTableExportSize

-- Test Case: TC_001.LwSciBufIpcTableExportSize.globalVar_endpointCount_and_ipcEndPoint_is_equal_to_Zero
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcTableExportSize.globalVar_endpointCount_and_ipcEndPoint_is_equal_to_Zero
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcTableExportSize.globalVar_endpointCount_is_equal_to_Zero}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when global variable ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount is equal to 0.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Add single entry to global ipcTableEntryArr with endpointCount,allocEntryCount and  validEntryCount to 0, 1 and 1
 * Add single entry to global ipcTableEntryArr array with ipcTableEntryArr[0].ipcRoute->ipcEndpointList[0] set to 1 .}
 *
 * @testinput{- ipcTable set to global ipcTable instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to false.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() API should return (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t))}
 *
 * @testcase{18856368}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint:0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:false
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) ) }}

TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcTableExportSize.globalVar_endpointCount_is_equal_to_one
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcTableExportSize.globalVar_endpointCount_is_equal_to_one
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcTableExportSize.globalVar_endpointCount_is_equal_to_one}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when global variable ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount is equal to 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Add single entry to global ipcTableEntryArr with endpointCount,allocEntryCount and  validEntryCount to 1, 1 and 1
 * Add single entry to global ipcTableEntryArr array with ipcTableEntryArr[0].ipcRoute->ipcEndpointList[0] set to 1 .}
 *
 * @testinput{- ipcTable set to global ipcTable instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to false.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() API should return (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t))}
 *
 * @testcase{18856371}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint:0x0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) + (sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t)) ) }}



TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcTableExportSize.ipcEndpoint_is_not_Zero
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcTableExportSize.ipcEndpoint_is_not_Zero
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcTableExportSize.globalVar_endpointCount_and_ipcEndPoint_is_equal_to_Zero}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when input ipcEndpoint is not equal to 0.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Add single entry to global ipcTableEntryArr with endpointCount,allocEntryCount and  validEntryCount to 1, 1 and 1
 * Add single entry to global ipcTableEntryArr array with ipcTableEntryArr[0].ipcRoute->ipcEndpointList[0] set to 1 .}
 *
 * @testinput{- ipcTable set to global ipcTable instance.
 * - ipcEndpoint set to 2.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to false.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() API should return (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t))}
 *
 * @testcase{18856374}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint:2
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) ) }}

TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when discardOuterEndpoint is set to true}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Add single entry to global ipcTableEntryArr with endpointCount,allocEntryCount and  validEntryCount to 1, 1 and 1
 * Add single entry to global ipcTableEntryArr array with ipcTableEntryArr[0].ipcRoute->ipcEndpointList[0] set to 1 .}
 *
 * @testinput{- ipcTable set to global ipcTable instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to true.
 * - discardNullIpcRoutes set to false.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() API should return (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) + ((sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t)) - sizeof(LwSciIpcEndpoint))}
 *
 * @testcase{18856377}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint:0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:true
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) + ((sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(uint8_t)) - sizeof(LwSciIpcEndpoint)) }}
//{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == ( sizeof(LwSciBufIpcTableExportHeader) + sizeof(LwSciBufIpcTableEntryExportHeader) - sizeof(LwSciIpcEndpoint) + <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].entryExportSize ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true_and_globalVar_endpointCount_is_zero
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true_and_globalVar_endpointCount_is_zero
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true_and_globalVar_endpointCount_is_zero}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when discardOuterEndpoint set to true and global variable ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount is equal to 0.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Set global variable endpointCount,allocEntryCount and  validEntryCount to 0, 1 and 1
 * Set global variable entryExportSize to 0
 * Add single entry to ipcTableEntryArr array with allocEntryCount and validEntryCount set to 1.}
 *
 * @testinput{ - ipcTable set to global ipcTable instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to true.
 * - discardNullIpcRoutes set to false.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() API should return (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t))}
 *
 * @testcase{18856380}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint:0x0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:true
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIpcTableExportSize.discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_006.LwSciBufIpcTableExportSize.discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIpcTableExportSize.discardOuterEndpoint_and_discardNullIpcRoutes_set_to_true}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when discardOuterEndpoint and discardNullIpcRoutes set to true}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Add single entry to global ipcTableEntryArr with endpointCount,allocEntryCount and validEntryCount to 1, 1 and 1
 * Add single entry to global ipcTableEntryArr array with ipcTableEntryArr[0].ipcRoute->ipcEndpointList[0] set to 1 .}
 *
 * @testinput{- ipcTable set to global ipcTable instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to true.
 * - discardNullIpcRoutes set to true.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() API should return sizeof(LwSciBufIpcTableExportHeader).}
 *
 * @testcase{18856383}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint:0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:true
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:true
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true_and_entryExportSize_globalVar_set_to_MaxValue
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_007.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true_and_entryExportSize_globalVar_set_to_MaxValue
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufIpcTableExportSize.discardOuterEndpoint_set_to_true_and_entryExportSize_globalVar_set_to_MaxValue}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when discardOuterEndpoint set to true  global variable ipcTable[0].ipcTableEntryArr[0].entryExportSize is 0xFFFFFFFFFFFFFFFF}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Add single entry to global ipcTableEntryArr with endpointCount,allocEntryCount and validEntryCount to 1, 1 and 1
 * Add single entry to global ipcTableEntryArr array with ipcTableEntryArr[0].ipcRoute->ipcEndpointList[0] set to 1 .}
 * Set global variable entryExportSize to UINT64_MAX}
 *
 * @testinput{- ipcTable set to global ipcTable instance.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to true.
 * - discardNullIpcRoutes set to false.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() should return (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t))}
 *
 * @testcase{18856386}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint:0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:true
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable.ipcTable[0].ipcTableEntryArr.ipcTableEntryArr[0].entryExportSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0].entryExportSize = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufIpcTableExportSize.ipcTable_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_008.LwSciBufIpcTableExportSize.ipcTable_is_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufIpcTableExportSize.ipcTable_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when ipcEndpoint set to NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Add single entry to global ipcTableEntryArr with endpointCount,allocEntryCount and  validEntryCount to 1, 1 and 1
 * Add single entry to global ipcTableEntryArr array with ipcTableEntryArr[0].ipcRoute->ipcEndpointList[0] set to 1 .}
 *
 * @testinput{- ipcTable set to NULL.
 * - ipcEndpoint set to 0.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to false.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() API shoud return 0}
 *
 * @testcase{18856389}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint:0
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return:0
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciBufIpcTableExportSize.ipcEndpoint_set_to_Max
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableExportSize
TEST.NEW
TEST.NAME:TC_009.LwSciBufIpcTableExportSize.ipcEndpoint_set_to_Max
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufIpcTableExportSize.ipcEndpoint_set_to_Max}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableExportSize() when input ipcEndpoint is not equal to UINT64_MAX}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Set global ipcTable instance with ipcTableEntryArr, ipcRoute and ipcEndpointList set to valid memory
 * Add single entry to global ipcTableEntryArr with endpointCount,allocEntryCount and validEntryCount to 1, 1 and 1
 * Add single entry to global ipcTableEntryArr array with ipcTableEntryArr[0].ipcRoute->ipcEndpointList[0] set to 1 .}
 *
 * @testinput{- ipcTable set to global ipcTable instance.
 * - ipcEndpoint set to UINT64_MAX.
 * - discardOuterEndpoint set to false.
 * - discardNullIpcRoutes set to false.}
 *
 * @testbehavior{LwSciBufIpcTableExportSize() API should return (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t))}
 *
 * @testcase{22060509}
 *
 * @verify{18843078}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].allocEntryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:1
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardOuterEndpoint:false
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.discardNullIpcRoutes:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
  lwscibuf_ipc_table.c.LwSciBufIpcTableExportSize
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcTable>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint
<<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.ipcEndpoint>> = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableExportSize.return>> == ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) ) }}

TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcTableImport

-- Test Case: TC_001.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when LwSciCommonCalloc returns NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL.
 * Create global ipcTable instance and point it to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.ipcEndpointSize and tableEntryExportHeader->keyCount to 8,1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciBufIpcTableImport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856392}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_second_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when LwSciCommonCalloc() return NULL on second invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on first invocation and NULL on second invocation.
 * Create global ipcTable, ipcTableEntry and dstIpcTable instances and point them to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableImport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856395}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_third_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when LwSciCommonCalloc() return NULL on third invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on first and second invocation and NULL on third invocation.
 * Create global ipcTable and ipcRoute instances and point them to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableImport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856398}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.size:1
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>  ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fourth_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when LwSciCommonCalloc return NULL on fourth invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on first, second and third invocation and NULL on fourth invocation.
 * Create global ipcTable and ipcRoute instances and point them to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{ - desc reference global tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciCommonFree() receives correct input arguments
 * LwSciBufIpcTableImport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856401}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_InsufficientMemory
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(*(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>)) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRoute) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 0 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>>  ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}
}
if ( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart[0].desc ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fifth_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fifth_ilwocation
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_fifth_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when LwSciCommonCalloc returns NULL on fifth invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on first, second, third and fourth invocation and NULL on fifth invocation.
 * Create global ipcTable , ipcTableEntry and ipcRoute instances and point them to valid memory
 * ipcTable->ipcTableEntryArr and ipcTableEntry->ipcRoute  points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcTableImport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856404}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_InsufficientMemory
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute[0].ipcEndpointList );
}
if ( i == 4 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>[0]) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0]) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRoute) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 0 ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute[0].ipcEndpointList ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> ) }}
}
if ( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}
}
if ( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (  &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>>  ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart[0].desc ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_Sixth_ilwocation
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_006.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_Sixth_ilwocation
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIpcTableImport.Failure_due_to_LwSciCommonCalloc_returns_NULL_on_Sixth_ilwocation}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when LwSciCommonCalloc return NULL on sixth invocation.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on first,second ,third, fourth, fifth invocation and NULL on Sixth invocation.
 * Create global ipcTable , ipcTableAttrData , ipcTableEntry and ipcRoute instances and point them to valid memory
 * ipcTable->ipcTableEntryArr and ipcTableEntry->ipcRoute  points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance..}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonFree() receives correct input arguments
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.
 * LwSciBufIpcTableImport() should return LwSciError_InsufficientMemory.}
 *
 * @testcase{18856407}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_InsufficientMemory
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute[0].ipcEndpointList );
}
if ( i == 4 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> );
}
if ( i == 5 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>[0]) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0]) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRoute) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
if( i == 5 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 0 ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 0 ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 5 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> ) }}
}
if ( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> ) }}
}
if ( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> ) }}
}
if ( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> = ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute[0].ipcEndpointList ) }}
}

i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.key ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.len ) }}
}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart[0].desc ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart[0].desc + sizeof(uint64_t) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( (const uint8_t *)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart[0].desc + sizeof(uint64_t))  + 8 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].key) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData>>[0].len) ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufIpcTableImport.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_007.LwSciBufIpcTableImport.Success_due_to_LwSciCommonCalloc_returns_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufIpcTableImport.Success_due_to_LwSciCommonCalloc_returns_allocated_memory}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when LwSciCommonCalloc return allocated memory for all ilwocations}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on all ilwocations
 * Create global ipcTable , ipcTableAttrData , ipcTableEntry and ipcRoute instances and point them to valid memory
 * ipcTable->ipcTableEntryArr and ipcTableEntry->ipcRoute  points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciCommonMemcpyS() is called to copy 'n' number of bytes from 'src' pointer address to 'dest' pointer address.}
 * LwSciBufIpcTableImport() should return LwSciError_Success.}
 *
 * @testcase{18856410}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj.value:VECTORCAST_BUFFER
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].allocEntryCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj[0].validEntryCount:1
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj.endpointCount:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_Success
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcTableExport.ipcEndpoint:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
if ( i == 2 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRouteObj>> );
}
if ( i == 3 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>>.ipcRoute[0].ipcEndpointList );
}
if ( i == 4 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>> );
}
if ( i == 5 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.value );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
memcpy(<<uut_prototype_stubs.LwSciCommonMemcpyS.dest>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.src>>, <<uut_prototype_stubs.LwSciCommonMemcpyS.n>>);

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>[0]) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute>>[0]) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcRoute) ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableAttrData) ) }}
}
if( i == 5 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 0 ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 0 ) }}
}
if( i == 4 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
if( i == 5 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.key ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.len ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrDataObj>>.value ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart[0].desc ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart[0].desc + sizeof(uint64_t) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( (const uint8_t *)(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart[0].desc + sizeof(uint64_t))  + 8 ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> != ( (const uint8_t *)(<<uut_prototype_stubs.LwSciCommonMemcpyS.src>>) + 16 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint32_t) ) }}
}
if( i == 2 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(uint64_t) ) }}
}
if( i == 3 )
{
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( 0 ) }}
}
i++;
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = ( (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>) );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufIpcTableImport.Failure_due_to_desc_is_null
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_008.LwSciBufIpcTableImport.Failure_due_to_desc_is_null
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufIpcTableImport.Failure_due_to_desc_is_null}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when desc is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable instance and point it to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc set to NULL.
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciBufIpcTableImport() returns LwSciError_BadParameter}
 *
 * @testcase{18856413}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc:<<null>>
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufIpcTableImport.NotSupported
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_009.LwSciBufIpcTableImport.NotSupported
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufIpcTableImport.NotSupported}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when entryCount in buffer being imported is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable instance and point it to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 0, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc points to tableExport global structure having entryCount equal to 0.
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciBufIpcTableImport() should return LwSciError_NotSupported}
 *
 * @testcase{18856416}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_NotSupported
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.totalSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.totalSize = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufIpcTableImport.Failure_due_to_ipcEndpointSize_is_large
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_010.LwSciBufIpcTableImport.Failure_due_to_ipcEndpointSize_is_large
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufIpcTableImport.Failure_due_to_ipcEndpointSize_is_large}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when tableExport.ipcEndpointSize set to 16}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Create global ipcTable instance and point it to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 16
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport union with member tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciBufIpcTableImport() should return LwSciError_NotSupported.}
 *
 * @testcase{18856419}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:16
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_NotSupported
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciBufIpcTableImport.Failure_due_to_ipcTable_is_null
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_011.LwSciBufIpcTableImport.Failure_due_to_ipcTable_is_null
TEST.NOTES:
/**
 * @testname{TC_011.LwSciBufIpcTableImport.Panic_due_to_ipcTable_is_null}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when ipcTable is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable instance and point it to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len reference size of tableExport union
 * - ipcTable set to NULL.}
 *
 * @testbehavior{LwSciBufIpcTableImport() returns LwSciError_BadParameter}
 *
 * @testcase{18856422}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_BadParameter
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciBufIpcTableImport.Failure_due_to_len_set_to_less_than_8
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_012.LwSciBufIpcTableImport.Failure_due_to_len_set_to_less_than_8
TEST.NOTES:
/**
 * @testname{TC_012.LwSciBufIpcTableImport.Failure_due_to_len_set_to_less_than_8}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when len is lower than the size in the descriptor header.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{Create global ipcTable instance and point it to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize and tableExport.arr to 1, 8 and 100
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len set to 7
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciBufIpcTableImport() returns LwSciError_BadParameter}
 *
 * @testcase{18856425}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.totalSize:100
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len:7
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_BadParameter
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciBufIpcTableImport.NotSupported_when_global_entryCount_is_0
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_013.LwSciBufIpcTableImport.NotSupported_when_global_entryCount_is_0
TEST.NOTES:
/**
 * @testname{TC_013.LwSciBufIpcTableImport.NotSupported_when_global_entryCount_is_0}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when entryCount in buffer being imported is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable instance and point it to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize and totalSize to 0, 8, 3
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc points to tableExport global structure having entryCount equal to 0.
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciBufIpcTableImport() should return LwSciError_NotSupported}
 *
 * @testcase{22060512}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.totalSize:3
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_NotSupported
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciBufIpcTableImport.NotSupported_when_global_ipcEndpoint_is_not_8
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_014.LwSciBufIpcTableImport.NotSupported_when_global_ipcEndpoint_is_not_8
TEST.NOTES:
/**
 * @testname{TC_014.LwSciBufIpcTableImport.NotSupported_when_global_ipcEndpoint_is_not_8}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when entryCount in buffer being imported is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Create global ipcTable instance and point it to valid memory
 * ipcTable->ipcTableEntryArr points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize and totalSize to 5, 2, 3
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc points to tableExport global structure having entryCount equal to 0.
 * - len reference size of tableExport union
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciBufIpcTableImport() should return LwSciError_NotSupported}
 *
 * @testcase{22060515}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:5
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.totalSize:3
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_NotSupported
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = (sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>));
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_015.LwSciBufIpcTableImport.OverFlow_if_computing_operation_exceeds_max_dataType
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableImport
TEST.NEW
TEST.NAME:TC_015.LwSciBufIpcTableImport.OverFlow_if_computing_operation_exceeds_max_dataType
TEST.NOTES:
/**
 * @testname{TC_015.LwSciBufIpcTableImport.OverFlow_if_computing_operation_exceeds_max_dataType}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableImport() when computing operation exceeds MAX dataType}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary values}
 *
 * @testsetup{LwSciCommonCalloc() returns allocated memory on first,second ilwocations.
 * Create global ipcTable , ipcTableAttrData , ipcTableEntry and ipcRoute instances and point them to valid memory
 * ipcTable->ipcTableEntryArr and ipcTableEntry->ipcRoute  points to valid memory
 * Set union members tableExport.entryCount, tableExport.ipcEndpointSize to 1, 8
 * Set uint64DestBufSize to sizeof(uint64_t) * 10.
 * Set global tableEntryExportHeader to address of tableExportHeader.entryStart and tableEntryExportHeader->keyCount to 1}
 *
 * @testinput{- desc reference global tableExport.arr
 * - len set to (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t))[OverFlow condition]
 * - ipcTable reference ipcTable global instance.}
 *
 * @testbehavior{LwSciCommonCalloc() is called to allocate memory for an array of 'n' elements with each of the element 'size' number of bytes.
 * LwSciBufIpcTableImport() should return LwSciError_Overflow.}
 *
 * @testcase{22060519}
 *
 * @verify{18843069}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].validEntryCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableAttrData:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf:<<malloc 10>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj.ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.entryCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport.tableExportHeader.ipcEndpointSize:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonMemcpyS.dest:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableImport.return:LwSciError_Overflow
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_ipc_table.c.LwSciBufIpcTableImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntryObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableObj>> );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(LwSciBufIpcTableEntry) ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( sizeof(<<USER_GLOBALS_VCAST.<<GLOBAL>>.dstIpcTable>>[0]) ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 1 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize
<<USER_GLOBALS_VCAST.<<GLOBAL>>.uint64DestBufSize>> = ( sizeof(uint64_t) * 10 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>> = (&<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.tableExportHeader.entryStart);
<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableEntryExportHeader>>->keyCount = 1;
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.desc
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.desc>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.tableExport>>.arr);
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.len
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.len>> = ( (sizeof(LwSciBufIpcTableExportHeader) - sizeof(uint8_t)) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
<<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> = ( (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>) );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable
{{ <<lwscibuf_ipc_table.LwSciBufIpcTableImport.ipcTable>> == ( <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciBufIpcTableIterNext

-- Test Case: TC_001.LwSciBufIpcTableIterNext.false_due_to_validEntryCount_set_to_three
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_001.LwSciBufIpcTableIterNext.false_due_to_validEntryCount_set_to_three
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufIpcTableIterNext.false_due_to_validEntryCount_set_to_three}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when global variable 'validEntryCount' of ipcTable set to 3}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute set to valid memory
 * Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * Initialize members of ipcTableEntryArr[0], ipcTableEntryArr[1] and ipcTableEntryArr[2]
 * Set allocEntryCount and validEntryCount of ipcTableEntryArr with 3
 * Set ipcIter->ipcIter[0].ipcTable->ipcEndpoint to 1
 * Set ipcIter->ipcIter[0].ipcTable->routeIsEndpointOnly to false}
 *
 * @testinput{- ipcIter reference global ipcIter instance.}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return false.}
 *
 * @testcase{18856428}
 *
 * @verify{18843087}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcEndpoint:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].routeIsEndpointOnly:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:false
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufIpcTableIterNext.false_due_to_validEntryCount_set_to_Zero
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_002.LwSciBufIpcTableIterNext.false_due_to_validEntryCount_set_to_Zero
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufIpcTableIterNext.false_due_to_validEntryCount_set_to_Zero}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when global variable 'validEntryCount' of ipcTable set to 0}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute set to valid memory
 * Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * Initialize members of ipcTableEntryArr[0], ipcTableEntryArr[1] and ipcTableEntryArr[2]
 * Set ipcTableEntryArr[0] entry with endpointCount to value 2
 * Set ipcTableEntryArr[1] entry with endpointCount equal to 0
 * Set ipcTableEntryArr[0] entry with endpointCount equal to 2
 * Set ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] with 1
 * Set ipcEndpointList[1] and ipcEndpointList[1] of ipcTableEntryArr[0] with 1
 * Set ipcEndpointList[2] and ipcEndpointList[1] of ipcTableEntryArr[0] with 1
 * Set allocEntryCount and validEntryCount of ipcTableEntryArr with 3 and 0
 * Set ipcIter->ipcIter[0].ipcTable->ipcEndpoint to 1
 * Set ipcIter->ipcIter[0].ipcTable->routeIsEndpointOnly to false}
 *
 * @testinput{- ipcIter reference global ipcTable instance.}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return false.}
 *
 * @testcase{18856431}
 *
 * @verify{18843087}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].validEntryCount:0
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcEndpoint:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].routeIsEndpointOnly:false
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:false
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable[0].ipcTableEntryArr[0].entryExportSize:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_UINT32_MAX
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_003.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_UINT32_MAX
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_UINT32_MAX}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when global variable lwrrMatchEntryIdx is equal to UNIT32_MAX.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute set to valid memory
 * Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * Initialize members of ipcTableEntryArr[0], ipcTableEntryArr[1].
 * Set ipcTableEntryArr[0] entry with endpointCount to 1
 * Set ipcTableEntryArr[1] entry with endpointCount equal to 1
 * Set allocEntryCount and validEntryCount of ipcTable with 2
 * Set ipcIter->ipcIter[0].ipcTable->ipcEndpoint to 1
 * Set ipcIter->ipcIter[0].ipcTable->lwrrMatchEntryIdx to UINT32_MAX}
 *
 * @testinput{- ipcIter reference global ipcTable instance.}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return false.}
 *
 * @testcase{18856434}
 *
 * @verify{18843087}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcEndpoint:1
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:false
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].lwrrMatchEntryIdx
<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].lwrrMatchEntryIdx = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufIpcTableIterNext.true_due_to_ipcEndpoint_is_0
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_004.LwSciBufIpcTableIterNext.true_due_to_ipcEndpoint_is_0
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufIpcTableIterNext.true_due_to_ipcEndpoint_is_0}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when global variable ipcEndPoint set to 0}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute set to valid memory
 * Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * Initialize members of ipcTableEntryArr[0], ipcTableEntryArr[1].
 * Set ipcTableEntryArr[0] entry with endpointCount to 1
 * Set ipcTableEntryArr[1] entry with endpointCount equal to 1
 * Set allocEntryCount and validEntryCount of ipcTable with 2
 * Set ipcIter->ipcIter[0].ipcTable->ipcEndpoint to 0}
 *
 * @testinput{- ipcIter reference global ipcIter instance.}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return true.}
 *
 * @testcase{18856437}
 *
 * @verify{18843087}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcEndpoint:0
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].lwrrMatchEntryIdx:1
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:true
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTableEntry>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcTable>>[0].ipcTableEntryArr[0] ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_Zero
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_005.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_Zero
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_Zero}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when global variable lwrrMatchEntryIdx set to 0}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute set to valid memory
 * Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * ipcIter->ipcIter[0].lwrrMatchEntryIdx set to 0
 * Initialize members of ipcTableEntryArr[0], ipcTableEntryArr[1].
 * Set ipcTableEntryArr[0] entry with endpointCount to 1
 * Set ipcTableEntryArr[1] entry with endpointCount equal to 1
 * Set allocEntryCount and validEntryCount of ipcTable with 2
 * Set ipcIter->ipcIter[0].ipcTable->ipcEndpoint to 0}
 *
 * @testinput{- ipcIter reference global ipcIter instance}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return false.}
 *
 * @testcase{18856440}
 *
 * @verify{18843087}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcEndpoint:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].lwrrMatchEntryIdx:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_two
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_006.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_two
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufIpcTableIterNext.false_due_to_lwrrMatchEntryIdx_set_to_two}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when global variable lwrrMatchEntryIdx is equal to 2.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute set to valid memory
 * Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * Initialize members of ipcTableEntryArr[0], ipcTableEntryArr[1].
 * Set ipcTableEntryArr[0] entry with endpointCount to 1
 * Set ipcTableEntryArr[1] entry with endpointCount equal to 1
 * Set allocEntryCount and validEntryCount of ipcTableEntryArray with 2
 * Set ipcIter->ipcIter[0].ipcTable->ipcEndpoint to 1
 * Set ipcIter->ipcIter[0].ipcTable->lwrrMatchEntryIdx to 2.
 * Set ipcIter->ipcIter[0].ipcTable->routeIsEndpointOnly to false}
 *
 * @testinput{- ipcIter reference global ipcIter instance}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return false.}
 *
 * @testcase{18856443}
 *
 * @verify{18843087}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcEndpoint:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].routeIsEndpointOnly:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].lwrrMatchEntryIdx:2
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:false
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufIpcTableIterNext.false_due_to_routeIsEndpointOnly_set_to_true
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_007.LwSciBufIpcTableIterNext.false_due_to_routeIsEndpointOnly_set_to_true
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufIpcTableIterNext.false_due_to_routeIsEndpointOnly_set_to_true}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when global variable 'routeIsEndpointOnly' set to true}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute set to valid memory
 * Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * Initialize members of ipcTableEntryArr[0], ipcTableEntryArr[1].
 * Set ipcTableEntryArr[0] entry with endpointCount to value 2
 * Set ipcTableEntryArr[1] entry with endpointCount equal to 2
 * Set allocEntryCount and validEntryCount of ipcTableEntryArr with 2
 * Set ipcIter->ipcIter[0].ipcTable->ipcEndpoint to 1
 * Set ipcIter->ipcIter[0].ipcTable->routeIsEndpointOnly to true}
 *
 * @testinput{- ipcIter reference global ipcIter instance.}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return false.
 * ipcIter->ipcIter[0].ipcTable->lwrrMatchEntryIdx should be equal to UINT32_MAX}
 *
 * @testcase{18856446}
 *
 * @verify{18843087}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].allocEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].validEntryCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcEndpoint:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].routeIsEndpointOnly:true
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:false
TEST.ATTRIBUTES:lwscibuf_ipc_table.LwSciBufIpcIterLwrrGetAttrKey.attrKey:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter.ipcIter[0].lwrrMatchEntryIdx
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>[0].lwrrMatchEntryIdx == ( UINT32_MAX ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufIpcTableIterNext.true_due_to_lwrrMatchEntryIdx_set_to_Zero
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_008.LwSciBufIpcTableIterNext.true_due_to_lwrrMatchEntryIdx_set_to_Zero
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufIpcTableIterNext.true_due_to_lwrrMatchEntryIdx_set_to_Zero}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when when global variable lwrrMatchEntryIdx is equal to 0}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute set to valid memory
 * Set ipcIter->ipcTable->ipcTableEntryArr->ipcRoute->ipcEndpointList set to valid memory
 * ipcIter->ipcTable and ipcIter->ipcTableEntryArr set to valid memory
 * Initialize members of ipcTableEntryArr[0], ipcTableEntryArr[1].
 * Set ipcTableEntryArr[0] entry with endpointCount to 2
 * Set ipcTableEntryArr[1] entry with endpointCount equal to 2
 * Set ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[0] with 1
 * Set ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[1] with 1
 * Set ipcEndpointList[0] and ipcEndpointList[1] of ipcTableEntryArr[2] with 1
 * Set allocEntryCount and validEntryCount of ipcTableEntryArr with 3
 * Set ipcIter->ipcIter[0].ipcTable->ipcEndpoint to 1
 * Set Set ipcIter->ipcIter[0].ipcTable->lwrrMatchEntryIdx to 0
 * Set ipcIter->ipcIter[0].ipcTable->routeIsEndpointOnly to false}
 *
 * @testinput{- ipcIter reference global ipcIter instance.}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return true.}
 *
 * @testcase{18856449}
 *
 * @verify{18843087}
 */

TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr:<<malloc 3>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[0].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[1].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute:<<malloc 1>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList:<<malloc 2>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[0]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].ipcEndpointList[1]:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].ipcTableEntryArr[2].ipcRoute[0].endpointCount:2
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].allocEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcTable[0].validEntryCount:3
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].ipcEndpoint:1
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].routeIsEndpointOnly:false
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter[0].lwrrMatchEntryIdx:0
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:true
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter
<<lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter>> = (<<USER_GLOBALS_VCAST.<<GLOBAL>>.ipcIter>>);
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufIpcTableIterNext.false_due_to_ipcIter_is_NULL
TEST.UNIT:lwscibuf_ipc_table
TEST.SUBPROGRAM:LwSciBufIpcTableIterNext
TEST.NEW
TEST.NAME:TC_009.LwSciBufIpcTableIterNext.false_due_to_ipcIter_is_NULL
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufIpcTableIterNext.false_due_to_ipcIter_is_NULL}
 *
 * @verifyFunction{This test case verifies functionality of API LwSciBufIpcTableIterNext() when ipcIter set to NULL
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{Not needed.}
 *
 * @testinput{- ipcIter set to NULL.}
 *
 * @testbehavior{LwSciBufIpcTableIterNext() should return false}
 *
 * @testcase{18856452}
 *
 * @verify{18843087}
 */

TEST.END_NOTES:
TEST.VALUE:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.ipcIter:<<null>>
TEST.EXPECTED:lwscibuf_ipc_table.LwSciBufIpcTableIterNext.return:false
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[0]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[1]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[2]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[3]:INPUT_BASE=16
TEST.ATTRIBUTES:USER_GLOBALS_VCAST.<<GLOBAL>>.puint64DestBuf[4]:INPUT_BASE=16
TEST.FLOW
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
  lwscibuf_ipc_table.c.LwSciBufIpcTableIterNext
TEST.END_FLOW
TEST.END

