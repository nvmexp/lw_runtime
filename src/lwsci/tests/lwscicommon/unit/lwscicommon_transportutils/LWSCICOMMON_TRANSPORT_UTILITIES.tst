-- VectorCAST 19.sp3 (11/13/19)
-- Test Case Script
--
-- Environment    : LWSCICOMMON_TRANSPORT_UTILITIES
-- Unit(s) Under Test: lwscicommon_transportutils
--
-- Script Features
TEST.SCRIPT_FEATURE:C_DIRECT_ARRAY_INDEXING
TEST.SCRIPT_FEATURE:CPP_CLASS_OBJECT_REVISION
TEST.SCRIPT_FEATURE:MULTIPLE_UUT_SUPPORT
TEST.SCRIPT_FEATURE:MIXED_CASE_NAMES
TEST.SCRIPT_FEATURE:STATIC_HEADER_FUNCS_IN_UUTS
TEST.SCRIPT_FEATURE:VCAST_MAIN_NOT_RENAMED
--

-- Unit: lwscicommon_transportutils

-- Subprogram: LwSciCommonTransportAllocTxBufferForKeys

-- Test Case: TC_001.LwSciCommonTransportAllocTxBufferForKeys.Success_use_case
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAllocTxBufferForKeys
TEST.NEW
TEST.NAME:TC_001.LwSciCommonTransportAllocTxBufferForKeys.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonTransportAllocTxBufferForKeys.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAllocTxBufferForKeys() - Success case when creates a new LwSciCommonTransportBuf successfuly}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns memory to 'paramLwSciCommonTransportRec'.
 * LwSciCommonCalloc() returns memory to 'paramLwSciCommonTransportRec.bufPtr'}
 *
 * @testinput{'bufParams.keyCount' driven as '1'.
 * 'bufParams.msgVersion' driven as '1'.
 * 'bufParams.msgMagic' driven as '1'.
 * 'totalValueSize' driven as '10'.
 * 'txbuf' set to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as 'sizeof(LwSciCommonTransportBufPriv)'.
 * LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as '(sizeof(LwSciCommonTransportHeader) - sizeof(uint8_t))  + (10 * (sizeof(LwSciCommonTransportKey) - sizeof(uint8_t))) + 10)'.
 * 'txbuf' pointing to valid address.
 * returns LwSciError_Success.}
 *
 * @testcase{18859746}
 *
 * @verify{18851142}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.msgVersion:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.msgMagic:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:10
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:10
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf[0]:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>>);
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>>.bufPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr=0;
cntr++;
if(cntr==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciCommonTransportBufPriv) ) }}
else if(cntr==2)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ((sizeof(LwSciCommonTransportHeader) - sizeof(uint8_t))  + (10 * (sizeof(LwSciCommonTransportKey) - sizeof(uint8_t))) + 10) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ *<<lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> == ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>>) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Transportbuffer_BufPtr_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAllocTxBufferForKeys
TEST.NEW
TEST.NAME:TC_002.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Transportbuffer_BufPtr_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Transportbuffer_BufPtr_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAllocTxBufferForKeys() - Failure due to allocation for transport buffer data failed.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{LwSciCommonCalloc() returns memory to 'paramLwSciCommonTransportRec'.
 * LwSciCommonCalloc() returns NULL.'}
 *
 * @testinput{'bufParams.keyCount' driven as 'UINT32_MAX'.
 * 'bufParams.msgVersion' driven as '1'.
 * 'bufParams.msgMagic' driven as 'UINT64_MAX'.
 * 'totalValueSize' driven as '10'.
 * 'txbuf' set to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as 'sizeof(LwSciCommonTransportBufPriv)'.
 * LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as  '(sizeof(LwSciCommonTransportHeader) - sizeof(uint8_t)) + (UINT32_MAX* sizeof(LwSciCommonTransportKey) - sizeof(uint8_t))+UINT64_MAX'.
 * LwSciCommonFree() receives 'ptr' as valid address.
 * 'txbuf' pointing to valid address.
 * returns LwSciError_InsufficientMemory.}
 *
 * @testcase{18859749}
 *
 * @verify{18851142}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.msgVersion:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.msgMagic:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:<<MAX>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:<<MAX>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf[0]:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Unknown
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>>);
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr=0;
cntr++;
if(cntr==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciCommonTransportBufPriv) ) }}
else if(cntr==2)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ((sizeof(LwSciCommonTransportHeader) - sizeof(uint8_t))  + (UINT32_MAX * (sizeof(LwSciCommonTransportKey) - sizeof(uint8_t))) + UINT64_MAX) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Transportbuffer_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAllocTxBufferForKeys
TEST.NEW
TEST.NAME:TC_003.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Transportbuffer_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Transportbuffer_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAllocTxBufferForKeys() - Failure due to allocation for transport buffer object failed.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{ LwSciCommonCalloc() returns 'NULL'.}
 *
 * @testinput{'bufParams.keyCount' driven as '1'.
 * 'bufParams.msgVersion' driven as '1'.
 * 'bufParams.msgMagic' driven as '1'.
 * 'totalValueSize' driven as '1'.
 * 'txbuf' set to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as 'sizeof(LwSciCommonTransportBufPriv)'.
 * returns LwSciError_InsufficientMemory.}
 *
 * @testcase{18859752}
 *
 * @verify{18851142}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.msgVersion:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.msgMagic:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf[0]:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Unknown
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonCalloc
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr=0;
cntr++;
if(cntr==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciCommonTransportBufPriv) ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_KeyCount_Is_0
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAllocTxBufferForKeys
TEST.NEW
TEST.NAME:TC_004.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_KeyCount_Is_0
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_KeyCount_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAllocTxBufferForKeys() - Failure due to bufParams.keyCount is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'bufParams.keyCount' driven as '0'.
 * 'totalValueSize' driven as '1'.
 * 'txbuf' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859755}
 *
 * @verify{18851142}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:0
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_TotalValueSize_Is_0
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAllocTxBufferForKeys
TEST.NEW
TEST.NAME:TC_005.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_TotalValueSize_Is_0
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_TotalValueSize_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAllocTxBufferForKeys() - Failure due to totalValueSize is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'bufParams.keyCount' driven as '1'.
 * 'totalValueSize' driven as '0'.
 * 'txbuf' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859758}
 *
 * @verify{18851142}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:0
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Txbuf_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAllocTxBufferForKeys
TEST.NEW
TEST.NAME:TC_006.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Txbuf_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonTransportAllocTxBufferForKeys.Failure_due_to_Txbuf_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAllocTxBufferForKeys() - Failure due to txbuf is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'bufParams.keyCount' driven as '1'.
 * 'totalValueSize' driven as '0'.
 * 'txbuf' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859761}
 *
 * @verify{18851142}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAllocTxBufferForKeys.txbuf:<<null>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonTransportAppendKeyValuePair

-- Test Case: TC_001.LwSciCommonTransportAppendKeyValuePair.Success_use_case
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAppendKeyValuePair
TEST.NEW
TEST.NAME:TC_001.LwSciCommonTransportAppendKeyValuePair.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonTransportAppendKeyValuePair.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAppendKeyValuePair() - Success case when appends key-value pair in LwSciCommonTransportBuf.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values. }
 *
 * @testsetup{ None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' driven as 1.
 * 'txbuf[0].sizewr' driven as '1'.
 * 'length' driven as 'UINT64_MAX'.
 * 'sizeAllocated' driven as '16'.
 * 'txbuf[0].wrKeyCount' driven as '1'.
 * 'value' set to valid memory.}
 *
 * @testbehavior{ LwSciCommonMemcpyS() receives 'dest' as valid address.
 * LwSciCommonMemcpyS() receives 'src' as valid address.
 * LwSciCommonMemcpyS() receives 'n' as 'UINT64_MAX'.
 * LwSciCommonMemcpyS() receives 'destSize' as 'UINT64_MAX
 * 'txbuf[0].wrKeyCount' receives valid value '2'.
 * 'txbuf[0].sizewr' receives valid value '12'.
 * returns LwSciError_Success.}
 *
 * @testcase{18859764}
 *
 * @verify{18851148}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].wrKeyCount:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].sizeAllocated:16
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].sizewr:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.key:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length:<<MAX>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].wrKeyCount:2
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].sizewr:12
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:<<MAX>>
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:<<MAX>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempvalue>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf.txbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf>>[0].bufPtr = ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempbufPtr>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Transportbuffer_WriteKeyCount_Is_Equal_To_UINT32_MAX
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAppendKeyValuePair
TEST.NEW
TEST.NAME:TC_002.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Transportbuffer_WriteKeyCount_Is_Equal_To_UINT32_MAX
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Transportbuffer_WriteKeyCount_Is_Equal_To_UINT32_MAX}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAppendKeyValuePair() - Failure due to txbuf[0].wrKeyCount is equal to UINT32_MAX.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values. }
 *
 * @testsetup{ None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' driven as 1.
 * 'txbuf[0].sizewr' driven as '1'.
 * 'length' driven as '1'.
 * 'sizeAllocated' driven as '16'.
 * 'txbuf[0].wrKeyCount' driven as 'UINT32_MAX'.
 * 'value' set to valid memory.}
 *
 * @testbehavior{returns LwSciError_Overflow.}
 *
 * @testcase{18859767}
 *
 * @verify{18851148}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].sizeAllocated:16
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].sizewr:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.key:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length:1
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Overflow
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf.txbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf>>[0].bufPtr = ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempbufPtr>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf.txbuf[0].wrKeyCount
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf>>[0].wrKeyCount = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Transportbuffer_Sizeallocated_Is_13
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAppendKeyValuePair
TEST.NEW
TEST.NAME:TC_003.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Transportbuffer_Sizeallocated_Is_13
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Transportbuffer_Sizeallocated_Is_13.}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAppendKeyValuePair() - Failure due to temsum is greater than txbuf[0].sizeAllocated.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' driven as 1.
 * 'txbuf[0].sizewr' driven as '1'.
 * 'length' driven as 'INT64_MAX'.
 * 'sizeAllocated' driven as '13'.
 * 'value' set to valid memory.}
 *
 * @testbehavior{returns LwSciError_NoSpace.}
 *
 * @testcase{18859770}
 *
 * @verify{18851148}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].sizeAllocated:13
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].sizewr:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.key:1
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_NoSpace
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf.txbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf>>[0].bufPtr = ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempbufPtr>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length>> = ( INT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_temSum_LessThan_TransportbufferSizewr
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAppendKeyValuePair
TEST.NEW
TEST.NAME:TC_004.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_temSum_LessThan_TransportbufferSizewr
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_temSum_LessThan_TransportbufferSizewr}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAppendKeyValuePair() - Failure due to tempSum less than transportbuffer.sizewr.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' driven as 1.
 * 'txbuf[0].sizewr' driven as 'UINT64_MAX'.
 * 'length' driven as 'UINT64_MAX/2'.
 * 'sizeAllocated' driven as '1'.
 * 'txbuf[0].wrKeyCount' driven as '1'.
 * 'value' set to valid memory.}
 *
 * @testbehavior{returns LwSciError_Overflow.}
 *
 * @testcase{18859773}
 *
 * @verify{18851148}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].wrKeyCount:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].sizeAllocated:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.key:1
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Overflow
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf.txbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf>>[0].bufPtr = ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempbufPtr>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf.txbuf[0].sizewr
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf>>[0].sizewr = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length>> = (  UINT64_MAX/2 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_txbuf.magic_Is_1
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAppendKeyValuePair
TEST.NEW
TEST.NAME:TC_005.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_txbuf.magic_Is_1
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_txbuf.magic_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAppendKeyValuePair() - Failure due to txbuf.magic is 1.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory
 * 'txbuf[0].magic' driven as '1'.
 * 'length' driven as '1'.
 * 'value' set to valid memory.}
 *
 * @testbehavior{ LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859776}
 *
 * @verify{18851148}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].magic:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value:VECTORCAST_INT1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_value_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAppendKeyValuePair
TEST.NEW
TEST.NAME:TC_006.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_value_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonTransportAppendKeyValuePair.value_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAppendKeyValuePair() - Failure due to value is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory
 * 'txbuf[0].magic' driven as '1'.
 * 'length' driven as '1'.
 * 'value' set to NULL.}
 *
 * @testbehavior{ LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859779}
 *
 * @verify{18851148}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value:<<null>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Length_Is_0
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAppendKeyValuePair
TEST.NEW
TEST.NAME:TC_007.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Length_Is_0
TEST.NOTES:
/**
 * @testname{TC_007.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_Length_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAppendKeyValuePair() - Failure due to length is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'length' driven as '0'.
 * 'value' set to valid memory.}
 *
 * @testbehavior{ LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859782}
 *
 * @verify{18851148}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length:0
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value:VECTORCAST_INT1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_txbuf_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportAppendKeyValuePair
TEST.NEW
TEST.NAME:TC_008.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_txbuf_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciCommonTransportAppendKeyValuePair.Failure_due_to_txbuf_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportAppendKeyValuePair() - Failure due to txbuf is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'txbuf set to NULL.
 * 'length' driven as '1'.
 * 'value' set to valid memory.}
 *
 * @testbehavior{ LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{22060174}
 *
 * @verify{18851148}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.txbuf:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.length:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportAppendKeyValuePair.value:VECTORCAST_INT1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonTransportBufferFree

-- Test Case: TC_001.LwSciCommonTransportBufferFree.Success_use_case
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportBufferFree
TEST.NEW
TEST.NAME:TC_001.LwSciCommonTransportBufferFree.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonTransportBufferFree.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportBufferFree() - Success case when deallocates LwSciCommonTransportBuf successfuly.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'buf set to valid memory.
 * 'buf.magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.}
 *
 * @testbehavior{ LwSciCommonFree() receives ptr as valid address.}
 *
 * @testcase{18859785}
 *
 * @verify{18851151}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_transportutils.c.LwSciCommonTransportBufferFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonTransportBufferFree.Failure_due_to_buf_magic_Is_1
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportBufferFree
TEST.NEW
TEST.NAME:TC_002.LwSciCommonTransportBufferFree.Failure_due_to_buf_magic_Is_1
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonTransportBufferFree.Failure_due_to_buf_magic_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportBufferFree() - Failure due to buf.magic is 1.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'buf set to valid memory.
 * 'buf.magic' driven as '1'.}
 *
 * @testbehavior{ LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859788}
 *
 * @verify{18851151}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf[0].magic:1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_003.LwSciCommonTransportBufferFree.Failure_due_to_bufPtr_Is_Not_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportBufferFree
TEST.NEW
TEST.NAME:TC_003.LwSciCommonTransportBufferFree.Failure_due_to_bufPtr_Is_Not_NULL
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonTransportBufferFree.Failure_due_to_bufPtr_Is_Not_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportBufferFree() - Failure due to bufPtr is not null.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'buf set to valid memory.
 * buf.bufPtr set to ''VECTORCAST_BUFFER'.
 * 'buf.magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.}
 *
 * @testbehavior{LwSciCommonFree() receives ptr as valid address.}
 *
 * @testcase{22060233}
 *
 * @verify{18851151}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf[0].bufPtr:VECTORCAST_BUFFER
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_transportutils.c.LwSciCommonTransportBufferFree
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == (<<lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf>>[0].bufPtr) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonTransportBufferFree.Failure_due_to_buf_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportBufferFree
TEST.NEW
TEST.NAME:TC_004.LwSciCommonTransportBufferFree.Failure_due_to_buf_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonTransportBufferFree.Failure_due_to_buf_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportBufferFree() - Failure due to buf is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'buf set to NULL}
 *
 * @testbehavior{ LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{22060239}
 *
 * @verify{18851151}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportBufferFree.buf:<<null>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciCommonTransportGetNextKeyValuePair

-- Test Case: TC_001.LwSciCommonTransportGetNextKeyValuePair.Success_use_case
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_001.LwSciCommonTransportGetNextKeyValuePair.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonTransportGetNextKeyValuePair.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Success case when retrieves next keyvalue pair from LwSciCommonTransportBuf.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rxbuf[0].sizerd' driven as '2'.
 * 'rxbuf[0].sizewr' driven as '10'.}
 *
 * @testbehavior{'key[0]' pointing to valid value.
 * 'length[0]' pointing to valid value.
 * 'value' pointing to valid memory.
 * 'rdFinish[0]' pointing to valid value.
 * returns LwSciError_Success.}
 *
 * @testcase{18859791}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].bufPtr:VECTORCAST_BUFFER
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:2
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizewr:10
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].rdKeyCount:1
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:14
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length.length[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_sizewr_Is_Equal_To_sizerd
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_002.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_sizewr_Is_Equal_To_sizerd
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_sizewr_Is_Equal_To_sizerd}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to rxbuf->sizerd is equal to rxbuf->sizewr.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rxbuf[0].sizerd' driven as '2'.
 * 'rxbuf[0].sizewr' driven as '10'.}
 *
 * @testbehavior{'key[0]' pointing to valid value.
 * 'length[0]' pointing to valid value.
 * 'value' pointing to valid memory.
 * 'rdFinish[0]' pointing to valid value.
 * returns LwSciError_Success.}
 *
 * @testcase{18859794}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].bufPtr:VECTORCAST_BUFFER
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizewr:13
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].rdKeyCount:1
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:32
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length.length[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_rdKeyCount_Is_MAX
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_003.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_rdKeyCount_Is_MAX
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_rdKeyCount_Is_MAX}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to rxbuf->rdKeyCount is equal to UINT32_MAX.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rxbuf[0].sizerd' driven as '2'.
 * 'rxbuf[0].sizewr' driven as '11'.
 * 'rxbuf[0].rdKeyCount' driven as 'UINT32_MAX'.}
 *
 * @testbehavior{'key[0]' pointing to valid value.
 * 'length[0]' pointing to valid value.
 * 'value' pointing to valid memory.
 * 'rdFinish[0]' pointing to valid value.
 * returns LwSciError_Overflow.}
 *
 * @testcase{18859797}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].bufPtr:VECTORCAST_BUFFER
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:2
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizewr:11
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Overflow
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf.rxbuf[0].rdKeyCount
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf>>[0].rdKeyCount = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length.length[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_ReceiverBuffer_Sizewr_Is_1
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_004.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_ReceiverBuffer_Sizewr_Is_1
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_ReceiverBuffer_Sizewr_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to rxbuf->sizewr is less than the tmpSum.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rxbuf[0].sizerd' driven as '2'.
 * 'rxbuf[0].sizewr' driven as '1'.}
 *
 * @testbehavior{returns LwSciError_Overflow.}
 *
 * @testcase{18859800}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].bufPtr:VECTORCAST_BUFFER
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:2
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizewr:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Overflow
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_magic_Is_4
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_005.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_magic_Is_4
TEST.NOTES:
/**
 * @testname{TC_008.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_magic_Is_4}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to magic is 4 so LwSciCommonTransportIsValidBuf() returns false.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as '4'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rdFinish' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859803}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:4
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf.rxbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf>>[0].bufPtr = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_Key_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_006.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_Key_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_009.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_Key_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to key is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rdFinish' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859806}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf.rxbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf>>[0].bufPtr = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_Length_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_007.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_Length_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_010.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_Length_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to length is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to NULL.
 * 'value' set to valid memory
 * 'rdFinish' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859809}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf.rxbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf>>[0].bufPtr = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_value_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_008.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_value_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_011.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_value_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to value is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory
 * 'value' set to NULL.
 * 'rdFinish' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859812}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf.rxbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf>>[0].bufPtr = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rdfinish_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_009.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rdfinish_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_012.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rdfinish_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to rdFinish is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory
 * 'value' set to valid memory.
 * 'rdFinish' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859815}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<null>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf.rxbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf>>[0].bufPtr = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_tmpSum_lessthan_rxbuf_sizerd
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_010.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_tmpSum_lessthan_rxbuf_sizerd
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_tmpSum_lessthan_rxbuf_sizerd}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to tmpSum is less than rxbuf->sizerd.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rxbuf[0].sizerd' driven as 'SIZE_MAX'.
 * 'rxbuf[0].sizewr' driven as 'SIZE_MAX'.}
 *
 * @testbehavior{'key[0]' pointing to valid value.
 * 'length[0]' pointing to valid value.
 * 'value' pointing to valid memory.
 * 'rdFinish[0]' pointing to valid value.
 * returns LwSciError_Overflow.}
 *
 * @testcase{18859818}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].bufPtr:VECTORCAST_BUFFER
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:<<MAX>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizewr:<<MAX>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].rdKeyCount:1
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:18446744073709551615
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Overflow
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length.length[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_tmpSum_lessthan_rxbuf_sizerd
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_011.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_tmpSum_lessthan_rxbuf_sizerd
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_tmpSum_lessthan_rxbuf_sizerd}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to tmpSum is less than rxbuf->sizerd.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rxbuf[0].sizerd' driven as 'SIZE_MAX'.
 * 'rxbuf[0].sizewr' driven as 'SIZE_MAX'.}
 *
 * @testbehavior{returns LwSciError_Overflow.}
 *
 * @testcase{18859821}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:<<MAX>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].bufPtr:VECTORCAST_INT1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:<<MAX>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizewr:<<MAX>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].rdKeyCount:0
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:18446744073709551615
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Overflow
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length.length[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_012.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_013.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_rxbuf_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to rxbuf is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ None.}
 *
 * @testinput{'rxbuf' set to NULL.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory
 * 'value' set to valid memory.
 * 'rdFinish' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{22060177}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_013.LWSCICOMMONTRANSPORTGETNEXTKEYVALUEPAIR.FAILURE_DUE_TO_TMPSUM_LESSTHAN_ADJUSTED_SIZEOF
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetNextKeyValuePair
TEST.NEW
TEST.NAME:TC_013.LWSCICOMMONTRANSPORTGETNEXTKEYVALUEPAIR.FAILURE_DUE_TO_TMPSUM_LESSTHAN_ADJUSTED_SIZEOF
TEST.NOTES:
/**
 * @testname{TC_013.LwSciCommonTransportGetNextKeyValuePair.Failure_due_to_tmpSum_lessthan_ADJUSTED_SIZEOF}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetNextKeyValuePair() - Failure due to tmpSum less than ADJUSTED_SIZEOF.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'rxbuf' set to valid memory.
 * 'rxbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'key' set to valid memory.
 * 'length' set to valid memory.
 * 'value' set to valid memory
 * 'rxbuf[0].sizerd' driven as '2'.
 * 'rxbuf[0].sizewr' driven as '10'.}
 *
 * @testbehavior{'key[0]' pointing to valid value.
 * 'length[0]' pointing to valid value.
 * 'value' pointing to valid memory.
 * 'rdFinish[0]' pointing to valid value.
 * returns LwSciError_Success.}
 *
 * @testcase{22060179}
 *
 * @verify{18851154}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.VECTORCAST_INT1:<<ZERO>>
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.tempbufPtr:<<MIN-1>>
TEST.VALUE:lwscicommon_transportutils.<<GLOBAL>>.test_LwSciCommonTransportKey.length:<<MIN-1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:<<MAX+1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizewr:10
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].rdKeyCount:0
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf[0].sizerd:0
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Overflow
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
  lwscicommon_transportutils.c.LwSciCommonTransportGetNextKeyValuePair
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf.rxbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.rxbuf>>[0].bufPtr = ( &<<lwscicommon_transportutils.<<GLOBAL>>.test_LwSciCommonTransportKey>> );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
<<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key.key[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.key>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length.length[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.length>>[0] == ( 0 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetNextKeyValuePair.value>> == ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.tempvalue>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonTransportGetRxBufferAndParams

-- Test Case: TC_001.LwSciCommonTransportGetRxBufferAndParams.Success_use_case
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_001.LwSciCommonTransportGetRxBufferAndParams.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonTransportGetRxBufferAndParams.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Success case when verifies data integrity of buffer array and copy into txbuf and params.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ LwSciCommonCalloc() returns memory to 'paramLwSciCommonTransportRec'.
 *  LwSciCommonCalloc() returns memory to 'paramLwSciCommonTransportRec.bufPtr'.}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as '9'.
 * 'rxbuf' set to valid memory.
 * 'params' set to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as 'sizeof(LwSciCommonTransportBufPriv)'.
 * LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as '9'.
 * LwSciCommonMemcpyS() receives 'dest' as valid address.
 * LwSciCommonMemcpyS() receives 'src' as valid address.
 * LwSciCommonMemcpyS() receives 'n' as '9'.
 * LwSciCommonMemcpyS() receives 'destSize' as '9'.
 * 'rxbuf[0]' pointing to valid address.
 * 'params[0].msgVersion' pointing to valid value '0'.
 * 'params[0].msgMagic' pointing to valid value '4294967295'.
 * 'params[0].keyCount' pointing to valid value '0'.
 * returns LwSciError_Success.}
 *
 * @testcase{18859824}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:9
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf[0]:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params[0].msgVersion:0
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params[0].msgMagic:4294967295
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:0
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.destSize:9
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonMemcpyS.n:9
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>> );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>>.bufPtr );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciCommonTransportBufPriv) ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 9 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == (  &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>>.bufPtr ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( VECTORCAST_BUFFER ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->size=9;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->msgMagic=0xFFFFFFFF;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->checksum=0xFF000000;
<<lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] == ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Transportbuffer_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_002.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Transportbuffer_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Transportbuffer_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to if memory allocation failed.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{ LwSciCommonCalloc() returns NULL.}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as '9'.
 * 'rxbuf' set to valid memory.
 * 'params' set to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as 'sizeof(LwSciCommonTransportBufPriv)'.
 * LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as '9'.
 * LwSciCommonFree() receives 'ptr' as valid address.
 * returns LwSciError_InsufficientMemory.}
 *
 * @testcase{18859827}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:9
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf[0]:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonCalloc
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciCommonTransportBufPriv) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->size=9;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->msgMagic=0xFFFFFFFF;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->checksum=0xFF000000;
<<lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_hdr_checksum_Is_Not_Equal_To_ComputedCrc
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_003.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_hdr_checksum_Is_Not_Equal_To_ComputedCrc
TEST.NOTES:
/**
 * @testname{TC_003.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_hdr_checksum_Is_Not_Equal_To_ComputedCrc}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to LwSciCommolwerifyChecksum() returns false.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{None}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as 'UINT64_MAX'.
 * 'rxbuf' set to valid memory.
 * 'params' set to valid memory.}
 *
 * @testbehavior{returns LwSciError_BadParameter.}
 *
 * @testcase{18859830}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:<<MAX>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf[0]:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_BadParameter
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->size=9;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->msgMagic=0xFFFFFFFF;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->checksum=0xFFF00000;
<<lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_SubStatus_Is_OP_FAIL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_004.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_SubStatus_Is_OP_FAIL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_SubStatus_Is_OP_FAIL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to SUB_FUNC() macro returns OP_FAIL.}
 *
 * @casederiv{Analysis of Requirements.
 * Analysis of Boundary Values.}
 *
 * @testsetup{None}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as '1'.
 * 'rxbuf' set to valid memory.
 * 'params' set to valid memory.}
 *
 * @testbehavior{returns LwSciError_BadParameter.}
 *
 * @testcase{18859833}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_BadParameter
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportHeader>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_hdr_size_Is_greater_Than_BufZize
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_005.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_hdr_size_Is_greater_Than_BufZize
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_hdr_size_Is_greater_Than_BufZize}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to bufsize is less than the hdr->size.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as '8'.
 * 'rxbuf' set to valid memory.
 * 'params' set to valid memory.}
 *
 * @testbehavior{returns LwSciError_BadParameter.}
 *
 * @testcase{18859836}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:8
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf[0]:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_BadParameter
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->size=10;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->msgMagic=0xFFFFFFFF;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->checksum=0xFFF00000;
<<lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Bufptr_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_006.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Bufptr_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Bufptr_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to bufPtr is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{'bufPtr' set to NULL.
 * 'bufSize' driven as '1'.
 * 'rxbuf' set to valid memory.
 * 'params' set to valid memory.}
 *
 * @testbehavior{returns LwSciError_BadParameter.}
 *
 * @testcase{18859839}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_BadParameter
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_bufsize_Is_0
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_007.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_bufsize_Is_0
TEST.NOTES:
/**
 * @testname{TC_007.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_bufsize_Is_0}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to bufSize is 0.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as '1'.
 * 'rxbuf' set to valid memory.
 * 'params' set to valid memory.}
 *
 * @testbehavior{returns LwSciError_BadParameter.}
 *
 * @testcase{18859842}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr:VECTORCAST_INT1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:0
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_BadParameter
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
TEST.END_FLOW
TEST.END

-- Test Case: TC_008.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_rxbuf_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_008.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_rxbuf_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_rxbuf_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to rxbuf is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as '1'.
 * 'rxbuf' set to NULL.
 * 'params' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859845}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr:VECTORCAST_INT1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_params_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_009.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_params_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_009.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_params_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to params is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as '1'.
 * 'rxbuf' set to valid memory.
 * 'params' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859848}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr:VECTORCAST_INT1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<null>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_010.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Transportbuffer_bufptr_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportGetRxBufferAndParams
TEST.NEW
TEST.NAME:TC_010.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Transportbuffer_bufptr_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_010.LwSciCommonTransportGetRxBufferAndParams.Failure_due_to_Transportbuffer_bufptr_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportGetRxBufferAndParams() - Failure due to if memory allocation failed.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{LwSciCommonCalloc() returns memory to 'paramLwSciCommonTransportRec'.
 * LwSciCommonCalloc() returns NULL.}
 *
 * @testinput{'bufPtr' set to valid memory.
 * 'bufSize' driven as '9'.
 * 'rxbuf' set to valid memory.
 * 'params' set to valid memory.}
 *
 * @testbehavior{LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as 'sizeof(LwSciCommonTransportBufPriv)'.
 * LwSciCommonCalloc() receives 'numItems' as '1'.
 * LwSciCommonCalloc() receives 'size' as '9'.
 * LwSciCommonFree() receives 'ptr' as valid address.
 * returns LwSciError_InsufficientMemory.}
 *
 * @testcase{22060236}
 *
 * @verify{18851145}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufSize:9
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.rxbuf[0]:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params:<<malloc 1>>
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params[0].msgVersion:0
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params[0].msgMagic:0
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:0
TEST.EXPECTED:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscicommon_transportutils.c.LwSciCommonTransportGetRxBufferAndParams
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cnt=0;
cnt++;
if(cnt==1)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>> );
else if(cnt==2)
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cnt=0;
cnt++;
if(cnt==1)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciCommonTransportBufPriv) ) }}
else if(cnt==2)
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( 9 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportRec>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->size=9;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->msgMagic=0xFFFFFFFF;
((LwSciCommonTransportHeader*)VECTORCAST_BUFFER)->checksum=0xFF000000;
<<lwscicommon_transportutils.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciCommonTransportPrepareBufferForTx

-- Test Case: TC_001.LwSciCommonTransportPrepareBufferForTx.Success_use_case
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportPrepareBufferForTx
TEST.NEW
TEST.NAME:TC_001.LwSciCommonTransportPrepareBufferForTx.Success_use_case
TEST.NOTES:
/**
 * @testname{TC_001.LwSciCommonTransportPrepareBufferForTx.Success_use_case}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportPrepareBufferForTx() - Success case when serializes LwSciCommonTransportBuf into binary buffer.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory.
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'descBufPtr' set to valid memory.
 * 'descBufSize' set to valid memory.
 * 'txbuf[0].sizewr' driven as '9'.
 * 'txbuf[0].sizeAllocated' driven as '1'.
 * 'txbuf[0].wrKeyCount' driven as '8'.
 * 'txbuf[0].allocatedKeyCount' driven as '1'.}
 *
 * @testbehavior{'descBufPtr[0]' points to valid memory of 'txbuf[0].bufPtr'.
 * 'descBufSizetr[0]' points to valid memory of 'txbuf[0].sizewr'. }
 *
 * @testcase{18859851}
 *
 * @verify{18851157}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].allocatedKeyCount:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].wrKeyCount:8
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].sizeAllocated:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].sizewr:9
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufSize:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportPrepareBufferForTx
  lwscicommon_transportutils.c.LwSciCommonTransportPrepareBufferForTx
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf.txbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf>>[0].bufPtr = ( VECTORCAST_BUFFER );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufPtr.descBufPtr[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufPtr>>[0] == ( &VECTORCAST_BUFFER ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufSize.descBufSize[0]
{{ <<lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufSize>>[0] == ( <<lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf>>[0].sizewr ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_SubStatus_Is_OP_FAIL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportPrepareBufferForTx
TEST.NEW
TEST.NAME:TC_002.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_SubStatus_Is_OP_FAIL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_SubStatus_Is_OP_FAIL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportPrepareBufferForTx() - Failure due to sub status is OP_FAIL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory.
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'descBufPtr' set to valid memory.
 * 'descBufSize' set to valid memory.
 * 'txbuf[0].sizewr' driven as '8'.
 * 'txbuf[0].sizeAllocated' driven as '8'.
 * 'txbuf[0].wrKeyCount' driven as '8'.
 * 'txbuf[0].allocatedKeyCount' driven as '8'.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859854}
 *
 * @verify{18851157}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].allocatedKeyCount:8
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].wrKeyCount:8
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].sizeAllocated:8
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].sizewr:8
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufSize:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf.txbuf[0].bufPtr
<<lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf>>[0].bufPtr = ( &<<lwscicommon_transportutils.<<GLOBAL>>.paramLwSciCommonTransportHeader>> );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_magic_Is_1
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportPrepareBufferForTx
TEST.NEW
TEST.NAME:TC_004.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_magic_Is_1
TEST.NOTES:
/**
 * @testname{TC_004.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_magic_Is_1}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportPrepareBufferForTx() - Failure due to txbuf.magic is 1 so LwSciCommonTransportIsValidBuf() is false.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory.
 * 'txbuf[0].magic' driven as '1'.
 * 'descBufPtr' set to valid memory.
 * 'descBufSize' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859860}
 *
 * @verify{18851157}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].magic:1
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufSize:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_Descbufptr_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportPrepareBufferForTx
TEST.NEW
TEST.NAME:TC_005.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_Descbufptr_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_Descbufptr_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportPrepareBufferForTx() - Failure due to descbufptr is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory.
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'descBufPtr' set to NULL.
 * 'descBufSize' set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859863}
 *
 * @verify{18851157}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufPtr:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufSize:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_006.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_Descbufsize_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportPrepareBufferForTx
TEST.NEW
TEST.NAME:TC_006.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_Descbufsize_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_Descbufsize_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportPrepareBufferForTx() - Failure due to descbufsize is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'txbuf[0].bufPtr' pointing to valid memory.
 * 'txbuf[0].magic' driven as 'LWSCICOMMON_TRANSPORT_MAGIC'.
 * 'descBufPtr' set to valid memory..
 * 'descBufSize' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{18859866}
 *
 * @verify{18851157}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf[0].magic:MACRO=LWSCICOMMON_TRANSPORT_MAGIC
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufSize:<<null>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_txbuf_Is_NULL
TEST.UNIT:lwscicommon_transportutils
TEST.SUBPROGRAM:LwSciCommonTransportPrepareBufferForTx
TEST.NEW
TEST.NAME:TC_007.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_txbuf_Is_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciCommonTransportPrepareBufferForTx.Failure_due_to_txbuf_Is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciCommonTransportPrepareBufferForTx() - Failure due to txbuf is NULL.}
 *
 * @casederiv{Analysis of Requirements.}
 *
 * @testsetup{None.}
 *
 * @testinput{'txbuf[0] set to NULL.
 * 'descBufPtr' set to valid memory..
 * 'descBufSize' set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called and exits from test case.}
 *
 * @testcase{22060237}
 *
 * @verify{18851157}
 */
TEST.END_NOTES:
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.txbuf:<<null>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufPtr:<<malloc 1>>
TEST.VALUE:lwscicommon_transportutils.LwSciCommonTransportPrepareBufferForTx.descBufSize:<<malloc 1>>
TEST.FLOW
  lwscicommon_transportutils.c.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END
