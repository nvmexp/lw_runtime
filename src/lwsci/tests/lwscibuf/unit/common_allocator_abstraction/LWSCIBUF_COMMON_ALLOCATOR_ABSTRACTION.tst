-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCIBUF_COMMON_ALLOCATOR_ABSTRACTION
-- Unit(s) Under Test: lwscibuf_alloc_interface
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

-- Subprogram: LwSciBufAllocIfaceAlloc

-- Test Case: TC_001.LwSciBufAllocIfaceAlloc.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceAlloc.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceAlloc.Panic.allocTypesetToLwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857709}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_ExternalCarveout
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 77;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_002.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() when LwSciCommonCalloc returns NULL on first invocation }
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized.
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer is initialized}
 *
 * @testbehavior{Returns LwSciError_InsufficientMemory if there is insufficient memory to complete the operation.}
 *
 * @testcase{18857712}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:9
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_IOMMU
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemVal) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 76;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciCommonCalloc_returns_allocated_memory_on_first_ilwocation
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciCommonCalloc_returns_allocated_memory_on_first_ilwocation
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciCommonCalloc_returns_allocated_memory_on_first_ilwocation}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() when LwSciCommonCalloc returns allocated memory on first invocation}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciCommonFree() returns valid memory}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized.
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer is initialized}
 *
 * @testbehavior{Returns LwSciError_InsufficientMemory if there is insufficient memory to complete the operation.}
 *
 * @testcase{18857715}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:9
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_IOMMU
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.return:LwSciError_Unknown
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if ( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.sysMemallocValObj>> );
}
if ( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
}
i++;

TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 0xFFFFFFFF  ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemVal) ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemHeapType) ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
uint32_t var1 = 9;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal>>.numHeaps = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceAlloc.Panic_due_to_heap_set_to_LwSciBufAllocIfaceHeapType_Ilwalid
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceAlloc.Panic_due_to_heap_set_to_LwSciBufAllocIfaceHeapType_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceAlloc.Panic_due_to_heap_set_to_LwSciBufAllocIfaceHeapType_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics}
 *
 * @casederiv{Analysis of Requirement
 * Generation and Analysis of Equivalence Classes
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer is initialized}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857718}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:9
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_Ilwalid
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.sysMemallocValObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 4 ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemVal) ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemHeapType) ) }}
}
i++;


TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 86;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceAlloc.Success_due_to_LwSciBufSysMemAlloc_returns_Success
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceAlloc.Success_due_to_LwSciBufSysMemAlloc_returns_Success
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceAlloc.Success_due_to_LwSciBufSysMemAlloc_returns_Success}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() buffer allocated successfully}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciBufSysMemAlloc() returns LwSciError_Success
 * LwSciCommonFree() returns valid memory}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized.
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer is initialized}
 *
 * @testbehavior{Returns LwSciError_Success if buffer is allocated for specified LwSciBufAllocIfaceType}
 *
 * @testcase{18857721}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:9
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_IOMMU
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.return:LwSciError_AccessDenied
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemAlloc.rmHandle[0].memHandle:1
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemAlloc.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle[0].memHandle:1
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.return:LwSciError_Success
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciBufSysMemAlloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int i = 0;
if( i == 0 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.sysMemallocValObj>> );
}
if( i == 1 )
{
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.sysMemHeapTypeObj>> );
}
i++;
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 4 ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemVal) ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemHeapType) ) }}
}
i++;


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemAlloc.context
{{ <<uut_prototype_stubs.LwSciBufSysMemAlloc.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemAlloc.allocVal
{{ <<uut_prototype_stubs.LwSciBufSysMemAlloc.allocVal>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemAlloc.devHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemAlloc.devHandle>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 74;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciBufSysMemAlloc_returns_InsufficientMemory
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciBufSysMemAlloc_returns_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceAlloc.Failure_due_to_LwSciBufSysMemAlloc_returns_InsufficientMemory}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() when LwSciBufSysMemAlloc() returns LwSciError_InsufficientMemory }
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciBufSysMemAlloc() returns LwSciError_InsufficientMemory
 * LwSciCommonFree() returns valid memory}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized.
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer is initialized}
 *
 * @testbehavior{Returns LwSciError_InsufficientMemory}
 *
 * @testcase{18857724}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:9
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_IOMMU
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.return:LwSciError_AccessDenied
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemAlloc.rmHandle[0].memHandle:7
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemAlloc.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.return:LwSciError_InsufficientMemory
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciBufSysMemAlloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.sysMemallocValObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 4 ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemVal) ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemHeapType) ) }}
}
i++;


TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemAlloc.context
{{ <<uut_prototype_stubs.LwSciBufSysMemAlloc.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemAlloc.allocVal
{{ <<uut_prototype_stubs.LwSciBufSysMemAlloc.allocVal>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemAlloc.devHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemAlloc.devHandle>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 44;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal>>.alignment = ( UINT64_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufAllocIfaceAlloc.Panic_due_to_context_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_007.LwSciBufAllocIfaceAlloc.Panic_due_to_context_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufAllocIfaceAlloc.Panic_due_to_context_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when context pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to NULL
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857727}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context:<<null>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_ExternalCarveout
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufAllocIfaceAlloc.Panic_due_to_size_Set_to_Zero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_008.LwSciBufAllocIfaceAlloc.Panic_due_to_size_Set_to_Zero
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufAllocIfaceAlloc.Panic_due_to_size_Set_to_Zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when iFaceAllocVal.size set to Min value 0}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857730}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:3
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_ExternalCarveout
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 43;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufAllocIfaceAlloc.Panic_due_to_heap_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_009.LwSciBufAllocIfaceAlloc.Panic_due_to_heap_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufAllocIfaceAlloc.Panic_due_to_heap_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when iFaceAllocVal.heap set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857733}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<null>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 88;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufAllocIfaceAlloc.Panic_due_to_numHeaps_Set_to_Zero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_010.LwSciBufAllocIfaceAlloc.Panic_due_to_numHeaps_Set_to_Zero
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufAllocIfaceAlloc.Panic_due_to_numHeaps_Set_to_Zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when iFaceAllocVal.numHeaps set to 0}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857736}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:2
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_ExternalCarveout
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 22;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciBufAllocIfaceAlloc.Panic_due_to_devHandle_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_011.LwSciBufAllocIfaceAlloc.Panic_due_to_devHandle_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_011.LwSciBufAllocIfaceAlloc.Panic_due_to_devHandle_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when devHandle pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to NULL
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857739}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_ExternalCarveout
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 33;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciBufAllocIfaceAlloc.Panic_due_to_rmHandle_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_012.LwSciBufAllocIfaceAlloc.Panic_due_to_rmHandle_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_012.LwSciBufAllocIfaceAlloc.Panic_due_to_rmHandle_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when rmHandle pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857742}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_ExternalCarveout
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<null>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 33;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciBufAllocIfaceAlloc.Panic_due_to_heapElement_Set_to_LwSciBufAllocIfaceHeapType_Ilwalid
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_013.LwSciBufAllocIfaceAlloc.Panic_due_to_heapElement_Set_to_LwSciBufAllocIfaceHeapType_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_013.LwSciBufAllocIfaceAlloc.Panic_due_to_heapElement_Set_to_LwSciBufAllocIfaceHeapType_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when heap in array iFaceAllocVal.heap set to 'LwSciBufAllocIfaceHeapType_Ilwalid'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857745}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_Ilwalid
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.sysMemallocValObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.numItems
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 1 ) }}
}
if( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.numItems>> == ( 4 ) }}
}
i++;

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int i = 0;
if ( i == 0 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemVal) ) }}
}
if ( i == 1 )
{
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemHeapType) ) }}
}
i++;


TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 55;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciBufAllocIfaceAlloc.Panic.allocTypesetTolessthanzero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceAlloc
TEST.NEW
TEST.NAME:TC_014.LwSciBufAllocIfaceAlloc.Panic.allocTypesetTolessthanzero
TEST.NOTES:
/**
 * @testname{TC_014.LwSciBufAllocIfaceAlloc.Panic.allocTypesetTolessthanzero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceAlloc() Panics when 'allocType' set to less than 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType enum set to less than 0
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'iFaceAllocVal' - LwSciBufAllocIfaceVal struct is initialized
 * 4. 'devHandle' - devHandle pointer set to valid memory
 * 5. 'rmHandle' - LwSciBufRmHandle struct pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060690}
 *
 * @verify{18842928}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.size:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.alignment:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.coherency:true
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.heap[0]:LwSciBufAllocIfaceHeapType_ExternalCarveout
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:0x4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.cpuMapping:false
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.rmHandle:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context
int var = 41;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufAllocIfaceClose

-- Test Case: TC_001.LwSciBufAllocIfaceClose.Panic_due_to_context_set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceClose
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceClose.Panic_due_to_context_set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceClose.Panic_due_to_context_set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceClose() Panics}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation interface context should be closed set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque open context pointer set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857751}
 *
 * @verify{18842949}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceClose.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceClose.context:<<null>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceClose
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_002.LwSciBufAllocIfaceClose.context_set_to_allocated_memory
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceClose
TEST.NEW
TEST.NAME:TC_002.LwSciBufAllocIfaceClose.context_set_to_allocated_memory
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufAllocIfaceClose.context_set_to_allocated_memory}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceClose()}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciBufSysMemClose() returns valid address.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation interface context should be closed set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque open context pointer set to valid memory}
 *
 * @testbehavior{None}
 *
 * @testcase{18857754}
 *
 * @verify{18842949}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceClose.allocType:LwSciBufAllocIfaceType_SysMem
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceClose
  uut_prototype_stubs.LwSciBufSysMemClose
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceClose
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemClose.context
{{ <<uut_prototype_stubs.LwSciBufSysMemClose.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceClose.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceClose.context
int var = 4444;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceClose.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufAllocIfaceCpuCacheFlush

-- Test Case: TC_001.LwSciBufAllocIfaceCpuCacheFlush.Panic.allocTypesetToLwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceCpuCacheFlush.Panic.allocTypesetToLwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceCpuCacheFlush.Panic.allocTypesetToLwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to valid memory.
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 2.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060694}
 *
 * @verify{18842949}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:2
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
int var = 3333;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
int var = 885;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceCpuCacheFlush.Failure_due_to_LwSciBufSysMemCpuCacheFlush_returns_ResourceError
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceCpuCacheFlush.Failure_due_to_LwSciBufSysMemCpuCacheFlush_returns_ResourceError
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceCpuCacheFlush.Failure_due_to_LwSciBufSysMemCpuCacheFlush_returns_ResourceError}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() memHandle set to Mid value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemCpuCacheFlush() returns LwSciError_ResourceError.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to valid memory.
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 1.}
 *
 * @testbehavior{Returns LwSciError_ResourceError if LWPU driver stack failed to flush the buffer.}
 *
 * @testcase{18857763}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:0xFF
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:8
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.return:LwSciError_NotInitialized
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.return:LwSciError_ResourceError
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:0xFF
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.len:8
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.context
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.cpuPtr
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.cpuPtr>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
int var = 3309;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
int var = 33998;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_LwSciBufSysMemCpuCacheFlush_returns_Success
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_LwSciBufSysMemCpuCacheFlush_returns_Success
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_LwSciBufSysMemCpuCacheFlush_returns_Success}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() memHandle set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemCpuCacheFlush() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to valid memory.
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 1.}
 *
 * @testbehavior{Returns LwSciError_Success if Flushing the given len bytes of the mapped buffer is done}
 *
 * @testcase{18857766}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:9
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.return:LwSciError_NotInitialized
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.len:9
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.context
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.cpuPtr
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.cpuPtr>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
int var = 7751;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
int var = 4422;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_len_set_to_Min
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_len_set_to_Min
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_len_set_to_Min}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() length set to Min value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemCpuCacheFlush() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to valid memory.
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 1.}
 *
 * @testbehavior{Returns LwSciError_Success if Flushing the given len bytes of the mapped buffer is done}
 *
 * @testcase{18857769}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.return:LwSciError_NotInitialized
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.len:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.context
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.cpuPtr
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.cpuPtr>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
int var = 9980;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
int var = 305;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_len_set_to_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_len_set_to_Max
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceCpuCacheFlush.Success_due_to_len_set_to_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() length set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemCpuCacheFlush() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to valid memory.
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 1.}
 *
 * @testbehavior{Returns LwSciError_Success if Flushing the given len bytes of the mapped buffer is done}
 *
 * @testcase{22060656}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:19
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.return:LwSciError_NotInitialized
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.len:19
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.context
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.cpuPtr
{{ <<uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.cpuPtr>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
int var = 5567;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
int var = 33980;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufAllocIfaceCpuCacheFlush.Panic.allocTypesetTolessthanzero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_007.LwSciBufAllocIfaceCpuCacheFlush.Panic.allocTypesetTolessthanzero
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufAllocIfaceCpuCacheFlush.Panic.allocTypesetTolessthanzero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() Panics when 'allocType' set to less than 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to valid memory.
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 2.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857772}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:2
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
int var = 3333;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
int var = 885;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufAllocIfaceCpuCacheFlush.Panic.contextSettoNULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_008.LwSciBufAllocIfaceCpuCacheFlush.Panic.contextSettoNULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufAllocIfaceCpuCacheFlush.Panic.contextSettoNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() Panics when context pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to NULL
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to valid memory.
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 2.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060659}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:2
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
int var = 885;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufAllocIfaceCpuCacheFlush.Panic.cpuPtrSettoNULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_009.LwSciBufAllocIfaceCpuCacheFlush.Panic.cpuPtrSettoNULL
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufAllocIfaceCpuCacheFlush.Panic.cpuPtrSettoNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() Panics when cpu pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to NULL
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 2.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060662}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:2
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
int var = 7776;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufAllocIfaceCpuCacheFlush.Panic.lenSettozero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceCpuCacheFlush
TEST.NEW
TEST.NAME:TC_010.LwSciBufAllocIfaceCpuCacheFlush.Panic.lenSettozero
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufAllocIfaceCpuCacheFlush.Panic.lenSettozero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceCpuCacheFlush() Panics when len set to 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer is initialized
 * 4. 'cpuPtr' - CPU virtual address pointer set to valid memory
 * 5. 'len' - Length (in bytes) of the buffer to flush set to 0.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060664}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.len:0
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceCpuCacheFlush
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context
int var = 7776;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr
int var = 9999;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.cpuPtr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufAllocIfaceDeAlloc

-- Test Case: TC_001.LwSciBufAllocIfaceDeAlloc.Panic.allocTypesetToLwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDeAlloc
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceDeAlloc.Panic.allocTypesetToLwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceDeAlloc.Panic.allocTypesetToLwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDeAlloc() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer struct is initialized with memHandle value 1}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060667}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16,EXPECTED_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context
int var = 55;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufAllocIfaceDeAlloc.Success.memHandleMin
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDeAlloc
TEST.NEW
TEST.NAME:TC_002.LwSciBufAllocIfaceDeAlloc.Success.memHandleMin
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufAllocIfaceDeAlloc.Success.memHandleMin}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDeAlloc() memHandle set to Min value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemDealloc() returns LwSciError_NotSupported.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer struct is initialized with memHandle value 0}
 *
 * @testbehavior{Returns LwSciError_NotSupported if LwSciBufAllocIfaceType is not supported.}
 *
 * @testcase{22060671}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:0x0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.return:LwSciError_AccessDenied
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDealloc.rmHandle.memHandle:0
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
  uut_prototype_stubs.LwSciBufSysMemDealloc
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDealloc.context
{{ <<uut_prototype_stubs.LwSciBufSysMemDealloc.context>> == (  <<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context
int var = 8881;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceDeAlloc.Success.memHandleMid
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDeAlloc
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceDeAlloc.Success.memHandleMid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceDeAlloc.Success.memHandleMid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDeAlloc() memHandle set to Mid value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemDealloc() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer struct is initialized with memHandle value 255}
 *
 * @testbehavior{Returns LwSciError_Success if buffer is deallocated successfully}
 *
 * @testcase{22060675}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:0xFF
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.return:LwSciError_AccessDenied
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemDealloc.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDealloc.rmHandle.memHandle:0xFF
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
  uut_prototype_stubs.LwSciBufSysMemDealloc
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDealloc.context
{{ <<uut_prototype_stubs.LwSciBufSysMemDealloc.context>> == (  <<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context
int var = 77712;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceDeAlloc.Success.memHandleMax
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDeAlloc
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceDeAlloc.Success.memHandleMax
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceDeAlloc.Success.memHandleMax}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDeAlloc() memHandle set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemDealloc() returns LwSciError_ResourceError.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer struct is initialized with memHandle value '0xFFFFFFFF'}
 *
 * @testbehavior{Returns LwSciError_ResourceError if LWPU driver stack failed to deallocate the buffer.}
 *
 * @testcase{22060676}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.return:LwSciError_AccessDenied
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.return:LwSciError_Success
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
  uut_prototype_stubs.LwSciBufSysMemDealloc
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDealloc.context
{{ <<uut_prototype_stubs.LwSciBufSysMemDealloc.context>> == (  <<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDealloc.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemDealloc.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context
int var = 777713;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceDeAlloc.Panic.allocTypesetTolessthanzero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDeAlloc
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceDeAlloc.Panic.allocTypesetTolessthanzero
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceDeAlloc.Panic.allocTypesetTolessthanzero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDeAlloc() Panics when 'allocType' set to less than 0.}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to less than 0.
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer struct is initialized with memHandle value 1}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060678}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:0x1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context
int var = 111;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceDeAlloc.Panic.contextSetToNULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDeAlloc
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceDeAlloc.Panic.contextSetToNULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceDeAlloc.Panic.contextSetToNULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDeAlloc() Panics when context pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to NULL
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer struct is initialized with memHandle value 1}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060681}
 *
 * @verify{18842946}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.context:<<null>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDeAlloc
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciBufAllocIfaceDupHandle

-- Test Case: TC_001.LwSciBufAllocIfaceDupHandle.Panic_due_to_allocTypes_set_To_LwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceDupHandle.Panic_due_to_allocTypes_set_To_LwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceDupHandle.Panic.allocTypesetToLwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_ReadWrite'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value 1
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857805}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context
int var = 66;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufAllocIfaceDupHandle.Success.memHandleMin
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_002.LwSciBufAllocIfaceDupHandle.Success.memHandleMin
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufAllocIfaceDupHandle.Success.memHandleMin}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() memHandle set to Min value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemDupHandle() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_Readonly'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value 0
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer struct is initialized with memHandle value 2}
 *
 * @testbehavior{Returns LwSciError_Success }
 *
 * @testcase{22060683}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_Readonly
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:0x0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:12
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDupHandle.newPerm:LwSciBufAccessPerm_Readonly
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:0x0
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciBufSysMemDupHandle
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDupHandle.context
{{ <<uut_prototype_stubs.LwSciBufSysMemDupHandle.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context
int var = 4444;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceDupHandle.Success.memHandleMid
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceDupHandle.Success.memHandleMid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceDupHandle.Success.memHandleMid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() memHandle set to Mid value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemDupHandle() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_ReadWrite'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value 255
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer struct is initialized with memHandle value 2}
 *
 * @testbehavior{Returns LwSciError_Success }
 *
 * @testcase{22060685}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:0xFF
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:11
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDupHandle.newPerm:LwSciBufAccessPerm_ReadWrite
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:0xFF
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciBufSysMemDupHandle
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDupHandle.context
{{ <<uut_prototype_stubs.LwSciBufSysMemDupHandle.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context
int var = 3321;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceDupHandle.Success.memHandleMax
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceDupHandle.Success.memHandleMax
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceDupHandle.Success.memHandleMax}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() memHandle set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemDupHandle() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_Auto'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value '0xFFFFFFFF'
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer struct is initialized with memHandle value 2}
 *
 * @testbehavior{Returns LwSciError_Success if buffer handle is duplicated with the specified LwSciBufAttrValAccessPerm}
 *
 * @testcase{22060688}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_Auto
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:5
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemDupHandle.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:0x5
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDupHandle.newPerm:LwSciBufAccessPerm_Auto
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciBufSysMemDupHandle
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDupHandle.context
{{ <<uut_prototype_stubs.LwSciBufSysMemDupHandle.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context
int var = 6666;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceDupHandle.Panic_due_to_allocType_set_To_less_than_zero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceDupHandle.Panic_due_to_allocType_set_To_less_than_zero
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceDupHandle.Panic_due_to_allocType_set_To_less_than_zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() Panics when 'allocType' set to less than 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to less than 0.
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_ReadWrite'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value 1
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857817}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context
int var = 66;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceDupHandle.Panic_due_to_context_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceDupHandle.Panic_due_to_context_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceDupHandle.Panic_due_to_context_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() Panics when context pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to NULL
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_ReadWrite'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value 1
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857820}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context:<<null>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_007.LwSciBufAllocIfaceDupHandle.Panic_due_to_newParam_Set_to_LwSciBufAccessPerm_Ilwalid
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_007.LwSciBufAllocIfaceDupHandle.Panic_due_to_newParam_Set_to_LwSciBufAccessPerm_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufAllocIfaceDupHandle.Panic_due_to_newParam_Set_to_LwSciBufAccessPerm_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() Panics when newParam set to 'LwSciBufAccessPerm_Ilwalid'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_Ilwalid'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value 1
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857823}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_Ilwalid
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context
int var = 99;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufAllocIfaceDupHandle.Panic_due_to_dupHandle_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_008.LwSciBufAllocIfaceDupHandle.Panic_due_to_dupHandle_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufAllocIfaceDupHandle.Panic_due_to_dupHandle_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() Panics when dupHandle set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_Ilwalid'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value 1
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857826}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_Ilwalid
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<null>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context
int var = 99;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufAllocIfaceDupHandle.ResourceError
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceDupHandle
TEST.NEW
TEST.NAME:TC_009.LwSciBufAllocIfaceDupHandle.ResourceError
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufAllocIfaceDupHandle.ResourceError}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceDupHandle() memHandle set to Mid value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemDupHandle() returns LwSciError_ResourceError.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'newPerm' - New LwSciBufAttrValAccessPerm enum set to 'LwSciBufAccessPerm_ReadWrite'
 * 4. 'rmHandle' - LwSciBufRmHandle struct of the buffer to duplicate is initialized with memHandle value 255
 * 5. 'dupHandle' - Duplicated LwSciBufRmHandle pointer struct is initialized with memHandle value 2}
 *
 * @testbehavior{Returns LwSciError_ResourceError if LWPU driver stack failed to duplicate the LwSciBufRmHandle.}
 *
 * @testcase{22060643}
 *
 * @verify{18842934}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.newPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:0xFF
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:11
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemDupHandle.return:LwSciError_ResourceError
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDupHandle.newPerm:LwSciBufAccessPerm_ReadWrite
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:0xFF
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
  uut_prototype_stubs.LwSciBufSysMemDupHandle
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceDupHandle
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemDupHandle.context
{{ <<uut_prototype_stubs.LwSciBufSysMemDupHandle.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context
int var = 3321;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufAllocIfaceGetAllocContext

-- Test Case: TC_001.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetAllocContext
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceGetAllocContext.Panic.allocTypesetToLwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetAllocContext() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation context should be created set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'iFaceAllocContextParams'- LwSciBufAllocIfaceAllocContextParams struct is initialized with gpuId count 6
 * 3. 'openContext'- Opaque open context set to valid memory
 * 4. 'allocContext'- Opaque allocation context set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857829}
 *
 * @verify{18842925}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.iFaceAllocContextParams.gpuIdsCount:6
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext
int var = 77745;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufAllocIfaceGetAllocContext.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetAllocContext
TEST.NEW
TEST.NAME:TC_002.LwSciBufAllocIfaceGetAllocContext.Failure_due_to_LwSciCommonCalloc_returns_NULL
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufAllocIfaceGetAllocContext.Failure_due_to_LwSciCommonCalloc_returns_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetAllocContext() gpuIdCount set to Min value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns NULL}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'iFaceAllocContextParams'- LwSciBufAllocIfaceAllocContextParams struct is initialized with gpuId count 1
 * 3. 'openContext'- Opaque open context set to valid memory
 * 4. 'allocContext'- Opaque allocation context pointer set to valid memory}
 *
 * @testbehavior{Returns LwSciError_InsufficientMemory if there is insufficient memory to complete the operation.}
 *
 * @testcase{18857832}
 *
 * @verify{18842925}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.iFaceAllocContextParams.gpuIdsCount:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.return:LwSciError_NotImplemented
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
  uut_prototype_stubs.LwSciCommonCalloc
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemAllocContextParam) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext
int var = 6666;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceGetAllocContext.Success_due_to_gpuIdCount_set_to_Mid
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetAllocContext
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceGetAllocContext.Success_due_to_gpuIdCount_set_to_Mid
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceGetAllocContext.Success_due_to_gpuIdCount_set_to_Mid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetAllocContext() gpuIdCount set to Mid value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciBufSysMemGetAllocContext() returns LwSciError_NotSupported
 * LwSciCommonFree() points to valid memory}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'iFaceAllocContextParams'- LwSciBufAllocIfaceAllocContextParams struct is initialized with gpuId count 4
 * 3. 'openContext'- Opaque open context set to valid memory
 * 4. 'allocContext'- Opaque allocation context pointer set to valid memory}
 *
 * @testbehavior{Returns LwSciError_Success }
 *
 * @testcase{18857835}
 *
 * @verify{18842925}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.iFaceAllocContextParams.gpuIdsCount:4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.return:LwSciError_NotImplemented
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciBufSysMemGetAllocContext
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.allocContextParamObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemAllocContextParam) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.allocContextParam
{{ <<uut_prototype_stubs.LwSciBufSysMemGetAllocContext.allocContextParam>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.openContext
{{ <<uut_prototype_stubs.LwSciBufSysMemGetAllocContext.openContext>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext
int var = 44789;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceGetAllocContext.Success_due_to_gpuIdCount_set_to_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetAllocContext
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceGetAllocContext.Success_due_to_gpuIdCount_set_to_Max
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceGetAllocContext.Success_due_to_gpuIdCount_set_to_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetAllocContext() gpuIdCount set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciBufSysMemGetAllocContext() returns LwSciError_Success
 * LwSciCommonFree() points to valid memory}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'iFaceAllocContextParams'- LwSciBufAllocIfaceAllocContextParams struct is initialized with gpuId count 14
 * 3. 'openContext'- Opaque open context set to valid memory
 * 4. 'allocContext'- Opaque allocation context pointer set to valid memory}
 *
 * @testbehavior{Returns LwSciError_Success if an opaque allocation context for the specified LwSciBufAllocIfaceType is created}
 *
 * @testcase{18857838}
 *
 * @verify{18842925}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.iFaceAllocContextParams.gpuIdsCount:14
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.return:LwSciError_NotImplemented
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciBufSysMemGetAllocContext
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.allocContextParamObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemAllocContextParam) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.allocContextParam
{{ <<uut_prototype_stubs.LwSciBufSysMemGetAllocContext.allocContextParam>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.openContext
{{ <<uut_prototype_stubs.LwSciBufSysMemGetAllocContext.openContext>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext
int var = 5540;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext
{{ <<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext>> == ( <<uut_prototype_stubs.LwSciBufSysMemGetAllocContext.allocContext>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_allocType_set_To_lessthanzero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetAllocContext
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_allocType_set_To_lessthanzero
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_allocType_set_To_lessthanzero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetAllocContext() Panics when 'allocType' set to less than 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation context should be created set to less than 0
 * 2. 'iFaceAllocContextParams'- LwSciBufAllocIfaceAllocContextParams struct is initialized with gpuId count 6
 * 3. 'openContext'- Opaque open context set to valid memory
 * 4. 'allocContext'- Opaque allocation context set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857841}
 *
 * @verify{18842925}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.iFaceAllocContextParams.gpuIdsCount:6
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext
int var = 77745;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_openContext_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetAllocContext
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_openContext_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_openContext_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetAllocContext() Panics when openContext pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'iFaceAllocContextParams'- LwSciBufAllocIfaceAllocContextParams struct is initialized with gpuId count 6
 * 3. 'openContext'- Opaque open context set to NULL
 * 4. 'allocContext'- Opaque allocation context set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857844}
 *
 * @verify{18842925}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.iFaceAllocContextParams.gpuIdsCount:6
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext:<<malloc 1>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_allocContext_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetAllocContext
TEST.NEW
TEST.NAME:TC_007.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_allocContext_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufAllocIfaceGetAllocContext.Panic_due_to_allocContext_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetAllocContext() Panics when openContext pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'iFaceAllocContextParams'- LwSciBufAllocIfaceAllocContextParams struct is initialized with gpuId count 6
 * 3. 'openContext'- Opaque open context set to valid memory
 * 4. 'allocContext'- Opaque allocation context set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857847}
 *
 * @verify{18842925}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.iFaceAllocContextParams.gpuIdsCount:6
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext
int var = 553;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufAllocIfaceGetAllocContext.InsufficientMemory
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetAllocContext
TEST.NEW
TEST.NAME:TC_008.LwSciBufAllocIfaceGetAllocContext.InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufAllocIfaceGetAllocContext.InsufficientMemory}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetAllocContext() when LwSciBufSysMemGetAllocContext() returns LwSciError_InsufficientMemory }
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciCommonCalloc() returns valid memory
 * LwSciBufSysMemGetAllocContext() returns LwSciError_InsufficientMemory
 * LwSciCommonFree() points to valid memory}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the allocation context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'iFaceAllocContextParams'- LwSciBufAllocIfaceAllocContextParams struct is initialized with gpuId count 4
 * 3. 'openContext'- Opaque open context set to valid memory
 * 4. 'allocContext'- Opaque allocation context pointer set to valid memory}
 *
 * @testbehavior{Returns LwSciError_InsufficientMemory }
 *
 * @testcase{22060646}
 *
 * @verify{18842925}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.iFaceAllocContextParams.gpuIdsCount:4
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.allocContext:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.return:LwSciError_NotImplemented
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciBufSysMemGetAllocContext
  uut_prototype_stubs.LwSciCommonFree
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetAllocContext
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &<<USER_GLOBALS_VCAST.<<GLOBAL>>.allocContextParamObj>> );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciBufAllocSysMemAllocContextParam) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.allocContextParam
{{ <<uut_prototype_stubs.LwSciBufSysMemGetAllocContext.allocContextParam>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetAllocContext.openContext
{{ <<uut_prototype_stubs.LwSciBufSysMemGetAllocContext.openContext>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext
int var = 44789;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetAllocContext.openContext>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufAllocIfaceGetSize

-- Test Case: TC_001.LwSciBufAllocIfaceGetSize.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetSize
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceGetSize.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceGetSize.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetSize() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 5. 'size' -  Size pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857850}
 *
 * @verify{18842943}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.size:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetSize
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context
int var = 555;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceGetSize.Failure_due_to_LwSciBufSysMemGetSize_returns_ResourceError
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetSize
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceGetSize.Failure_due_to_LwSciBufSysMemGetSize_returns_ResourceError
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceGetSize.Failure_due_to_LwSciBufSysMemGetSize_returns_ResourceError}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetSize() memHandle set to Mid value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemGetSize() returns LwSciError_ResourceError.}
 *
 * @testinput{1. 'allocType' set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with memHandle value 255
 * 5. 'size' -  Size pointer of the buffer set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_ResourceError if LWPU driver stack failed to get the size of the buffer.}
 *
 * @testcase{18857856}
 *
 * @verify{18842943}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:0xFF
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.size:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.return:LwSciError_AccessDenied
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemGetSize.size[0]:4
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemGetSize.return:LwSciError_ResourceError
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:0xFF
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetSize
  uut_prototype_stubs.LwSciBufSysMemGetSize
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetSize
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetSize.context
{{ <<uut_prototype_stubs.LwSciBufSysMemGetSize.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context
int var = 3214;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceGetSize.Success_due_to_LwSciBufAllocIfaceGetSize_returns_Success
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetSize
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceGetSize.Success_due_to_LwSciBufAllocIfaceGetSize_returns_Success
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceGetSize.Success_due_to_LwSciBufAllocIfaceGetSize_returns_Success}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetSize() memHandle set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemGetSize() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType' set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with memHandle value '0xFFFFFFFF'
 * 5. 'size' -  Size pointer of the buffer set to valid memory.}
 *
 * @testbehavior{Returns LwSciError_Success if size of the buffer got successfully}
 *
 * @testcase{18857859}
 *
 * @verify{18842943}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.size:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.return:LwSciError_AccessDenied
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemGetSize.size[0]:2
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemGetSize.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.size[0]:2
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.return:LwSciError_Success
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetSize
  uut_prototype_stubs.LwSciBufSysMemGetSize
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetSize
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetSize.context
{{ <<uut_prototype_stubs.LwSciBufSysMemGetSize.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context
int var = 3345;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceGetSize.Panic.allocTypesetTolessthanzero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetSize
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceGetSize.Panic.allocTypesetTolessthanzero
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceGetSize.Panic.allocTypesetTolessthanzero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetSize() Panics when 'allocType' set to less than 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to less than 0
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 5. 'size' -  Size pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060654}
 *
 * @verify{18842943}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.size:<<malloc 1>>
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetSize
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context
int var = 555;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceGetSize.Panic_due_to_context_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetSize
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceGetSize.Panic_due_to_context_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceGetSize.Panic_due_to_context_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetSize() Panics when context pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to NULL
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 5. 'size' -  Size pointer of the buffer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857865}
 *
 * @verify{18842943}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.size:<<malloc 1>>
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetSize
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufAllocIfaceGetSize.Panic_due_to_size_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceGetSize
TEST.NEW
TEST.NAME:TC_007.LwSciBufAllocIfaceGetSize.Panic_due_to_size_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufAllocIfaceGetSize.Panic_due_to_size_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceGetSize() Panics when size pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 5. 'size' -  Size pointer of the buffer set to NULL.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857868}
 *
 * @verify{18842943}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:0x5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.size:<<null>>
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceGetSize
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context
int var = 999;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.context>> = ( &var );

TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufAllocIfaceMemMap

-- Test Case: TC_001.LwSciBufAllocIfaceMemMap.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceMemMap.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceMemMap.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 4. 'offset' - starting offset of the buffer set to 1
 * 5. 'len' -  length (in bytes) of the buffer to map set to 3.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_ReadWrite'.
 * 7. 'ptr' - CPU virtual address pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857871}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:3
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceCpuCacheFlush.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemCpuCacheFlush.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 88;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceMemMap.Failure_due_to_LwSciBufSysMemMemMap_returns_ResourceError
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceMemMap.Failure_due_to_LwSciBufSysMemMemMap_returns_ResourceError
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceMemMap.Failure_due_to_LwSciBufSysMemMemMap_returns_ResourceError}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() memHandle set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemMap() returns LwSciError_ResourceError.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer points to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with value '0xFFFFFFFF'
 * 4. 'offset' - starting offset of the buffer set to 2
 * 5. 'len' -  length (in bytes) of the buffer to map set to 3.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_ReadWrite'.
 * 7. 'ptr' - CPU virtual address pointer points to valid memory}
 *
 * @testbehavior{Returns LwSciError_ResourceError if LWPU driver stack failed to map the buffer.}
 *
 * @testcase{18857877}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:0xFFFFFFFF
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:2
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:3
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_NotPermitted
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemMap.return:LwSciError_ResourceError
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:0xFFFFFFFF
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.offset:2
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.len:3
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.accPerm:LwSciBufAccessPerm_ReadWrite
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciBufSysMemMemMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 5555;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceMemMap.Success_due_to_offset_set_to_Min_value
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceMemMap.Success_due_to_offset_set_to_Min_value
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceMemMap.Success_due_to_offset_set_to_Min_value}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() offset set to Min value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemMap() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer points to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with value '0xFFFFFFFF'
 * 4. 'offset' - starting offset of the buffer set to 0
 * 5. 'len' -  length (in bytes) of the buffer to map set to 15.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_Auto'.
 * 7. 'ptr' - CPU virtual address pointer points to valid memory}
 *
 * @testbehavior{Returns LwSciError_Success if Mapping of buffer is done.}
 *
 * @testcase{18857880}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:15
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_Auto
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_NotPermitted
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemMap.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.offset:0
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.len:15
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.accPerm:LwSciBufAccessPerm_Auto
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciBufSysMemMemMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 8888;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr
{{ <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr>> == ( <<uut_prototype_stubs.LwSciBufSysMemMemMap.ptr>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceMemMap.Success_due_to_offset_set_to_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceMemMap.Success_due_to_offset_set_to_Max
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceMemMap.Success_due_to_offset_set_to_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() offset set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemMap() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer points to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with value '0xFFFFFFFF'
 * 4. 'offset' - starting offset of the buffer set to 10
 * 5. 'len' -  length (in bytes) of the buffer to map set to 15.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_Auto'.
 * 7. 'ptr' - CPU virtual address pointer points to valid memory}
 *
 * @testbehavior{Returns LwSciError_Success if Mapping of buffer is done.}
 *
 * @testcase{18857883}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:10
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:15
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_Auto
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_NotPermitted
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemMap.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.offset:10
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.len:15
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.accPerm:LwSciBufAccessPerm_Auto
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciBufSysMemMemMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 99;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr
{{ <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr>> == ( <<uut_prototype_stubs.LwSciBufSysMemMemMap.ptr>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceMemMap.Success_due_to_length_set_to_Min_value
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceMemMap.Success_due_to_length_set_to_Min_value
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceMemMap.Success_due_to_length_set_to_Min_value}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() length set to Min value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemMap() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer points to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with value '0xFFFFFFFF'
 * 4. 'offset' - starting offset of the buffer set to 10
 * 5. 'len' -  length (in bytes) of the buffer to map set to 1.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_Auto'.
 * 7. 'ptr' - CPU virtual address pointer points to valid memory}
 *
 * @testbehavior{Returns LwSciError_Success if Mapping of buffer is done.}
 *
 * @testcase{18857886}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:10
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_Auto
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_NotPermitted
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemMap.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.offset:10
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.len:1
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.accPerm:LwSciBufAccessPerm_Auto
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciBufSysMemMemMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 29;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr
{{ <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr>> == ( <<uut_prototype_stubs.LwSciBufSysMemMemMap.ptr>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufAllocIfaceMemMap.Success_due_to_len_set_to_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_007.LwSciBufAllocIfaceMemMap.Success_due_to_len_set_to_Max
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufAllocIfaceMemMap.Success_due_to_len_set_to_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() length set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemMap() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer points to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with value '0xFFFFFFFF'
 * 4. 'offset' - starting offset of the buffer set to 10
 * 5. 'len' -  length (in bytes) of the buffer to map set to 15.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_Auto'.
 * 7. 'ptr' - CPU virtual address pointer points to valid memory}
 *
 * @testbehavior{Returns LwSciError_Success if Mapping of buffer is done.}
 *
 * @testcase{18857889}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:10
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:15
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_Auto
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_NotPermitted
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemMap.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.offset:10
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.len:15
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.accPerm:LwSciBufAccessPerm_Auto
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciBufSysMemMemMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 1123;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr
{{ <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr>> == ( <<uut_prototype_stubs.LwSciBufSysMemMemMap.ptr>> ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufAllocIfaceMemMap.ResourceError.memHandleMin
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_008.LwSciBufAllocIfaceMemMap.ResourceError.memHandleMin
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufAllocIfaceMemMap.ResourceError.memHandleMin}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() memHandle set to Min value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemMap() returns LwSciError_ResourceError.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer points to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with value '0'
 * 4. 'offset' - starting offset of the buffer set to 2
 * 5. 'len' -  length (in bytes) of the buffer to map set to 3.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_Auto'.
 * 7. 'ptr' - CPU virtual address pointer points to valid memory}
 *
 * @testbehavior{Returns LwSciError_ResourceError if LWPU driver stack failed to map the buffer.}
 *
 * @testcase{22060651}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:0x0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:2
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:3
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_Auto
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_NotPermitted
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemMap.return:LwSciError_ResourceError
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:0x0
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.offset:2
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.len:3
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemMap.accPerm:LwSciBufAccessPerm_Auto
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciBufSysMemMemMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 5543;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufAllocIfaceMemMap.Panic_due_to_allocType_set_To_less_than_zero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_009.LwSciBufAllocIfaceMemMap.Panic_due_to_allocType_set_To_less_than_zero
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufAllocIfaceMemMap.Panic_due_to_allocType_set_To_less_than_zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() Panics when 'allocType' set to less than 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to less than 0
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 4. 'offset' - starting offset of the buffer set to 1
 * 5. 'len' -  length (in bytes) of the buffer to map set to 3.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_ReadWrite'.
 * 7. 'ptr' - CPU virtual address pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857895}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:3
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 88;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufAllocIfaceMemMap.Panic_due_to_context_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_010.LwSciBufAllocIfaceMemMap.Panic_due_to_context_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufAllocIfaceMemMap.Panic_due_to_context_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() Panics when context set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to NULL
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 4. 'offset' - starting offset of the buffer set to 1
 * 5. 'len' -  length (in bytes) of the buffer to map set to 3.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_ReadWrite'.
 * 7. 'ptr' - CPU virtual address pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857898}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context:<<null>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:3
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_ReadWrite
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_011.LwSciBufAllocIfaceMemMap.Panic_due_to_iFaceAccPerm_Set_to_LwSciBufAccessPerm_Ilwalid
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_011.LwSciBufAllocIfaceMemMap.Panic_due_to_iFaceAccPerm_Set_to_LwSciBufAccessPerm_Ilwalid
TEST.NOTES:
/**
 * @testname{TC_011.LwSciBufAllocIfaceMemMap.Panic_due_to_iFaceAccPerm_Set_to_LwSciBufAccessPerm_Ilwalid}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() Panics when iFaceAccPerm set to LwSciBufAccessPerm_Ilwalid}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 4. 'offset' - starting offset of the buffer set to 1
 * 5. 'len' -  length (in bytes) of the buffer to map set to 3.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_Ilwalid'.
 * 7. 'ptr' - CPU virtual address pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857901}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:3
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_Ilwalid
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 778;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciBufAllocIfaceMemMap.Panic_due_to_len_Set_to_zero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_012.LwSciBufAllocIfaceMemMap.Panic_due_to_len_Set_to_zero
TEST.NOTES:
/**
 * @testname{TC_012.LwSciBufAllocIfaceMemMap.Panic_due_to_len_Set_to_zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() Panics when len set to 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 4. 'offset' - starting offset of the buffer set to 1
 * 5. 'len' -  length (in bytes) of the buffer to map set to 0.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_Auto'.
 * 7. 'ptr' - CPU virtual address pointer set to valid memory.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857904}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:0
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_Auto
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 778;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciBufAllocIfaceMemMap.Panic_due_to_ptr_is_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemMap
TEST.NEW
TEST.NAME:TC_013.LwSciBufAllocIfaceMemMap.Panic_due_to_ptr_is_NULL
TEST.NOTES:
/**
 * @testname{TC_013.LwSciBufAllocIfaceMemMap.Panic_due_to_ptr_is_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemMap() Panics when ptr set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized
 * 4. 'offset' - starting offset of the buffer set to 1
 * 5. 'len' -  length (in bytes) of the buffer to map set to 5.
 * 6. 'iFaceAccPerm' - enum LwSciBufAttrValAccessPerm of the mapped buffer set to 'LwSciBufAccessPerm_Auto'.
 * 7. 'ptr' - CPU virtual address pointer set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{22060647}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.offset:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.len:5
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.iFaceAccPerm:LwSciBufAccessPerm_Auto
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.ptr:<<null>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context
int var = 778;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufAllocIfaceMemUnMap

-- Test Case: TC_001.LwSciBufAllocIfaceMemUnMap.Panic.allocTypesetToLwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceMemUnMap.Panic.allocTypesetToLwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceMemUnMap.Panic.allocTypesetToLwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer to unmap is initialized
 * 4. 'ptr' - CPU virtual address pointer set to valid memory.
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857907}
 *
 * @verify{18842937}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
int var = 666;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr
int var = 777;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceMemUnMap.Failure_due_to_LwSciBufSysMemMemUnMap_returns_ResourceError
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceMemUnMap.Failure_due_to_LwSciBufSysMemMemUnMap_returns_ResourceError
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceMemUnMap.Failure_due_to_LwSciBufSysMemMemUnMap_returns_ResourceError}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() memHandle set to Mid value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemUnMap() returns LwSciError_ResourceError.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with memHandle value 255
 * 4. 'ptr' - CPU virtual address pointer set to valid memory
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 2.}
 *
 * @testbehavior{Returns LwSciError_ResourceError if LWPU driver stack failed to unmap the buffer.}
 *
 * @testcase{18857916}
 *
 * @verify{18842940}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:0xFF
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:2
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.return:LwSciError_IlwalidOperation
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.return:LwSciError_ResourceError
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:0xFF
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemUnMap.size:2
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceGetSize.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemGetSize.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciBufSysMemMemUnMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.ptr
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.ptr>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
int var = 1190;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr
int var = 5543;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceMemUnMap.Success_due_to_memHandle_set_to_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceMemUnMap.Success_due_to_memHandle_set_to_Max
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceMemUnMap.Success_due_to_memHandle_set_to_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() memHandle set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemUnMap() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with memHandle value '0xFFFFFFFF'
 * 4. 'ptr' - CPU virtual address pointer set to valid memory
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 2.}
 *
 * @testbehavior{Returns LwSciError_Success if unmapping of the mapped buffer is done}
 *
 * @testcase{18857919}
 *
 * @verify{18842940}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:2
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.return:LwSciError_IlwalidState
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemUnMap.size:2
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciBufSysMemMemUnMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.ptr
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.ptr>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
int var = 2212;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr
int var = 2212;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceMemUnMap.Success_due_to_size_set_to_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceMemUnMap.Success_due_to_size_set_to_Max
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceMemUnMap.Success_due_to_size_set_to_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() size set to Max value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemUnMap() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with memHandle value '0xFFFFFFFF'
 * 4. 'ptr' - CPU virtual address pointer set to valid memory
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 9.}
 *
 * @testbehavior{Returns LwSciError_Success if unmapping of the mapped buffer is done}
 *
 * @testcase{18857922}
 *
 * @verify{18842940}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:9
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.return:LwSciError_IlwalidState
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemUnMap.size:9
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciBufSysMemMemUnMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.ptr
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.ptr>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
int var = 7765;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr
int var = 2216;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceMemUnMap.Success_due_to_size_set_to_Min
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceMemUnMap.Success_due_to_size_set_to_Min
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceMemUnMap.Success_due_to_size_set_to_Min}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() size set to Min value}
 *
 * @casederiv{Analysis of Requirement
 * Analysis of Boundary Values}
 *
 * @testsetup{LwSciBufSysMemMemUnMap() returns LwSciError_Success.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle struct of the buffer to map is initialized with memHandle value '0xFFFFFFFF'
 * 4. 'ptr' - CPU virtual address pointer set to valid memory
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 1.}
 *
 * @testbehavior{Returns LwSciError_Success if unmapping of the mapped buffer is done}
 *
 * @testcase{18857925}
 *
 * @verify{18842940}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.return:LwSciError_IlwalidState
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciBufSysMemMemUnMap.size:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciBufSysMemMemUnMap
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.context
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.context>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle>>.memHandle == ( UINT32_MAX ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemMemUnMap.ptr
{{ <<uut_prototype_stubs.LwSciBufSysMemMemUnMap.ptr>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
int var = 3326;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle>>.memHandle = ( UINT32_MAX );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr
int var = 2267;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciBufAllocIfaceMemUnMap.Panic_due_to_allocType_set_To_lessthanzero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_007.LwSciBufAllocIfaceMemUnMap.Panic_due_to_allocType_set_To_lessthanzero
TEST.NOTES:
/**
 * @testname{TC_007.LwSciBufAllocIfaceMemUnMap.Panic_due_to_allocType_set_To_lessthanzero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() Panics when 'allocType' set to less than 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer to unmap is initialized
 * 4. 'ptr' - CPU virtual address pointer set to valid memory.
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857928}
 *
 * @verify{18842940}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
int var = 666;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr
int var = 777;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciBufAllocIfaceMemUnMap.Panic_due_to_context_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_008.LwSciBufAllocIfaceMemUnMap.Panic_due_to_context_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_008.LwSciBufAllocIfaceMemUnMap.Panic_due_to_context_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() Panics when context set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to NULL
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer to unmap is initialized
 * 4. 'ptr' - CPU virtual address pointer set to valid memory.
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 1.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857931}
 *
 * @verify{18842940}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:1
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr
int var = 777;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciBufAllocIfaceMemUnMap.Panic_due_to_size_Set_to_zero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_009.LwSciBufAllocIfaceMemUnMap.Panic_due_to_size_Set_to_zero
TEST.NOTES:
/**
 * @testname{TC_009.LwSciBufAllocIfaceMemUnMap.Panic_due_to_size_Set_to_zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() Panics when size set to 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer to unmap is initialized
 * 4. 'ptr' - CPU virtual address pointer set to valid memory.
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 0.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857934}
 *
 * @verify{18842940}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:0
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
int var = 666;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr
int var = 777;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciBufAllocIfaceMemUnMap.Panic_due_to_ptr_Set_to_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceMemUnMap
TEST.NEW
TEST.NAME:TC_010.LwSciBufAllocIfaceMemUnMap.Panic_due_to_ptr_Set_to_NULL
TEST.NOTES:
/**
 * @testname{TC_010.LwSciBufAllocIfaceMemUnMap.Panic_due_to_ptr_Set_to_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceMemUnMap() Panics when ptr set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'context'- Opaque allocation context pointer set to valid memory
 * 3. 'rmHandle' - LwSciBufRmHandle of the buffer to unmap is initialized
 * 4. 'ptr' - CPU virtual address pointer set to NULL
 * 5. 'size' -  length (in bytes) of the mapped buffer set to 2.}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857937}
 *
 * @verify{18842940}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.rmHandle.memHandle:0x1
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.ptr:<<null>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.size:2
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.dupHandle[0].memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.dupRmHandle[0].memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemUnMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceMemUnMap
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context
int var = 666;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceMemUnMap.context>> = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciBufAllocIfaceOpen

-- Test Case: TC_001.LwSciBufAllocIfaceOpen.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceOpen
TEST.NEW
TEST.NAME:TC_001.LwSciBufAllocIfaceOpen.Panic_due_to_allocType_set_To_LwSciBufAllocIfaceType_Max
TEST.NOTES:
/**
 * @testname{TC_001.LwSciBufAllocIfaceOpen.Panic.allocTypesetToLwSciBufAllocIfaceType_Max}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceOpen() Panics when 'allocType' set to 'LwSciBufAllocIfaceType_Max'}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType' - LwSciBufAllocIfaceType for which the opaque open context should be created set to enum 'LwSciBufAllocIfaceType_Max'
 * 2. 'devHandle' - devHandle pointer of LwSciBufDev struct set to valid memory
 * 3. 'context'- Opaque open context pointer set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857940}
 *
 * @verify{18842922}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.allocType:LwSciBufAllocIfaceType_Max
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceOpen
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciBufAllocIfaceOpen.Failure_due_to_LwSciBufSysMemOpen_returns_InsufficientMemory
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceOpen
TEST.NEW
TEST.NAME:TC_002.LwSciBufAllocIfaceOpen.Failure_due_to_LwSciBufSysMemOpen_returns_InsufficientMemory
TEST.NOTES:
/**
 * @testname{TC_002.LwSciBufAllocIfaceOpen.Failure_due_to_LwSciBufSysMemOpen_returns_InsufficientMemory}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceOpen() which returns LwSciError_InsufficientMemory}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciBufSysMemOpen() returns 'LwSciError_InsufficientMemory'.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the opaque open context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'devHandle' - devHandle pointer of LwSciBufDev struct set to valid memory
 * 3. 'context'- Opaque open context pointer set to valid memory}
 *
 * @testbehavior{Returns LwSciError_InsufficientMemory}
 *
 * @testcase{18857943}
 *
 * @verify{18842922}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemOpen.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.return:LwSciError_InsufficientMemory
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceOpen
  uut_prototype_stubs.LwSciBufSysMemOpen
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceOpen
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemOpen.devHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemOpen.devHandle>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciBufAllocIfaceOpen.Success_due_to_LwSciBufSysMemOpen_returns_Success
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceOpen
TEST.NEW
TEST.NAME:TC_003.LwSciBufAllocIfaceOpen.Success_due_to_LwSciBufSysMemOpen_returns_Success
TEST.NOTES:
/**
 * @testname{TC_003.LwSciBufAllocIfaceOpen.Success_due_to_LwSciBufSysMemOpen_returns_Success}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceOpen() opaque open context created}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{LwSciBufSysMemOpen() returns 'LwSciError_Success'.}
 *
 * @testinput{1. 'allocType'- LwSciBufAllocIfaceType for which the opaque open context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'devHandle' - devHandle pointer of LwSciBufDev struct set to valid memory
 * 3. 'context'- Opaque open context pointer set to valid memory}
 *
 * @testbehavior{Returns LwSciError_Success if opaque open context corresponding to specified LwSciBufAllocIfaceType is created}
 *
 * @testcase{18857946}
 *
 * @verify{18842922}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context:<<malloc 1>>
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.return:LwSciError_Unknown
TEST.VALUE:uut_prototype_stubs.LwSciBufSysMemOpen.return:LwSciError_Success
TEST.EXPECTED:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.return:LwSciError_Success
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDeAlloc.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceDupHandle.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceMemMap.rmHandle.memHandle:INPUT_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemDupHandle.rmHandle.memHandle:EXPECTED_BASE=16
TEST.ATTRIBUTES:uut_prototype_stubs.LwSciBufSysMemMemMap.rmHandle.memHandle:EXPECTED_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceOpen
  uut_prototype_stubs.LwSciBufSysMemOpen
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceOpen
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemOpen.devHandle
{{ <<uut_prototype_stubs.LwSciBufSysMemOpen.devHandle>> == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciBufSysMemOpen.context.context[0]
{{ <<uut_prototype_stubs.LwSciBufSysMemOpen.context>>[0] == ( <<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context>>[0] ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context.context[0]
int var = 8;
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context>>[0] = ( &var );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciBufAllocIfaceOpen.Panic_due_to_devHandle_set_To_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceOpen
TEST.NEW
TEST.NAME:TC_004.LwSciBufAllocIfaceOpen.Panic_due_to_devHandle_set_To_NULL
TEST.NOTES:
/**
 * @testname{TC_004.LwSciBufAllocIfaceOpen.Panic_due_to_devHandle_set_To_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceOpen() Panics when 'devHandle' pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType' - LwSciBufAllocIfaceType for which the opaque open context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'devHandle' - devHandle pointer of LwSciBufDev struct set to NULL
 * 3. 'context'- Opaque open context pointer set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857952}
 *
 * @verify{18842922}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceOpen
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciBufAllocIfaceOpen.Panic_due_to_context_set_To_NULL
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceOpen
TEST.NEW
TEST.NAME:TC_005.LwSciBufAllocIfaceOpen.Panic_due_to_context_set_To_NULL
TEST.NOTES:
/**
 * @testname{TC_005.LwSciBufAllocIfaceOpen.Panic_due_to_context_set_To_NULL}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceOpen() Panics when 'context' pointer set to NULL}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType' - LwSciBufAllocIfaceType for which the opaque open context should be created set to enum 'LwSciBufAllocIfaceType_SysMem'
 * 2. 'devHandle' - devHandle pointer of LwSciBufDev struct set to valid memory
 * 3. 'context'- Opaque open context pointer set to NULL}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857955}
 *
 * @verify{18842922}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.allocType:LwSciBufAllocIfaceType_SysMem
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context:<<null>>
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceOpen
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciBufAllocIfaceOpen.Panic_due_to_allocType_Set_lessthan_zero
TEST.UNIT:lwscibuf_alloc_interface
TEST.SUBPROGRAM:LwSciBufAllocIfaceOpen
TEST.NEW
TEST.NAME:TC_006.LwSciBufAllocIfaceOpen.Panic_due_to_allocType_Set_lessthan_zero
TEST.NOTES:
/**
 * @testname{TC_006.LwSciBufAllocIfaceOpen.Panic_due_to_allocType_Set_lessthan_zero}
 *
 * @verifyFunction{This test-case verifies the functionality of the API - LwSciBufAllocIfaceOpen() Panics when 'allocType' set less than 0}
 *
 * @casederiv{Analysis of Requirement}
 *
 * @testsetup{None.}
 *
 * @testinput{1. 'allocType' - LwSciBufAllocIfaceType for which the opaque open context should be created set less than 0
 * 2. 'devHandle' - devHandle pointer of LwSciBufDev struct set to valid memory
 * 3. 'context'- Opaque open context pointer set to valid memory}
 *
 * @testbehavior{LwSciCommonPanic() is called to terminate exelwtion of the program.}
 *
 * @testcase{18857958}
 *
 * @verify{18842922}
 */
TEST.END_NOTES:
TEST.VALUE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.context:<<malloc 1>>
TEST.ATTRIBUTES:lwscibuf_alloc_interface.LwSciBufAllocIfaceAlloc.iFaceAllocVal.numHeaps:INPUT_BASE=16
TEST.FLOW
  lwscibuf_alloc_interface.c.LwSciBufAllocIfaceOpen
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.allocType
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.allocType>> = ( -1 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle
<<lwscibuf_alloc_interface.LwSciBufAllocIfaceOpen.devHandle>> = ( (LwSciBufDev *) malloc(sizeof(LwSciBufDev)) );
TEST.END_VALUE_USER_CODE:
TEST.END



