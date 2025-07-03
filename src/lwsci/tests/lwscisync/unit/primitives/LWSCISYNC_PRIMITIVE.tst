-- VectorCAST 20.sp5 (12/16/20)
-- Test Case Script
--
-- Environment    : LWSCISYNC_PRIMITIVE
-- Unit(s) Under Test: lwscisync_primitive lwscisync_primitive_tegra lwscisync_semaphore_stub lwscisync_syncpoint
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

-- Unit: lwscisync_primitive

-- Subprogram: LwSciSyncCoreCopyCpuPrimitives

-- Test Case: TC_001.LwSciSyncCoreCopyCpuPrimitives.Cpu_supported_primitives_are_copied_into_buffer
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreCopyCpuPrimitives
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreCopyCpuPrimitives.Cpu_supported_primitives_are_copied_into_buffer
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreCopyCpuPrimitives.Cpu_supported_primitives_are_copied_into_buffer}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreCopyCpuPrimitives() for success use case when cpu supported primitives are copied into the provided buffer.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.}
 *
 * @testinput{- 'primitiveType' points to a valid memory buffer of '3' elements.
 * - 'len' is set to 'sizeof(LwSciSyncCoreSupportedPrimitives)'.}
 *
 * @testbehavior{- LwSciCommonMemcpyS() receives the address of 'primitiveType' in 'dest', 'len' in 'destSize', address of 'LwSciSyncCoreSupportedPrimitives' in 'src' and 'sizeof(LwSciSyncCoreSupportedPrimitives)' in 'n'.
 *
 * - primitiveType[0] is updated with 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - primitiveType[1] is updated with 'LwSciSyncInternalAttrValPrimitiveType_LowerBound'.
 * - primitiveType[2] is updated with 'LwSciSyncInternalAttrValPrimitiveType_LowerBound'.}
 *
 * @testcase{18854019}
 *
 * @verify{18844782}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.primitiveType:<<malloc 3>>
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.primitiveType[1]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.primitiveType[2]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_primitive.c.LwSciSyncCoreCopyCpuPrimitives
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( <<lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_primitive.<<GLOBAL>>.LwSciSyncCoreSupportedPrimitives>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciSyncCoreSupportedPrimitives) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.len
<<lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.len>> = ( sizeof(LwSciSyncCoreSupportedPrimitives) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreCopyCpuPrimitives.Panic_due_to_null_primitiveType
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreCopyCpuPrimitives
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreCopyCpuPrimitives.Panic_due_to_null_primitiveType
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreCopyCpuPrimitives.Panic_due_to_null_primitiveType}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreCopyCpuPrimitives() for panic use case when primitive type buffer does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'primitiveType' points to 'NULL'.
 * - 'len' is set to 'sizeof(LwSciSyncCoreSupportedPrimitives)'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCoreCopyCpuPrimitives() panics.}
 *
 * @testcase{22060800}
 *
 * @verify{18844782}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.primitiveType:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.len
<<lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.len>> = ( sizeof(LwSciSyncCoreSupportedPrimitives) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreCopyCpuPrimitives.Panic_due_to_primitiveType_buffer_overflow
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreCopyCpuPrimitives
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreCopyCpuPrimitives.Panic_due_to_primitiveType_buffer_overflow
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreCopyCpuPrimitives.Panic_due_to_primitiveType_buffer_overflow}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreCopyCpuPrimitives() for panic use case when overflow oclwrs in the provided buffer while copying cpu supported primitives.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciCommonMemcpyS() panics because 'destSize' is less than 'n'.}
 *
 * @testinput{- 'primitiveType' points to a valid memory buffer of '2' elements.
 * - 'len' is set to '1'.}
 *
 * @testbehavior{- LwSciCommonMemcpyS() receives the address of 'primitiveType' in 'dest', 'len' in 'destSize', address of 'LwSciSyncCoreSupportedPrimitives' in 'src' and 'sizeof(LwSciSyncCoreSupportedPrimitives)' in 'n'
 *
 * - LwSciSyncCoreCopyCpuPrimitives() panics.}
 *
 * @testcase{22060804}
 *
 * @verify{18844782}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.primitiveType:<<malloc 2>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.len:1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreCopyCpuPrimitives
  uut_prototype_stubs.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.primitiveType>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( <<lwscisync_primitive.LwSciSyncCoreCopyCpuPrimitives.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_primitive.<<GLOBAL>>.LwSciSyncCoreSupportedPrimitives>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciSyncCoreSupportedPrimitives) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreDeinitPrimitive

-- Test Case: TC_001.LwSciSyncCoreDeinitPrimitive.Free_LwSciSyncCorePrimitive_backed_by_Syncpoint
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreDeinitPrimitive
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreDeinitPrimitive.Free_LwSciSyncCorePrimitive_backed_by_Syncpoint
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreDeinitPrimitive.Free_LwSciSyncCorePrimitive_backed_by_Syncpoint}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreDeinitPrimitive() for success use case when:
 * - actual primitve exists.
 * - data specific to the actual primitive exists.
 * - primitive is owned.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_info.syncpt' points to the address '0x9999999999999999'.
 *
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'primitive->specificData' points to the address of 'test_info'.
 * - 'primitive->ownsPrimitive' is set to 'true'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{--- syncpoint free --------------------------------------------------------------------------------
 * - LwRmHost1xSyncpointFree() receives the address '0x9999999999999999'.
 * - LwSciCommonFree() receives the address of 'test_info'.
 *
 * --- primitive free --------------------------------------------------------------------------------
 * - LwSciCommonFree() receives the address of 'primitive'.}
 *
 * @testcase{18854022}
 *
 * @verify{18844767}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive[0].ownsPrimitive:true
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
  uut_prototype_stubs.LwRmHost1xSyncpointFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_info ) }}
else if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointFree.syncpt
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointFree.syncpt>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.syncpt
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.syncpt = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive.primitive[0].specificData
<<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>>[0].specificData = ( &test_info );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreDeinitPrimitive.Free_LwSciSyncCorePrimitive_without_any_backing_Syncpoint
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreDeinitPrimitive
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreDeinitPrimitive.Free_LwSciSyncCorePrimitive_without_any_backing_Syncpoint
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreDeinitPrimitive.Free_LwSciSyncCorePrimitive_without_any_backing_Syncpoint}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreDeinitPrimitive() for success use case when:
 * - actual primitve exists.
 * - data specific to the actual primitive exists.
 * - primitive is not owned.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'primitive->specificData' points to the address of 'test_info'.
 * - 'primitive->ownsPrimitive' is set to 'false'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{--- syncpoint free --------------------------------------------------------------------------------
 * - LwSciCommonFree() receives the address of 'test_info'.
 *
 * --- primitive free --------------------------------------------------------------------------------
 * - LwSciCommonFree() receives the address of 'primitive'.}
 *
 * @testcase{18854025}
 *
 * @verify{18844767}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive[0].ownsPrimitive:false
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_info ) }}
else if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive.primitive[0].specificData
<<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>>[0].specificData = ( &test_info );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreDeinitPrimitive.Free_primitive_with_null_specificData
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreDeinitPrimitive
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreDeinitPrimitive.Free_primitive_with_null_specificData
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreDeinitPrimitive.Free_primitive_with_null_specificData}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreDeinitPrimitive() for success use case when:
 * - actual primitve exists.
 * - data specific to the actual primitive does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'primitive->specificData' points to 'NULL'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonFree() receives the address of 'primitive'.}
 *
 * @testcase{18854028}
 *
 * @verify{18844767}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( <<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive.primitive[0].specificData
<<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>>[0].specificData = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreDeinitPrimitive.Free_null_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreDeinitPrimitive
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreDeinitPrimitive.Free_null_primitive
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreDeinitPrimitive.Free_null_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreDeinitPrimitive() for success use case when the actual primitive does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'primitive' points to 'NULL'.}
 *
 * @testbehavior{- LwSciSyncCoreDeinitPrimitive() returns successfully without any error.}
 *
 * @testcase{18854031}
 *
 * @verify{18844767}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
TEST.END_FLOW
TEST.END

-- Test Case: TC_005.LwSciSyncCoreDeinitPrimitive.Panic_due_to_ilwalid_primitive_type
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreDeinitPrimitive
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreDeinitPrimitive.Panic_due_to_ilwalid_primitive_type
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreDeinitPrimitive.Panic_due_to_ilwalid_primitive_type}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreDeinitPrimitive() for panic use case when:
 * - actual primitve exists.
 * - primitive is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->ops' points to 'NULL'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{- LwSciSyncCoreDeinitPrimitive() panics.}
 *
 * @testcase{22060807}
 *
 * @verify{18844767}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreDeinitPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreDeinitPrimitive.primitive>>[0].ops = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreGetSupportedPrimitives

-- Test Case: TC_001.LwSciSyncCoreGetSupportedPrimitives.Supported_primitives_are_copied_into_buffer
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreGetSupportedPrimitives
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreGetSupportedPrimitives.Supported_primitives_are_copied_into_buffer
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreGetSupportedPrimitives.Supported_primitives_are_copied_into_buffer}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreGetSupportedPrimitives() for success use case when supported primitives are copied into the provided buffer.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciCommonMemcpyS() copies 'n' number of bytes from 'src' address to 'dest' address.}
 *
 * @testinput{- 'primitiveType' points to a valid memory buffer of '3' elements.
 * - 'len' is set to 'sizeof(LwSciSyncCoreSupportedPrimitives)'.}
 *
 * @testbehavior{- LwSciCommonMemcpyS() receives the address of 'primitiveType' in 'dest', 'len' in 'destSize', address of 'LwSciSyncCoreSupportedPrimitives' in 'src' and 'sizeof(LwSciSyncCoreSupportedPrimitives)' in 'n'.
 *
 * - primitiveType[0] is updated with 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - primitiveType[1] is updated with 'LwSciSyncInternalAttrValPrimitiveType_LowerBound'.
 * - primitiveType[2] is updated with 'LwSciSyncInternalAttrValPrimitiveType_LowerBound'.}
 *
 * @testcase{18854034}
 *
 * @verify{18844785}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.primitiveType:<<malloc 3>>
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.primitiveType[0]:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.primitiveType[1]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.primitiveType[2]:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreGetSupportedPrimitives
  uut_prototype_stubs.LwSciCommonMemcpyS
  lwscisync_primitive.c.LwSciSyncCoreGetSupportedPrimitives
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.primitiveType>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( <<lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_primitive.<<GLOBAL>>.LwSciSyncCoreSupportedPrimitives>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciSyncCoreSupportedPrimitives) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.len
<<lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.len>> = ( sizeof(LwSciSyncCoreSupportedPrimitives) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreGetSupportedPrimitives.Panic_due_to_null_primitiveType
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreGetSupportedPrimitives
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreGetSupportedPrimitives.Panic_due_to_null_primitiveType
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreGetSupportedPrimitives.Panic_due_to_null_primitiveType}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreGetSupportedPrimitives() for panic use case when primitive type buffer does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'primitiveType' points to 'NULL'.
 * - 'len' is set to 'sizeof(LwSciSyncCoreSupportedPrimitives)'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCoreGetSupportedPrimitives() panics.}
 *
 * @testcase{22060810}
 *
 * @verify{18844785}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.primitiveType:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreGetSupportedPrimitives
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.len
<<lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.len>> = ( sizeof(LwSciSyncCoreSupportedPrimitives) );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreGetSupportedPrimitives.Panic_due_to_primitiveType_buffer_overflow
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreGetSupportedPrimitives
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreGetSupportedPrimitives.Panic_due_to_primitiveType_buffer_overflow
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreGetSupportedPrimitives.Panic_due_to_primitiveType_buffer_overflow}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreGetSupportedPrimitives() for panic use case when overflow oclwrs in the provided buffer while copying supported primitives.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciCommonMemcpyS() panics because 'destSize' is less than 'n'.}
 *
 * @testinput{- 'primitiveType' points to a valid memory buffer of '2' elements.
 * - 'len' is set to '1'.}
 *
 * @testbehavior{- LwSciCommonMemcpyS() receives the address of 'primitiveType' in 'dest', 'len' in 'destSize', address of 'LwSciSyncCoreSupportedPrimitives' in 'src' and 'sizeof(LwSciSyncCoreSupportedPrimitives)' in 'n'
 *
 * - LwSciSyncCoreGetSupportedPrimitives() panics.}
 *
 * @testcase{22060813}
 *
 * @verify{18844785}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.primitiveType:<<malloc 2>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.len:1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreGetSupportedPrimitives
  uut_prototype_stubs.LwSciCommonMemcpyS
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.dest
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.dest>> == ( <<lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.primitiveType>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.destSize
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.destSize>> == ( <<lwscisync_primitive.LwSciSyncCoreGetSupportedPrimitives.len>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.src
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.src>> == ( <<lwscisync_primitive.<<GLOBAL>>.LwSciSyncCoreSupportedPrimitives>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonMemcpyS.n
{{ <<uut_prototype_stubs.LwSciCommonMemcpyS.n>> == ( sizeof(LwSciSyncCoreSupportedPrimitives) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreInitPrimitive

-- Test Case: TC_001.LwSciSyncCoreInitPrimitive.Init_LwSciSyncCorePrimitive_backed_by_Syncpoint
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreInitPrimitive.Init_LwSciSyncCorePrimitive_backed_by_Syncpoint
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreInitPrimitive.Init_LwSciSyncCorePrimitive_backed_by_Syncpoint}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for success use case when the primitive is owned.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_syncptAllocAttrs.pool' is set to 'LwRmHost1xSyncpointPool_Default'.
 *
 * --- primitive init --------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciSyncCoreAttrListValidate() returns 'LwSciError_Success'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns the address of 'test_info'.
 *
 * - LwSciSyncCoreAttrListGetModule() writes the address '0x8888888888888888' into memory pointed by 'module'.
 *
 * - LwSciSyncCoreModuleGetRmBackEnd() writes the address '0x7777777777777777' into memory pointed by 'backEnd'.
 *
 * - LwSciSyncCoreRmGetHost1xHandle() returns the address '0x6666666666666666'.
 *
 * - LwRmHost1xGetDefaultSyncpointAllocateAttrs() returns the struct 'test_syncptAllocAttrs'.
 *
 * - LwRmHost1xSyncpointAllocate() writes the address '0x5555555555555555' into memory pointed by 'syncptp'.
 * - LwRmHost1xSyncpointAllocate() returns 'LwError_Success'.
 *
 * - LwRmHost1xSyncpointGetId() returns '0x12345'.
 *
 * - LwRmHost1xSyncpointRead() writes '0x1A2B3C4D' into memory pointed by 'value'.
 * - LwRmHost1xSyncpointRead() returns 'LwError_Success'.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{--- primitive init --------------------------------------------------------------------------------
 * - LwSciSyncCoreAttrListValidate() receives the address '0x9999999999999999' in 'attrList'.
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciSyncCoreAttrListGetModule() receives the address '0x9999999999999999' in 'attrList' and a non-null address in 'module'.
 * - LwSciSyncCoreModuleGetRmBackEnd() receives the address '0x8888888888888888' in  'module' and a non-null address in 'backEnd'.
 * - LwSciSyncCoreRmGetHost1xHandle() receives the address '0x7777777777777777' in 'backEnd'.
 * - LwRmHost1xGetDefaultSyncpointAllocateAttrs() is called with no args.
 * - LwRmHost1xSyncpointAllocate() receives a non-null addresses in 'syncptp', '0x6666666666666666' in 'host1x' and 'LwRmHost1xSyncpointPool_Default' in 'attrs.pool'.
 * - LwRmHost1xSyncpointGetId() receives the address '0x5555555555555555' in 'syncpt'.
 * - LwRmHost1xSyncpointRead() receives the address '0x6666666666666666' in 'host1x', '0x12345' in 'id' and a non-null address in 'value'.
 *
 * --- primitive init --------------------------------------------------------------------------------
 * - memory pointed by 'primitive' is updated to point to the address of 'test_resultPrimitive'.
 * - LwSciSyncCoreInitPrimitive() returns 'LwSciError_Success'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'true'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'test_resultPrimitive.id' is updated to '0x12345'.
 * - 'test_resultPrimitive.lastFence' is updated to '0x1A2B3C4D'.
 * - 'test_resultPrimitive.specificData' is updated to point to the address of 'test_info'.
 *
 * - 'test_info.host1x' is updated to point to the address '0x6666666666666666'.
 * - 'test_info.syncpt' is updated to point to the address '0x5555555555555555'.}
 *
 * @testcase{18854037}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_syncptAllocAttrs.pool:LwRmHost1xSyncpointPool_Default
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.return:LwError_Success
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointGetId.return:0x12345
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointRead.value[0]:0x1A2B3C4D
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointRead.return:LwError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.id:0x12345
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.lastFence:0x1A2B3C4D
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:true
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.attrs.pool:LwRmHost1xSyncpointPool_Default
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointRead.id:0x12345
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle
  uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs
  uut_prototype_stubs.LwRmHost1xSyncpointAllocate
  uut_prototype_stubs.LwRmHost1xSyncpointGetId
  uut_prototype_stubs.LwRmHost1xSyncpointRead
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_info );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs.return
<<uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs.return>> = ( test_syncptAllocAttrs );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp.syncptp[0]
<<uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp>>[0] = ( 0x5555555555555555 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0] = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return>> = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd.backEnd[0]
<<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.host1x
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointAllocate.host1x>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointGetId.syncpt
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointGetId.syncpt>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointRead.host1x
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointRead.host1x>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointRead.value
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointRead.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive>>[0] == ( &test_resultPrimitive ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.specificData
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.specificData == ( &test_info ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.syncpt
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.syncpt == ( 0x5555555555555555 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.host1x
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.host1x == ( 0x6666666666666666 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreInitPrimitive.Init_LwSciSyncCorePrimitive_without_backing_Syncpoint
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreInitPrimitive.Init_LwSciSyncCorePrimitive_without_backing_Syncpoint
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreInitPrimitive.Init_LwSciSyncCorePrimitive_without_backing_Syncpoint}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for success use case when the primitive is not owned.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_syncptAllocAttrs.pool' is set to 'LwRmHost1xSyncpointPool_Default'.
 *
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciSyncCoreAttrListValidate() returns 'LwSciError_Success'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'false'.}
 *
 * @testbehavior{- LwSciSyncCoreAttrListValidate() receives the address '0x9999999999999999' in 'attrList'.
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 *
 * - memory pointed by 'primitive' is updated to point to the address of 'test_resultPrimitive'.
 * - LwSciSyncCoreInitPrimitive() returns 'LwSciError_Success'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'test_resultPrimitive.specificData' is updated to point to 'NULL'.}
 *
 * @testcase{18854046}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:false
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive>>[0] == ( &test_resultPrimitive ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.specificData
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.specificData == ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreInitPrimitive.Failed_to_read_syncpoint_value
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreInitPrimitive.Failed_to_read_syncpoint_value
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreInitPrimitive.Failed_to_read_syncpoint_value}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for failure use case when:
 * - primitive is owned.
 * - failure oclwrs in reading syncpoint value.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_syncptAllocAttrs.pool' is set to 'LwRmHost1xSyncpointPool_Default'.
 *
 * --- primitive init --------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciSyncCoreAttrListValidate() returns 'LwSciError_Success'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns the address of 'test_info'.
 *
 * - LwSciSyncCoreAttrListGetModule() writes the address '0x8888888888888888' into memory pointed by 'module'.
 *
 * - LwSciSyncCoreModuleGetRmBackEnd() writes the address '0x7777777777777777' into memory pointed by 'backEnd'.
 *
 * - LwSciSyncCoreRmGetHost1xHandle() returns the address '0x6666666666666666'.
 *
 * - LwRmHost1xGetDefaultSyncpointAllocateAttrs() returns the struct 'test_syncptAllocAttrs'.
 *
 * - LwRmHost1xSyncpointAllocate() writes the address '0x5555555555555555' into memory pointed by 'syncptp'.
 * - LwRmHost1xSyncpointAllocate() returns 'LwError_Success'.
 *
 * - LwRmHost1xSyncpointGetId() returns '0x12345'.
 *
 * - LwRmHost1xSyncpointRead() returns 'LwError_IoctlFailed'.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{--- primitive init --------------------------------------------------------------------------------
 * - LwSciSyncCoreAttrListValidate() receives the address '0x9999999999999999' in 'attrList'.
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciSyncCoreAttrListGetModule() receives the address '0x9999999999999999' in 'attrList' and a non-null address in 'module'.
 * - LwSciSyncCoreModuleGetRmBackEnd() receives the address '0x8888888888888888' in  'module' and a non-null address in 'backEnd'.
 * - LwSciSyncCoreRmGetHost1xHandle() receives the address '0x7777777777777777' in 'backEnd'.
 * - LwRmHost1xGetDefaultSyncpointAllocateAttrs() is called with no args.
 * - LwRmHost1xSyncpointAllocate() receives a non-null addresses in 'syncptp', '0x6666666666666666' in 'host1x' and 'LwRmHost1xSyncpointPool_Default' in 'attrs.pool'.
 * - LwRmHost1xSyncpointGetId() receives the address '0x5555555555555555' in 'syncpt'.
 * - LwRmHost1xSyncpointRead() receives the address '0x6666666666666666' in 'host1x', '0x12345' in 'id' and a non-null address in 'value'.
 * - LwRmHost1xSyncpointFree() receives the address '0x5555555555555555' in 'syncpt'.
 * - LwSciCommonFree() receives the address of 'test_info' in 'ptr'.
 *
 * --- primitive init --------------------------------------------------------------------------------
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is updated to point to 'NULL'.
 * - LwSciSyncCoreInitPrimitive() returns 'LwSciError_ResourceError'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'true'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * - 'test_info.host1x' is updated to point to the address '0x6666666666666666'.
 * - 'test_info.syncpt' is updated to point to the address '0x5555555555555555'.}
 *
 * @testcase{22060814}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_syncptAllocAttrs.pool:LwRmHost1xSyncpointPool_Default
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.return:LwError_Success
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointGetId.return:0x12345
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointRead.return:LwError_IoctlFailed
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:true
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive[0]:<<null>>
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.attrs.pool:LwRmHost1xSyncpointPool_Default
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointRead.id:0x12345
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle
  uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs
  uut_prototype_stubs.LwRmHost1xSyncpointAllocate
  uut_prototype_stubs.LwRmHost1xSyncpointGetId
  uut_prototype_stubs.LwRmHost1xSyncpointRead
  uut_prototype_stubs.LwRmHost1xSyncpointFree
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_info );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs.return
<<uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs.return>> = ( test_syncptAllocAttrs );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp.syncptp[0]
<<uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp>>[0] = ( 0x5555555555555555 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0] = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return>> = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd.backEnd[0]
<<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_info ) }}
else if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.host1x
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointAllocate.host1x>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointGetId.syncpt
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointGetId.syncpt>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointRead.host1x
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointRead.host1x>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointRead.value
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointRead.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.syncpt
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.syncpt == ( 0x5555555555555555 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.host1x
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.host1x == ( 0x6666666666666666 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreInitPrimitive.Failure_in_syncpoint_allocation
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreInitPrimitive.Failure_in_syncpoint_allocation
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreInitPrimitive.Failure_in_syncpoint_allocation}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for failure use case when:
 * - primitive is owned.
 * - failure oclwrs in reserving syncpoint because of insufficient memory.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_syncptAllocAttrs.pool' is set to 'LwRmHost1xSyncpointPool_Default'.
 *
 * --- primitive init --------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciSyncCoreAttrListValidate() returns 'LwSciError_Success'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns the address of 'test_info'.
 *
 * - LwSciSyncCoreAttrListGetModule() writes the address '0x8888888888888888' into memory pointed by 'module'.
 *
 * - LwSciSyncCoreModuleGetRmBackEnd() writes the address '0x7777777777777777' into memory pointed by 'backEnd'.
 *
 * - LwSciSyncCoreRmGetHost1xHandle() returns the address '0x6666666666666666'.
 *
 * - LwRmHost1xGetDefaultSyncpointAllocateAttrs() returns the struct 'test_syncptAllocAttrs'.
 *
 * - LwRmHost1xSyncpointAllocate() returns 'LwError_InsufficientMemory'.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{--- primitive init --------------------------------------------------------------------------------
 * - LwSciSyncCoreAttrListValidate() receives the address '0x9999999999999999' in 'attrList'.
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciSyncCoreAttrListGetModule() receives the address '0x9999999999999999' in 'attrList' and a non-null address in 'module'.
 * - LwSciSyncCoreModuleGetRmBackEnd() receives the address '0x8888888888888888' in  'module' and a non-null address in 'backEnd'.
 * - LwSciSyncCoreRmGetHost1xHandle() receives the address '0x7777777777777777' in 'backEnd'.
 * - LwRmHost1xGetDefaultSyncpointAllocateAttrs() is called with no args.
 * - LwRmHost1xSyncpointAllocate() receives a non-null addresses in 'syncptp', '0x6666666666666666' in 'host1x' and 'LwRmHost1xSyncpointPool_Default' in 'attrs.pool'.
 * - LwSciCommonFree() receives the address of 'test_info' in 'ptr'.
 *
 * --- primitive init --------------------------------------------------------------------------------
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is updated to point to 'NULL'.
 * - LwSciSyncCoreInitPrimitive() returns 'LwSciError_ResourceError'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'true'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * - 'test_info.host1x' is updated to point to the address '0x6666666666666666'.}
 *
 * @testcase{22060817}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_syncptAllocAttrs.pool:LwRmHost1xSyncpointPool_Default
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.return:LwError_InsufficientMemory
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:true
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive[0]:<<null>>
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.attrs.pool:LwRmHost1xSyncpointPool_Default
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle
  uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs
  uut_prototype_stubs.LwRmHost1xSyncpointAllocate
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_info );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs.return
<<uut_prototype_stubs.LwRmHost1xGetDefaultSyncpointAllocateAttrs.return>> = ( test_syncptAllocAttrs );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0] = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return>> = ( 0x6666666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd.backEnd[0]
<<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_info ) }}
else if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointAllocate.syncptp>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointAllocate.host1x
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointAllocate.host1x>> == ( 0x6666666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.host1x
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.host1x == ( 0x6666666666666666 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_LwSciSyncModule
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_LwSciSyncModule
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_LwSciSyncModule}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for panic use case when:
 * - primitive is owned.
 * - LwSciSyncModule object is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{--- primitive init --------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciSyncCoreAttrListValidate() returns 'LwSciError_Success'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns the address of 'test_info'.
 *
 * - LwSciSyncCoreAttrListGetModule() writes the address '0xFFFFFFFFFFFFFFFF' into memory pointed by 'module'.
 * (Assuming the memory at the address '0xFFFFFFFFFFFFFFFF' is always an invalid LwSciSyncModule object.)
 *
 * - LwSciSyncCoreModuleGetRmBackEnd() panics because 'module' points to invalid LwSciSyncModule object.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{--- primitive init --------------------------------------------------------------------------------
 * - LwSciSyncCoreAttrListValidate() receives the address '0x9999999999999999' in 'attrList'.
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciSyncCoreAttrListGetModule() receives the address '0x9999999999999999' in 'attrList' and a non-null address in 'module'.
 * - LwSciSyncCoreModuleGetRmBackEnd() receives the address '0xFFFFFFFFFFFFFFFF' in  'module' and a non-null address in 'backEnd'.
 *
 * - LwSciSyncCoreInitPrimitive() panics.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'true'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.}
 *
 * @testcase{22060820}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:true
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_info );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0] = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreInitPrimitive.Failed_to_allocate_LwSciSyncCoreSyncpointInfo
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreInitPrimitive.Failed_to_allocate_LwSciSyncCoreSyncpointInfo
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreInitPrimitive.Failed_to_allocate_LwSciSyncCoreSyncpointInfo}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for failure use case when:
 * - primitive is owned.
 * - failure oclwrs in allocating memory for syncpoint info struct because of insufficient memory.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{--- primitive init --------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciSyncCoreAttrListValidate() returns 'LwSciError_Success'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns 'NULL'.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{--- primitive init --------------------------------------------------------------------------------
 * - LwSciSyncCoreAttrListValidate() receives the address '0x9999999999999999' in 'attrList'.
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 *
 * --- syncpoint init --------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciCommonFree() receives 'NULL' in 'ptr'.
 *
 * --- primitive init --------------------------------------------------------------------------------
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is updated to point to 'NULL'.
 * - LwSciSyncCoreInitPrimitive() returns 'LwSciError_InsufficientMemory'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'true'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.}
 *
 * @testcase{18854043}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:true
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive[0]:<<null>>
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreInitPrimitive.Failed_to_allocate_LwSciSyncCorePrimitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreInitPrimitive.Failed_to_allocate_LwSciSyncCorePrimitive
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreInitPrimitive.Failed_to_allocate_LwSciSyncCorePrimitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for failure use case when failure oclwrs in allocating memory for primitive struct because of insufficient memory.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to a valid memory address.
 *
 * - LwSciSyncCoreAttrListValidate() returns 'LwSciError_Success'.
 *
 * - LwSciCommonCalloc() returns 'NULL'.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{- LwSciSyncCoreAttrListValidate() receives the address '0x9999999999999999' in 'attrList'.
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 *
 * - memory pointed by 'primitive' is updated to point to 'NULL'.
 * - LwSciSyncCoreInitPrimitive() returns 'LwSciError_InsufficientMemory'.}
 *
 * @testcase{18854049}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive[0]:<<null>>
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreInitPrimitive.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCoreInitPrimitive.Panic_due_to_null_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreInitPrimitive.Panic_due_to_null_primitive
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreInitPrimitive.Panic_due_to_null_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for panic use case when the actual primitive does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to 'NULL'.
 *
 * - LwSciSyncCoreAttrListValidate() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{- LwSciSyncCoreAttrListValidate() receives the address '0x9999999999999999' in 'attrList'.
 * - LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCoreInitPrimitive() panics.}
 *
 * @testcase{22060823}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<null>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_Success
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_LwSciSyncAttrList
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_LwSciSyncAttrList
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_LwSciSyncAttrList}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for panic use case when LwSciSyncAttrList object is ilwaild.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreAttrListValidate() panics because 'attrList' points to invalid LwSciSyncAttrList object.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to the address '0xFFFFFFFFFFFFFFFF'.
 * (Assuming the memory at the address '0xFFFFFFFFFFFFFFFF' is always an invalid LwSciSyncAttrList object.)
 *
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{- LwSciSyncCoreAttrListValidate() receives the address '0xFFFFFFFFFFFFFFFF' in 'attrList'.
 *
 * - LwSciSyncCoreInitPrimitive() panics.}
 *
 * @testcase{22060826}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncCoreInitPrimitive.Panic_due_to_null_reconciledList
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_010.LwSciSyncCoreInitPrimitive.Panic_due_to_null_reconciledList
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncCoreInitPrimitive.Panic_due_to_null_reconciledList}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for panic use case when reconciled list does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- LwSciSyncCoreAttrListValidate() returns 'LwSciError_BadParameter'.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'reconciledList' points to 'NULL'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{- LwSciSyncCoreAttrListValidate() receives 'NULL' in 'attrList'.
 * - LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCoreInitPrimitive() panics.}
 *
 * @testcase{22060829}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.VALUE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciSyncCoreAttrListValidate
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListValidate.attrList>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_primitiveType
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreInitPrimitive
TEST.NEW
TEST.NAME:TC_011.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_primitiveType
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncCoreInitPrimitive.Panic_due_to_ilwalid_primitiveType}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreInitPrimitive() for panic use case when primitive type is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to a valid memory address.}
 *
 * @testinput{- 'primitiveType' is set to 'LwSciSyncInternalAttrValPrimitiveType_LowerBound'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'needsAllocation' is set to 'true'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCoreInitPrimitive() panics.}
 *
 * @testcase{22060832}
 *
 * @verify{18844764}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitiveType:LwSciSyncInternalAttrValPrimitiveType_LowerBound
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.needsAllocation:true
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreInitPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList
<<lwscisync_primitive.LwSciSyncCoreInitPrimitive.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCorePrimitiveExport

-- Test Case: TC_001.LwSciSyncCorePrimitiveExport.Primitive_is_exported
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCorePrimitiveExport.Primitive_is_exported
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCorePrimitiveExport.Primitive_is_exported}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for success use case when primitive is exported.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to a valid memory address.
 * - 'length' points to a valid memory address.
 *
 * - LwSciCommonTransportAllocTxBufferForKeys() writes the address '0x9999999999999999' into memory pointed by 'txbuf'.
 * - LwSciCommonTransportAllocTxBufferForKeys() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportAppendKeyValuePair() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportAppendKeyValuePair() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportPrepareBufferForTx() writes the address '0x8888888888888888' into memory pointed by 'descBufPtr'.
 * - LwSciCommonTransportPrepareBufferForTx() writes the value '0x123' into memory pointed by 'descBufSize'.
 * - LwSciCommonTransportPrepareBufferForTx) returns 'LwSciError_Success'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciCommonTransportAllocTxBufferForKeys() receives '2' in 'bufParams.keyCount', '12' in 'totalValueSize' and a non-null address in 'txbuf'.
 * - LwSciCommonTransportAppendKeyValuePair() receives the address '0x9999999999999999' in 'txbuf', 'LwSciSyncCorePrimitiveKey_Type' in 'key', 'sizeof(primtive->type)' in 'length' and the address of 'primitive->type' in 'value'.
 * - LwSciCommonTransportAppendKeyValuePair() receives the address '0x9999999999999999' in 'txbuf', 'LwSciSyncCorePrimitiveKey_Id' in 'key', 'sizeof(primtive->id)' in 'length' and the address of 'primitive->id' in 'value'.
 * - LwSciCommonTransportPrepareBufferForTx() receives the address '0x9999999999999999' in 'txbuf'. 'data' in 'descBufPtr' and 'length' in 'descBufSize'.
 * - LwSciCommonTransportBufferFree() receives the address '0x9999999999999999' in 'buf'.
 *
 * - memory pointed by 'data' is updated to point to the address '0x8888888888888888'.
 * - memory pointed by 'length' is updated with '0x123'.
 * - LwSciSyncCorePrimitiveExport() returns 'LwSciError_Success'.}
 *
 * @testcase{18854052}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:(2)LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize[0]:0x123
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length[0]:0x123
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveExport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:12
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:MACRO=LwSciSyncCorePrimitiveKey_Type,MACRO=LwSciSyncCorePrimitiveKey_Id
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf.txbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] = ( 0x9999999999999999 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
*<<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int cntr = 0;
cntr++;

if(cntr == 1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( sizeof(<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type) ) }}
else if(cntr == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( sizeof(<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].id) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type ) }}
else if(cntr == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].id ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.txbuf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufPtr>> == ( <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize
{{ <<uut_prototype_stubs.LwSciCommonTransportPrepareBufferForTx.descBufSize>> == ( <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.length>> ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCorePrimitiveExport.Failed_to_append_key_id_in_export_buffer_due_to_overflow
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCorePrimitiveExport.Failed_to_append_key_id_in_export_buffer_due_to_overflow
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCorePrimitiveExport.Failed_to_append_key_id_in_export_buffer_due_to_overflow}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for failure use case when overflow oclwrs in append key-value pair operation for the key - 'LwSciSyncCorePrimitiveKey_Id'.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to a valid memory address.
 * - memory pointed by 'data' points to the address '0x8888888888888888'.
 * - 'length' points to a valid memory address.
 * - memory pointed by 'length' is set to '0x123'.
 *
 * - LwSciCommonTransportAllocTxBufferForKeys() writes the address '0x9999999999999999' into memory pointed by 'txbuf'.
 * - LwSciCommonTransportAllocTxBufferForKeys() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportAppendKeyValuePair() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportAppendKeyValuePair() returns 'LwSciError_Overflow'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciCommonTransportAllocTxBufferForKeys() receives '2' in 'bufParams.keyCount', '12' in 'totalValueSize' and a non-null address in 'txbuf'.
 * - LwSciCommonTransportAppendKeyValuePair() receives the address '0x9999999999999999' in 'txbuf', 'LwSciSyncCorePrimitiveKey_Type' in 'key', 'sizeof(primtive->type)' in 'length' and the address of 'primitive->type' in 'value'.
 * - LwSciCommonTransportAppendKeyValuePair() receives the address '0x9999999999999999' in 'txbuf', 'LwSciSyncCorePrimitiveKey_Id' in 'key', 'sizeof(primtive->id)' in 'length' and the address of 'primitive->id' in 'value'.
 * - LwSciCommonTransportBufferFree() receives the address '0x9999999999999999' in 'buf'.
 *
 * - memory pointed by 'data' not updated, still points to the address '0x8888888888888888'.
 * - memory pointed by 'length' is not updated, still contains '0x123'.
 * - LwSciSyncCorePrimitiveExport() returns 'LwSciError_Overflow'.}
 *
 * @testcase{22060835}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success,LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:12
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:MACRO=LwSciSyncCorePrimitiveKey_Type,MACRO=LwSciSyncCorePrimitiveKey_Id
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf.txbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] = ( 0x9999999999999999 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int cntr = 0;
cntr++;

if(cntr == 1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( sizeof(<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type) ) }}
else if(cntr == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( sizeof(<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].id) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type ) }}
else if(cntr == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].id ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCorePrimitiveExport.Failed_to_append_key_id_in_export_buffer_due_to_no_space
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCorePrimitiveExport.Failed_to_append_key_id_in_export_buffer_due_to_no_space
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCorePrimitiveExport.Failed_to_append_key_id_in_export_buffer_due_to_no_space}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for failure use case when ran out of space in append key-value pair operation for the key for the key - 'LwSciSyncCorePrimitiveKey_Id'.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to a valid memory address.
 * - memory pointed by 'data' points to the address '0x8888888888888888'.
 * - 'length' points to a valid memory address.
 * - memory pointed by 'length' is set to '0x123'.
 *
 * - LwSciCommonTransportAllocTxBufferForKeys() writes the address '0x9999999999999999' into memory pointed by 'txbuf'.
 * - LwSciCommonTransportAllocTxBufferForKeys() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportAppendKeyValuePair() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportAppendKeyValuePair() returns 'LwSciError_NoSpace'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciCommonTransportAllocTxBufferForKeys() receives '2' in 'bufParams.keyCount', '12' in 'totalValueSize' and a non-null address in 'txbuf'.
 * - LwSciCommonTransportAppendKeyValuePair() receives the address '0x9999999999999999' in 'txbuf', 'LwSciSyncCorePrimitiveKey_Type' in 'key', 'sizeof(primtive->type)' in 'length' and the address of 'primitive->type' in 'value'.
 * - LwSciCommonTransportAppendKeyValuePair() receives the address '0x9999999999999999' in 'txbuf', 'LwSciSyncCorePrimitiveKey_Id' in 'key', 'sizeof(primtive->id)' in 'length' and the address of 'primitive->id' in 'value'.
 * - LwSciCommonTransportBufferFree() receives the address '0x9999999999999999' in 'buf'.
 *
 * - memory pointed by 'data' not updated, still points to the address '0x8888888888888888'.
 * - memory pointed by 'length' is not updated, still contains '0x123'.
 * - LwSciSyncCorePrimitiveExport() returns 'LwSciError_NoSpace'.}
 *
 * @testcase{18854061}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Success,LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:12
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:MACRO=LwSciSyncCorePrimitiveKey_Type,MACRO=LwSciSyncCorePrimitiveKey_Id
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf.txbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] = ( 0x9999999999999999 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
static int cntr = 0;
cntr++;

if(cntr == 1)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( sizeof(<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type) ) }}
else if(cntr == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( sizeof(<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].id) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type ) }}
else if(cntr == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].id ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCorePrimitiveExport.Failed_to_append_key_type_in_export_buffer_due_to_overflow
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCorePrimitiveExport.Failed_to_append_key_type_in_export_buffer_due_to_overflow
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCorePrimitiveExport.Failed_to_append_key_type_in_export_buffer_due_to_overflow}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for failure use case when overflow oclwrs in append key-value pair operation for the key - 'LwSciSyncCorePrimitiveKey_Type.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to a valid memory address.
 * - memory pointed by 'data' points to the address '0x8888888888888888'.
 * - 'length' points to a valid memory address.
 * - memory pointed by 'length' is set to '0x123'.
 *
 * - LwSciCommonTransportAllocTxBufferForKeys() writes the address '0x9999999999999999' into memory pointed by 'txbuf'.
 * - LwSciCommonTransportAllocTxBufferForKeys() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportAppendKeyValuePair() returns 'LwSciError_Overflow'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciCommonTransportAllocTxBufferForKeys() receives '2' in 'bufParams.keyCount', '12' in 'totalValueSize' and a non-null address in 'txbuf'.
 * - LwSciCommonTransportAppendKeyValuePair() receives the address '0x9999999999999999' in 'txbuf', 'LwSciSyncCorePrimitiveKey_Type' in 'key', 'sizeof(primtive->type)' in 'length' and the address of 'primitive->type' in 'value'.
 * - LwSciCommonTransportBufferFree() receives the address '0x9999999999999999' in 'buf'.
 *
 * - memory pointed by 'data' not updated, still points to the address '0x8888888888888888'.
 * - memory pointed by 'length' is not updated, still contains '0x123'.
 * - LwSciSyncCorePrimitiveExport() returns 'LwSciError_Overflow'.}
 *
 * @testcase{22060838}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:12
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:MACRO=LwSciSyncCorePrimitiveKey_Type
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf.txbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] = ( 0x9999999999999999 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( sizeof(<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type ) }}
else if(cntr == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].id ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCorePrimitiveExport.Failed_to_append_key_type_in_export_buffer_due_to_no_space
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCorePrimitiveExport.Failed_to_append_key_type_in_export_buffer_due_to_no_space
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCorePrimitiveExport.Failed_to_append_key_type_in_export_buffer_due_to_no_space}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for failure use case when ran out of space in append key-value pair operation for the key for the key - 'LwSciSyncCorePrimitiveKey_Type.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to a valid memory address.
 * - memory pointed by 'data' points to the address '0x8888888888888888'.
 * - 'length' points to a valid memory address.
 * - memory pointed by 'length' is set to '0x123'.
 *
 * - LwSciCommonTransportAllocTxBufferForKeys() writes the address '0x9999999999999999' into memory pointed by 'txbuf'.
 * - LwSciCommonTransportAllocTxBufferForKeys() returns 'LwSciError_Success'.
 *
 * - LwSciCommonTransportAppendKeyValuePair() returns 'LwSciError_NoSpace'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciCommonTransportAllocTxBufferForKeys() receives '2' in 'bufParams.keyCount', '12' in 'totalValueSize' and a non-null address in 'txbuf'.
 * - LwSciCommonTransportAppendKeyValuePair() receives the address '0x9999999999999999' in 'txbuf', 'LwSciSyncCorePrimitiveKey_Type' in 'key', 'sizeof(primtive->type)' in 'length' and the address of 'primitive->type' in 'value'.
 * - LwSciCommonTransportBufferFree() receives the address '0x9999999999999999' in 'buf'.
 *
 * - memory pointed by 'data' not updated, still points to the address '0x8888888888888888'.
 * - memory pointed by 'length' is not updated, still contains '0x123'.
 * - LwSciSyncCorePrimitiveExport() returns 'LwSciError_NoSpace'.}
 *
 * @testcase{18854064}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.return:LwSciError_NoSpace
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:12
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.key:MACRO=LwSciSyncCorePrimitiveKey_Type
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf.txbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>>[0] = ( 0x9999999999999999 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.txbuf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.length>> == ( sizeof(<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type) ) }}

TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].type ) }}
else if(cntr == 2)
{{ <<uut_prototype_stubs.LwSciCommonTransportAppendKeyValuePair.value>> == ( &<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive>>[0].id ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCorePrimitiveExport.Failed_to_allocate_transport_buffer
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCorePrimitiveExport.Failed_to_allocate_transport_buffer
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCorePrimitiveExport.Failed_to_allocate_transport_buffer}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for failure use case when failure oclwrs in allocating transport buffer due to insufficient memory.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to a valid memory address.
 * - memory pointed by 'data' points to the address '0x8888888888888888'.
 * - 'length' points to a valid memory address.
 * - memory pointed by 'length' is set to '0x123'.
 *
 * - LwSciCommonTransportAllocTxBufferForKeys() returns 'LwSciError_InsufficientMemory'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciCommonTransportAllocTxBufferForKeys() receives '2' in 'bufParams.keyCount', '12' in 'totalValueSize' and a non-null address in 'txbuf'.
 *
 * - memory pointed by 'data' not updated, still points to the address '0x8888888888888888'.
 * - memory pointed by 'length' is not updated, still contains '0x123'.
 * - LwSciSyncCorePrimitiveExport() returns 'LwSciError_InsufficientMemory'.}
 *
 * @testcase{18854067}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length[0]:0x123
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.return:LwSciError_InsufficientMemory
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length[0]:0x123
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveExport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.bufParams.keyCount:2
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.totalValueSize:12
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportAllocTxBufferForKeys.txbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCorePrimitiveExport.Panic_due_to_null_length
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCorePrimitiveExport.Panic_due_to_null_length
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCorePrimitiveExport.Panic_due_to_null_length}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for panic use case when the size of data blob does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to a valid memory address.
 * - memory pointed by 'data' points to the address '0x8888888888888888'.
 * - 'length' points to 'NULL'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciSyncCorePrimitiveExport() panics.}
 *
 * @testcase{18854058}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] == ( 0x8888888888888888 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCorePrimitiveExport.Panic_due_to_null_data
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCorePrimitiveExport.Panic_due_to_null_data
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCorePrimitiveExport.Panic_due_to_null_data}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for panic use case when the data blob does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to 'NULL'.
 * - 'length' points to a valid memory address.
 * - memory pointed by 'length' is set to '0x123'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciSyncCorePrimitiveExport() panics.}
 *
 * @testcase{18854055}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<null>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length[0]:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_009.LwSciSyncCorePrimitiveExport.Panic_due_to_null_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveExport
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCorePrimitiveExport.Panic_due_to_null_primitive
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCorePrimitiveExport.Panic_due_to_null_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveExport() for panic use case when the actual primitive does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'data' points to a valid memory address.
 * - memory pointed by 'data' points to the address '0x8888888888888888'.
 * - 'length' points to a valid memory address
 * - memory pointed by 'length' is set to '0x123'.}
 *
 * @testinput{- 'primitive' points to 'NULL'.
 * - 'ipcEndPoint' is set to '0x1A2B3C4D'}
 *
 * @testbehavior{- LwSciSyncCorePrimitiveExport() panics.}
 *
 * @testcase{22060841}
 *
 * @verify{18844770}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.primitive:<<null>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.length[0]:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveExport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveExport.data.data[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveExport.data>>[0] = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCorePrimitiveGetId

-- Test Case: TC_001.LwSciSyncCorePrimitiveGetId.Primitive_id_is_retrieved
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveGetId
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCorePrimitiveGetId.Primitive_id_is_retrieved
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCorePrimitiveGetId.Primitive_id_is_retrieved}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveGetId() for success use case when primitive id is retrieved.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->id' is set to '0x123'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{- LwSciSyncCorePrimitiveGetId() returns '0x123'.}
 *
 * @testcase{18854070}
 *
 * @verify{18844791}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetId.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetId.primitive[0].id:0x123
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveGetId.return:0x123
TEST.END

-- Test Case: TC_002.LwSciSyncCorePrimitiveGetId.Panic_due_to_null_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveGetId
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCorePrimitiveGetId.Panic_due_to_null_primitive
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCorePrimitiveGetId.Panic_due_to_null_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveGetId() for panic use case when:
 * - 'primitive' points to 'NULL'.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'primitive' points to 'NULL'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCorePrimitiveGetId() panics.}
 *
 * @testcase{22060844}
 *
 * @verify{18844791}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetId.primitive:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveGetId
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Subprogram: LwSciSyncCorePrimitiveGetNewFence

-- Test Case: TC_001.LwSciSyncCorePrimitiveGetNewFence.New_fence_value_is_retrieved
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveGetNewFence
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCorePrimitiveGetNewFence.New_fence_value_is_retrieved
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCorePrimitiveGetNewFence.New_fence_value_is_retrieved}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveGetNewFence() for success use case when a new fence value is retrieved.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'primitive->lastFence' is set to '0x123'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{- 'primitive->lastFence' is updated  to '0x124'.
 *
 * - LwSciSyncCorePrimitiveGetNewFence() returns '0x124'.}
 *
 * @testcase{18854073}
 *
 * @verify{18844788}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive[0].lastFence:0x123
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive[0].lastFence:0x124
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.return:0x124
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveGetNewFence
  lwscisync_primitive.c.LwSciSyncCorePrimitiveGetNewFence
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_primitive_lastFence_is_max
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveGetNewFence
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_primitive_lastFence_is_max
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_primitive_lastFence_is_max}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveGetNewFence() for panic use case when the last fence value is maximum.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'primitive->lastFence' is set to '0xFFFFFFFFFFFFFFFF'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCorePrimitiveGetNewFence() panics.}
 *
 * @testcase{22060847}
 *
 * @verify{18844788}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive[0].lastFence:0xFFFFFFFFFFFFFFFF
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveGetNewFence
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_null_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveGetNewFence
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_null_primitive
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_null_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveGetNewFence() for panic use case when the actual primitive does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'primitive' points to 'NULL'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCorePrimitiveGetNewFence() panics.}
 *
 * @testcase{22060850}
 *
 * @verify{18844788}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveGetNewFence
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_null_primitive_ops
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveGetNewFence
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_null_primitive_ops
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCorePrimitiveGetNewFence.Panic_due_to_null_primitive_ops}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveGetNewFence() for panic use case when getting new fence value is not supported.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->ops' points to 'NULL'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCorePrimitiveGetNewFence() panics.}
 *
 * @testcase{22060853}
 *
 * @verify{18844788}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveGetNewFence
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCorePrimitiveGetNewFence.primitive>>[0].ops = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCorePrimitiveImport

-- Test Case: TC_001.LwSciSyncCorePrimitiveImport.Primitive_is_imported
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCorePrimitiveImport.Primitive_is_imported
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCorePrimitiveImport.Primitive_is_imported}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for success use case when the primitive is imported.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{- 'test_type' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_Id' is set to '0x12345'.
 *
 * --- primitive import ------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '2' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Type' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '4' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_type' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'false' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Id' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '8' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_Id' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'true' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 *
 * --- syncpoint import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns the address of 'test_info'.
 *
 * - LwSciSyncCoreAttrListGetModule() writes the address '0x66666666666666' into memory pointed by 'module'.
 *
 * - LwSciSyncCoreModuleGetRmBackEnd() writes the address '0x5555555555555555' into memory pointed by 'backEnd'.
 *
 * - LwSciSyncCoreRmGetHost1xHandle() returns the address '0x4444444444444444'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{--- primitive import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * --- syncpoint import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciSyncCoreAttrListGetModule() receives the address '0x9999999999999999' in 'attrList' and a non-null address in 'module'.
 * - LwSciSyncCoreModuleGetRmBackEnd() receives the address '0x66666666666666' in  'module' and a non-null address in 'backEnd'.
 * - LwSciSyncCoreRmGetHost1xHandle() receives the address '0x5555555555555555' in 'backEnd'.
 *
 * --- primitive import ------------------------------------------------------------------------------
 * - LwSciCommonTransportBufferFree() receives the address '0x7777777777777777' in 'buf'.
 *
 * - memory pointed by 'primitive' is updated to point to the address of 'test_resultPrimitive'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_Success'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.id' is updated to '0x12345'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'test_resultPrimitive.specificData' is updated to point to the address of 'test_info'.
 *
 * - 'test_info.host1x' is updated to point to the address '0x4444444444444444'.
 * - 'test_info.syncpt' is updated to point to 'NULL'.}
 *
 * @testcase{22060856}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_Id:0x12345
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:2
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCorePrimitiveKey_Type,MACRO=LwSciSyncCorePrimitiveKey_Id
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:4,8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false,true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:(2)LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.id:0x12345
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_info );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static cntr = 0;
cntr++;

if(cntr == 1)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_type );
else if(cntr == 2)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_Id );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0] = ( 0x66666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return>> = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd.backEnd[0]
<<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] = ( 0x5555555555555555 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0x66666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( &test_resultPrimitive ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.specificData
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.specificData == ( &test_info ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.syncpt
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.syncpt == ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.host1x
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.host1x == ( 0x4444444444444444 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCorePrimitiveImport.Panic_due_to_ilwalid_LwSciSyncModule
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCorePrimitiveImport.Panic_due_to_ilwalid_LwSciSyncModule
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCorePrimitiveImport.Panic_due_to_ilwalid_LwSciSyncModule}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for panic use case when LwSciSyncModule object is invalid.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_type' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_Id' is set to '0x12345'.
 *
 * --- primitive import ------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '2' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Type' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '4' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_type' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'false' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Id' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '8' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_Id' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'true' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 *
 * --- syncpoint import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns the address of 'test_info'.
 *
 * - LwSciSyncCoreAttrListGetModule() writes the address '0xFFFFFFFFFFFFFFFF' into memory pointed by 'module'.
 * (Assuming the memory at the address '0xFFFFFFFFFFFFFFFF' is always an invalid LwSciSyncModule object.)
 *
 * - LwSciSyncCoreModuleGetRmBackEnd() panics because 'module' points to invalid LwSciSyncModule object.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{--- primitive import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * --- syncpoint import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciSyncCoreAttrListGetModule() receives the address '0x9999999999999999' in 'attrList' and a non-null address in 'module'.
 * - LwSciSyncCoreModuleGetRmBackEnd() receives the address '0xFFFFFFFFFFFFFFFF' in  'module' and a non-null address in 'backEnd'.
 *
 * - LwSciSyncCorePrimitiveImport() panics.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.id' is updated to '0x12345'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.}
 *
 * @testcase{22060859}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_Id:0x12345
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:2
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCorePrimitiveKey_Type,MACRO=LwSciSyncCorePrimitiveKey_Id
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:4,8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false,true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:(2)LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.id:0x12345
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_info );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static cntr = 0;
cntr++;

if(cntr == 1)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_type );
else if(cntr == 2)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_Id );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0] = ( 0xFFFFFFFFFFFFFFFF );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0xFFFFFFFFFFFFFFFF ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCorePrimitiveImport.Failed_to_allocate_syncpoint_info
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCorePrimitiveImport.Failed_to_allocate_syncpoint_info
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCorePrimitiveImport.Failed_to_allocate_syncpoint_info}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for failure use case when failure oclwrs in allocating memory for syncpoint info struct because of insufficient memory.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_type' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_Id' is set to '0x12345'.
 *
 * --- primitive import ------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 * - memory pointed by 'primitive' points to the address '0x9090909090909090'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '2' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Type' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '4' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_type' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'false' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Id' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '8' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_Id' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'true' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 *
 * --- syncpoint import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns 'NULL'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{--- primitive import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * --- syncpoint import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciCommonFree() receives 'NULL' in 'ptr'.
 *
 * --- primitive import ------------------------------------------------------------------------------
 * - LwSciCommonTransportBufferFree() receives the address '0x7777777777777777' in 'buf'.
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is not updated and still points to the addresss '0x9090909090909090'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_InsufficientMemory'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.id' is updated to '0x12345'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.}
 *
 * @testcase{22060863}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_Id:0x12345
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:2
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCorePrimitiveKey_Type,MACRO=LwSciSyncCorePrimitiveKey_Id
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:4,8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false,true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:(2)LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.id:0x12345
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonFree
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( NULL );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static cntr = 0;
cntr++;

if(cntr == 1)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_type );
else if(cntr == 2)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_Id );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( NULL ) }}
else if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] = ( 0x9090909090909090 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( 0x9090909090909090 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCorePrimitiveImport.Failure_due_to_missing_tag_in_rxbuf
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCorePrimitiveImport.Failure_due_to_missing_tag_in_rxbuf
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCorePrimitiveImport.Failure_due_to_missing_tag_in_rxbuf}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for failure use case when the tag - 'LwSciSyncCorePrimitiveKey_Id' is missed in the core primitive rx buffer.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_type' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 *
 * - 'primitive' points to a valid memory address.
 * - memory pointed by 'primitive' points to the address '0x9090909090909090'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '1' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Type' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '4' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_type' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'true' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * - LwSciCommonTransportBufferFree() receives the address '0x7777777777777777' in 'buf'.
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is not updated and still points to the addresss '0x9090909090909090'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_BadParameter'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.}
 *
 * @testcase{22060866}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:1
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCorePrimitiveKey_Type
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:4
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
*<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_type );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] = ( 0x9090909090909090 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( 0x9090909090909090 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCorePrimitiveImport.Failure_due_to_multiple_oclwrence_of_a_tag_in_rxbuf
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCorePrimitiveImport.Failure_due_to_multiple_oclwrence_of_a_tag_in_rxbuf
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCorePrimitiveImport.Failure_due_to_multiple_oclwrence_of_a_tag_in_rxbuf}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for failure use case when the tag 'LwSciSyncCorePrimitiveKey_Type' is found twice in the core primitive rx buffer.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_type' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 *
 * - 'primitive' points to a valid memory address
 * - memory pointed by 'primitive' points to the address '0x9090909090909090'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '2' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Type' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '4' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_type' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'false' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Type' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '4' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_type' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'true' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * - LwSciCommonTransportBufferFree() receives the address '0x7777777777777777' in 'buf'.
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is not updated and still points to the addresss '0x9090909090909090'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_BadParameter'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.}
 *
 * @testcase{22060870}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:2
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:(2)MACRO=LwSciSyncCorePrimitiveKey_Type
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:(2)4
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false,true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:(2)LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static cntr = 0;
cntr++;

if(cntr == 1)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_type );
else if(cntr == 2)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_type );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] = ( 0x9090909090909090 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( 0x9090909090909090 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCorePrimitiveImport.Failure_due_to_tag_in_rxbuf_is_key_Specific
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCorePrimitiveImport.Failure_due_to_tag_in_rxbuf_is_key_Specific
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCorePrimitiveImport.Failure_due_to_tag_in_rxbuf_is_key_Specific}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for failure use case when the tag - 'LwSciSyncCorePrimitiveKey_Specific' is found in the core primitive rx buffer.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to a valid memory address.
 * - memory pointed by 'primitive' points to the address '0x9090909090909090'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '1' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Specific' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '8' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address '0x8080808080808080' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'false' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * - LwSciCommonTransportBufferFree() receives the address '0x7777777777777777' in 'buf'.
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is not updated and still points to the addresss '0x9090909090909090'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_BadParameter'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.}
 *
 * @testcase{22060872}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:1
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCorePrimitiveKey_Specific
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( 0x8080808080808080 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] = ( 0x9090909090909090 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( 0x9090909090909090 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCorePrimitiveImport.Failure_due_to_ilwalid_tag_in_rxbuf
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCorePrimitiveImport.Failure_due_to_ilwalid_tag_in_rxbuf
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCorePrimitiveImport.Failure_due_to_ilwalid_tag_in_rxbuf}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for failure use case when an invalid tag is found in the core primitive rx buffer.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to a valid memory address.
 * - memory pointed by 'primitive' points to the address '0x9090909090909090'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '1' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes '0xFF' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '8' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address '0x8080808080808080' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'true' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * - LwSciCommonTransportBufferFree() receives the address '0x7777777777777777' in 'buf'.
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is not updated and still points to the addresss '0x9090909090909090'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_BadParameter'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.}
 *
 * @testcase{22060876}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:1
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:0xFF
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_BadParameter
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( 0x8080808080808080 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] = ( 0x9090909090909090 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( 0x9090909090909090 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCorePrimitiveImport.Failure_due_to_overflow_in_getNextKeyValue
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCorePrimitiveImport.Failure_due_to_overflow_in_getNextKeyValue
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCorePrimitiveImport.Failure_due_to_overflow_in_getNextKeyValue}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for failure use case when overflow oclwrs in get next key-value pair operation.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to a valid memory address.
 * - memory pointed by 'primitive' points to the address '0x9090909090909090'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '1' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Overflow'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * - LwSciCommonTransportBufferFree() receives the address '0x7777777777777777' in 'buf'.
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is not updated and still points to the addresss '0x9090909090909090'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_Overflow'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.}
 *
 * @testcase{22060878}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:1
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:LwSciError_Overflow
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] = ( 0x9090909090909090 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( 0x9090909090909090 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncCorePrimitiveImport.Failure_due_to_overflow_in_getRxBufferAndParams
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCorePrimitiveImport.Failure_due_to_overflow_in_getRxBufferAndParams
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCorePrimitiveImport.Failure_due_to_overflow_in_getRxBufferAndParams}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for failure use case when overflow oclwrs in get core primitive rx buffer and params.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to a valid memory address.
 * - memory pointed by 'primitive' points to the address '0x9090909090909090'.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Overflow'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x123' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 * - LwSciCommonFree() receives the address of 'test_resultPrimitive' in 'ptr'.
 *
 * - memory pointed by 'primitive' is not updated and still points to the addresss '0x9090909090909090'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_Overflow'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.}
 *
 * @testcase{22060881}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Overflow
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_Overflow
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x123
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
<<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonFree.ptr
{{ <<uut_prototype_stubs.LwSciCommonFree.ptr>> == ( &test_resultPrimitive ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] = ( 0x9090909090909090 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( 0x9090909090909090 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncCorePrimitiveImport.Failed_allocate_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_010.LwSciSyncCorePrimitiveImport.Failed_allocate_primitive
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncCorePrimitiveImport.Failed_allocate_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for failure use case when failure oclwrs in allocating memory for primitive struct because of insufficient memory.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to a valid memory address and memory pointed by 'primitive' points to the address '0x9090909090909090'.
 *
 * - LwSciCommonCalloc() returns 'NULL'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 *
 * - memory pointed by 'primitive' is not updated and still points to the addresss '0x9090909090909090'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_InsufficientMemory'.}
 *
 * @testcase{18854079}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonCalloc.return:<<null>>
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_InsufficientMemory
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
{{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] = ( 0x9090909090909090 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( 0x9090909090909090 ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncCorePrimitiveImport.Panic_due_to_null_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_011.LwSciSyncCorePrimitiveImport.Panic_due_to_null_primitive
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncCorePrimitiveImport.Panic_due_to_null_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for panic use case when the actual primitive does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to 'NULL'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCorePrimitiveImport() panics.}
 *
 * @testcase{22060884}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncCorePrimitiveImport.Panic_due_to_len_set_to_zero
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_012.LwSciSyncCorePrimitiveImport.Panic_due_to_len_set_to_zero
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncCorePrimitiveImport.Panic_due_to_len_set_to_zero}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for panic use case when the size of export descriptor is set to zero.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{- 'primitive' points to a valid memory address.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x0'.}
 *
 * @testbehavior{- LwSciSyncCorePrimitiveImport() panics.}
 *
 * @testcase{22060888}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x0
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncCorePrimitiveImport.Panic_due_to_null_data
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_013.LwSciSyncCorePrimitiveImport.Panic_due_to_null_data
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncCorePrimitiveImport.Panic_due_to_null_data}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() for panic use case when the export descriptor does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive' points to a valid memory address.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to 'NULL'.
 * - 'len' is set to '0x123'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCorePrimitiveImport() panics.}
 *
 * @testcase{18854076}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data:<<null>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncCorePrimitiveImport.len_set_to_LBV
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCorePrimitiveImport
TEST.NEW
TEST.NAME:TC_014.LwSciSyncCorePrimitiveImport.len_set_to_LBV
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncCorePrimitiveImport.len_set_to_LBV}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCorePrimitiveImport() when 'len' is set to lower boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- 'test_type' is set to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_Id' is set to '0x12345'.
 *
 * --- primitive import ------------------------------------------------------------------------------
 * - 'primitive' points to a valid memory address.
 *
 * - LwSciCommonCalloc() returns the address of 'test_resultPrimitive'.
 *
 * - LwSciCommonTransportGetRxBufferAndParams() writes '2' into 'params->keyCount'.
 * - LwSciCommonTransportGetRxBufferAndParams() writes the address '0x7777777777777777' into 'rxbuf'.
 * - LwSciCommonTransportGetRxBufferAndParams() returns 'LwSciError_Success'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Type' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '4' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_type' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'false' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'LwSciSyncCorePrimitiveKey_Id' into memory pointed by 'key'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes '8' into memory pointed by 'len'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes the address of 'test_Id' into 'value'.
 * - LwSciCommonTransportGetNextKeyValuePair() writes 'true' into memory pointed by 'rdFinish'.
 * - LwSciCommonTransportGetNextKeyValuePair() returns 'LwSciError_Success'.
 *
 * --- syncpoint import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() returns the address of 'test_info'.
 *
 * - LwSciSyncCoreAttrListGetModule() writes the address '0x66666666666666' into memory pointed by 'module'.
 *
 * - LwSciSyncCoreModuleGetRmBackEnd() writes the address '0x5555555555555555' into memory pointed by 'backEnd'.
 *
 * - LwSciSyncCoreRmGetHost1xHandle() returns the address '0x4444444444444444'.}
 *
 * @testinput{- 'ipcEndpoint' is set to '0x1A2B3C4D'.
 * - 'reconciledList' points to the address '0x9999999999999999'.
 * - 'data' points to the address '0x8888888888888888'.
 * - 'len' is set to '0x1'.}
 *
 * @testbehavior{--- primitive import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(struct LwSciSyncCorePrimitiveRec)' in 'size'.
 * - LwSciCommonTransportGetRxBufferAndParams() receives the address '0x8888888888888888' in 'bufPtr', '0x1' in 'len and non-NULL addresses in 'rxbuf' and 'params'.
 *
 * --- Iteration 1 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 * --- Iteration 2 ---
 * - LwSciCommonTransportGetNextKeyValuePair() receives the address '0x7777777777777777' in 'rxbuf' and non-NULL addresses in 'key', 'len', 'value' and 'rdFinish'.
 *
 * --- syncpoint import ------------------------------------------------------------------------------
 * - LwSciCommonCalloc() receives '1' in 'numItems' and 'sizeof(LwSciSyncCoreSyncpointInfo)' in 'size'.
 * - LwSciSyncCoreAttrListGetModule() receives the address '0x9999999999999999' in 'attrList' and a non-null address in 'module'.
 * - LwSciSyncCoreModuleGetRmBackEnd() receives the address '0x66666666666666' in  'module' and a non-null address in 'backEnd'.
 * - LwSciSyncCoreRmGetHost1xHandle() receives the address '0x5555555555555555' in 'backEnd'.
 *
 * --- primitive import ------------------------------------------------------------------------------
 * - LwSciCommonTransportBufferFree() receives the address '0x7777777777777777' in 'buf'.
 *
 * - memory pointed by 'primitive' is updated to point to the address of 'test_resultPrimitive'.
 * - LwSciSyncCorePrimitiveImport() returns 'LwSciError_Success'.
 *
 * - 'test_resultPrimitive.ownsPrimitive' is updated to 'false'.
 * - 'test_resultPrimitive.type' is updated to 'LwSciSyncInternalAttrValPrimitiveType_Syncpoint'.
 * - 'test_resultPrimitive.id' is updated to '0x12345'.
 * - 'test_resultPrimitive.ops' is updated to point to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'test_resultPrimitive.specificData' is updated to point to the address of 'test_info'.
 *
 * - 'test_info.host1x' is updated to point to the address '0x4444444444444444'.
 * - 'test_info.syncpt' is updated to point to 'NULL'.}
 *
 * @testcase{22060891}
 *
 * @verify{18844773}
 */
TEST.END_NOTES:
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.VALUE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_Id:0x12345
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.ipcEndpoint:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.len:0x1
TEST.VALUE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive:<<malloc 1>>
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params[0].keyCount:2
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.return:LwSciError_Success
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key[0]:MACRO=LwSciSyncCorePrimitiveKey_Type,MACRO=LwSciSyncCorePrimitiveKey_Id
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length[0]:4,8
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish[0]:false,true
TEST.VALUE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.return:(2)LwSciError_Success
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.type:LwSciSyncInternalAttrValPrimitiveType_Syncpoint
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.id:0x12345
TEST.EXPECTED:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ownsPrimitive:false
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCorePrimitiveImport.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonCalloc.numItems:(2)1
TEST.EXPECTED:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufSize:0x1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair
  uut_prototype_stubs.LwSciCommonMemcpyS
  uut_prototype_stubs.LwSciCommonCalloc
  uut_prototype_stubs.LwSciSyncCoreAttrListGetModule
  uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle
  uut_prototype_stubs.LwSciCommonTransportBufferFree
  lwscisync_primitive.c.LwSciSyncCorePrimitiveImport
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.return
static int cntr = 0;
cntr++;

if(cntr == 1)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_resultPrimitive );
else if(cntr == 2)
    <<uut_prototype_stubs.LwSciCommonCalloc.return>> = ( &test_info );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf.rxbuf[0]
<<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>>[0] = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
static cntr = 0;
cntr++;

if(cntr == 1)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_type );
else if(cntr == 2)
    *<<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> = ( &test_Id );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module.module[0]
<<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>>[0] = ( 0x66666666666666 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.return>> = ( 0x4444444444444444 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd.backEnd[0]
<<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>>[0] = ( 0x5555555555555555 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonCalloc.size
static int cntr = 0;
cntr++;

if(cntr == 1)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(struct LwSciSyncCorePrimitiveRec) ) }}
if(cntr == 2)
    {{ <<uut_prototype_stubs.LwSciCommonCalloc.size>> == ( sizeof(LwSciSyncCoreSyncpointInfo) ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.bufPtr>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.rxbuf>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params
{{ <<uut_prototype_stubs.LwSciCommonTransportGetRxBufferAndParams.params>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rxbuf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.key>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.length>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.value>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish
{{ <<uut_prototype_stubs.LwSciCommonTransportGetNextKeyValuePair.rdFinish>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciCommonTransportBufferFree.buf
{{ <<uut_prototype_stubs.LwSciCommonTransportBufferFree.buf>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.attrList>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module
{{ <<uut_prototype_stubs.LwSciSyncCoreAttrListGetModule.module>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmGetHost1xHandle.backEnd>> == ( 0x5555555555555555 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.module>> == ( 0x66666666666666 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreModuleGetRmBackEnd.backEnd>> != ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.reconciledList>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.data
<<lwscisync_primitive.LwSciSyncCorePrimitiveImport.data>> = ( 0x8888888888888888 );
TEST.END_VALUE_USER_CODE:
TEST.EXPECTED_USER_CODE:lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive.primitive[0]
{{ <<lwscisync_primitive.LwSciSyncCorePrimitiveImport.primitive>>[0] == ( &test_resultPrimitive ) }}
TEST.END_EXPECTED_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.ops
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.ops == ( &LwSciSyncBackEndSyncpoint ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive.specificData
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_resultPrimitive>>.specificData == ( &test_info ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.syncpt
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.syncpt == ( NULL ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.EXPECTED_GLOBALS_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.host1x
{{ <<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.host1x == ( 0x4444444444444444 ) }}
TEST.END_EXPECTED_GLOBALS_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreSignalPrimitive

-- Test Case: TC_001.LwSciSyncCoreSignalPrimitive.Primitive_is_signalled
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreSignalPrimitive
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreSignalPrimitive.Primitive_is_signalled
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreSignalPrimitive.Primitive_is_signalled}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreSignalPrimitive() for success use case when the primitve is signalled.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_info.syncpt' points to the address '0x9999999999999999'.
 *
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'primitive->specificData' points to the address of 'test_info'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{--- syncpoint signal ------------------------------------------------------------------------------
 * - LwRmHost1xSyncpointIncrement() receives the address '0x9999999999999999' in 'syncpt' and '1' in 'num'.}
 *
 * @testcase{22060894}
 *
 * @verify{18844776}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive:<<malloc 1>>
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointIncrement.num:1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreSignalPrimitive
  uut_prototype_stubs.LwRmHost1xSyncpointIncrement
  lwscisync_primitive.c.LwSciSyncCoreSignalPrimitive
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointIncrement.syncpt
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointIncrement.syncpt>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.syncpt
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.syncpt = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive.primitive[0].specificData
<<lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive>>[0].specificData = ( &test_info );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_syncpt
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreSignalPrimitive
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_syncpt
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_syncpt}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreSignalPrimitive() for panic use case when syncpoint does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'test_info.syncpt' points to 'NULL'.
 *
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 * - 'primitive->specificData' points to the address of 'test_info'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{- LwSciSyncCoreSignalPrimitive() panics.}
 *
 * @testcase{22060897}
 *
 * @verify{18844776}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreSignalPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:USER_GLOBALS_VCAST.<<GLOBAL>>.test_info.syncpt
<<USER_GLOBALS_VCAST.<<GLOBAL>>.test_info>>.syncpt = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive.primitive[0].specificData
<<lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive>>[0].specificData = ( &test_info );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreSignalPrimitive
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_primitive
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreSignalPrimitive() for panic use case when the actual primitive does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'primitive' points to 'NULL'.}
 *
 * @testbehavior{- LwSciSyncCoreSignalPrimitive() panics.}
 *
 * @testcase{22060900}
 *
 * @verify{18844776}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreSignalPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.END

-- Test Case: TC_004.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_primitive_ops
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreSignalPrimitive
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_primitive_ops
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreSignalPrimitive.Panic_due_to_null_primitive_ops}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreSignalPrimitive() for panic use case when signalling the primitive is not supported.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->ops' points to 'NULL'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.}
 *
 * @testbehavior{- LwSciSyncCoreSignalPrimitive() panics.}
 *
 * @testcase{22060903}
 *
 * @verify{18844776}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive:<<malloc 1>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreSignalPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreSignalPrimitive.primitive>>[0].ops = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Subprogram: LwSciSyncCoreWaitOnPrimitive

-- Test Case: TC_001.LwSciSyncCoreWaitOnPrimitive.Wait_on_primitive_with_finite_timeout
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_001.LwSciSyncCoreWaitOnPrimitive.Wait_on_primitive_with_finite_timeout
TEST.NOTES:
/**
 * @testname{TC_001.LwSciSyncCoreWaitOnPrimitive.Wait_on_primitive_with_finite_timeout}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for success use case when timeout for waiting on primitive in microseconds is finite.}
 *
 * @casederiv{Analysis of Requirements
 * Generation and Analysis of Equivalence Classes}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.
 *
 * - LwRmHost1xSyncpointWait() returns 'LwError_Success'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '0x12345'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 * - LwRmHost1xSyncpointWait() receives '0x7777777777777777' in 'waiter', '0x123' in 'id', '0x1A2B3C4D' in 'threshold', '0x12345' in 'timeout_us' and 'NULL' in 'completed_at'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_Success'.}
 *
 * @testcase{18854085}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x12345
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointWait.return:LwError_Success
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.id:0x123
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.threshold:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.timeout_us:0x12345
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.completed_at:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  uut_prototype_stubs.LwRmHost1xSyncpointWait
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_002.LwSciSyncCoreWaitOnPrimitive.Wait_on_primitive_with_infinite_timeout
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_002.LwSciSyncCoreWaitOnPrimitive.Wait_on_primitive_with_infinite_timeout
TEST.NOTES:
/**
 * @testname{TC_002.LwSciSyncCoreWaitOnPrimitive.Wait_on_primitive_with_infinite_timeout}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for success use case when timeout for waiting on primitive is infinite.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.
 *
 * - LwRmHost1xSyncpointWait() returns 'LwError_Success'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '-1'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 * - LwRmHost1xSyncpointWait() receives '0x7777777777777777' in 'waiter', '0x123' in 'id',  '0x1A2B3C4D' in 'threshold', 'LWRMHOST1X_MAX_WAIT' in 'timeout_us' and 'NULL' in 'completed_at'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_Success'.}
 *
 * @testcase{18854082}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointWait.return:LwError_Success
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.id:0x123
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.threshold:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.timeout_us:MACRO=LWRMHOST1X_MAX_WAIT
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.completed_at:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  uut_prototype_stubs.LwRmHost1xSyncpointWait
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_003.LwSciSyncCoreWaitOnPrimitive.Failure_in_syncpoint_wait
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_003.LwSciSyncCoreWaitOnPrimitive.Failure_in_syncpoint_wait
TEST.NOTES:
/**
 * @testname{TC_003.LwSciSyncCoreWaitOnPrimitive.Failure_in_syncpoint_wait}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for failure use case when:
 * - timeout for waiting on primitive is infinite.
 * - failure in setting the timeout for wait on syncpoint.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.
 *
 * - LwRmHost1xSyncpointWait() returns 'LwError_ResourceError'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '-1'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 * - LwRmHost1xSyncpointWait() receives '0x7777777777777777' in 'waiter', '0x123' in 'id',  '0x1A2B3C4D' in 'threshold', 'LWRMHOST1X_MAX_WAIT' in 'timeout_us' and 'NULL' in 'completed_at'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_ResourceError'.}
 *
 * @testcase{22060906}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointWait.return:LwError_ResourceError
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_ResourceError
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.id:0x123
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.threshold:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.timeout_us:MACRO=LWRMHOST1X_MAX_WAIT
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.completed_at:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  uut_prototype_stubs.LwRmHost1xSyncpointWait
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_004.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_on_wait
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_004.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_on_wait
TEST.NOTES:
/**
 * @testname{TC_004.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_on_wait}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for failure use case when:
 * - timeout for waiting on primitive is infinite.
 * - failure due to timeout on wait operation on syncpoint.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.
 *
 * - LwRmHost1xSyncpointWait() returns 'LwError_Timeout'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '-1'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 * - LwRmHost1xSyncpointWait() receives '0x7777777777777777' in 'waiter', '0x123' in 'id',  '0x1A2B3C4D' in 'threshold', 'LWRMHOST1X_MAX_WAIT' in 'timeout_us' and 'NULL' in 'completed_at'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_Timeout'.}
 *
 * @testcase{18854088}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointWait.return:LwError_Timeout
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Timeout
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.id:0x123
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.threshold:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.timeout_us:MACRO=LWRMHOST1X_MAX_WAIT
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.completed_at:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  uut_prototype_stubs.LwRmHost1xSyncpointWait
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_005.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_overflow_in_value
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_005.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_overflow_in_value
TEST.NOTES:
/**
 * @testname{TC_005.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_overflow_in_value}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for failure use case when:
 * - timeout for waiting on primitive is infinite.
 * - 'value' is set to '0x100000000', which causes overflow.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x100000000'.
 * - 'timeout_us' is set to '-1'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_Overflow'.}
 *
 * @testcase{22060909}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x100000000
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Overflow
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_006.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_overflow_in_id
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_006.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_overflow_in_id
TEST.NOTES:
/**
 * @testname{TC_006.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_overflow_in_id}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for failure use case when:
 * - timeout for waiting on primitive is infinite.
 * - 'id' is set to '0x100000000', which causes overflow.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0x100000000'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '-1'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_Overflow'.}
 *
 * @testcase{22060914}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x100000000
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Overflow
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_007.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_waitContext
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_007.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_waitContext
TEST.NOTES:
/**
 * @testname{TC_007.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_waitContext}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for panic use case when the wait context does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() panics because 'context' points to 'NULL'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to 'NULL'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '-1'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 *
 * - LwSciSyncCoreWaitOnPrimitive() panics.}
 *
 * @testcase{22060917}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
TEST.END_FLOW
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( NULL ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_008.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_us_is_set_below_LBV
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_008.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_us_is_set_below_LBV
TEST.NOTES:
/**
 * @testname{TC_008.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_us_is_set_below_LBV}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for failure use case when 'timeout_us' is set to '-2', just below lower boundary value.}
 *
 * @casederiv{Analysis of Requirements
 * Analysis of Boundary Values}
 *
 * @testsetup{- 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '-2'.}
 *
 * @testbehavior{- LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_BadParameter'.}
 *
 * @testcase{22060919}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-2
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_009.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_primitive
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_009.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_primitive
TEST.NOTES:
/**
 * @testname{TC_009.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_primitive}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for panic use case when the actual primitive does not exist.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{None}
 *
 * @testinput{- 'primitive' points to 'NULL'
 * - 'waitContext' points to '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '-1'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCoreWaitOnPrimitive() panics.}
 *
 * @testcase{22060922}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<null>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_010.LwSciSyncCoreWaitOnPrimitive.id_set_to_UBV
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_010.LwSciSyncCoreWaitOnPrimitive.id_set_to_UBV
TEST.NOTES:
/**
 * @testname{TC_010.LwSciSyncCoreWaitOnPrimitive.id_set_to_UBV}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() when 'id' is set to '0xFFFFFFFF', which is the upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.
 *
 * - LwRmHost1xSyncpointWait() returns 'LwError_Success'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0xFFFFFFFF'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '0x12345'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 * - LwRmHost1xSyncpointWait() receives '0x7777777777777777' in 'waiter', '0xFFFFFFFF' in 'id',  '0x1A2B3C4D' in 'threshold', '0x12345' in 'timeout_us' and 'NULL' in 'completed_at'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_Success'.}
 *
 * @testcase{22060925}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0xFFFFFFFF
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x12345
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointWait.return:LwError_Success
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.id:0xFFFFFFFF
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.threshold:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.timeout_us:0x12345
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.completed_at:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  uut_prototype_stubs.LwRmHost1xSyncpointWait
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_011.LwSciSyncCoreWaitOnPrimitive.value_set_to_UBV
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_011.LwSciSyncCoreWaitOnPrimitive.value_set_to_UBV
TEST.NOTES:
/**
 * @testname{TC_011.LwSciSyncCoreWaitOnPrimitive.value_set_to_UBV}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for success use case when 'value' is set to '0xFFFFFFFF', which is the upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.
 *
 * - LwRmHost1xSyncpointWait() returns 'LwError_Success'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0xFFFFFFFF'.
 * - 'timeout_us' is set to '0x12345'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 * - LwRmHost1xSyncpointWait() receives '0x7777777777777777' in 'waiter', '0x123' in 'id', '0xFFFFFFFF' in 'threshold', '0x12345' in 'timeout_us' and 'NULL' in 'completed_at'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_Success'.}
 *
 * @testcase{22060927}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0xFFFFFFFF
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x12345
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointWait.return:LwError_Success
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.id:0x123
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.threshold:0xFFFFFFFF
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.timeout_us:0x12345
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.completed_at:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  uut_prototype_stubs.LwRmHost1xSyncpointWait
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_012.LwSciSyncCoreWaitOnPrimitive.timeout_us_set_to_UBV
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_012.LwSciSyncCoreWaitOnPrimitive.timeout_us_set_to_UBV
TEST.NOTES:
/**
 * @testname{TC_012.LwSciSyncCoreWaitOnPrimitive.timeout_us_set_to_UBV}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for success use case when 'timeout_us' is set to '0x20C49BA5E353F7', which is the upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{--- primitive wait on -----------------------------------------------------------------------------
 * - 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.
 *
 * --- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() returns the address '0x8888888888888888'.
 *
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() returns the address '0x7777777777777777'.
 *
 * - LwRmHost1xSyncpointWait() returns 'LwError_Success'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to the address '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '0x20C49BA5E353F7'.}
 *
 * @testbehavior{--- syncpoint wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreCpuWaitContextGetBackEnd() receives the address '0x9999999999999999' in 'context'.
 * - LwSciSyncCoreRmWaitCtxGetWaiterHandle() receives the address '0x8888888888888888' in 'waitContextBackEnd'.
 * - LwRmHost1xSyncpointWait() receives '0x7777777777777777' in 'waiter', '0x123' in 'id', '0x1A2B3C4D' in 'threshold', '0x20C49BA5E353F7' in 'timeout_us' and 'NULL' in 'completed_at'.
 *
 * --- primitive wait on -----------------------------------------------------------------------------
 * - LwSciSyncCoreWaitOnPrimitive() return 'LwSciError_Success'.}
 *
 * @testcase{22060930}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x20C49BA5E353F7
TEST.VALUE:uut_prototype_stubs.LwRmHost1xSyncpointWait.return:LwError_Success
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_Success
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.id:0x123
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.threshold:0x1A2B3C4D
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.timeout_us:0x20C49BA5E353F7
TEST.EXPECTED:uut_prototype_stubs.LwRmHost1xSyncpointWait.completed_at:<<null>>
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd
  uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle
  uut_prototype_stubs.LwRmHost1xSyncpointWait
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return
<<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.return>> = ( 0x8888888888888888 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_VAL_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return
<<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.return>> = ( 0x7777777777777777 );
TEST.END_STUB_VAL_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter
{{ <<uut_prototype_stubs.LwRmHost1xSyncpointWait.waiter>> == ( 0x7777777777777777 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context
{{ <<uut_prototype_stubs.LwSciSyncCoreCpuWaitContextGetBackEnd.context>> == ( 0x9999999999999999 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.STUB_EXP_USER_CODE:uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd
{{ <<uut_prototype_stubs.LwSciSyncCoreRmWaitCtxGetWaiterHandle.waitContextBackEnd>> == ( 0x8888888888888888 ) }}
TEST.END_STUB_EXP_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_013.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_us_is_set_above_UBV
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_013.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_us_is_set_above_UBV
TEST.NOTES:
/**
 * @testname{TC_013.LwSciSyncCoreWaitOnPrimitive.Failure_due_to_timeout_us_is_set_above_UBV}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for failure use case when 'timeout_us' is set to '0x20C49BA5E353F8', just above the upper boundary value.}
 *
 * @casederiv{Analysis of Boundary Values}
 *
 * @testsetup{- 'primitive->ops' points to the address of 'LwSciSyncBackEndSyncpoint'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '0x20C49BA5E353F8'.}
 *
 * @testbehavior{- LwSciSyncCoreWaitOnPrimitive() returns 'LwSciError_BadParameter'.}
 *
 * @testcase{22060933}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:0x20C49BA5E353F8
TEST.EXPECTED:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.return:LwSciError_BadParameter
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( &LwSciSyncBackEndSyncpoint );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END

-- Test Case: TC_014.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_primitive_ops
TEST.UNIT:lwscisync_primitive
TEST.SUBPROGRAM:LwSciSyncCoreWaitOnPrimitive
TEST.NEW
TEST.NAME:TC_014.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_primitive_ops
TEST.NOTES:
/**
 * @testname{TC_014.LwSciSyncCoreWaitOnPrimitive.Panic_due_to_null_primitive_ops}
 *
 * @verifyFunction{This test-case verifies the fuctionality of the API - LwSciSyncCoreWaitOnPrimitive() for panic use case when waiting on primitive is not supported.}
 *
 * @casederiv{Analysis of Requirements}
 *
 * @testsetup{- 'primitive->ops' points to 'NULL'.}
 *
 * @testinput{- 'primitive' points to a valid memory address.
 * - 'waitContext' points to '0x9999999999999999'.
 * - 'id' is set to '0x123'.
 * - 'value' is set to '0x1A2B3C4D'.
 * - 'timeout_us' is set to '-1'.}
 *
 * @testbehavior{- LwSciCommonPanic() is called with no args.
 *
 * - LwSciSyncCoreWaitOnPrimitive() panics.}
 *
 * @testcase{22060936}
 *
 * @verify{18844779}
 */
TEST.END_NOTES:
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive:<<malloc 1>>
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.id:0x123
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.value:0x1A2B3C4D
TEST.VALUE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.timeout_us:-1
TEST.FLOW
  lwscisync_primitive.c.LwSciSyncCoreWaitOnPrimitive
  uut_prototype_stubs.LwSciCommonPanic
TEST.END_FLOW
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive.primitive[0].ops
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.primitive>>[0].ops = ( NULL );
TEST.END_VALUE_USER_CODE:
TEST.VALUE_USER_CODE:lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext
<<lwscisync_primitive.LwSciSyncCoreWaitOnPrimitive.waitContext>> = ( 0x9999999999999999 );
TEST.END_VALUE_USER_CODE:
TEST.END
