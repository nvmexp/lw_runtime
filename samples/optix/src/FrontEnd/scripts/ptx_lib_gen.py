#!/usr/bin/elw python

from __future__ import print_function


import ptx_instr_enum
import cpp_util

import sys
import tempfile 
import subprocess
import os




################################################################################
#
#  ptx instruction to llvm assembly 
#
################################################################################

from string import Template

def debug_print(str):
    sys.stderr.write(str)
    return None


'''
define [linkage] [visibility] [DLLStorageClass]
       [ccolw] [ret attrs]
       <ResultType> @<FunctionName> ([argument list])
       [fn Attrs] [section "name"] [align N]
       [gc] [prefix Constant] { ... }
'''

ptx_type_to_reg = {
        "f16"      : "h",
        "f16x2"    : "r",
        "f32"      : "f",
        "f64"      : "d",
        "u8"       : "h",
        "u16"      : "h",
        "u32"      : "r",
        "u64"      : "l",
        "s8"       : "h",
        "s16"      : "h",
        "s32"      : "r",
        "s64"      : "l",
        "b8"       : "h",
        "b16"      : "h",
        "b32"      : "r",
        "b64"      : "l",
        "b128"     : "l",
        "bf16"     : "b",
        "bf16x2"   : "b",
        "tf32"     : "b",
        "pred"     : "b",
        "trgt"     : "l",
        "texaddr"  : "l",
        "memaddr"  : "r",
        "void"     : "l",
        "smbl"     : "l",
        "pram"     : "l",
        "coord_off": "r",
        "lod_grad" : "f",
        "label"    : "l",
        }

ptx_type_to_llvm = {
        "f16"      : "half",
        "f16x2"    : "i32",
        "f32"      : "float",
        "f64"      : "double",
        "u8"       : "i8",
        "u16"      : "i16",
        "u32"      : "i32",
        "u64"      : "i64",
        "s8"       : "i8",
        "s16"      : "i16",
        "s32"      : "i32",
        "s64"      : "i64",
        "b8"       : "i8",
        "b16"      : "i16",
        "b32"      : "i32",
        "b64"      : "i64",
        "b128"     : "i64", # TODO: ???
        "bf16"     : "i16",
        "bf16x2"   : "i32",
        "tf32"     : "i32",
        "pred"     : "i1",
        "trgt"     : "i64",
        "texaddr"  : "i64",
        "memaddr"  : "MEMADDR",
        "void"     : "i64",
        "smbl"     : "i64",
        "pram"     : "i64",
        "coord_off": "i32",
        "lod_grad" : "float",
        "label"    : "i64", # TODO: ??
        }

insts_with_sideeffects = [
    'addc',
    'subc',
    'madc',
    'ld',
    'ldu',
    'st',
    'prefetch',
    'prefetchu',
    'cctl',
    'cctlu',
    'sust',
    'suld',
    'sured',
    'bar',
    'bar.red',
    'atom',
    'red',
    'vote',
    'match',
    'trap',
    'brkpt',
    'intr',
    'activemask',
    ]

ptx_state_space_to_addrspace = {
        "global"    : 1,
        "shared"    : 3,
        "const"     : 4,
        "local"     : 5,
        }

cost_to_metadata_node = { 'low': '!0', 'medium': '!1', 'high': '!2' }

llvm_template = Template( '''
define linkonce $RESULT_TYPE @$FUNC_NAME( $PARAM_LIST ) nounwind alwaysinline {
$UNPACK${RESULT}call $CALL_TYPE asm $SIDEEFFECT"$PTX_STRING", "$CONSTRAINTS"($FUNCTION_ARGS) $CALL_ATTRIBUTE$CALL_METADATA
$PACK$RETURN_STATEMENT
}''' )


def to_struct_or_scalar( items ):
    if len( items ) == 0:
        return 'void'
    if len( items ) > 1:
        return "{{ {} }}".format( ", ".join( [ ptx_type_to_llvm[r] for r in items ] ) )
    else:
        return ptx_type_to_llvm[ items[0] ]

def to_array_or_scalar( param ):
    if len( param ) == 0:
        return 'void'
    if len( param ) > 1:
        return "< {} x {} >".format( len( param ), ptx_type_to_llvm[ param[0] ] )
    else:
        return ptx_type_to_llvm[ param[0] ]

def to_array( param ):
    return "< {} x {} >".format( len( param ), ptx_type_to_llvm[ param[0] ] )

# Note: This is used for return type only
def to_agg_or_scalar( params ):
    if len( params ) == 0:
        return 'void'
    if len( params ) > 1:
        # In this case, the instruction has a predicate return.
        # With vectorized instructions that have this, it is not correct to flatten
        # the types here. Instead, the first needs to be the vector type and the second
        # needs to be the predicate.
        flat_params = [ to_array_or_scalar( i ) for i in params ]
        return "{{ {} }}".format( ", ".join( [ r for r in flat_params ] ) )
    else:
        return to_array_or_scalar( params[0] )


def to_ptx_arg( arg, asm_args, arg_idx ):
    v = len( arg )
    if v > 1:
        vec_args = [ "${}".format( i ) for i in range( arg_idx, arg_idx+v ) ] 
        asm_args.append( "{{ {} }}".format( ",".join( vec_args ) ) )
    else:
        if 'memaddr' in arg[0]: 
            asm_args.append( "[${}]".format( arg_idx ) )
        elif 'texaddr' in arg[0]: 
            asm_args.append( "[ ${} ".format( arg_idx ) )
        else:
            asm_args.append( "${}".format( arg_idx ) )
    return arg_idx + v

def to_ptx_arg_vector( arg, asm_args, arg_idx ):
    v = len(arg)
    vec_args = [ "${}".format( i ) for i in range( arg_idx, arg_idx+v ) ] 
    asm_args.append( "{{ {} }}".format( ",".join( vec_args ) ) )
    return arg_idx + v

def memaddr_to_ptr( instr, type_fields ):
    arg_type = [type_fields[0]]
    #debug_print("memaddr_to_ptr() argType: %s\n" % (arg_type))
    if '.v2' in instr:
        arg_type *= 2
    if '.v4' in instr:
        arg_type *= 4
    return to_array_or_scalar( tuple( arg_type ) )


def to_addrspace( instr ):
    parts = instr.split('.')
    state_space = ''

    for key in ptx_state_space_to_addrspace:
        if key in parts:
            state_space = key
            break

    return ptx_state_space_to_addrspace.get(state_space,0)

def add_default_rounding_modes( instr ):
    '''Depending on the instruction we might need to add default rounding modes'''

    instr_parts = instr.split('.')
    op = instr_parts[0]
    if (op == 'mad' and 'f32' in instr_parts):
        if ('rn' in instr_parts): return instr
        if ('rz' in instr_parts): return instr
        if ('rm' in instr_parts): return instr
        if ('rp' in instr_parts): return instr
        result = '.'.join(instr_parts[:1] + ['rn'] + instr_parts[1:])
        return result
    return instr

def create_llvm( rets, instr, arg_types, type_fields, rematCost ):
    # This script's instruction permutation generation doesn't distinguish
    # between singleton vectors and scalars. The compiler doesn't care most of
    # the time, but it only accepts singleton vectors for the derivatives in 1D
    # tex.grad fetches. In that case, we need to adjust the parameters,
    # parameter unpacking, and PTX ASM so the derivative arguments are always
    # vectors.
    is_tex_grad_1d = 'tex.grad.1d' in instr or 'tex.grad.a1d' in instr

    # Flat lists of rets and args for easier processing
    flat_args = []
    for i in arg_types:
        for j in i:
            if type(j) == tuple:
                # struct args are packed into a nested tuple.
                for k in j:
                    flat_args.append( k )
            else:
                flat_args.append( j )

    flat_rets = [ j for i in rets for j in i ]

    result_type = to_agg_or_scalar( rets )
    
    func_name = 'optix.ptx.'+instr  

    # Once we've generated the LLVM function name, we can remove the
    # "nonsparse." identifier, which was used to differentiate tex
    # instructions that don't have a predicate output, since it's not used in
    # the PTX instruction itself
    instr = instr.replace( "nonsparse.", "" )

    op = instr.split('.')[0]
        
    # Generate param list for LLVM function
    param_list = []
    for i, p in enumerate( arg_types ):
        if p == ( 'memaddr', ):
            param_list.append( "{} addrspace({})* %p{}".format( memaddr_to_ptr( instr, type_fields ), to_addrspace( instr ), i ) )
        elif is_tex_grad_1d and i > 1:
            param_list.append( "{} %p{}".format( to_array( p ), i ) )
        elif type( p[0] ) == tuple:
            param_list.append( "{} %p{}".format( to_struct_or_scalar( p[0] ), i ) )            
        else:
            param_list.append( "{} %p{}".format( to_array_or_scalar( p ), i ) )
            
    # unpack any vector input params into local scalar regs
    unpack = ''
    for pidx, arg_type in enumerate( arg_types ):
        if len( arg_type ) > 1:
            array_type = to_array_or_scalar( arg_type )
            for aidx in range( len( arg_type ) ):
                unpack += '    %p{}_{} = extractelement {} %p{}, i32 {}\n'.format( pidx, aidx, array_type, pidx, aidx )
        elif is_tex_grad_1d and pidx > 1:
            array_type = to_array( arg_type )
            unpack += '    %p{}_{} = extractelement {} %p{}, i32 {}\n'.format( pidx, 0, array_type, pidx, 0 )
        elif len( arg_type ) == 1 and op == 'cvt' and arg_type[0] == 'f16' and flat_rets[0] == 'f32':
            unpack = '    %t = bitcast half %p0 to i16\n';
        elif type( arg_type[0]) == tuple:
            struct_type = to_struct_or_scalar( arg_type[0] )
            for aidx in range( len( arg_type[0] ) ):
                unpack += '    %p{}_{} = extractvalue {} %p{}, {}\n'.format( pidx, aidx, struct_type, pidx, aidx )
    unpack += '    '
  
    result = "%r0 = " if rets else ""

    call_type = to_struct_or_scalar( flat_rets )

    # Mark instructions that have implicit side-efffects.  Note: it would be safer to parse this from the table but we do not have it here.
    sideeffect = " "
    call_attribute = "#0"
#    sideeffect = "sideeffect "
    if '.cc' in instr or op in insts_with_sideeffects:
        sideeffect = "sideeffect "
        call_attribute = ""

    # dest registers for PTX ASM string 
    ptx_string  = ''
    ptx_args    = [] 
    ptx_arg_idx = 0
    for ridx, r in enumerate( rets ):
        if r[0] == 'pred' and ridx == 1:
            # hackity hack for p0|p1 output syntax (setp, tex, and tld)
            ptx_args[-1] = ptx_args[-1] + '|${}'.format( ptx_arg_idx )
            ptx_arg_idx += 1
        else:
            ptx_arg_idx = to_ptx_arg( r, ptx_args, ptx_arg_idx )

    # input registers for PTX ASM string
    has_texaddr = False
    for pidx, p in enumerate( arg_types ):
        if 'texaddr' in p[0]:
            has_texaddr = True
        if is_tex_grad_1d and pidx > 1:
            ptx_arg_idx = to_ptx_arg_vector( p, ptx_args, ptx_arg_idx )
        elif type(p[0]) == tuple:
            ptx_arg_idx = to_ptx_arg( p[0], ptx_args, ptx_arg_idx )
        else:
            ptx_arg_idx = to_ptx_arg( p, ptx_args, ptx_arg_idx )

    ptx_instr = add_default_rounding_modes( instr )
    if has_texaddr:
        # The position of the closing bracket depends on the operation.
        # It is after arg 2 for tld4, tex and suld, and after arg 1
        # for the others (txq, sust, sured, suq). Note that this only 
        # works because we do not support the tex and tld4 overloads with 
        # the explicit texture sampler.
        last_bracket_arg = 1
        if op == 'tld4' or op == 'tex' or op == 'suld':
            last_bracket_arg = 2
        ptx_args[last_bracket_arg] += ' ]'
        ptx_string += ' {} {};'.format( ptx_instr, ', '.join( ptx_args ) )
    else:
        if op == 'cvt' and flat_args[0] == 'f16' and flat_rets[0] == 'f32':
          ptx_string += "{.reg .f16 %t; mov.b16 %t, $1; " + ptx_instr + " $0, %t;}";
        elif op == 'cvt' and flat_args[0] == 'f32' and flat_rets[0] == 'f16':
          ptx_string += "{.reg .f16 %t; " + ptx_instr + " %t, $1; mov.b16 $0, %t;}";
          call_type = 'i16';
        else:
          ptx_string += ' {} {};'.format( ptx_instr, ', '.join( ptx_args ) )
    
    outputs = [ '={}'.format( ptx_type_to_reg[ p ] ) for p in flat_rets ]
    inputs  = [  '{}'.format( ptx_type_to_reg[ p ] ) for p in flat_args ]

    # Generate list of LLVM registers to pass to ASM string
    function_args = []
    for i, p in enumerate( arg_types ):
        if type(p[0]) == tuple:
            p = p[0]
        if p == ( 'memaddr', ):
            function_args.append( "{} addrspace({})* %p{}".format( memaddr_to_ptr( instr, type_fields ), to_addrspace( instr ), i ) )
        else:
            if is_tex_grad_1d and i > 1:
                for j in range( len( p ) ):
                    p_type = ptx_type_to_llvm[ p[j] ]
                    function_args.append( "{} %p{}_{}".format( p_type, i, j ) )
            elif len( p ) == 1:
                p_type = ptx_type_to_llvm[ p[0] ]
                function_args.append( "{} %p{}".format( p_type, i ) )
            else:
                for j in range( len( p ) ):
                    p_type = ptx_type_to_llvm[ p[j] ]
                    function_args.append( "{} %p{}_{}".format( p_type, i, j ) )

    # if necessary, pack return values into array
    pack = ''
    ret_reg  = 0

    # ASM instructions always return structs when multiple values are returned.
    # Repack them into arrays, if necessary
    if rets and len( rets[0] ) > 1:
        # The return type is a vector.
        # This got more complicated due to the introduction of instructions that support
        # VECTORIZABLE and RESULTP.
        struct_type = to_struct_or_scalar( flat_rets )

        # In the return type the vector type may be wrapped in a struct if a predicate is present (e.g. { < float, float >, pred }).
        # Identify each struct element's type.
        ret_elements = []
        for r in rets:
            ret_elements.append(to_array_or_scalar(r))
        struct_idx = 0
        # Go through each element of the return struct. If len( rets ) == 1, the return type is ret_elements[0]
        for ret_idx in range( len( rets ) ):
            ret_element_type = ret_elements[ret_idx]
            # Go through the values of the element type (the vector)
            for ridx, el in enumerate( rets[ret_idx] ):
                elem_type = ptx_type_to_llvm[ el ]
                # Extract the value from the struct that was returned by the ASM instruction.
                pack += '    %r{} = extractvalue {} %r0, {}\n'.format( ret_reg + 1, struct_type, struct_idx )
                if len( rets[ret_idx] ) > 1:
                    # If the element type in the return struct is a vector, insert the extracted value into it.
                    array_reg = '%r{}'.format( ret_reg ) if ridx else 'undef'
                    ret_reg += 1
                    pack += '    %r{} = insertelement {} {}, {} %r{}, i32 {}\n'.format(
                                ret_reg + 1, ret_element_type, array_reg, elem_type, ret_reg, ridx )
                struct_idx += 1
                ret_reg += 1

            if len( rets ) > 1:
                # Fill the return struct element.
                array_reg = '%r{}'.format( ret_reg - 1 ) if ret_idx else 'undef'
                pack += '    %r{} = insertvalue {} {}, {} %r{}, {}\n'.format(
                                ret_reg + 1, result_type, array_reg, ret_element_type, ret_reg, ret_idx )
                ret_reg += 1
            elif ret_element_type != result_type:
                raise ValueError

    elif rets and len(rets[0]) == 1 and rets[0][0] == 'f16' and flat_args[0] == 'f32' and op == 'cvt':
      pack = '    %r1 = bitcast i16 %r0 to half\n'
      ret_reg += 1
    else:
      pass
    pack += '    '

    # If we've specified a rematerialization cost for the instruction, add it as metadata
    # to the ASM call. Otherwise, don't add any metadata.
    if rematCost is not None:
        call_metadata = ', !rematCost {}'.format( cost_to_metadata_node[rematCost] )
    else:
        call_metadata = ''

    return llvm_template.substitute(
            RESULT_TYPE      = result_type,
            FUNC_NAME        = func_name,
            PARAM_LIST       = ', '.join( param_list),
            PACK             = pack,
            RESULT           = result,
            CALL_TYPE        = call_type,
            SIDEEFFECT       = sideeffect,
            CALL_ATTRIBUTE   = call_attribute, 
            CALL_METADATA    = call_metadata,
            PTX_STRING       = ptx_string,
            CONSTRAINTS      = ",".join( outputs + inputs ),
            FUNCTION_ARGS    = ", ".join( function_args ), 
            UNPACK           = unpack,
            RETURN_STATEMENT = "ret void" if not rets else "ret {} %r{}".format( result_type, ret_reg)
            )

            
def update_tables( address_size ):
    if address_size == 32:
        ptx_type_to_reg['memaddr'] = 'r'
        ptx_type_to_llvm['texaddr'] = 'i32'
        ptx_type_to_llvm['label'] = 'i32'
    elif address_size == 64:
        ptx_type_to_reg['memaddr'] = 'l'        # This only valid for generic and global pointers, but OCG will clean it up
        ptx_type_to_llvm['texaddr'] = 'i64'
        ptx_type_to_llvm['label'] = 'i64'
    else:
        raise ValueError        
       

def print_llvm( table_fname, llvm_version ):
    print("attributes #0 = { nounwind readnone }")

    # Metadata used to specify rematerialization costs.
    # TODO: These will likely need to be changed when we switch to LWVM 70.
    print("")
    if llvm_version == 34:
        print("!0 = metadata !{metadata !\"low\"}")
        print("!1 = metadata !{metadata !\"medium\"}")
        print("!2 = metadata !{metadata !\"high\"}")
    elif llvm_version == 70:
        print("!0 = !{!\"low\"}")
        print("!1 = !{!\"medium\"}")
        print("!2 = !{!\"high\"}")
    else:
        raise ValueError

    # TODO Why are we not ignoring these much earlier, as soon as there is a match for instance?
    ignored_instrs      = set( [ 'bar', 'barrier', '_sulea', 'trap', 'cctl', 'cctlu', 'isspacep', 'exit', 'prefetch', 'prefetchu' ] )
    ignored_param_types = set( [ 'pram', 'smbl', 'immed', 'void' ] )

    perms = ptx_instr_enum.create_instruction_permutations( table_fname )

    for perm in perms:
        flat_args = [ j for i in perm[2] for j in i ]
        if perm[1].split( '.' )[0] in ignored_instrs :
            continue
        if ignored_param_types.intersection( flat_args ):
            continue
        #debug_print("perm: rets: %s instr: %s arg types: %s type_fields: %s)\n" % (perm[0], perm[1], perm[2], perm[3], perm[4]) )
        print( create_llvm( perm[0], perm[1], perm[2], perm[3], perm[4] ) )





###############################################################################
#
# main
#
###############################################################################

import argparse

parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter )
parser.add_argument('ptx_table_fname',
                    type=str,
                    metavar='PTX_INSTR_DEFS_FNAME',
                    help='Path to input instruction def table file' )
parser.add_argument('-a', '--assembly-fname',
                    metavar='ASSEMBLY_FNAME',
                    default='./ptx_instructions.ll',
                    help="Output llvm assembly filename." )
parser.add_argument('-c', '--cpp-fname-base',
                    metavar='CPP_FNAME_BASE',
                    help="Enables generation of C++ header/src files with bc lib embedded as char[] vars." )
parser.add_argument('-b', '--bin-dir',
                    metavar='BIN-DIR',
                    default='',
                    help='Binary directory for LLVM tools' )
parser.add_argument('-p', '--pointer-size',
                    metavar='POINTER-SIZE',
                    default=64,
                    help='Pointer size in bits (32 or 64)' )
parser.add_argument('-l', '--llvm-version',
                    metavar='LLVM_VERSION',
                    default=34,
                    help='LLVM version (34 or 70)')

a = parser.parse_args( sys.argv[1:] )

llvm_as = os.path.join(a.bin_dir, 'llvm-as')

update_tables( int(a.pointer_size) )

print("Generating LLVM assembly...")
with cpp_util.stdout_to_file( a.assembly_fname ):
    print_llvm( a.ptx_table_fname, int( a.llvm_version ) )
    pass

if a.cpp_fname_base:
    print( "Compiling LLVM assembly to bitcode ..." )
    bitcode_fname = tempfile.mktemp( suffix='.bc' )
    var_name = 'ptxInstructionsBitcode'
    try:
        subprocess.call( [llvm_as, '-o', bitcode_fname, a.assembly_fname ] )
    except Exception(e):
        print ( e )
        print ( "Error while running " + " ".join([llvm_as, '-o', bitcode_fname, a.assembly_fname ]))
        raise

    print( "Colwerting bitcode to cpp..." )
    data=open( bitcode_fname,'rb').read()
    cpp_util.binary_to_cpp(data, var_name, a.cpp_fname_base)
    os.unlink( bitcode_fname )

print( 'Done.' )


