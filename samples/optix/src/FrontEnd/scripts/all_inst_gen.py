#!/usr/bin/elw python

import optixpy.ptx_instr_enum
import optixpy.util
import os
import sys



header = '''
.version 3.1
.target sm_35

.entry test_kernel
{
  .reg .f16  f16_<6>;
  .reg .f32  f32_<6>;
  .reg .f64  f64_<6>;
  .reg .u8   u8_<6>;
  .reg .u16  u16_<6>;
  .reg .u32  u32_<6>;
  .reg .u64  u64_<6>;
  .reg .s8   s8_<6>;
  .reg .s16  s16_<6>;
  .reg .s32  s32_<6>;
  .reg .s64  s64_<6>;
  .reg .b8   b8_<6>;
  .reg .b16  b16_<6>;
  .reg .b32  b32_<6>;
  .reg .b64  b64_<6>;
  .reg .pred pred_<6>;

'''

footer = '''
}
'''

type_to_var_prefix = {
        "f16"      : "f16_",
        "f32"      : "f32_",
        "f64"      : "f64_",
        "u8"       : "u8_",
        "u16"      : "u16_",
        "u32"      : "u32_",
        "u64"      : "u64_",
        "s8"       : "s8_",
        "s16"      : "s16_",
        "s32"      : "s32_",
        "s64"      : "s64_",
        "b8"       : "b8_",
        "b16"      : "b16_",
        "b32"      : "b32_",
        "b64"      : "b64_",
        #"b128"     : "b128_",
        "pred"     : "pred_",
        "smbl"     : "b32_",
        "pram"     : "b32_",
        "immed"    : "",
        "trgt"     : "b32_",
        "void"     : "b32_",
        "memaddr"  : "u32_",
        }


def create_ptx( rets, inst, args ):
    params = []
    
    for i, ret in enumerate( rets ):
        if len( ret ) > 1:
            vret = [ type_to_var_prefix[x] + str(i) for i,x in  enumerate( ret ) ]
            params.append( '{{ {} }}'.format( ', '.join( vret ) ) )
        else:
            if i > 0 and ret[0] == 'pred':
                params[-1] += '|{}{}'.format( type_to_var_prefix[ ret[0] ], i )
            else:
                params.append( '{}{}'.format( type_to_var_prefix[ ret[0] ], i ) )

    for i, p in enumerate( args ):
        i += len( rets )
        if len( p ) == 1:
            if 'memaddr' in p[0]:
                params.append( '[{}{}]'.format( type_to_var_prefix[ p[0] ], i ) )
            else:
                params.append( '{}{}'.format( type_to_var_prefix[ p[0] ], i ) )
        else:
            vparm = [ type_to_var_prefix[x] + str(i) for i,x in  enumerate( p ) ]
            params.append( '{{ {} }}'.format( ', '.join( vparm ) ) )

    return '  {}\t{};'.format( inst, ',\t'.join( params ) )


def generate_kernel( table_fname ):

    print header

    #ignored_instrs      = set( [ 'bar', '_sulea', 'trap', 'cctl', 'cctlu', 'isspacep', 'exit' ] )
    #ignored_param_types = set( [ 'pram', 'smbl', 'immed', 'void' ] )
    tex_instrs  = [ 'tex', 'tld4', 'txq' ]
    surf_instrs = [ 'suld', 'sust', 'sured', 'suq' ]
    ignored_instrs      = set( [ '_sulea', '_ldldu', 'cctl', 'cctlu', 'setlmembase', 'getlmembase'  ] +
                                 tex_instrs + surf_instrs )
    ignored_param_types = set( )

    perms = optixpy.ptx_instr_enum.create_instruction_permutations( table_fname )

    for perm in perms:
        flat_args = [ j for i in perm[2] for j in i ]
        if perm[1].split( '.' )[0] in ignored_instrs :
            continue
        if ignored_param_types.intersection( flat_args ):
            continue
        #print '  // {}'.format( perm )
        print create_ptx( perm[0], perm[1], perm[2] )

    print footer

###############################################################################
#
# main
#
###############################################################################

if len( sys.argv ) != 2:
    print "Usage: {} path/to/ptxInstructionDefs.table".format( sys.argv[0] )
    sys.exit( 0 )

generate_kernel( sys.argv[1] )    
