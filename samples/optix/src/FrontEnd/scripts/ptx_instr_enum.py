#!/usr/bin/elw python

import sys
import re
from contextlib import contextmanager
import subprocess
import array
import datetime
import os

# Old format
##clearFeatures(); RESULT TESTP
##addInstructionTemplate("F23","testp","P0",getFeatures(),ptx_testp_Instr);
#clearFeactures_re = re.compile('clearFeatures\(\);(.*)$') 
#addInstructionTemplate_re = re.compile('addInstructionTemplate\(\"(\w*)\",\"([\w.]*)\",\"(\w*)\",getFeatures\(\),(\w+)\);')

#format of entry in ptxInstructionDefs.incl
# Note (2020-08-19): There was a change in the instruction type formatting of instruction types. The types
#       are now spelled out including their bit size instead of log2 of size in bytes (e.g. F64 instead of F3).
#       Also, multiple possible bit sizes are now specified "regular-expression-like" (e.g. F[16|32|64] instead of F123).
#stdMEMCLEAR(&features);
#features.RESULT = 1;
#features.VECTORIZABLE = 1;
#features.SHAPE = 1;
#features.SAT = 1;
#features.SATF = 1;
#addInstructionTemplate("F16F16","_mma","hhhh",features,ptx__mma_Instr, False);
#   or
#addInstructionTemplate("F[16|32|64]F[16|32|64]","cvt","01",features,ptx_cvt_Instr, False);
#clearFeatures_re = re.compile('stdMEMCLEAR\(&features\);$') 
#features_re = re.compile('features\.(\w*) = 1;$')
#addInstructionTemplate_re = re.compile('addInstructionTemplate\(\"([\w|\[|\]|\|]*)\",\"([\w.]*)\",\"(\w*)\",features,(\w+), False\);')

#format of entry in ptxInstructionDefs.incl
# Note (2021-08-04): There was yet another change in the instruction - the global state gets passed in as ptxIR
#stdMEMCLEAR(& features)
#features.RESULT = 1
#features.VECTORIZABLE = 1
#features.SHAPE = 1
#features.SAT = 1
#features.SATF = 1
#addInstructionTemplate(parseData, "F16F16", "_mma", "hhhh", features, ptx__mma_Instr, False)
#   or
#addInstructionTemplate(parseData, "F[16|32|64]F[16|32|64]","cvt","01",features,ptx_cvt_Instr, False);
clearFeatures_re = re.compile('stdMEMCLEAR\(&features\);$') 
features_re = re.compile('features\.(\w*) = 1;$')
addInstructionTemplate_re = re.compile('addInstructionTemplate\(parseData, \"([\w|\[|\]|\|]*)\",\"([\w.]*)\",\"(\w*)\",features,(\w+), False\);')


def debug_print(str):
    sys.stderr.write(str)
    return None
    
class Instruction:

    # OK, so we don't necessarily want to map the type directly to the corresponding type.
    # Instead we want to map it to something we can do something with later.  For example,
    # 'i' should be treated as a texture address.

    # Find the mappings in apps\optix\src\FrontEnd\PTX\PTXStitch\ptxparse\ptxInstructionTemplates.c
    type_map = {
            'x' : 'u16',
            'u' : 'u32',
            'U' : 'u64',
            'd' : 'b32',
            'e' : 'b64',
            's' : 'coord_off',  # s32 or vector of s32 (note: only used in texture and surface instructions for now)
            'f' : 'lod_grad',   # f32 or vector of f32 (note: only used in tex.grad for now)
            'l' : 'depth_comp', # scalar .f32 (note: only used in texture instructions for now)
            'i' : 'texaddr',    # match with texture, sampler, surface, .u64, .s64 and
                                # .b64 (only for texture and surface instructions)
            'h' : 'f16',        # f16x2
            'C' : 'immed',      # int immediate
            'D' : 'immed',      # float immediate
            'P' : 'pred',
            'Q' : 'pred',       # Predicate vector (TODO)
            'M' : 'memaddr',
            'S' : 'smbl',
            'T' : 'trgt',
            'A' : 'pram',
            'V' : 'void',
            'L' : 'label'
            }

    # Here's the deal.  These enums represent rules an instruction should follow.  During
    # parsing these rules are combined with the set of modifiers that were parsed out to
    # determine if the instruction is correct.  Since we don't want to reinvent all those
    # rules here, we assume that each rule represents a set of possibly legal modifiers
    # and produce instructions for all possiblities including the ones that aren't legal.
    # This is OK, because the parser will filter out the bad ones, so nobody will try and
    # use them anyway.
    feature_map = {

            'APRX'    : [ '.approx' ],

            'APRXROUNDF' : [ '.approx' ] + [ '.rn', '.rz', '.rm', '.rp' ],

            'APRXROUNDF_DIV' : [ '.approx', '.full' ] + [ '.rn', '.rz', '.rm', '.rp' ],

            'ATOMOPB' : [ '.add', '.and', '.or', '.xor', '.exch' ],

            'ATOMOPI' : [ '.add', '.inc', '.dec', '.min', '.max', '.sub' ],

            'ATOMOPF' : [ '.add' ],

            'CAS'     : ['.cas'],

            'TESTP'   : [ '.finite', '.infinite', '.number', '.notanumber', '.normal', '.subnormal' ],


            'ROUNDI'  : [ '', '.rni', '.rzi', '.rmi', '.rpi' ],

            'ROUNDF'  : [ '', '.rn', '.rz', '.rm', '.rp' ],

            'FTZ'     : [ '', '.ftz' ],

            'SAT'     : [ '', '.sat' ],

            'VSAT'    : [ '', '.sat' ],

            'CC'      : [ '', '.cc' ],

            'CMP'     : [ '.eq', '.ne', '.le', '.lt', '.gt', '.ge', '.lo', '.ls', '.hi', '.hs',
                          '.equ', '.neu', '.ltu', '.leu', '.gtu', '.geu', '.num', '.nan' ],

            'BOP'     : [ '.and', '.or', '.xor' ],

            'CLAMP'   : [ '.clamp', '.wrap', '.trap', '.zero' ],

            'ARITHOP' : [ '.add', '.min', '.max' ],

            'SHFL'    : [ '.up', '.down', '.bfly', '.idx' ],

            'SHAMT'   : [ '', '.shiftamt' ],

            'PRMT'    : [ '', '.f4e', '.b4e', '.rc8', '.ecl', '.ecr', '.rc16' ],

            'NC'     : [ '', '.nc' ],

            'CACHEOP': [ '', '.ca', '.cg', '.cs', '.lu', '.cv', '.wb', '.wt'  ],

            'MEMSPACE': [ '', '.const', '.global', '.local', '.param', '.shared' ],

            'VECTORIZABLE' : [ '', '.v2', '.v4' ],

            'LVL'     : [ '.L1', '.L2' ],

            'COMPMOD' : [ '.r', '.g', '.b', '.a' ],

            'TEXMOD'  : [ '.1d', '.2d', '.3d', '.a1d', '.a2d', '.lwbe', '.alwbe', '.2dms', '.a2dms' ],

            # For txq SURFQ includes '.width', '.height', '.depth', '.channel_data_type', '.channel_order', '.normalized_coords', '.array_size', '.num_mipmap_levels', '.num_samples'
            # For suq SURFQ includes '.width', '.height', '.depth', '.channel_data_type', '.channel_order', '.array_size', '.memory_layout'
            # The features of txq are mutual exclusive, so only add those for suq here and handle txq further down.
            'SURFQ'   : [ '.width', '.height', '.depth', '.channel_data_type', '.channel_order', '.array_size', '.memory_layout' ],

            # Used by txq and txq.level, but since txq.level only has TEXQ, we put the modifiers that txq.level needs in here. Unused for txq because see below.
            'TEXQ'   : [ '.width', '.height', '.depth' ],

            # Only supported by txq, but not used at all. See below.
            'SMPLQ'    : [ '.force_unnormalized_coords', '.filter_mode',
                           '.addr_mode_0', '.addr_mode_1', '.addr_mode_2' ],

            # txq supports SURFQ, TEXQ and SMPLQ but only one at a time. So we use this instead which is the union of those three.
            'TXQ'   : [ '.width', '.height', '.depth', '.channel_data_type', '.channel_order', '.normalized_coords', '.array_size', '.num_mipmap_levels', 
                          '.num_samples', '.force_unnormalized_coords', '.filter_mode', '.addr_mode_0', '.addr_mode_1', '.addr_mode_2' ],

            'UNANI'   : [ '', '.uni' ],

            'BAR'     : [ '.cta', '.gl', '.sys' ],

            'VOTE'    : [ '.all', '.any', '.uni', '.ballot' ],

            'SHR'     : [ '', '.shr7', '.shr15' ],

            'VMAD'    : [ '', '.po' ],

            'SYNC'    : [ '.sync' ],

            'ALIGN'   : [ '', '.aligned' ],

            'NANMODE' : [ '', '.NaN' ],

            'DESC'    : [ '.desc' ],

            'SCOPE'   : [ '', '.cta', '.gpu', '.sys' ],

            # ignoring __obfuscate(".mmio"))
            'ORDER'   : [ '', '.weak', '.relaxed', '.release', '.acquire', '.acq_rel', '.sc', '.volatile' ],

            'NOFTZ'   : [ '.noftz' ],
            }

    # The relative rematerialization costs of various instructions.
    remat_cost_map = {
        'abs': 'low',
        'add': 'low',
        'and': 'low',
        'bfe': 'low',
        'bfi': 'low',
        'bfind': 'low',
        'clz': 'low',
        'cnot': 'low',
        'copysign': 'low',
        'cos': 'medium',
        'cvt': 'low',
        'cvta': 'low',
        'div': 'medium',
        'div.full': 'medium',
        'dp2a': 'medium',
        'dp2a.hi': 'medium',
        'dp2a.lo': 'medium',
        'dp4a': 'medium',
        'ex2': 'medium',
        'fma': 'medium',
        'fns': 'low',
        'lg2': 'medium',
        'lop3': 'low',
        'mad': 'low',
        'mad.hi': 'low',
        'mad.lo': 'low',
        'mad24.hi': 'low',
        'mad24.lo': 'low',
        'max': 'low',
        'min': 'low',
        'mov': 'low',
        'mul': 'low',
        'mul.hi': 'low',
        'mul.lo': 'low',
        'mul.wide': 'low',
        'mul24.hi': 'low',
        'mul24.lo': 'low',
        'neg': 'low',
        'not': 'low',
        'or': 'low',
        'popc': 'low',
        'prmt': 'low',
        'rcp': 'medium',
        'rem': 'low',
        'rsqrt': 'medium',
        'sad': 'low',
        'selp': 'low',
        'shf.l': 'low',
        'shf.r': 'low',
        'shl': 'low',
        'shr': 'low',
        'sin': 'medium',
        'sqrt': 'low',
        'sub': 'low',
        'testp': 'low',
        'txq': 'high',
        'txq.level': 'high',
        'xor': 'low'
    }

    def __init__( self, instr, index_types, operand_types, features):
        self.instr          = instr
        self.cost           = self.remat_cost_map[self.instr] if self.instr in self.remat_cost_map else None
        self.index_types    = index_types
        self.features       = features[:]

        vid_instructions =  [ 'vabsdiff', 'vabsdiff2', 'vabsdiff4', 'vadd',
                'vadd2', 'vadd4', 'vavrg2', 'vavrg4', 'vmad', 'vmax', 'vmax2',
                'vmax4', 'vmin', 'vmin2', 'vmin4', 'vset', 'vset2', 'vset4',
                'vshl', 'vshr', 'vsub', 'vsub2', 'vsub4' ]

        if 'RESULT' in features or self.instr in vid_instructions:
            self.return_type   = operand_types[0]
            self.param_types   = operand_types[1:]
        else:
            self.return_type   = None
            self.param_types   = operand_types


    def __str__( self ):
        return "{}:\n\t{}\n\t{}\n\t{}\n\t{}".format(
                self.instr,
                self.index_types,
                self.return_type,
                self.param_types,
                self.features
                )


    def is_blacklisted( self, instr_type, fields, type_fields ):
        #debug_print("// {} | {} | {} | {}\n".format( self.instr, instr_type, fields, type_fields ))

        # CC only applies to s32 and u32 instrs
        if 'cc' in fields and instr_type not in [ 's32', 'u32' ]:
            return True

        # sat only applies to s32 and float instrs
        #if 'sat' in fields and instr_type != 's32' and instr_type[0] != 'f':
        #    return True

        # ftz not allowed for rsqrt until ptx_isa_40 -- need to switch this eventually
        #if self.instr == 'rsqrt' and 'ftz' in fields:
        #    return True

        # rcp.approx must have ftz when it is f64 type
        if self.instr == 'rcp' and instr_type == 'f64' and 'approx' in fields and 'ftz' not in fields:
            return True

        # bitfield types can only have eq, ne cmps
        if set( ['le', 'lt', 'ge', 'gt' ] ) & set( fields ) and type_fields[-1][0] == 'b':
            return True

        # lo,ls,hi,hs are applicable only to unsigned operands
        if set( ['lo', 'ls', 'hi', 'hs' ] ) & set( fields ) and type_fields[-1][0] != 'u':
            return True

        # equ, neu, ltu, leu, gtu, geu, num, nan are applicable only to floats operands
        if set( ['equ', 'neu', 'ltu', 'leu', 'gtu', 'geu', 'num', 'nan' ] ) & set( fields ) and type_fields[-1][0] != 'f':
            return True

        # Can't combine cc and sat
        if 'sat' in fields and 'cc' in fields:
            return True

        # Can't go over 128 total bits with our vector ops
        if 'v4' in fields:
            if instr_type:
                if '64' in instr_type:
                    return True
            elif '64' in type_fields[-1]:
                return True

        # volatile only valid in global, shared, surf spaces
        if 'volatile' in fields and set( ['const', 'param' ] ) & set( fields ):
            return True

        # prefetch must be in local or global
        if self.instr == 'prefetch' and set( ['const', 'shared', 'param' ] ) & set( fields ):
            return True

        if self.instr == 'prefetchu' and 'L2' in fields:
            return True

        # no caching + voloatile
        if 'volatile' in fields and set( ['cg', 'ca', 'cs', 'cv', 'lu', 'wb', 'wt' ] ) & set( fields ):
            return True

        # ldu only valid for global memspace
        if self.instr == 'ldu' and 'global' not in fields:
            return True

        # filter out write cache ops
        if self.instr == 'ld' and set( [ 'wb', 'wt' ] ) & set( fields ):
            return True

        # ld with .nc requires .global
        if self.instr == 'ld' and 'nc' in fields and not 'global' in fields:
            return True

        # ld.global.nc can only have .cop .vec and .type modifiers. And .cop can only be .ca, .cg, .cs
        if self.instr == 'ld' and 'nc' in fields and set( [ 'volatile', 'relaxed', 'acquire', 'cta', 'gpu', 'sys', 'lu', 'cv', 'wb', 'wt' ] ) & set( fields ):
            return True

        # filter out read cache ops
        if self.instr == 'st' and set( [ 'ca', 'lu', 'cv' ] ) & set( fields ):
            return True

        # cant store to const
        if self.instr == 'st' and 'const' in fields:
            return True

        # isspacep must specify a memspace (param not valid)
        if self.instr == 'isspacep' and not set( [ 'const', 'local', 'global', 'shared' ] ) & set( fields ):
            return True

        # cvta, cvta.to take only u32 or u64
        if 'cvta' in self.instr and instr_type not in [ 'u32', 'u64' ]:
            return True

        # cvta{.to} must specify a memspace (param not valid)
        if 'cvta' in self.instr and not set( [ 'const', 'local', 'global', 'shared' ] ) & set( fields ):
            return True

        # Cant have both an integer mode round and float round -- but dont tell that to ptxInstructionDefs.table
        if set( [ 'rni', 'rzi', 'rmi', 'rpi' ] ) & set( fields ) and set( [ 'rn', 'rz', 'rm', 'rp' ] ) & set( fields ):
            return True

        # No round with ftz
        # TODO: this needs to be applied in some, but not all, cases.
        #if 'ftz' in fields and set( [ 'rn', 'rz', 'rm', 'rp' ] ) & set( fields ):
        #    return True
        #if 'ftz' in fields and set( [ 'rni', 'rzi', 'rmi', 'rpi' ] ) & set( fields ):
        #    return True

        if( self.instr == 'cvt' and ( instr_type[0] == 's' or instr_type[0] == 'u' ) and ( type_fields[-1][0] == 's' or type_fields[-1][0] == 'u' ) and
            int( instr_type[1:] ) >= int( type_fields[-1][1:] ) and 'sat' in fields):
            return True

        # Integer rounding is required for float-to-integer colwersions
        # cvt.ftz.sat.u32.f32 - nope
        # cvt.sat.u32.f32 - nope
        # cvt.u32.f32 - nope
        if( self.instr == 'cvt' and ( instr_type[0] == 's' or instr_type[0] == 'u' ) and type_fields[-1][0] == 'f' and not set( ['rni', 'rzi', 'rmi', 'rpi'] ) & set( fields ) ) :
            return True;

        if( 'tex'  in self.instr or
            'tld4' in self.instr):
            # Texture family of instructions added optional arguments 's' and 'l' which we
            # aren't going to handle at the moment (they require overloading the
            # instruction name).
            if( 's' in self.param_types or
                'l' in self.param_types):
                return True
            # In addition we want to avoid the instructions with an optional texture
            # sampler argument.  These are all the ones with 'ii' in the argument
            # signature.
            if( 'ii' in self.param_types):
                return True
    
        # From ptxInstructionDefs.table:
        #   if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_FUTURE)
        #   The below templates for fma with empty instruction type are for supporting fe8m7/fe8m7x2.
        # Same for min/max. The empty instruction type is causing issues with overloaded wrappers being
        # generated, so omit for now. 
        if "fma" in self.instr or "min" in self.instr or "max" in self.instr:
            if 'x' in self.param_types or 'd' in self.param_types:
                return True

        # ptxInstructionDefs.table lists new variants of many instructions marked with the DESC feature:
        #   The memory descriptor variants :
        #   if LWCFG(GLOBAL_ARCH_AMPERE) && LWCFG(GLOBAL_FEATURE_PTX_ISA_INTERNAL)
        # I am not sure how to handle these, omit for now.
        #if "DESC" in self.features:
        #    return True

        # ptxInstructionDefs.table adds a variant of bra with a mask operand
        #   if LWCFG(GLOBAL_ARCH_AMPERE)
        #   bra.colw/div with mask operand
        # This results in an overloaded wrapper with an extra arg, so omit for now
        if self.instr == 'bra' and self.param_types == 'Tu':
            return True 

        # Only '', 'global' and 'shared' are viable for red
        if self.instr == 'red' and set( [ 'const', 'local', 'param' ] ) & set( fields ):
            return True 


        return False


    def get_permutations( self ):

        def combine( s, x, res ):
            if not x:
                res.append( s )
                return
            lwr = x[0]
            for val in lwr:
                combine( s+val, x[1:], res )

        # move VECTORIZABLE to end of mods since ptxIR does not follow same
        # ordering as ptxInstructionDefs.table for this case
        if 'VECTORIZABLE' in self.features:
            self.features.remove( 'VECTORIZABLE' )
            self.features.append( 'VECTORIZABLE' )
        if 'ROUNDF' in self.features:
            self.features.remove( 'ROUNDF' )
            self.features = ['ROUNDF'] + self.features
        if 'ROUNDI' in self.features:
            self.features.remove( 'ROUNDI' )
            self.features = ['ROUNDI'] + self.features

        if self.instr == 'prefetchu':
            self.features.remove( 'MEMSPACE' )

        # ORDER and SCOPE are new additions here, hence we limit it to 'atom' and 'red' only
        if self.instr == 'atom' or self.instr == 'red':
            self.features.remove( 'MEMSPACE' )
            self.features.insert( 0, 'MEMSPACE' )
            self.features.remove( 'SCOPE' )
            self.features.insert( 0, 'SCOPE' )
            self.features.remove( 'ORDER' )
            self.features.insert( 0, 'ORDER' )
        else:
            if 'SCOPE' in self.features:
                self.features.remove( 'SCOPE' )
            if 'ORDER' in self.features:
                self.features.remove( 'ORDER' )

        # Add the NC feature for ld (this is not an actual listed feature in ptxInstructionDefs.incl 
        # for some reason but behaves like one otherwise). Insert before the CACHEOP modifier.
        if self.instr == 'ld':
            i = 0
            for feature in self.features:
                if feature == 'CACHEOP':
                    insert_at = i;
                    break
                i = i + 1
            self.features.insert( i, 'NC' )

        if 'ROUNDF' in self.features and 'APRX' in self.features:
            self.features.remove( 'APRX' )
            self.features.remove( 'ROUNDF' )
            if self.instr == 'div':
                self.features.insert( 0, 'APRXROUNDF_DIV' )
            else:
                self.features.insert( 0, 'APRXROUNDF' )

        if 'SYNC' in self.features:
            self.features.remove( 'SYNC' )
            self.features.insert( 0, 'SYNC')

        # txq supports TEXQ, SMPLQ, and SURFQ, but they are mutually exclusive. So they would
        # generate way too many permutations and all of them would be wrong. Remove those
        # three features and add the "artificial" feature TXQ which is the union of the three.
        if self.instr == 'txq':
            self.features.remove( 'TEXQ' )
            self.features.remove( 'SMPLQ' )
            self.features.remove( 'SURFQ' )
            self.features.insert( 0, 'TXQ' )

        # Add feature modifiers
        combiners = [ self.feature_map[ f ] for f in self.features if f in self.feature_map ]

        # Add type modifiers
        combiners += self.index_types[:]

        # Add .volatile and normal modifiers for ld/st
        if self.instr == 'ld' or self.instr == 'st':
            combiners = [['', '.volatile']] + combiners[0:]

        # Generate modifier permutations
        mod_perms  = []
        combine( self.instr, combiners, mod_perms )
        #if 'ld' in self.instr:
            #debug_print("%s\n" % (self))
            #debug_print("%s: index_types: %s param_types: %s features: %s combiners: %s\n" % (self.instr, self.index_types, self.param_types, self.features, combiners))
            #debug_print("%s\n--------------------------------------\n" % (mod_perms))
        def get_type( p, fields ):
            if p in '0123456789':
                return fields[ int(p) ]
            else:
                return self.type_map[p]

        def widen( ptx_type ):
            if '32' in ptx_type:
                return ptx_type.replace( '32', '64' )
            else:
                return ptx_type.replace( '16', '32' )

        perms = []
        for mod_perm in mod_perms:
            fields      = mod_perm[len(self.instr)+1:].split( '.' )
            type_fields = fields[len(fields)-len(self.index_types):]
            is_wide     = '.wide' in self.instr
            ret_type    = get_type( self.return_type, type_fields ) if self.return_type else None

            if self.is_blacklisted( ret_type, fields, type_fields ):
                #debug_print("BLACKLISTED %s ret_type: %s fields: %s type_len: %s\n" %(self.instr, ret_type, fields, len(type_fields)))
                continue

            ret_type    = widen( ret_type ) if is_wide else ret_type
            vec_count   =  2 if 'v2' in fields else 4 if 'v4' in fields else 1
            rets = []
            if ret_type:
               rets.append( ( ret_type, ) * vec_count )
            
            # Some instructions like setp and match can have an additional pred return value. 
            # Need to make sure only match.all gets this extra arg, since match.any can't have
            # it, but this is not reflected in ptxInstructionDefs.incl
            if 'RESULTP' in self.features and not ( self.instr == 'match' and 'any' in fields ):
               rets.append( ( 'pred', ) )

            params = []
            # We need to create a mangled name based on the parameter types, because PTX
            # supports parameter overloading with the same function name and return type.
            for i, p in enumerate( self.param_types ):
                ty = get_type( p, type_fields )
                vec_mult = vec_count
                if (ty == 'memaddr' or
                    ty == 'texaddr'):
                    vec_mult = 1
                elif ty == 'depth_comp':
                    ty = 'f32'
                    vec_mult = 1
                elif ty == 'coord_off':
                    # This is used in txq, suld, sust, sured and suq.
                    # TODO: It is also used in mma which I do not find mentioned in the ptx_isa_6.4 (only wmma.mma),
                    #       so I am unsure if it works correctly for that one.
                    ty = 's32'
                    # The number of arguments are based on the geometry modifier
                    if '1d' in fields:
                        vec_mult = 1
                    if '2d' in fields or 'a1d' in fields:
                        vec_mult = 2
                    if '3d' in fields or 'a2d' in fields:
                        vec_mult = 4
                elif ty == 'lod_grad':
                    # This is used in tex.grad.
                    # TODO: It is also used in mma which I do not find mentioned in the ptx_isa_6.4 (only wmma.mma),
                    #       so I am unsure if it works correctly for that one.
                    ty = 'f32'
                    # The number of arguments are based on the geometry modifier
                    if '1d' in fields or 'a1d' in fields:
                        vec_mult = 1
                    if '2d' in fields or 'a2d' in fields:
                        vec_mult = 2
                    if '3d' in fields:
                        vec_mult = 4
                elif 'tex.level' in self.instr and i == 2:
                    # the third arg of tex.level follows the intruction type, but is no vector.
                    vec_mult = 1
                elif p in '0123456789' and ('tex' in self.instr  or 'tld4' in self.instr):
                    # tex, tld4 get their coordinate vector size from the geometry.
                    # also the type of the vector elements changes depending on the geometry.
                    if '1d' in fields:
                        vec_mult = 1
                    elif '2d' in fields:
                        vec_mult = 2
                    elif '3d' in fields:
                        vec_mult = 4
                    elif 'a1d' in fields:
                        # Requires a struct type to support mixed types.
                        # Pack into a nested tuple.
                        ty = ('b32', ty)
                        vec_mult = 1
                    elif 'a2d' in fields:
                        # Requires a struct type to support mixed types.
                        # Pack into a nested tuple.
                        ty = ('b32', ty, ty, ty)
                        vec_mult = 1
                    elif 'lwbe' in fields:
                        ty = 'f32'
                        vec_mult = 4
                    elif 'alwbe' in fields:
                        ty = ('u32', 'f32', 'f32', 'f32')
                        vec_mult = 1
                    elif '2dms' in fields or 'a2dms' in fields:
                        ty = 'b32'
                        vec_mult = 4
                params.append( ( ty, ) * vec_mult )
            if self.instr == 'mad.wide':
                params[-1] = ( widen( params[-1][0] ), )

            #if 'fma' in self.instr:
                #debug_print("mod_perm: %s params: %s rets: %s\n" % (mod_perm, params, rets));
                #debug_print("mod_perm: %s fields: %s type_fields: %s is_wide: %s ret_type: %s\n" % (mod_perm, fields, type_fields, is_wide, ret_type))

            perms.append( ( tuple(rets), mod_perm, tuple(params), tuple(type_fields), self.cost ) )

        return tuple(perms)



################################################################################
#
#  ptxInstructionDefs parsing
#
################################################################################

def get_type( type_str, signed ):
    types = []
    type_types = []
    if type_str[0] == 'F':
        type_types.append( '.f' )
    elif type_str[0] == 'H':
        return [ '.f16x2' ]
    elif type_str[0] == 'I':
        type_types.append( '.s' )
        if not signed:
            type_types.append( '.u' )
    elif type_str[0] == 'B':
        type_types.append( '.b' )
    elif type_str[0] == 'P':
        return [ '.pred' ]
    elif type_str[0] == 'A':
        # manually added type identifier for bf16 and bf16x2
        type_types.append( '.bf' )
    elif type_str[0] == 'T':
        #manually added type identifier for tf32
        type_types.append( '.tf' )
    else:
        #print "****FOUND TYPE: {}".format( chars[0] )
        pass

    if len(type_types) != 0:
        end = len(type_str)
        start = 1
        if type_str[1] == '[':
            start = 2
            end -= 1
        for type_type in type_types:
            for size in type_str[start:end].split('|'):
                if type_type == '.bf' and size == '32':
                    types.append( '.bf16x2' )
                else:
                    types.append( type_type + size )
    return types


def get_index_types( type_str, signed ):
    type_strs = []
    idx = 0
    while idx < len( type_str ):
        type_type = type_str[idx]
        type_size = ""
        idx += 1
        while idx < len( type_str ) and type_str[idx] not in "FHIBPOAT":
            type_size += type_str[idx]
            idx += 1
        if not type_size:
            if type_type == 'I' or type_type == 'B':
                type_size = '[16|32|64]'
            elif type_type == 'F':
                type_size = '[32|64]'
            else:
                type_size = ''
        type_strs.append( type_type + type_size )

    types = []
    for s in type_strs:
        types.append( get_type( s, signed ) )

    return types


def create_instruction_permutations( table_fname ):
    with open( table_fname, 'r' ) as f:
        instructions = []
        lines = f.readlines()
        features = []
        lineCount = 0
        matchCounter = 0
        for line in lines:
            lineCount += 1
            match = clearFeatures_re.match( line )
            if match:
                #debug_print("cf: match = %s\n" % (match.group(),))
                features    = [ ]
                continue
            match = features_re.match( line )
            if match:
                #debug_print("fe: match = %s\n" % (match.group(),))
                feature = match.group( 1 )
                features.append( match.group( 1 ) )
                continue
            match = addInstructionTemplate_re.match( line )
            if match:
                #debug_print("ai: match = %s\n" % (match.group(),))
                signed      = False#'SIGNED' in features
                index_types = get_index_types( match.group( 1 ), signed )
                instr       = match.group( 2 )
                operands    = match.group( 3 )
                #debug_print("ai: features    = %s\n" % (features,))
                #debug_print("ai: index_types = %s\n" % (index_types,))
                #debug_print("ai: instr       = %s\n" % (instr,))
                #debug_print("ai: operands    = %s\n" % (operands,))
                # filter out instructions with empty index_types which would cause issues later on when indexed accesses to it
                if instr.startswith( 'atom' ) or instr.startswith( 'red' ) or instr.startswith( 'set' ):
                    if not len(index_types):
                        #debug_print("ignoring Instruction( %s, %s, %s, %s ) with no index_types\n" % (instr,index_types, operands, features,))
                        continue

                if instr.startswith( 'cp.async' ) or instr.startswith( 'cp.reduce.async' ):
                    #debug_print("ignoring Instruction( %s, %s, %s, %s )\n" % (instr,index_types, operands, features,))
                    continue

                # Ignored, as it loads/stores one or more matrices from/to shared memory for mma instruction
                if instr.startswith( 'stmatrix' ) or instr.startswith( 'ldmatrix' ):
                    #debug_print("ignoring Instruction( %s, %s, %s, %s )\n" % (instr,index_types, operands, features,))
                    continue

                if instr.startswith( 'createpolicy' ):
                    #debug_print("ignoring Instruction( %s, %s, %s, %s )\n" % (instr,index_types, operands, features,))
                    continue

                if 'CACHEHINT' in features:
                    if not( instr.startswith( 'atom' ) or instr.startswith( 'red' ) ):
                        #debug_print("ignoring Instruction( %s, %s, %s, %s )\n" % (instr,index_types, operands, features,))
                        continue
                    else:
                        # in this case the last operand should always be 'U', which we lwrrently ignore
                        operands = operands[:-1]

                #if 'DESC' in features:
                #    debug_print("ignoring Instruction( %s, %s, %s, %s )\n" % (instr,index_types, operands, features,))
                #    continue

                #debug_print("adding Instruction( %s, %s, %s, %s )\n" % (instr,index_types, operands, features,))
                instructions.append( Instruction( instr, index_types, operands, features ) )

                # For every tex instruction, we need to add a version that does
                # not include the optional result predicate. This is done to
                # support older architectures, as well as to allow for both
                # types of instructions to be called by a program. The term
                # "nonsparse" refers to the term used by the lwvm intrinsics to
                # differentiate these variants.
                if instr.startswith( "tex" ):
                    features.remove( "RESULTP" )
                    instructions.append( Instruction( "nonsparse." + instr, index_types, operands, features ) )
                #debug_print("instr: %s\n" %(instr))
                   

        perms = []
        for instruction in instructions:
            # TODO(jbigler 4/2/2019) I have no idea how to deal with these new instructions
            # that have _ in the begining.
            #
            # _mma : multiply-and-accumulate
            # _ldsm : ? - I didn't even find mention of these in the 6.4 PTX ISA doc
            # _movm : ?
            # _ldldu : ?
            # (ali 08/10/2021)
            #  ignoring new instructions cp.async[.*] and cp.reduce.async[.*]
            if instruction.instr[0] == '_':
                #debug_print( "skipping instruction %s\n" % (instruction.instr,))
                continue
            if instruction.instr == 'cachepolicy':
                #debug_print( "skipping instruction %s\n" % (instruction.instr,))
                continue
            if instruction.instr.find('wmma.') != -1:
                #debug_print( "skipping instruction %s\n" % (instruction.instr,))
                continue
            if instruction.instr.find('mma.') != -1:
                #debug_print( "skipping instruction %s\n" % (instruction.instr,))
                continue
            if instruction.instr.find('mbarrier.') != -1:
                #debug_print( "skipping instruction %s\n" % (instruction.instr,))
                continue
            perms += instruction.get_permutations()

        # remove duplicates
        seen = set()
        seen_add = seen.add
        perms = [ x for x in perms  if x not in seen and not seen_add(x)]

        return perms

