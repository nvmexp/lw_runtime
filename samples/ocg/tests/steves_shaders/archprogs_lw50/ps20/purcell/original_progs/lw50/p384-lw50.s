!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     3
.MAX_ATTR    4
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p384-lw40.s -o allprogs-new32//p384-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic Coit1m35rlpml7.Coit1m35rlpml7
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 Coit1m35rlpml7 :  : c[1] : -1 : 0
#var float4 f[COL0] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[COL0].x
#tram 2 = f[COL0].y
#tram 3 = f[COL0].z
#tram 4 = f[COL0].w
BB0:
IPA      R0, 0;
RCP      R3, R0;
IPA      R0, 1, R3;
IPA      R1, 2, R3;
IPA      R2, 3, R3;
FMUL32   R0, R0, c[4];
FMUL32   R1, R1, c[5];
FMUL32   R2, R2, c[6];
IPA      R3, 4, R3;
FMUL32   R3, R3, c[7];
END
# 10 instructions, 4 R-regs, 5 interpolants
# 10 inst, (0 mov, 0 mvi, 0 tex, 5 ipa, 1 complex, 4 math)
#    6 64-bit, 4 32-bit, 0 32-bit-const
