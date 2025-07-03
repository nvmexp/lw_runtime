!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     15
.MAX_ATTR    2
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p368-lw40.s -o allprogs-new32//p368-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#semantic <null atom>
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var sampler2D  : texunit 0 : -1 : 0
#var float4 f[TEX0] : $vin.F : F[0] : -1 : 0
#tram 0 = f[WPOS].w
#tram 1 = f[TEX0].x
#tram 2 = f[TEX0].y
BB0:
IPA      R0, 0;
RCP      R1, R0;
IPA      R0, 1, R1;
IPA      R1, 2, R1;
FADD32I  R2, R0, -0.000975;
FADD32I  R3, R0, 0.000975;
FADD32I  R4, R1, -0.000975;
MOV32    R0, R2;
MOV32    R8, R2;
FADD32I  R2, R1, 0.000975;
MOV32    R1, R4;
MOV32    R5, R4;
MOV32    R4, R3;
MOV32    R12, R3;
MOV32    R9, R2;
MOV32    R13, R2;
TEX      R0, 0, 0, 2D;
TEX      R4, 0, 0, 2D;
TEX      R8, 0, 0, 2D;
TEX      R12, 0, 0, 2D;
MOV32    R1, R4;
MOV32    R2, R8;
MOV32    R3, R12;
END
# 23 instructions, 16 R-regs, 3 interpolants
# 23 inst, (11 mov, 0 mvi, 4 tex, 3 ipa, 1 complex, 4 math)
#    8 64-bit, 11 32-bit, 4 32-bit-const
