!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     3
.MAX_ATTR    0
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p576-lw40.s -o allprogs-new32//p576-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
BB0:
MVI      R1, 0.0;
MVI      R0, 1.0;
F2F.SAT  R1, R1;
F2F.SAT  R3, R0;
MOV32    R0, R1;
MOV32    R2, R1;
MOV32    R1, R3;
END
# 7 instructions, 4 R-regs, 0 interpolants
# 7 inst, (3 mov, 2 mvi, 0 tex, 0 ipa, 0 complex, 2 math)
#    4 64-bit, 3 32-bit, 0 32-bit-const
