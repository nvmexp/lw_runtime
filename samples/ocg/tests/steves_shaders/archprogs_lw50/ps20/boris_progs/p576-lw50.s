!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     3
.MAX_ATTR    0
# parseasm build date Feb 13 2004 14:20:40
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i progs//p576-lw40.s -o progs//p576-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
BB0:
MVI      R0, 0.0;
MVI      R1, 1.0;
MVI      R2, 0.0;
MVI      R3, 1.0;
END
# 4 instructions, 4 R-regs, 0 interpolants
# 4 inst, (0 mov, 4 mvi, 0 tex, 0 ipa, 0 complex, 0 math)
#    4 64-bit, 0 32-bit, 0 32-bit-const
