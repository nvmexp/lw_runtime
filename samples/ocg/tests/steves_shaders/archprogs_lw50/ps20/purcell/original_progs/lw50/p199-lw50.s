!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE PIXEL
.MAX_REG     3
.MAX_ATTR    0
# parseasm build date Mar 10 2004 15:40:49
# -profile fp50 -po tbat3 -po lat3 -if ps2x -i allprogs-new32//p199-lw40.s -o allprogs-new32//p199-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile fp50
#program fp30entry
#semantic C76pv1sbdfq7lf.C76pv1sbdfq7lf
#var float4 o[COLH0] : $vout.O : O[0] : -1 : 0
#var float4 C76pv1sbdfq7lf :  : c[320] : -1 : 0
BB0:
MOV32    R0, c[1280];
MOV32    R1, c[1281];
MOV32    R2, c[1282];
MOV32.SAT R0, R0;
MOV32.SAT R1, R1;
MOV32.SAT R2, R2;
F2F.SAT  R0, R0;
F2F.SAT  R1, R1;
F2F.SAT  R2, R2;
MOV32    R3, c[1283];
MOV32.SAT R3, R3;
F2F.SAT  R3, R3;
END
# 12 instructions, 4 R-regs, 0 interpolants
# 12 inst, (8 mov, 0 mvi, 0 tex, 0 ipa, 0 complex, 4 math)
#    4 64-bit, 8 32-bit, 0 32-bit-const
