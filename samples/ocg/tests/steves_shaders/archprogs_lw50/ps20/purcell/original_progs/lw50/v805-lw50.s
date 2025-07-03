!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    10
.MAX_OBUF    9
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v805-lw40.s -o allprogs-new32//v805-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic c.c
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[NOR].x
#ibuf 8 = v[NOR].y
#ibuf 9 = v[NOR].z
#ibuf 10 = v[NOR].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX0].z
#obuf 7 = o[TEX1].x
#obuf 8 = o[TEX1].y
#obuf 9 = o[TEX1].z
BB0:
F2I.FLOOR R0, v[10];
F2I.FLOOR R1, v[8];
I2I.M4   R0, R0;
I2I.M4   R1, R1;
R2A      A0, R0;
R2A      A1, R1;
FMUL     R0, v[1], c[A0 + 1];
FMUL     R1, v[1], c[A1 + 1];
FMAD     R0, v[0], c[A0], R0;
FMAD     R2, v[0], c[A1], R1;
FMUL     R1, v[1], c[A0 + 5];
FMAD     R0, v[2], c[A0 + 2], R0;
FMAD     R2, v[2], c[A1 + 2], R2;
FMAD     R1, v[0], c[A0 + 4], R1;
FMAD     R0, v[3], c[A0 + 3], R0;
FMAD     R2, v[3], c[A1 + 3], R2;
FMAD     R1, v[2], c[A0 + 6], R1;
FMUL     R3, v[1], c[A1 + 5];
FMUL     R2, v[7], R2;
FMAD     R1, v[3], c[A0 + 7], R1;
FMAD     R3, v[0], c[A1 + 4], R3;
FMAD     R0, v[9], R0, R2;
FMUL     R2, v[1], c[A0 + 9];
FMAD     R3, v[2], c[A1 + 6], R3;
FMAD     R2, v[0], c[A0 + 8], R2;
FMAD     R4, v[3], c[A1 + 7], R3;
FMUL     R3, v[1], c[A1 + 9];
FMAD     R2, v[2], c[A0 + 10], R2;
FMUL     R4, v[7], R4;
FMAD     R3, v[0], c[A1 + 8], R3;
FMAD     R2, v[3], c[A0 + 11], R2;
FMAD     R1, v[9], R1, R4;
FMAD     R3, v[2], c[A1 + 10], R3;
FMUL32   R4, R1, c[1];
FMAD     R3, v[3], c[A1 + 11], R3;
FMUL32   R5, R1, c[5];
FMAD     R4, R0, c[0], R4;
FMUL     R3, v[7], R3;
FMAD     R5, R0, c[4], R5;
FMUL32   R6, R1, c[9];
FMAD     R2, v[9], R2, R3;
FMUL32   R3, R1, c[13];
FMAD     R6, R0, c[8], R6;
FMAD     R4, R2, c[2], R4;
FMAD     R5, R2, c[6], R5;
FMAD     R6, R2, c[10], R6;
FMAD     o[0], v[3], c[3], R4;
FMAD     o[1], v[3], c[7], R5;
FMAD     o[2], v[3], c[11], R6;
FMAD     R3, R0, c[12], R3;
FADD32   o[7], -R0, c[16];
FADD32   o[8], -R1, c[17];
FMAD     R1, R2, c[14], R3;
FADD32   o[9], -R2, c[18];
FMUL     R0, v[5], c[A0 + 1];
FMAD     o[3], v[3], c[15], R1;
FMUL     R1, v[5], c[A1 + 1];
FMAD     R0, v[4], c[A0], R0;
FMUL     R2, v[5], c[A0 + 5];
FMAD     R1, v[4], c[A1], R1;
FMAD     R0, v[6], c[A0 + 2], R0;
FMAD     R2, v[4], c[A0 + 4], R2;
FMAD     R1, v[6], c[A1 + 2], R1;
FMUL     R3, v[5], c[A1 + 5];
FMAD     R2, v[6], c[A0 + 6], R2;
FMUL     R1, v[7], R1;
FMAD     R3, v[4], c[A1 + 4], R3;
FMUL     R4, v[5], c[A0 + 9];
FMAD     o[4], v[9], R0, R1;
FMAD     R0, v[6], c[A1 + 6], R3;
FMAD     R1, v[4], c[A0 + 8], R4;
FMUL     R3, v[5], c[A1 + 9];
FMUL     R0, v[7], R0;
FMAD     R1, v[6], c[A0 + 10], R1;
FMAD     R3, v[4], c[A1 + 8], R3;
FMAD     o[5], v[9], R2, R0;
FMAD     R0, v[6], c[A1 + 10], R3;
FMUL     R0, v[7], R0;
FMAD     o[6], v[9], R1, R0;
END
# 79 instructions, 8 R-regs
# 79 inst, (0 mov, 0 mvi, 0 tex, 0 complex, 79 math)
#    72 64-bit, 7 32-bit, 0 32-bit-const
