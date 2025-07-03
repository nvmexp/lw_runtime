!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     15
.MAX_IBUF    23
.MAX_OBUF    3
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v304-lw40.s -o allprogs-new32//v304-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic C[4].C[4]
#semantic c.c
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[NOR].x
#ibuf 5 = v[NOR].y
#ibuf 6 = v[NOR].z
#ibuf 7 = v[NOR].w
#ibuf 8 = v[COL0].x
#ibuf 9 = v[COL0].y
#ibuf 10 = v[COL0].z
#ibuf 11 = v[COL0].w
#ibuf 12 = v[COL1].x
#ibuf 13 = v[COL1].y
#ibuf 14 = v[COL1].z
#ibuf 15 = v[COL1].w
#ibuf 16 = v[FOG].x
#ibuf 17 = v[FOG].y
#ibuf 18 = v[FOG].z
#ibuf 19 = v[FOG].w
#ibuf 20 = v[UNUSED0].x
#ibuf 21 = v[UNUSED0].y
#ibuf 22 = v[UNUSED0].z
#ibuf 23 = v[UNUSED0].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
BB0:
F2I.FLOOR R0, v[19];
F2I.FLOOR R1, v[17];
I2I.M4   R0, R0;
I2I.M4   R1, R1;
R2A      A4, R0;
R2A      A5, R1;
FMUL     R1, v[9], c[A4 + 5];
FMUL     R0, v[9], c[A5 + 5];
FMAD     R1, v[8], c[A4 + 4], R1;
FMAD     R2, v[8], c[A5 + 4], R0;
F2I.FLOOR R0, v[7];
FMAD     R1, v[10], c[A4 + 6], R1;
FMAD     R2, v[10], c[A5 + 6], R2;
I2I.M4   R0, R0;
FMAD     R1, v[11], c[A4 + 7], R1;
FMAD     R2, v[11], c[A5 + 7], R2;
R2A      A0, R0;
F2I.FLOOR R0, v[5];
FMUL     R3, v[16], R2;
FMUL     R2, v[1], c[A0 + 5];
I2I.M4   R0, R0;
FMAD     R1, v[18], R1, R3;
FMAD     R2, v[0], c[A0 + 4], R2;
R2A      A1, R0;
FMAD     R3, v[2], c[A0 + 6], R2;
FMUL     R2, v[1], c[A1 + 5];
F2I.FLOOR R0, v[23];
FMAD     R3, v[3], c[A0 + 7], R3;
FMAD     R2, v[0], c[A1 + 4], R2;
I2I.M4   R0, R0;
FMAD     R2, v[2], c[A1 + 6], R2;
R2A      A2, R0;
FMAD     R4, v[3], c[A1 + 7], R2;
FMUL     R2, v[13], c[A2 + 9];
F2I.FLOOR R0, v[21];
FMUL     R4, v[4], R4;
FMAD     R2, v[12], c[A2 + 8], R2;
I2I.M4   R0, R0;
FMAD     R4, v[6], R3, R4;
FMAD     R2, v[14], c[A2 + 10], R2;
R2A      A3, R0;
FADD32   R5, R1, -R4;
FMAD     R2, v[15], c[A2 + 11], R2;
FMUL     R1, v[13], c[A3 + 9];
FMUL     R0, v[1], c[A0 + 9];
FMAD     R3, v[12], c[A3 + 8], R1;
FMAD     R0, v[0], c[A0 + 8], R0;
FMUL     R1, v[1], c[A1 + 9];
FMAD     R3, v[14], c[A3 + 10], R3;
FMAD     R0, v[2], c[A0 + 10], R0;
FMAD     R1, v[0], c[A1 + 8], R1;
FMAD     R3, v[15], c[A3 + 11], R3;
FMAD     R0, v[3], c[A0 + 11], R0;
FMAD     R1, v[2], c[A1 + 10], R1;
FMUL     R6, v[20], R3;
FMUL     R3, v[13], c[A2 + 5];
FMAD     R1, v[3], c[A1 + 11], R1;
FMAD     R2, v[22], R2, R6;
FMAD     R3, v[12], c[A2 + 4], R3;
FMUL     R6, v[4], R1;
FMUL     R1, v[13], c[A3 + 5];
FMAD     R3, v[14], c[A2 + 6], R3;
FMAD     R0, v[6], R0, R6;
FMAD     R1, v[12], c[A3 + 4], R1;
FMAD     R3, v[15], c[A2 + 7], R3;
FADD32   R10, R2, -R0;
FMAD     R2, v[14], c[A3 + 6], R1;
FMUL     R1, v[9], c[A4 + 9];
FMUL32   R7, R5, R10;
FMAD     R6, v[15], c[A3 + 7], R2;
FMAD     R1, v[8], c[A4 + 8], R1;
FMUL     R2, v[9], c[A5 + 9];
FMUL     R6, v[20], R6;
FMAD     R1, v[10], c[A4 + 10], R1;
FMAD     R2, v[8], c[A5 + 8], R2;
FMAD     R3, v[22], R3, R6;
FMAD     R1, v[11], c[A4 + 11], R1;
FMAD     R2, v[10], c[A5 + 10], R2;
FADD32   R8, R3, -R4;
FMUL     R3, v[13], c[A2 + 1];
FMAD     R6, v[11], c[A5 + 11], R2;
FMUL     R2, v[13], c[A3 + 1];
FMAD     R3, v[12], c[A2], R3;
FMUL     R6, v[16], R6;
FMAD     R2, v[12], c[A3], R2;
FMAD     R3, v[14], c[A2 + 2], R3;
FMAD     R1, v[18], R1, R6;
FMAD     R2, v[14], c[A3 + 2], R2;
FMAD     R3, v[15], c[A2 + 3], R3;
FADD32   R1, R1, -R0;
FMAD     R6, v[15], c[A3 + 3], R2;
FMUL     R2, v[1], c[A0 + 1];
FMAD     R11, -R8, R1, R7;
FMUL     R7, v[20], R6;
FMAD     R2, v[0], c[A0], R2;
FMUL     R6, v[1], c[A1 + 1];
FMAD     R3, v[22], R3, R7;
FMAD     R2, v[2], c[A0 + 2], R2;
FMAD     R7, v[0], c[A1], R6;
FMUL     R6, v[9], c[A4 + 1];
FMAD     R2, v[3], c[A0 + 3], R2;
FMAD     R9, v[2], c[A1 + 2], R7;
FMAD     R6, v[8], c[A4], R6;
FMUL     R7, v[9], c[A5 + 1];
FMAD     R9, v[3], c[A1 + 3], R9;
FMAD     R6, v[10], c[A4 + 2], R6;
FMAD     R7, v[8], c[A5], R7;
FMUL     R9, v[4], R9;
FMAD     R6, v[11], c[A4 + 3], R6;
FMAD     R7, v[10], c[A5 + 2], R7;
FMAD     R2, v[6], R2, R9;
FADD32   R12, -R4, c[17];
FMAD     R7, v[11], c[A5 + 3], R7;
FADD32   R3, R3, -R2;
FADD32   R9, -R2, c[16];
FMUL     R7, v[16], R7;
FMUL32   R13, R1, R3;
FADD32   R1, -R0, c[18];
FMAD     R6, v[18], R6, R7;
FADD32   R6, R6, -R2;
FMAD     R7, -R10, R6, R13;
FMUL32   R6, R6, R8;
FADD32   R8, R2, -c[16];
FMUL32   R7, R12, R7;
FMAD     R3, -R3, R5, R6;
FADD32   R5, R4, -c[17];
FMAD     R6, R9, R11, R7;
FMAD     R1, R1, R3, R6;
FSET     R1, R1, c[19], LT;
FADD32   R3, R0, -c[18];
FMUL32   R6, R8, R1;
FMUL32   R5, R5, R1;
FMUL32   R7, R3, R1;
FADD     R3, v[3], -R1;
FMUL32   R1, R1, c[19];
FMAD     R2, R2, R3, R6;
FMAD     R4, R4, R3, R5;
FMAD     R0, R0, R3, R7;
FMAD     R1, v[3], R3, R1;
FMUL32   R3, R4, c[1];
FMUL32   R5, R4, c[5];
FMUL32   R6, R4, c[9];
FMAD     R3, R2, c[0], R3;
FMAD     R5, R2, c[4], R5;
FMAD     R6, R2, c[8], R6;
FMAD     R3, R0, c[2], R3;
FMAD     R5, R0, c[6], R5;
FMAD     R6, R0, c[10], R6;
FMAD     o[0], R1, c[3], R3;
FMAD     o[1], R1, c[7], R5;
FMAD     o[2], R1, c[11], R6;
FMUL32   R3, R4, c[13];
FMAD     R2, R2, c[12], R3;
FMAD     R0, R0, c[14], R2;
FMAD     o[3], R1, c[15], R0;
END
# 155 instructions, 16 R-regs
# 155 inst, (0 mov, 0 mvi, 0 tex, 0 complex, 155 math)
#    131 64-bit, 24 32-bit, 0 32-bit-const
