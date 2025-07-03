!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     11
.MAX_IBUF    20
.MAX_OBUF    20
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1007-lw40.s -o allprogs-new32//v1007-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic c.c
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
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
#ibuf 10 = v[COL0].x
#ibuf 11 = v[COL0].y
#ibuf 12 = v[COL0].z
#ibuf 13 = v[COL1].x
#ibuf 14 = v[COL1].y
#ibuf 15 = v[COL1].z
#ibuf 16 = v[COL1].w
#ibuf 17 = v[FOG].x
#ibuf 18 = v[FOG].y
#ibuf 19 = v[FOG].z
#ibuf 20 = v[FOG].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[BCOL0].w
#obuf 8 = o[TEX0].x
#obuf 9 = o[TEX0].y
#obuf 10 = o[TEX0].z
#obuf 11 = o[TEX0].w
#obuf 12 = o[TEX1].x
#obuf 13 = o[TEX1].y
#obuf 14 = o[TEX1].z
#obuf 15 = o[TEX2].x
#obuf 16 = o[TEX2].y
#obuf 17 = o[TEX2].z
#obuf 18 = o[TEX3].x
#obuf 19 = o[TEX3].y
#obuf 20 = o[TEX3].z
BB0:
F2I.FLOOR R0, v[20];
F2I.FLOOR R1, v[18];
I2I.M4   R0, R0;
I2I.M4   R1, R1;
R2A      A0, R0;
R2A      A1, R1;
FMUL     R1, v[5], c[A0 + 1];
FMUL     R0, v[5], c[A1 + 1];
FMUL     R2, v[5], c[A0 + 5];
FMAD     R1, v[4], c[A0], R1;
FMAD     R0, v[4], c[A1], R0;
FMAD     R2, v[4], c[A0 + 4], R2;
FMAD     R1, v[6], c[A0 + 2], R1;
FMAD     R0, v[6], c[A1 + 2], R0;
FMAD     R2, v[6], c[A0 + 6], R2;
FMUL     R3, v[5], c[A1 + 5];
FMUL     R4, v[17], R0;
FMUL     R0, v[5], c[A0 + 9];
FMAD     R3, v[4], c[A1 + 4], R3;
FMAD     R1, v[19], R1, R4;
FMAD     R0, v[4], c[A0 + 8], R0;
FMAD     R4, v[6], c[A1 + 6], R3;
FMUL     R3, v[5], c[A1 + 9];
FMAD     R0, v[6], c[A0 + 10], R0;
FMUL     R4, v[17], R4;
FMAD     R3, v[4], c[A1 + 8], R3;
FMAD     R2, v[19], R2, R4;
FMAD     R4, v[6], c[A1 + 10], R3;
FMUL32   R3, R2, R2;
FMUL     R4, v[17], R4;
FMAD     R3, R1, R1, R3;
FMAD     R0, v[19], R0, R4;
FMAD     R4, R0, R0, R3;
FMUL     R3, v[8], c[A0 + 1];
RSQ      R5, |R4|;
FMAD     R3, v[7], c[A0], R3;
FMUL     R4, v[8], c[A1 + 1];
FMUL32   R1, R1, R5;
FMUL32   R2, R2, R5;
FMUL32   R0, R0, R5;
FMAD     R3, v[9], c[A0 + 2], R3;
FMAD     R4, v[7], c[A1], R4;
FMUL32   R5, R2, c[25];
FMUL     R6, v[8], c[A0 + 5];
FMAD     R4, v[9], c[A1 + 2], R4;
FMAD     R5, R1, c[24], R5;
FMAD     R6, v[7], c[A0 + 4], R6;
FMUL     R4, v[17], R4;
FMAD     o[18], R0, c[26], R5;
FMAD     R5, v[9], c[A0 + 6], R6;
FMAD     R4, v[19], R3, R4;
FMUL     R6, v[8], c[A1 + 5];
FMUL     R3, v[8], c[A0 + 9];
FMAD     R7, v[7], c[A1 + 4], R6;
FMAD     R3, v[7], c[A0 + 8], R3;
FMUL     R6, v[8], c[A1 + 9];
FMAD     R7, v[9], c[A1 + 6], R7;
FMAD     R3, v[9], c[A0 + 10], R3;
FMAD     R6, v[7], c[A1 + 8], R6;
FMUL     R7, v[17], R7;
FMAD     R6, v[9], c[A1 + 10], R6;
FMAD     R5, v[19], R5, R7;
FMUL     R7, v[17], R6;
FMUL32   R6, R5, R5;
FMAD     R3, v[19], R3, R7;
FMAD     R6, R4, R4, R6;
FMAD     R7, R3, R3, R6;
FMUL     R6, v[11], c[A0 + 1];
RSQ      R8, |R7|;
FMAD     R6, v[10], c[A0], R6;
FMUL     R7, v[11], c[A1 + 1];
FMUL32   R4, R4, R8;
FMUL32   R5, R5, R8;
FMUL32   R3, R3, R8;
FMAD     R6, v[12], c[A0 + 2], R6;
FMAD     R7, v[10], c[A1], R7;
FMUL32   R8, R5, c[25];
FMUL     R9, v[11], c[A0 + 5];
FMAD     R7, v[12], c[A1 + 2], R7;
FMAD     R8, R4, c[24], R8;
FMAD     R9, v[10], c[A0 + 4], R9;
FMUL     R7, v[17], R7;
FMAD     o[19], R3, c[26], R8;
FMAD     R8, v[12], c[A0 + 6], R9;
FMAD     R7, v[19], R6, R7;
FMUL     R9, v[11], c[A1 + 5];
FMUL     R6, v[11], c[A0 + 9];
FMAD     R10, v[10], c[A1 + 4], R9;
FMAD     R6, v[10], c[A0 + 8], R6;
FMUL     R9, v[11], c[A1 + 9];
FMAD     R10, v[12], c[A1 + 6], R10;
FMAD     R6, v[12], c[A0 + 10], R6;
FMAD     R9, v[10], c[A1 + 8], R9;
FMUL     R10, v[17], R10;
FMAD     R9, v[12], c[A1 + 10], R9;
FMAD     R8, v[19], R8, R10;
FMUL     R10, v[17], R9;
FMUL32   R9, R8, R8;
FMAD     R6, v[19], R6, R10;
FMAD     R9, R7, R7, R9;
FMAD     R9, R6, R6, R9;
FMUL32   R10, R2, c[21];
RSQ      R9, |R9|;
FMAD     R10, R1, c[20], R10;
FMUL32   R11, R5, c[21];
FMUL32   R7, R7, R9;
FMUL32   R8, R8, R9;
FMUL32   R6, R6, R9;
FMAD     o[15], R0, c[22], R10;
FMAD     R9, R4, c[20], R11;
FMUL32   R10, R2, c[17];
FMUL32   R2, R8, c[25];
FMAD     o[16], R3, c[22], R9;
FMAD     R1, R1, c[16], R10;
FMAD     R2, R7, c[24], R2;
FMUL32   R5, R5, c[17];
FMAD     o[12], R0, c[18], R1;
FMAD     o[20], R6, c[26], R2;
FMAD     R1, R4, c[16], R5;
FMUL32   R0, R8, c[21];
FMUL32   R2, R8, c[17];
FMAD     o[13], R3, c[18], R1;
FMAD     R0, R7, c[20], R0;
FMAD     R1, R7, c[16], R2;
FMUL     R2, v[1], c[A0 + 1];
FMAD     o[17], R6, c[22], R0;
FMAD     o[14], R6, c[18], R1;
FMAD     R2, v[0], c[A0], R2;
FMUL     R0, v[1], c[A1 + 1];
FMUL     R1, v[1], c[A0 + 5];
FMAD     R2, v[2], c[A0 + 2], R2;
FMAD     R0, v[0], c[A1], R0;
FMAD     R1, v[0], c[A0 + 4], R1;
FMAD     R2, v[3], c[A0 + 3], R2;
FMAD     R0, v[2], c[A1 + 2], R0;
FMAD     R3, v[2], c[A0 + 6], R1;
FMUL     R1, v[1], c[A1 + 5];
FMAD     R0, v[3], c[A1 + 3], R0;
FMAD     R3, v[3], c[A0 + 7], R3;
FMAD     R1, v[0], c[A1 + 4], R1;
FMUL     R4, v[17], R0;
FMUL     R0, v[1], c[A0 + 9];
FMAD     R1, v[2], c[A1 + 6], R1;
FMAD     R2, v[19], R2, R4;
FMAD     R0, v[0], c[A0 + 8], R0;
FMAD     R4, v[3], c[A1 + 7], R1;
FMUL     R1, v[1], c[A1 + 9];
FMAD     R0, v[2], c[A0 + 10], R0;
FMUL     R4, v[17], R4;
FMAD     R1, v[0], c[A1 + 8], R1;
FMAD     R0, v[3], c[A0 + 11], R0;
FMAD     R3, v[19], R3, R4;
FMAD     R1, v[2], c[A1 + 10], R1;
FMUL32   R4, R3, c[1];
FMAD     R1, v[3], c[A1 + 11], R1;
FMUL32   R5, R3, c[5];
FMAD     R4, R2, c[0], R4;
FMUL     R1, v[17], R1;
FMAD     R5, R2, c[4], R5;
FMUL32   R6, R3, c[9];
FMAD     R0, v[19], R0, R1;
FMUL32   R1, R3, c[13];
FMAD     R3, R2, c[8], R6;
FMAD     R4, R0, c[2], R4;
FMAD     R1, R2, c[12], R1;
FMAD     R2, R0, c[6], R5;
FMAD     o[0], v[3], c[3], R4;
FMAD     R3, R0, c[10], R3;
FMAD     R0, R0, c[14], R1;
FMAD     o[1], v[3], c[7], R2;
FMAD     o[2], v[3], c[11], R3;
FMAD     o[3], v[3], c[15], R0;
MOV      o[8], v[13];
MOV      o[9], v[14];
MOV      o[10], v[15];
MOV      o[11], v[16];
MOV      o[4], v[3];
MOV      o[5], v[3];
MOV      o[6], v[3];
MOV      o[7], v[3];
END
# 180 instructions, 12 R-regs
# 180 inst, (8 mov, 0 mvi, 0 tex, 3 complex, 169 math)
#    155 64-bit, 25 32-bit, 0 32-bit-const
