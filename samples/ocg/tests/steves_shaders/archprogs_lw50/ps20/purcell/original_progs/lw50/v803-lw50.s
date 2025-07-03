!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    15
.MAX_OBUF    26
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v803-lw40.s -o allprogs-new32//v803-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[15].C[15]
#semantic C[14].C[14]
#semantic C[13].C[13]
#semantic C[8].C[8]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[1].C[1]
#semantic C[0].C[0]
#semantic c.c
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[15] :  : c[15] : -1 : 0
#var float4 C[14] :  : c[14] : -1 : 0
#var float4 C[13] :  : c[13] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
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
#ibuf 7 = v[WGT].w
#ibuf 8 = v[NOR].x
#ibuf 9 = v[NOR].y
#ibuf 10 = v[NOR].z
#ibuf 11 = v[NOR].w
#ibuf 12 = v[COL0].x
#ibuf 13 = v[COL0].y
#ibuf 14 = v[COL0].z
#ibuf 15 = v[COL0].w
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
#obuf 15 = o[TEX1].w
#obuf 16 = o[TEX2].x
#obuf 17 = o[TEX2].y
#obuf 18 = o[TEX2].z
#obuf 19 = o[TEX2].w
#obuf 20 = o[TEX3].x
#obuf 21 = o[TEX3].y
#obuf 22 = o[TEX3].z
#obuf 23 = o[TEX3].w
#obuf 24 = o[TEX4].x
#obuf 25 = o[TEX4].y
#obuf 26 = o[TEX4].z
BB0:
F2I.FLOOR R0, v[11];
F2I.FLOOR R1, v[9];
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
FMAD     R3, v[2], c[A0 + 6], R1;
FMUL     R1, v[1], c[A1 + 5];
FMUL     R2, v[8], R2;
FMAD     R3, v[3], c[A0 + 7], R3;
FMAD     R1, v[0], c[A1 + 4], R1;
FMAD     R2, v[10], R0, R2;
FMUL     R0, v[1], c[A0 + 9];
FMAD     R1, v[2], c[A1 + 6], R1;
FMAD     R0, v[0], c[A0 + 8], R0;
FMAD     R4, v[3], c[A1 + 7], R1;
FMUL     R1, v[1], c[A1 + 9];
FMAD     R0, v[2], c[A0 + 10], R0;
FMUL     R4, v[8], R4;
FMAD     R1, v[0], c[A1 + 8], R1;
FMAD     R0, v[3], c[A0 + 11], R0;
FMAD     R3, v[10], R3, R4;
FMAD     R1, v[2], c[A1 + 10], R1;
FMUL32   R4, R3, c[53];
FMAD     R1, v[3], c[A1 + 11], R1;
FMUL32   R5, R3, c[57];
FMAD     R4, R2, c[52], R4;
FMUL     R1, v[8], R1;
FMAD     R5, R2, c[56], R5;
FMUL32   R6, R3, c[61];
FMAD     R0, v[10], R0, R1;
FMUL32   R1, R3, c[1];
FMAD     R6, R2, c[60], R6;
FMAD     R4, R0, c[54], R4;
FMAD     R5, R0, c[58], R5;
FMAD     R6, R0, c[62], R6;
FMAD     o[12], v[3], c[55], R4;
FMAD     o[13], v[3], c[59], R5;
FMAD     o[14], v[3], c[63], R6;
FMAD     R1, R2, c[0], R1;
FMUL32   R4, R3, c[5];
FMUL32   R5, R3, c[9];
FMAD     R1, R0, c[2], R1;
FMAD     R4, R2, c[4], R4;
FMAD     R5, R2, c[8], R5;
FMAD     o[0], v[3], c[3], R1;
FMAD     R1, R0, c[6], R4;
FMAD     R4, R0, c[10], R5;
FMUL32   R3, R3, c[13];
FMAD     o[1], v[3], c[7], R1;
FMAD     o[2], v[3], c[11], R4;
FMAD     R3, R2, c[12], R3;
FMUL     R1, v[8], c[A1 + 12];
FMUL     R2, v[8], c[A1 + 13];
FMAD     R0, R0, c[14], R3;
FMAD     R1, v[10], c[A0 + 12], R1;
FMAD     R2, v[10], c[A0 + 13], R2;
FMAD     o[3], v[3], c[15], R0;
FADD     R0, -v[0], R1;
FADD     R2, -v[1], R2;
FMUL     R1, v[8], c[A1 + 14];
FMUL32   R3, R2, R2;
FMAD     R1, v[10], c[A0 + 14], R1;
MOV      o[20], v[0];
FMAD     R3, R0, R0, R3;
FADD     R1, -v[2], R1;
MOV      o[21], v[1];
MOV      o[22], v[2];
FMAD     R3, R1, R1, R3;
MOV      o[23], v[3];
MOV      o[16], v[12];
RSQ      R3, |R3|;
MOV      o[17], v[13];
MOV      o[18], v[14];
FMUL32   o[24], R0, R3;
FMUL32   o[25], R2, R3;
FMUL32   o[26], R1, R3;
MOV      o[19], v[15];
MOV      o[15], v[3];
MOV      o[8], v[4];
MOV      o[9], v[5];
MOV      o[10], v[6];
MOV      o[11], v[7];
MOV32    o[4], c[32];
MOV32    o[5], c[33];
MOV32    o[6], c[34];
MOV32    o[7], c[35];
END
# 97 instructions, 8 R-regs
# 97 inst, (17 mov, 0 mvi, 0 tex, 1 complex, 79 math)
#    82 64-bit, 15 32-bit, 0 32-bit-const
