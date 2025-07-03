!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     15
.MAX_IBUF    13
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1063-lw40.s -o allprogs-new32//v1063-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[3].C[3]
#semantic C[2].C[2]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic c.c
#semantic C[0].C[0]
#semantic C[1].C[1]
#var float4 o[TEX7] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX6] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX5] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL1] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].z
#ibuf 5 = v[NOR].x
#ibuf 6 = v[NOR].y
#ibuf 7 = v[NOR].z
#ibuf 8 = v[COL0].x
#ibuf 9 = v[COL0].y
#ibuf 10 = v[FOG].x
#ibuf 11 = v[FOG].y
#ibuf 12 = v[FOG].z
#ibuf 13 = v[FOG].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[BCOL0].w
#obuf 8 = o[BCOL1].x
#obuf 9 = o[BCOL1].y
#obuf 10 = o[BCOL1].z
#obuf 11 = o[BCOL1].w
#obuf 12 = o[TEX0].x
#obuf 13 = o[TEX0].y
#obuf 14 = o[TEX1].x
#obuf 15 = o[TEX1].y
#obuf 16 = o[TEX2].x
#obuf 17 = o[TEX2].y
#obuf 18 = o[TEX3].x
#obuf 19 = o[TEX3].y
#obuf 20 = o[TEX3].z
#obuf 21 = o[TEX4].x
#obuf 22 = o[TEX4].y
#obuf 23 = o[TEX4].z
#obuf 24 = o[TEX5].x
#obuf 25 = o[TEX5].y
#obuf 26 = o[TEX5].z
#obuf 27 = o[TEX6].x
#obuf 28 = o[TEX6].y
#obuf 29 = o[TEX6].z
#obuf 30 = o[TEX7].x
#obuf 31 = o[TEX7].y
#obuf 32 = o[TEX7].z
#obuf 33 = o[FOGC].x
#obuf 34 = o[FOGC].y
#obuf 35 = o[FOGC].z
#obuf 36 = o[FOGC].w
BB0:
FMUL     R0, v[4], c[4];
R2A      A0, R0;
FMUL     R0, v[6], c[A0 + 177];
FMUL     R2, v[11], c[A0 + 173];
FMUL     R1, v[6], c[A0 + 173];
FMAD     R0, v[5], c[A0 + 176], R0;
FMAD     R2, v[10], c[A0 + 172], R2;
FMAD     R1, v[5], c[A0 + 172], R1;
FMAD     R0, v[7], c[A0 + 178], R0;
FMAD     R7, v[12], c[A0 + 174], R2;
FMAD     R2, v[7], c[A0 + 174], R1;
FMUL     R4, v[11], c[A0 + 177];
FMUL32   R3, R0, R7;
FMUL     R1, v[6], c[A0 + 169];
FMAD     R5, v[10], c[A0 + 176], R4;
FMUL     R4, v[11], c[A0 + 169];
FMAD     R1, v[5], c[A0 + 168], R1;
FMAD     R5, v[12], c[A0 + 178], R5;
FMAD     R4, v[10], c[A0 + 168], R4;
FMAD     R1, v[7], c[A0 + 170], R1;
FMAD     R3, R2, R5, -R3;
FMAD     R6, v[12], c[A0 + 170], R4;
FMUL32   R4, R1, R5;
FMUL     R9, v[13], R3;
FMUL32   R3, R2, R6;
FMAD     R4, R0, R6, -R4;
FMAD     R3, R1, R7, -R3;
FMUL     R10, v[13], R4;
FMUL     R8, v[13], R3;
FMUL32   R3, R10, c[7];
FMUL32   R4, R7, c[7];
FMAD     R3, R9, c[7], R3;
FMAD     R4, R6, c[7], R4;
FMAD     R3, R8, c[7], R3;
FMAD     R11, R5, c[7], R4;
FMUL32   R13, R3, R3;
FSET     R4, R3, c[0], LT;
FMUL32   R3, R11, R11;
FSET     R11, R11, c[0], LT;
R2A      A2, R4;
FMUL32   R4, R2, c[7];
R2A      A1, R11;
FMUL32   R11, R13, c[A2 + 92];
FMAD     R4, R1, c[7], R4;
FMUL32   R12, R13, c[A2 + 93];
FMAD     R11, R3, c[A1 + 84], R11;
FMUL32   R13, R13, c[A2 + 94];
FMAD     R4, R0, c[7], R4;
FMAD     R12, R3, c[A1 + 85], R12;
FMAD     R13, R3, c[A1 + 86], R13;
FMUL32   R3, R4, R4;
FSET     R4, R4, c[0], LT;
FMUL32   R14, R10, c[12];
R2A      A1, R4;
FMAD     R14, R8, c[13], R14;
FMUL32   R4, R7, c[12];
FMAD     o[30], R3, c[A1 + 100], R11;
FMAD     o[31], R3, c[A1 + 101], R12;
FMAD     o[32], R3, c[A1 + 102], R13;
FMAD     R4, R5, c[13], R4;
FMUL32   R3, R14, R14;
FSET     R13, R14, c[0], LT;
FMUL32   R11, R4, R4;
FSET     R12, R4, c[0], LT;
R2A      A2, R13;
FMUL32   R4, R2, c[12];
R2A      A1, R12;
FMUL32   R12, R3, c[A2 + 92];
FMAD     R4, R0, c[13], R4;
FMUL32   R13, R3, c[A2 + 93];
FMUL32   R3, R3, c[A2 + 94];
FMAD     R12, R11, c[A1 + 84], R12;
FMAD     R13, R11, c[A1 + 85], R13;
FMAD     R11, R11, c[A1 + 86], R3;
FMUL32   R3, R4, R4;
FSET     R4, R4, c[0], LT;
FMUL32   R14, R10, c[6];
R2A      A1, R4;
FMAD     R4, R9, c[5], R14;
FMAD     o[8], R3, c[A1 + 100], R12;
FMAD     o[9], R3, c[A1 + 101], R13;
FMAD     o[10], R3, c[A1 + 102], R11;
FMAD     R4, R8, c[6], R4;
FMUL32   R3, R7, c[6];
FMUL32   R14, R4, R4;
FSET     R11, R4, c[0], LT;
FMAD     R4, R6, c[5], R3;
FMUL32   R3, R2, c[6];
R2A      A2, R11;
FMAD     R4, R5, c[6], R4;
FMAD     R3, R1, c[5], R3;
FMUL32   R12, R14, c[A2 + 92];
FMUL32   R11, R4, R4;
FSET     R4, R4, c[0], LT;
FMAD     R3, R0, c[6], R3;
FMUL32   R13, R14, c[A2 + 93];
R2A      A1, R4;
FMUL32   R14, R14, c[A2 + 94];
FMUL32   R4, R3, R3;
FMAD     R12, R11, c[A1 + 84], R12;
FSET     R3, R3, c[0], LT;
FMAD     R13, R11, c[A1 + 85], R13;
FMAD     R11, R11, c[A1 + 86], R14;
R2A      A1, R3;
MOV32    o[28], R8;
MOV32    o[25], R10;
FMAD     o[4], R4, c[A1 + 100], R12;
FMAD     o[5], R4, c[A1 + 101], R13;
FMAD     o[6], R4, c[A1 + 102], R11;
MOV32    o[22], R9;
FMUL     R3, v[1], c[A0 + 169];
FMUL     R4, v[1], c[A0 + 173];
FMAD     R3, v[0], c[A0 + 168], R3;
FMAD     R4, v[0], c[A0 + 172], R4;
FMUL     R8, v[1], c[A0 + 177];
FMAD     R3, v[2], c[A0 + 170], R3;
FMAD     R4, v[2], c[A0 + 174], R4;
FMAD     R8, v[0], c[A0 + 176], R8;
FMAD     R3, v[3], c[A0 + 171], R3;
FMAD     R4, v[3], c[A0 + 175], R4;
FMAD     R8, v[2], c[A0 + 178], R8;
MOV32    R9, c[43];
FMUL32   R10, R4, c[41];
FMAD     R8, v[3], c[A0 + 179], R8;
FMAD     R11, R3, c[40], R10;
MOV32    R10, c[67];
FMUL32   R12, R4, c[33];
FMAD     R11, R8, c[42], R11;
FMAD     R12, R3, c[32], R12;
FMAD     R9, R9, c[1], R11;
MOV32    R11, c[35];
FMAD     R12, R8, c[34], R12;
FMAD     R10, -R9, R10, c[64];
FMUL32   R13, R4, c[37];
MOV32    o[33], R10;
FMAD     o[0], R11, c[1], R12;
FMAD     R11, R3, c[36], R13;
MOV32    o[2], R9;
MOV32    o[34], R10;
FMAD     R9, R8, c[38], R11;
MOV32    o[35], R10;
MOV32    o[36], R10;
MOV32    R10, c[39];
FMUL32   R11, R4, c[45];
MOV32    o[27], R5;
MOV32    R5, R10;
FMAD     R10, R3, c[44], R11;
MOV32    o[29], R0;
FMAD     o[1], R5, c[1], R9;
FMAD     R0, R8, c[46], R10;
MOV32    o[24], R7;
MOV32    o[26], R2;
MOV32    o[21], R6;
MOV32    o[23], R1;
FADD32   o[18], -R3, c[8];
FADD32   o[19], -R4, c[9];
FADD32   o[20], -R8, c[10];
MOV32    R3, c[47];
FMUL     R1, v[8], c[360];
FMUL     R2, v[8], c[364];
FMAD     R1, v[9], c[361], R1;
FMAD     R2, v[9], c[365], R2;
FMAD     o[3], R3, c[1], R0;
MOV32    o[16], R1;
MOV32    o[17], R2;
MOV32    o[14], R1;
MOV32    o[12], R1;
MOV32    o[15], R2;
MOV32    o[13], R2;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    o[11], R0;
MOV32    o[7], R1;
END
# 173 instructions, 16 R-regs
# 173 inst, (40 mov, 0 mvi, 0 tex, 0 complex, 133 math)
#    106 64-bit, 67 32-bit, 0 32-bit-const
