!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     27
.MAX_IBUF    17
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1095-lw40.s -o allprogs-new32//v1095-lw50.s
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
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[NOR].x
#ibuf 7 = v[NOR].y
#ibuf 8 = v[NOR].z
#ibuf 9 = v[COL0].x
#ibuf 10 = v[COL0].y
#ibuf 11 = v[COL0].z
#ibuf 12 = v[COL1].x
#ibuf 13 = v[COL1].y
#ibuf 14 = v[UNUSED0].x
#ibuf 15 = v[UNUSED0].y
#ibuf 16 = v[UNUSED0].z
#ibuf 17 = v[UNUSED0].w
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
FMUL     R0, v[6], c[4];
FADD     R2, -v[4], c[1];
FMUL     R1, v[6], c[4];
R2A      A0, R0;
FADD     R0, -v[5], R2;
R2A      A1, R1;
FMUL     R1, v[7], c[4];
R2A      A2, R1;
FMUL     R3, v[5], c[A2 + 172];
FMUL     R2, v[5], c[A2 + 173];
FMUL     R1, v[5], c[A2 + 174];
FMAD     R3, v[4], c[A1 + 172], R3;
FMAD     R2, v[4], c[A1 + 173], R2;
FMAD     R1, v[4], c[A1 + 174], R1;
FMAD     R10, c[A0 + 172], R0, R3;
FMAD     R11, c[A0 + 173], R0, R2;
FMAD     R9, c[A0 + 174], R0, R1;
FMUL     R2, v[5], c[A2 + 176];
FMUL     R3, v[15], R11;
FMUL     R1, v[5], c[A2 + 177];
FMAD     R2, v[4], c[A1 + 176], R2;
FMAD     R3, v[14], R10, R3;
FMAD     R1, v[4], c[A1 + 177], R1;
FMAD     R2, c[A0 + 176], R0, R2;
FMAD     R17, v[16], R9, R3;
FMAD     R3, c[A0 + 177], R0, R1;
FMUL     R1, v[5], c[A2 + 178];
FMUL     R5, v[10], R11;
FMUL     R4, v[10], R3;
FMAD     R1, v[4], c[A1 + 178], R1;
FMAD     R5, v[9], R10, R5;
FMAD     R4, v[9], R2, R4;
FMAD     R1, c[A0 + 178], R0, R1;
FMAD     R12, v[11], R9, R5;
FMUL     R5, v[15], R3;
FMAD     R4, v[11], R1, R4;
FMUL     R6, v[5], c[A2 + 168];
FMAD     R5, v[14], R2, R5;
FMUL32   R7, R17, R4;
FMAD     R6, v[4], c[A1 + 168], R6;
FMAD     R15, v[16], R1, R5;
FMUL     R5, v[5], c[A2 + 169];
FMAD     R6, c[A0 + 168], R0, R6;
FMAD     R8, R12, R15, -R7;
FMAD     R7, v[4], c[A1 + 169], R5;
FMUL     R5, v[5], c[A2 + 170];
FMUL     R19, v[17], R8;
FMAD     R7, c[A0 + 169], R0, R7;
FMAD     R5, v[4], c[A1 + 170], R5;
FMUL     R8, v[10], R7;
FMAD     R5, c[A0 + 170], R0, R5;
FMUL     R13, v[15], R7;
FMAD     R8, v[9], R6, R8;
FMAD     R13, v[14], R6, R13;
FMAD     R8, v[11], R5, R8;
FMAD     R16, v[16], R5, R13;
FMUL32   R14, R15, R8;
FMUL32   R13, R16, R12;
FMAD     R14, R4, R16, -R14;
FMAD     R13, R8, R17, -R13;
FMUL     R20, v[17], R14;
FMUL     R18, v[17], R13;
FMUL32   R13, R20, c[7];
FMUL32   R14, R17, c[7];
FMAD     R13, R19, c[7], R13;
FMAD     R14, R16, c[7], R14;
FMAD     R13, R18, c[7], R13;
FMAD     R21, R15, c[7], R14;
FMUL32   R23, R13, R13;
FSET     R14, R13, c[0], LT;
FMUL32   R13, R21, R21;
FSET     R21, R21, c[0], LT;
R2A      A4, R14;
FMUL32   R14, R12, c[7];
R2A      A3, R21;
FMUL32   R21, R23, c[A4 + 92];
FMAD     R14, R8, c[7], R14;
FMUL32   R22, R23, c[A4 + 93];
FMAD     R21, R13, c[A3 + 84], R21;
FMUL32   R23, R23, c[A4 + 94];
FMAD     R14, R4, c[7], R14;
FMAD     R22, R13, c[A3 + 85], R22;
FMAD     R23, R13, c[A3 + 86], R23;
FMUL32   R13, R14, R14;
FSET     R14, R14, c[0], LT;
FMUL32   R24, R20, c[12];
R2A      A3, R14;
FMAD     R24, R18, c[13], R24;
FMUL32   R14, R17, c[12];
FMAD     o[30], R13, c[A3 + 100], R21;
FMAD     o[31], R13, c[A3 + 101], R22;
FMAD     o[32], R13, c[A3 + 102], R23;
FMAD     R14, R15, c[13], R14;
FMUL32   R13, R24, R24;
FSET     R23, R24, c[0], LT;
FMUL32   R21, R14, R14;
FSET     R22, R14, c[0], LT;
R2A      A4, R23;
FMUL32   R14, R12, c[12];
R2A      A3, R22;
FMUL32   R22, R13, c[A4 + 92];
FMAD     R14, R4, c[13], R14;
FMUL32   R23, R13, c[A4 + 93];
FMUL32   R13, R13, c[A4 + 94];
FMAD     R22, R21, c[A3 + 84], R22;
FMAD     R23, R21, c[A3 + 85], R23;
FMAD     R21, R21, c[A3 + 86], R13;
FMUL32   R13, R14, R14;
FSET     R14, R14, c[0], LT;
FMUL32   R24, R20, c[6];
R2A      A3, R14;
FMAD     R14, R19, c[5], R24;
FMAD     o[8], R13, c[A3 + 100], R22;
FMAD     o[9], R13, c[A3 + 101], R23;
FMAD     o[10], R13, c[A3 + 102], R21;
FMAD     R14, R18, c[6], R14;
FMUL32   R13, R17, c[6];
FMUL32   R24, R14, R14;
FSET     R21, R14, c[0], LT;
FMAD     R14, R16, c[5], R13;
FMUL32   R13, R12, c[6];
R2A      A4, R21;
FMAD     R14, R15, c[6], R14;
FMAD     R13, R8, c[5], R13;
FMUL32   R22, R24, c[A4 + 92];
FMUL32   R21, R14, R14;
FSET     R14, R14, c[0], LT;
FMAD     R13, R4, c[6], R13;
FMUL32   R23, R24, c[A4 + 93];
R2A      A3, R14;
FMUL32   R24, R24, c[A4 + 94];
FMUL32   R14, R13, R13;
FMAD     R22, R21, c[A3 + 84], R22;
FSET     R13, R13, c[0], LT;
FMAD     R23, R21, c[A3 + 85], R23;
FMAD     R21, R21, c[A3 + 86], R24;
R2A      A3, R13;
MOV32    o[28], R18;
MOV32    o[25], R20;
FMAD     o[4], R14, c[A3 + 100], R22;
FMAD     o[5], R14, c[A3 + 101], R23;
FMAD     o[6], R14, c[A3 + 102], R21;
MOV32    o[22], R19;
FMUL     R7, v[1], R7;
FMUL     R13, v[1], R11;
FMUL     R11, v[5], c[A2 + 171];
FMAD     R6, v[0], R6, R7;
FMAD     R7, v[0], R10, R13;
FMAD     R10, v[4], c[A1 + 171], R11;
FMAD     R5, v[2], R5, R6;
FMAD     R6, v[2], R9, R7;
FMAD     R7, c[A0 + 171], R0, R10;
FMUL     R9, v[1], R3;
FMUL     R3, v[5], c[A2 + 175];
FMAD     R5, v[3], R7, R5;
FMAD     R2, v[0], R2, R9;
FMAD     R3, v[4], c[A1 + 175], R3;
FMUL     R7, v[5], c[A2 + 179];
FMAD     R1, v[2], R1, R2;
FMAD     R2, c[A0 + 175], R0, R3;
FMAD     R3, v[4], c[A1 + 179], R7;
MOV32    R7, c[43];
FMAD     R2, v[3], R2, R6;
FMAD     R6, c[A0 + 179], R0, R3;
MOV32    R0, R7;
FMUL32   R3, R2, c[41];
FMAD     R1, v[3], R6, R1;
MOV32    R6, c[67];
FMAD     R3, R5, c[40], R3;
FMUL32   R7, R2, c[33];
FMAD     R3, R1, c[42], R3;
FMAD     R9, R5, c[32], R7;
MOV32    R7, c[35];
FMAD     R0, R0, c[1], R3;
FMAD     R3, R1, c[34], R9;
FMAD     R6, -R0, R6, c[64];
FMUL32   R9, R2, c[37];
FMAD     o[0], R7, c[1], R3;
MOV32    o[33], R6;
FMAD     R3, R5, c[36], R9;
MOV32    o[2], R0;
MOV32    o[34], R6;
FMAD     R0, R1, c[38], R3;
MOV32    o[35], R6;
MOV32    o[36], R6;
MOV32    R3, c[39];
FMUL32   R6, R2, c[45];
MOV32    o[27], R15;
FMAD     R6, R5, c[44], R6;
MOV32    o[29], R4;
FMAD     o[1], R3, c[1], R0;
FMAD     R0, R1, c[46], R6;
MOV32    o[24], R17;
MOV32    o[26], R12;
MOV32    o[21], R16;
MOV32    o[23], R8;
FADD32   o[18], -R5, c[8];
FADD32   o[19], -R2, c[9];
FADD32   o[20], -R1, c[10];
MOV32    R3, c[47];
FMUL     R1, v[12], c[360];
FMUL     R2, v[12], c[364];
FMAD     R1, v[13], c[361], R1;
FMAD     R2, v[13], c[365], R2;
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
# 215 instructions, 28 R-regs
# 215 inst, (42 mov, 0 mvi, 0 tex, 0 complex, 173 math)
#    148 64-bit, 67 32-bit, 0 32-bit-const
