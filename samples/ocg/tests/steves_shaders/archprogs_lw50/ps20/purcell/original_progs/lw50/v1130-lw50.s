!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     27
.MAX_IBUF    17
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1130-lw40.s -o allprogs-new32//v1130-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[32].C[32]
#semantic C[33].C[33]
#semantic C[3].C[3]
#semantic C[27].C[27]
#semantic C[30].C[30]
#semantic C[28].C[28]
#semantic C[31].C[31]
#semantic C[2].C[2]
#semantic C[16].C[16]
#semantic C[29].C[29]
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
#var float4 C[32] :  : c[32] : -1 : 0
#var float4 C[33] :  : c[33] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
#var float4 C[30] :  : c[30] : -1 : 0
#var float4 C[28] :  : c[28] : -1 : 0
#var float4 C[31] :  : c[31] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[29] :  : c[29] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
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
FMUL     R1, v[6], c[4];
FADD     R2, -v[4], c[1];
FMUL     R0, v[6], c[4];
R2A      A0, R1;
FADD     R6, -v[5], R2;
R2A      A1, R0;
FMUL     R0, v[7], c[4];
R2A      A2, R0;
FMUL     R1, v[5], c[A2 + 168];
FMUL     R2, v[5], c[A2 + 169];
FMUL     R0, v[5], c[A2 + 170];
FMAD     R1, v[4], c[A1 + 168], R1;
FMAD     R2, v[4], c[A1 + 169], R2;
FMAD     R0, v[4], c[A1 + 170], R0;
FMAD     R1, c[A0 + 168], R6, R1;
FMAD     R5, c[A0 + 169], R6, R2;
FMAD     R3, c[A0 + 170], R6, R0;
FMUL     R0, v[5], c[A2 + 171];
FMUL     R4, v[1], R5;
FMUL     R2, v[5], c[A2 + 172];
FMAD     R0, v[4], c[A1 + 171], R0;
FMAD     R4, v[0], R1, R4;
FMAD     R2, v[4], c[A1 + 172], R2;
FMAD     R0, c[A0 + 171], R6, R0;
FMAD     R4, v[2], R3, R4;
FMAD     R9, c[A0 + 172], R6, R2;
FMUL     R2, v[5], c[A2 + 173];
FMAD     R12, v[3], R0, R4;
FMUL     R0, v[5], c[A2 + 174];
FMAD     R2, v[4], c[A1 + 173], R2;
FADD32   R16, -R12, c[116];
FMAD     R0, v[4], c[A1 + 174], R0;
FMAD     R18, c[A0 + 173], R6, R2;
FMUL     R4, v[5], c[A2 + 175];
FMAD     R2, c[A0 + 174], R6, R0;
FMUL     R0, v[1], R18;
FMAD     R4, v[4], c[A1 + 175], R4;
FMUL     R7, v[5], c[A2 + 176];
FMAD     R0, v[0], R9, R0;
FMAD     R4, c[A0 + 175], R6, R4;
FMAD     R7, v[4], c[A1 + 176], R7;
FMAD     R8, v[2], R2, R0;
FMUL     R0, v[5], c[A2 + 177];
FMAD     R7, c[A0 + 176], R6, R7;
FMAD     R13, v[3], R4, R8;
FMAD     R4, v[4], c[A1 + 177], R0;
FMUL     R0, v[5], c[A2 + 178];
FADD32   R15, -R13, c[117];
FMAD     R8, c[A0 + 177], R6, R4;
FMAD     R4, v[4], c[A1 + 178], R0;
FMUL     R10, v[5], c[A2 + 179];
FMUL32   R0, R15, R15;
FMAD     R4, c[A0 + 178], R6, R4;
FMAD     R11, v[4], c[A1 + 179], R10;
FMAD     R0, R16, R16, R0;
FMUL     R10, v[1], R8;
FMAD     R11, c[A0 + 179], R6, R11;
MOV32    R6, c[125];
FMAD     R10, v[0], R7, R10;
FMAD     R10, v[2], R4, R10;
FMAD     R11, v[3], R11, R10;
FADD32   R14, -R11, c[118];
FMAD     R0, R14, R14, R0;
RSQ      R10, |R0|;
FMUL32   R16, R16, R10;
FMUL32   R17, R15, R10;
FMUL32   R15, R14, R10;
FMUL32   R10, R0, R10;
FMUL32   R14, -R17, c[113];
FMAD     R6, R6, R10, c[124];
FMAD     R10, -R16, c[112], R14;
FMAD     R0, R0, c[126], R6;
FMAD     R6, -R15, c[114], R10;
RCP      R14, R0;
FADD32   R0, R6, -c[122];
FMUL32   R0, R0, c[123];
FMAX     R6, R0, c[0];
FMUL     R0, v[15], R18;
LG2      R10, |R6|;
FMAD     R6, v[14], R9, R0;
FMUL     R0, v[10], R8;
FMUL32   R10, R10, c[120];
FMAD     R6, v[16], R2, R6;
FMAD     R0, v[9], R7, R0;
RRO      R10, R10, 1;
FMUL     R18, v[10], R18;
FMAD     R0, v[11], R4, R0;
EX2      R10, R10;
FMAD     R9, v[9], R9, R18;
FMUL     R8, v[15], R8;
FMIN     R18, R10, c[1];
FMAD     R2, v[11], R2, R9;
FMAD     R9, v[14], R7, R8;
FMUL32   R7, R18, c[108];
FMUL32   R8, R6, R0;
FMAD     R4, v[16], R4, R9;
FMUL32   R19, R14, R7;
FMUL     R7, v[10], R5;
FMAD     R8, R2, R4, -R8;
FMUL     R5, v[15], R5;
FMAD     R7, v[9], R1, R7;
FMUL     R8, v[17], R8;
FMAD     R5, v[14], R1, R5;
FMAD     R1, v[11], R3, R7;
FMAD     R5, v[16], R3, R5;
FMUL32   R3, R4, R1;
FMUL32   R7, R5, R2;
FMAD     R9, R0, R5, -R3;
FMUL32   R3, R6, c[7];
FMAD     R7, R1, R6, -R7;
FMUL     R9, v[17], R9;
FMAD     R3, R5, c[7], R3;
FMUL     R7, v[17], R7;
FMUL32   R20, R9, c[7];
FMAD     R10, R4, c[7], R3;
FMUL32   R3, R2, c[7];
FMAD     R20, R8, c[7], R20;
FMAD     R3, R1, c[7], R3;
FMAD     R21, R7, c[7], R20;
FMAD     R3, R0, c[7], R3;
FMUL32   R20, R21, R17;
FMUL32   R24, R21, R21;
FSET     R22, R21, c[0], LT;
FMAD     R20, R10, R16, R20;
FMUL32   R23, R10, R10;
R2A      A2, R22;
FMAD     R20, R3, R15, R20;
FSET     R22, R10, c[0], LT;
FMUL32   R25, R24, c[A2 + 92];
FMAX     R20, R20, c[0];
R2A      A1, R22;
FMUL32   R22, R21, -c[133];
FMUL32   R21, R3, R3;
FMAD     R25, R23, c[A1 + 84], R25;
FMAD     R10, R10, -c[132], R22;
FSET     R22, R3, c[0], LT;
FMUL32   R26, R18, c[109];
FMAD     R3, R3, -c[134], R10;
R2A      A0, R22;
FMUL32   R10, R14, R26;
FMAX     R3, R3, c[0];
FMAD     R22, R21, c[A0 + 100], R25;
FMUL32   R25, R18, c[110];
FMUL32   R18, R24, c[A2 + 93];
FMUL32   R24, R24, c[A2 + 94];
FMUL32   R14, R14, R25;
FMAD     R22, R19, R20, R22;
FMAD     R18, R23, c[A1 + 85], R18;
FMAD     R23, R23, c[A1 + 86], R24;
FMAD     o[30], R3, c[128], R22;
FMAD     R18, R21, c[A0 + 101], R18;
FMAD     R21, R21, c[A0 + 102], R23;
FMUL32   R22, R9, c[12];
FMAD     R18, R10, R20, R18;
FMAD     R20, R14, R20, R21;
FMAD     R21, R7, c[13], R22;
FMAD     o[31], R3, c[129], R18;
FMAD     o[32], R3, c[130], R20;
FMUL32   R20, R21, R17;
FMUL32   R18, R6, c[12];
FMUL32   R3, R2, c[12];
FMUL32   R24, R21, R21;
FMAD     R18, R4, c[13], R18;
FMAD     R3, R0, c[13], R3;
FSET     R22, R21, c[0], LT;
FMAD     R20, R18, R16, R20;
FMUL32   R23, R18, R18;
R2A      A2, R22;
FMAD     R20, R3, R15, R20;
FSET     R22, R18, c[0], LT;
FMUL32   R25, R24, c[A2 + 92];
FMAX     R20, R20, c[0];
R2A      A1, R22;
FMUL32   R22, R21, -c[133];
FMUL32   R21, R3, R3;
FMAD     R25, R23, c[A1 + 84], R25;
FMAD     R18, R18, -c[132], R22;
FSET     R22, R3, c[0], LT;
FMUL32   R26, R24, c[A2 + 93];
FMAD     R3, R3, -c[134], R18;
R2A      A0, R22;
FMAD     R18, R23, c[A1 + 85], R26;
FMUL32   R22, R24, c[A2 + 94];
FMAD     R24, R21, c[A0 + 100], R25;
FMAX     R3, R3, c[0];
FMAD     R22, R23, c[A1 + 86], R22;
FMAD     R23, R19, R20, R24;
FMAD     R18, R21, c[A0 + 101], R18;
FMAD     R21, R21, c[A0 + 102], R22;
FMAD     o[8], R3, c[128], R23;
FMAD     R18, R10, R20, R18;
FMAD     R20, R14, R20, R21;
FMUL32   R21, R9, c[6];
FMAD     o[9], R3, c[129], R18;
FMAD     o[10], R3, c[130], R20;
FMAD     R3, R8, c[5], R21;
FMUL32   R18, R6, c[6];
FMUL32   R20, R2, c[6];
FMAD     R3, R7, c[6], R3;
FMAD     R18, R5, c[5], R18;
FMAD     R20, R1, c[5], R20;
FMUL32   R21, R3, R17;
FMAD     R18, R4, c[6], R18;
FMAD     R17, R0, c[6], R20;
FMUL32   R20, R3, R3;
FMAD     R21, R18, R16, R21;
FSET     R22, R3, c[0], LT;
FMUL32   R16, R18, R18;
FMAD     R15, R17, R15, R21;
R2A      A2, R22;
FSET     R22, R18, c[0], LT;
FMAX     R15, R15, c[0];
FMUL32   R21, R20, c[A2 + 92];
R2A      A1, R22;
FMUL32   R3, R3, -c[133];
FMUL32   R22, R17, R17;
FMAD     R21, R16, c[A1 + 84], R21;
FMAD     R3, R18, -c[132], R3;
FSET     R18, R17, c[0], LT;
FMUL32   R23, R20, c[A2 + 93];
FMAD     R3, R17, -c[134], R3;
R2A      A0, R18;
FMAD     R17, R16, c[A1 + 85], R23;
FMUL32   R18, R20, c[A2 + 94];
FMAD     R20, R22, c[A0 + 100], R21;
FMAX     R3, R3, c[0];
FMAD     R16, R16, c[A1 + 86], R18;
FMAD     R18, R19, R15, R20;
FMAD     R17, R22, c[A0 + 101], R17;
FMAD     R16, R22, c[A0 + 102], R16;
FMAD     o[4], R3, c[128], R18;
FMAD     R10, R10, R15, R17;
FMAD     R14, R14, R15, R16;
MOV32    o[28], R7;
FMAD     o[5], R3, c[129], R10;
FMAD     o[6], R3, c[130], R14;
MOV32    o[25], R9;
MOV32    o[22], R8;
FMUL32   R8, R13, c[41];
MOV32    R3, c[43];
MOV32    R7, c[67];
FMAD     R8, R12, c[40], R8;
FMUL32   R9, R13, c[33];
FMAD     R8, R11, c[42], R8;
FMAD     R9, R12, c[32], R9;
FMAD     R3, R3, c[1], R8;
MOV32    R8, c[35];
FMAD     R9, R11, c[34], R9;
FMAD     R7, -R3, R7, c[64];
FMUL32   R10, R13, c[37];
MOV32    o[33], R7;
FMAD     o[0], R8, c[1], R9;
FMAD     R8, R12, c[36], R10;
MOV32    o[2], R3;
MOV32    o[34], R7;
FMAD     R3, R11, c[38], R8;
MOV32    o[35], R7;
MOV32    o[36], R7;
MOV32    R7, c[39];
FMUL32   R8, R13, c[45];
MOV32    o[27], R4;
MOV32    R4, R7;
FMAD     R7, R12, c[44], R8;
MOV32    o[29], R0;
FMAD     o[1], R4, c[1], R3;
FMAD     R0, R11, c[46], R7;
MOV32    o[24], R6;
MOV32    o[26], R2;
MOV32    o[21], R5;
MOV32    o[23], R1;
FADD32   o[18], -R12, c[8];
FADD32   o[19], -R13, c[9];
FADD32   o[20], -R11, c[10];
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
# 289 instructions, 28 R-regs
# 289 inst, (43 mov, 0 mvi, 0 tex, 4 complex, 242 math)
#    197 64-bit, 92 32-bit, 0 32-bit-const
