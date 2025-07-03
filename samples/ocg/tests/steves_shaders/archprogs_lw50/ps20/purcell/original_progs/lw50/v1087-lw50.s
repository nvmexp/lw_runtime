!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     27
.MAX_IBUF    16
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1087-lw40.s -o allprogs-new32//v1087-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[3].C[3]
#semantic C[27].C[27]
#semantic C[31].C[31]
#semantic C[2].C[2]
#semantic C[30].C[30]
#semantic C[28].C[28]
#semantic C[16].C[16]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[29].C[29]
#semantic C[8].C[8]
#semantic C[0].C[0]
#semantic c.c
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
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[31] :  : c[31] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[30] :  : c[30] : -1 : 0
#var float4 C[28] :  : c[28] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[29] :  : c[29] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[NOR].y
#ibuf 7 = v[NOR].z
#ibuf 8 = v[COL0].x
#ibuf 9 = v[COL0].y
#ibuf 10 = v[COL0].z
#ibuf 11 = v[COL1].x
#ibuf 12 = v[COL1].y
#ibuf 13 = v[UNUSED0].x
#ibuf 14 = v[UNUSED0].y
#ibuf 15 = v[UNUSED0].z
#ibuf 16 = v[UNUSED0].w
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
FMUL     R0, v[7], c[4];
FMUL     R1, v[6], c[4];
R2A      A0, R0;
R2A      A1, R1;
FMUL     R2, v[5], c[A1 + 168];
FMUL     R1, v[5], c[A1 + 169];
FMUL     R0, v[5], c[A1 + 170];
FMAD     R10, v[4], c[A0 + 168], R2;
FMAD     R15, v[4], c[A0 + 169], R1;
FMAD     R13, v[4], c[A0 + 170], R0;
FMUL     R1, v[5], c[A1 + 171];
FMUL     R0, v[1], R15;
FMUL     R2, v[5], c[A1 + 172];
FMAD     R1, v[4], c[A0 + 171], R1;
FMAD     R0, v[0], R10, R0;
FMAD     R17, v[4], c[A0 + 172], R2;
FMUL     R3, v[5], c[A1 + 173];
FMAD     R2, v[2], R13, R0;
FMUL     R0, v[5], c[A1 + 174];
FMAD     R18, v[4], c[A0 + 173], R3;
FMAD     R1, v[3], R1, R2;
FMAD     R11, v[4], c[A0 + 174], R0;
FMUL     R0, v[1], R18;
FADD32   R5, -R1, c[116];
FMUL     R2, v[5], c[A1 + 175];
FMAD     R3, v[0], R17, R0;
FMUL     R0, v[5], c[A1 + 176];
FMAD     R2, v[4], c[A0 + 175], R2;
FMAD     R3, v[2], R11, R3;
FMAD     R12, v[4], c[A0 + 176], R0;
FMUL     R0, v[5], c[A1 + 177];
FMAD     R2, v[3], R2, R3;
FMUL     R3, v[5], c[A1 + 178];
FMAD     R16, v[4], c[A0 + 177], R0;
FMUL     R0, v[5], c[A1 + 179];
FMAD     R8, v[4], c[A0 + 178], R3;
FADD32   R6, -R2, c[117];
FMAD     R0, v[4], c[A0 + 179], R0;
FMUL     R4, v[1], R16;
FMUL32   R3, R6, R6;
FMAD     R4, v[0], R12, R4;
FMAD     R3, R5, R5, R3;
MOV32    R7, c[125];
FMAD     R4, v[2], R8, R4;
FMAD     R0, v[3], R0, R4;
FADD32   R4, -R0, c[118];
FMAD     R3, R4, R4, R3;
RSQ      R9, |R3|;
FMUL32   R5, R5, R9;
FMUL32   R6, R6, R9;
FMUL32   R4, R4, R9;
FMUL32   R9, R3, R9;
FMUL32   R14, -R6, c[113];
FMAD     R7, R7, R9, c[124];
FMAD     R9, -R5, c[112], R14;
FMAD     R3, R3, c[126], R7;
FMAD     R7, -R4, c[114], R9;
RCP      R3, R3;
FADD32   R7, R7, -c[122];
FMUL32   R7, R7, c[123];
FMAX     R7, R7, c[0];
FMUL     R9, v[14], R18;
LG2      R7, |R7|;
FMAD     R14, v[13], R17, R9;
FMUL     R9, v[9], R16;
FMUL32   R7, R7, c[120];
FMAD     R14, v[15], R11, R14;
FMAD     R9, v[8], R12, R9;
RRO      R7, R7, 1;
FMUL     R18, v[9], R18;
FMAD     R9, v[10], R8, R9;
EX2      R7, R7;
FMAD     R17, v[8], R17, R18;
FMUL     R16, v[14], R16;
FMIN     R7, R7, c[1];
FMAD     R11, v[10], R11, R17;
FMAD     R12, v[13], R12, R16;
FMUL32   R16, R7, c[108];
FMUL32   R17, R14, R9;
FMAD     R12, v[15], R8, R12;
FMUL32   R8, R3, R16;
FMUL     R16, v[9], R15;
FMAD     R17, R11, R12, -R17;
FMUL     R15, v[14], R15;
FMAD     R18, v[8], R10, R16;
FMUL     R16, v[16], R17;
FMAD     R15, v[13], R10, R15;
FMAD     R10, v[10], R13, R18;
FMAD     R13, v[15], R13, R15;
FMUL32   R17, R12, R10;
FMUL32   R15, R13, R11;
FMAD     R17, R9, R13, -R17;
FMUL32   R18, R14, c[7];
FMAD     R15, R10, R14, -R15;
FMUL     R17, v[16], R17;
FMAD     R18, R13, c[7], R18;
FMUL     R15, v[16], R15;
FMUL32   R20, R17, c[7];
FMAD     R19, R12, c[7], R18;
FMUL32   R18, R11, c[7];
FMAD     R20, R16, c[7], R20;
FMAD     R18, R10, c[7], R18;
FMAD     R21, R15, c[7], R20;
FMAD     R20, R9, c[7], R18;
FMUL32   R18, R21, R6;
FMUL32   R22, R21, R21;
FSET     R23, R21, c[0], LT;
FMAD     R18, R19, R5, R18;
FMUL32   R21, R19, R19;
R2A      A2, R23;
FMAD     R18, R20, R4, R18;
FSET     R19, R19, c[0], LT;
FMUL32   R23, R22, c[A2 + 92];
FMAX     R18, R18, c[0];
R2A      A1, R19;
FMUL32   R19, R20, R20;
FSET     R20, R20, c[0], LT;
FMAD     R23, R21, c[A1 + 84], R23;
FMUL32   R24, R7, c[109];
R2A      A0, R20;
FMUL32   R7, R7, c[110];
FMUL32   R20, R3, R24;
FMAD     R23, R19, c[A0 + 100], R23;
FMUL32   R3, R3, R7;
FMUL32   R7, R22, c[A2 + 93];
FMUL32   R22, R22, c[A2 + 94];
FMAD     o[30], R8, R18, R23;
FMAD     R7, R21, c[A1 + 85], R7;
FMAD     R21, R21, c[A1 + 86], R22;
FMUL32   R22, R17, c[12];
FMAD     R7, R19, c[A0 + 101], R7;
FMAD     R19, R19, c[A0 + 102], R21;
FMAD     R21, R15, c[13], R22;
FMAD     o[31], R20, R18, R7;
FMAD     o[32], R3, R18, R19;
FMUL32   R7, R21, R6;
FMUL32   R18, R14, c[12];
FMUL32   R19, R11, c[12];
FMUL32   R22, R21, R21;
FMAD     R18, R12, c[13], R18;
FMAD     R19, R9, c[13], R19;
FSET     R23, R21, c[0], LT;
FMAD     R7, R18, R5, R7;
FMUL32   R21, R18, R18;
R2A      A2, R23;
FMAD     R7, R19, R4, R7;
FSET     R18, R18, c[0], LT;
FMUL32   R23, R22, c[A2 + 92];
FMAX     R7, R7, c[0];
R2A      A1, R18;
FMUL32   R18, R19, R19;
FSET     R19, R19, c[0], LT;
FMAD     R23, R21, c[A1 + 84], R23;
FMUL32   R24, R22, c[A2 + 93];
R2A      A0, R19;
FMUL32   R19, R22, c[A2 + 94];
FMAD     R22, R21, c[A1 + 85], R24;
FMAD     R23, R18, c[A0 + 100], R23;
FMAD     R19, R21, c[A1 + 86], R19;
FMAD     R21, R18, c[A0 + 101], R22;
FMAD     o[8], R8, R7, R23;
FMAD     R18, R18, c[A0 + 102], R19;
FMAD     o[9], R20, R7, R21;
FMUL32   R19, R17, c[6];
FMAD     o[10], R3, R7, R18;
FMUL32   R7, R14, c[6];
FMAD     R19, R16, c[5], R19;
FMUL32   R18, R11, c[6];
FMAD     R7, R13, c[5], R7;
FMAD     R19, R15, c[6], R19;
FMAD     R18, R10, c[5], R18;
FMAD     R7, R12, c[6], R7;
FMUL32   R6, R19, R6;
FMAD     R18, R9, c[6], R18;
FMUL32   R21, R19, R19;
FMAD     R5, R7, R5, R6;
FSET     R19, R19, c[0], LT;
FMUL32   R6, R7, R7;
FMAD     R4, R18, R4, R5;
R2A      A2, R19;
FSET     R5, R7, c[0], LT;
FMAX     R4, R4, c[0];
FMUL32   R7, R21, c[A2 + 92];
R2A      A1, R5;
FMUL32   R5, R18, R18;
FSET     R18, R18, c[0], LT;
FMAD     R7, R6, c[A1 + 84], R7;
FMUL32   R19, R21, c[A2 + 93];
R2A      A0, R18;
FMUL32   R18, R21, c[A2 + 94];
FMAD     R19, R6, c[A1 + 85], R19;
FMAD     R7, R5, c[A0 + 100], R7;
FMAD     R6, R6, c[A1 + 86], R18;
FMAD     R18, R5, c[A0 + 101], R19;
FMAD     o[4], R8, R4, R7;
FMAD     R5, R5, c[A0 + 102], R6;
FMAD     o[5], R20, R4, R18;
MOV32    o[28], R15;
FMAD     o[6], R3, R4, R5;
MOV32    o[25], R17;
MOV32    o[22], R16;
FMUL32   R5, R2, c[41];
MOV32    R3, c[43];
MOV32    R4, c[67];
FMAD     R5, R1, c[40], R5;
FMUL32   R6, R2, c[33];
FMAD     R5, R0, c[42], R5;
FMAD     R6, R1, c[32], R6;
FMAD     R3, R3, c[1], R5;
MOV32    R5, c[35];
FMAD     R6, R0, c[34], R6;
FMAD     R4, -R3, R4, c[64];
FMUL32   R7, R2, c[37];
MOV32    o[33], R4;
FMAD     o[0], R5, c[1], R6;
FMAD     R5, R1, c[36], R7;
MOV32    o[2], R3;
MOV32    o[34], R4;
FMAD     R3, R0, c[38], R5;
MOV32    o[35], R4;
MOV32    o[36], R4;
MOV32    R4, c[39];
FMUL32   R5, R2, c[45];
MOV32    o[27], R12;
FMAD     R5, R1, c[44], R5;
MOV32    o[29], R9;
FMAD     o[1], R4, c[1], R3;
FMAD     R3, R0, c[46], R5;
MOV32    o[24], R14;
MOV32    o[26], R11;
MOV32    o[21], R13;
MOV32    o[23], R10;
FADD32   o[18], -R1, c[8];
FADD32   o[19], -R2, c[9];
FADD32   o[20], -R0, c[10];
MOV32    R2, c[47];
FMUL     R0, v[11], c[360];
FMUL     R1, v[11], c[364];
FMAD     R0, v[12], c[361], R0;
FMAD     R1, v[12], c[365], R1;
FMAD     o[3], R2, c[1], R3;
MOV32    o[16], R0;
MOV32    o[17], R1;
MOV32    o[14], R0;
MOV32    o[12], R0;
MOV32    o[15], R1;
MOV32    o[13], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    o[11], R0;
MOV32    o[7], R1;
END
# 251 instructions, 28 R-regs
# 251 inst, (41 mov, 0 mvi, 0 tex, 4 complex, 206 math)
#    163 64-bit, 88 32-bit, 0 32-bit-const
