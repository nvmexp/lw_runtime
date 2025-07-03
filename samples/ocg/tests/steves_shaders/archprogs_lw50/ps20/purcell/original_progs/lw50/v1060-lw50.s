!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     27
.MAX_IBUF    16
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1060-lw40.s -o allprogs-new32//v1060-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[32].C[32]
#semantic C[33].C[33]
#semantic C[27].C[27]
#semantic C[30].C[30]
#semantic C[28].C[28]
#semantic C[31].C[31]
#semantic C[2].C[2]
#semantic C[29].C[29]
#semantic C[16].C[16]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
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
#var float4 C[32] :  : c[32] : -1 : 0
#var float4 C[33] :  : c[33] : -1 : 0
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[30] :  : c[30] : -1 : 0
#var float4 C[28] :  : c[28] : -1 : 0
#var float4 C[31] :  : c[31] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[29] :  : c[29] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
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
FMUL     R1, v[5], c[A1 + 168];
FMUL     R2, v[5], c[A1 + 169];
FMUL     R0, v[5], c[A1 + 170];
FMAD     R1, v[4], c[A0 + 168], R1;
FMAD     R2, v[4], c[A0 + 169], R2;
FMAD     R0, v[4], c[A0 + 170], R0;
FMUL     R4, v[5], c[A1 + 171];
FMUL     R3, v[1], R2;
FMUL     R5, v[5], c[A1 + 172];
FMAD     R4, v[4], c[A0 + 171], R4;
FMAD     R3, v[0], R1, R3;
FMAD     R5, v[4], c[A0 + 172], R5;
FMUL     R6, v[5], c[A1 + 173];
FMAD     R7, v[2], R0, R3;
FMUL     R3, v[5], c[A1 + 174];
FMAD     R6, v[4], c[A0 + 173], R6;
FMAD     R12, v[3], R4, R7;
FMAD     R4, v[4], c[A0 + 174], R3;
FMUL     R3, v[1], R6;
FADD32   R17, -R12, c[116];
FMUL     R7, v[5], c[A1 + 175];
FMAD     R8, v[0], R5, R3;
FMUL     R3, v[5], c[A1 + 176];
FMAD     R7, v[4], c[A0 + 175], R7;
FMAD     R8, v[2], R4, R8;
FMAD     R9, v[4], c[A0 + 176], R3;
FMUL     R3, v[5], c[A1 + 177];
FMAD     R13, v[3], R7, R8;
FMUL     R7, v[5], c[A1 + 178];
FMAD     R10, v[4], c[A0 + 177], R3;
FMUL     R3, v[5], c[A1 + 179];
FMAD     R8, v[4], c[A0 + 178], R7;
FADD32   R15, -R13, c[117];
FMAD     R3, v[4], c[A0 + 179], R3;
FMUL     R11, v[1], R10;
FMUL32   R7, R15, R15;
FMAD     R11, v[0], R9, R11;
FMAD     R7, R17, R17, R7;
MOV32    R14, c[1];
FMAD     R11, v[2], R8, R11;
FMUL32   R16, R14, c[1];
FMAD     R14, v[3], R3, R11;
FADD32   R3, -R14, c[118];
FMAD     R18, R3, R3, R7;
MOV32    R7, c[1];
RSQ      R11, |R18|;
MOV32    R20, R7;
FMUL32   R7, R18, c[1];
FMUL32   R19, R18, R11;
FMAD     R20, R18, -c[127], R20;
FMUL32   R18, R19, c[125];
FMAX     R20, R20, c[0];
FMUL32   R19, R17, R11;
FMAD     R16, R16, c[124], R18;
FMIN     R18, R20, c[1];
FMUL32   R17, R15, R11;
FMAD     R7, R7, c[126], R16;
FMUL32   R15, R3, R11;
FMUL32   R3, -R17, c[113];
RCP      R7, R7;
FMAD     R3, -R19, c[112], R3;
FMUL32   R16, R7, R18;
FMAD     R3, -R15, c[114], R3;
FMUL32   R18, R16, c[108];
FADD32   R3, R3, -c[122];
FMUL32   R7, R3, c[123];
FMUL     R3, v[9], R6;
FMAX     R7, R7, c[0];
FMAD     R3, v[8], R5, R3;
LG2      R11, |R7|;
FMAD     R7, v[10], R4, R3;
FMUL     R3, v[9], R2;
FMUL32   R11, R11, c[120];
FMUL32   R20, R7, R17;
FMAD     R3, v[8], R1, R3;
RRO      R17, R11, 1;
FMUL     R11, v[9], R10;
FMAD     R3, v[10], R0, R3;
EX2      R17, R17;
FMAD     R11, v[8], R9, R11;
FMAD     R19, R3, R19, R20;
FMIN     R17, R17, c[1];
FMAD     R11, v[10], R8, R11;
FMUL32   R21, R3, R3;
FMUL32   R18, R18, R17;
FMAD     R19, R11, R15, R19;
FSET     R15, R3, c[0], LT;
FMUL32   R22, R7, R7;
FMAX     R19, R19, c[0];
R2A      A1, R15;
FSET     R15, R7, c[0], LT;
FMUL32   R20, R11, R11;
FSET     R23, R11, c[0], LT;
R2A      A2, R15;
FMUL32   R15, R7, -c[133];
R2A      A0, R23;
FMUL32   R23, R22, c[A2 + 92];
FMAD     R15, R3, -c[132], R15;
FMUL32   R24, R16, c[109];
FMAD     R23, R21, c[A1 + 84], R23;
FMAD     R15, R11, -c[134], R15;
FMUL32   R24, R24, R17;
FMAD     R23, R20, c[A0 + 100], R23;
FMAX     R15, R15, c[0];
FMUL32   R16, R16, c[110];
FMAD     R18, R18, R19, R23;
FMUL32   R23, R22, c[A2 + 93];
FMUL32   R16, R16, R17;
FMUL32   R17, R22, c[A2 + 94];
FMAD     o[4], R15, c[128], R18;
FMAD     R18, R21, c[A1 + 85], R23;
FMAD     R17, R21, c[A1 + 86], R17;
FMUL     R6, v[14], R6;
FMAD     R18, R20, c[A0 + 101], R18;
FMAD     R17, R20, c[A0 + 102], R17;
FMAD     R5, v[13], R5, R6;
FMAD     R6, R24, R19, R18;
FMAD     R16, R16, R19, R17;
FMAD     R4, v[15], R4, R5;
FMAD     o[5], R15, c[129], R6;
FMAD     o[6], R15, c[130], R16;
FMUL     R5, v[14], R10;
FMUL     R6, v[14], R2;
FMUL32   R2, R4, R11;
FMAD     R5, v[13], R9, R5;
FMAD     R1, v[13], R1, R6;
FMAD     R5, v[15], R8, R5;
FMAD     R0, v[15], R0, R1;
FMAD     R1, R7, R5, -R2;
FMUL32   R2, R5, R3;
FMUL32   R6, R0, R7;
FMUL     o[24], v[16], R1;
FMAD     R1, R11, R0, -R2;
FMAD     R2, R3, R4, -R6;
FMUL32   R6, R13, c[41];
FMUL     o[25], v[16], R1;
FMUL     o[26], v[16], R2;
FMAD     R2, R12, c[40], R6;
MOV32    R1, c[43];
MOV32    R6, c[67];
FMAD     R2, R14, c[42], R2;
FMUL32   R8, R13, c[33];
FMAD     R1, R1, c[1], R2;
FMAD     R8, R12, c[32], R8;
MOV32    R2, c[35];
FMAD     R6, -R1, R6, c[64];
FMAD     R8, R14, c[34], R8;
MOV32    o[33], R6;
MOV32    o[34], R6;
FMAD     o[0], R2, c[1], R8;
MOV32    o[35], R6;
MOV32    o[36], R6;
MOV32    o[2], R1;
FMUL32   R2, R13, c[37];
MOV32    R1, c[39];
FMUL32   R6, R13, c[45];
FMAD     R2, R12, c[36], R2;
FMAD     R6, R12, c[44], R6;
FMAD     R2, R14, c[38], R2;
MOV32    o[27], R3;
FMAD     R3, R14, c[46], R6;
FMAD     o[1], R1, c[1], R2;
MOV32    o[28], R7;
MOV32    o[29], R11;
MOV32    o[21], R0;
MOV32    o[22], R4;
MOV32    o[23], R5;
FADD32   o[18], -R12, c[8];
FADD32   o[19], -R13, c[9];
FADD32   o[20], -R14, c[10];
MOV32    R1, c[47];
MOV32    o[30], c[0];
MOV32    R0, c[0];
MOV32    R2, R1;
MOV32    R1, c[0];
MOV32    o[31], R0;
FMAD     o[3], R2, c[1], R3;
MOV32    o[32], R1;
FMUL     R0, v[11], c[360];
FMUL     R1, v[11], c[364];
FMAD     R0, v[12], c[361], R0;
FMAD     R1, v[12], c[365], R1;
MOV32    o[8], c[0];
MOV32    o[16], R0;
MOV32    o[17], R1;
MOV32    o[14], R0;
MOV32    o[12], R0;
MOV32    o[15], R1;
MOV32    o[13], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    R2, c[0];
MOV32    o[9], R0;
MOV32    o[10], R1;
MOV32    o[11], R2;
MOV32    R0, c[0];
MOV32    o[7], R0;
END
# 201 instructions, 28 R-regs
# 201 inst, (45 mov, 0 mvi, 0 tex, 4 complex, 152 math)
#    121 64-bit, 80 32-bit, 0 32-bit-const
