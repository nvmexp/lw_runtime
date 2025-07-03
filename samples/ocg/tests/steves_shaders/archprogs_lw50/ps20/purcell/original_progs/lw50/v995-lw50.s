!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     27
.MAX_IBUF    19
.MAX_OBUF    35
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v995-lw40.s -o allprogs-new32//v995-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[95].C[95]
#semantic C[94].C[94]
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[1].C[1]
#semantic C[30].C[30]
#semantic C[28].C[28]
#semantic C[27].C[27]
#semantic C[31].C[31]
#semantic C[29].C[29]
#semantic C[16].C[16]
#semantic C[0].C[0]
#semantic C[2].C[2]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic c.c
#semantic C[3].C[3]
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
#var float4 C[95] :  : c[95] : -1 : 0
#var float4 C[94] :  : c[94] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[30] :  : c[30] : -1 : 0
#var float4 C[28] :  : c[28] : -1 : 0
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[31] :  : c[31] : -1 : 0
#var float4 C[29] :  : c[29] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX8] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
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
#ibuf 14 = v[COL1].z
#ibuf 15 = v[COL1].w
#ibuf 16 = v[UNUSED0].x
#ibuf 17 = v[UNUSED0].y
#ibuf 18 = v[UNUSED0].z
#ibuf 19 = v[UNUSED0].w
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
#obuf 11 = o[TEX0].x
#obuf 12 = o[TEX0].y
#obuf 13 = o[TEX1].x
#obuf 14 = o[TEX1].y
#obuf 15 = o[TEX2].x
#obuf 16 = o[TEX2].y
#obuf 17 = o[TEX3].x
#obuf 18 = o[TEX3].y
#obuf 19 = o[TEX3].z
#obuf 20 = o[TEX4].x
#obuf 21 = o[TEX4].y
#obuf 22 = o[TEX4].z
#obuf 23 = o[TEX5].x
#obuf 24 = o[TEX5].y
#obuf 25 = o[TEX5].z
#obuf 26 = o[TEX6].x
#obuf 27 = o[TEX6].y
#obuf 28 = o[TEX6].z
#obuf 29 = o[TEX7].x
#obuf 30 = o[TEX7].y
#obuf 31 = o[TEX7].z
#obuf 32 = o[FOGC].x
#obuf 33 = o[FOGC].y
#obuf 34 = o[FOGC].z
#obuf 35 = o[FOGC].w
BB0:
FMUL     R1, v[6], c[12];
FADD     R2, -v[4], c[13];
FMUL     R0, v[6], c[12];
R2A      A0, R1;
FADD     R3, -v[5], R2;
R2A      A1, R0;
FMUL     R0, v[7], c[12];
R2A      A2, R0;
FMUL     R1, v[5], c[A2 + 168];
FMUL     R2, v[5], c[A2 + 169];
FMUL     R0, v[5], c[A2 + 170];
FMAD     R1, v[4], c[A1 + 168], R1;
FMAD     R2, v[4], c[A1 + 169], R2;
FMAD     R0, v[4], c[A1 + 170], R0;
FMAD     R1, c[A0 + 168], R3, R1;
FMAD     R2, c[A0 + 169], R3, R2;
FMAD     R0, c[A0 + 170], R3, R0;
FMUL     R4, v[5], c[A2 + 171];
FMUL     R6, v[1], R2;
FMUL     R5, v[5], c[A2 + 172];
FMAD     R4, v[4], c[A1 + 171], R4;
FMAD     R6, v[0], R1, R6;
FMAD     R5, v[4], c[A1 + 172], R5;
FMAD     R4, c[A0 + 171], R3, R4;
FMAD     R7, v[2], R0, R6;
FMAD     R5, c[A0 + 172], R3, R5;
FMUL     R6, v[5], c[A2 + 173];
FMAD     R12, v[3], R4, R7;
FMUL     R4, v[5], c[A2 + 174];
FMAD     R6, v[4], c[A1 + 173], R6;
FADD32   R17, -R12, c[116];
FMAD     R4, v[4], c[A1 + 174], R4;
FMAD     R6, c[A0 + 173], R3, R6;
FMUL     R8, v[5], c[A2 + 175];
FMAD     R4, c[A0 + 174], R3, R4;
FMUL     R7, v[1], R6;
FMAD     R8, v[4], c[A1 + 175], R8;
FMUL     R9, v[5], c[A2 + 176];
FMAD     R7, v[0], R5, R7;
FMAD     R8, c[A0 + 175], R3, R8;
FMAD     R9, v[4], c[A1 + 176], R9;
FMAD     R10, v[2], R4, R7;
FMUL     R7, v[5], c[A2 + 177];
FMAD     R9, c[A0 + 176], R3, R9;
FMAD     R13, v[3], R8, R10;
FMAD     R8, v[4], c[A1 + 177], R7;
FMUL     R7, v[5], c[A2 + 178];
FADD32   R15, -R13, c[117];
FMAD     R10, c[A0 + 177], R3, R8;
FMAD     R8, v[4], c[A1 + 178], R7;
FMUL     R11, v[5], c[A2 + 179];
FMUL32   R7, R15, R15;
FMAD     R8, c[A0 + 178], R3, R8;
FMAD     R11, v[4], c[A1 + 179], R11;
FMAD     R7, R17, R17, R7;
FMUL     R14, v[1], R10;
FMAD     R3, c[A0 + 179], R3, R11;
FMAD     R11, v[0], R9, R14;
MOV32    R14, c[13];
FMAD     R11, v[2], R8, R11;
FMUL32   R16, R14, c[13];
FMAD     R14, v[3], R3, R11;
FADD32   R3, -R14, c[118];
FMAD     R18, R3, R3, R7;
MOV32    R7, c[13];
RSQ      R11, |R18|;
MOV32    R20, R7;
FMUL32   R7, R18, c[13];
FMUL32   R19, R18, R11;
FMAD     R18, R18, -c[127], R20;
FMUL32   R19, R19, c[125];
FMAX     R20, R18, c[14];
FMUL32   R18, R17, R11;
FMAD     R17, R16, c[124], R19;
FMIN     R19, R20, c[13];
FMUL32   R16, R15, R11;
FMAD     R7, R7, c[126], R17;
FMUL32   R17, R3, R11;
FMUL32   R3, -R16, c[113];
RCP      R7, R7;
FMAD     R3, -R18, c[112], R3;
FMUL32   R15, R7, R19;
FMAD     R3, -R17, c[114], R3;
FMUL32   R19, R15, c[109];
FADD32   R3, R3, -c[122];
FMUL32   R7, R3, c[123];
FMUL     R3, v[10], R6;
FMAX     R7, R7, c[0];
FMAD     R3, v[9], R5, R3;
LG2      R11, |R7|;
FMAD     R7, v[11], R4, R3;
FMUL     R3, v[10], R2;
FMUL32   R11, R11, c[120];
FMUL32   R20, R7, R16;
FMAD     R3, v[9], R1, R3;
RRO      R16, R11, 1;
FMUL     R11, v[10], R10;
FMAD     R3, v[11], R0, R3;
EX2      R16, R16;
FMAD     R11, v[9], R9, R11;
FMAD     R18, R3, R18, R20;
FMIN     R16, R16, c[1];
FMAD     R11, v[11], R8, R11;
FMUL32   R20, R3, R3;
FMUL32   R23, R19, R16;
FMAD     R18, R11, R17, R18;
FSET     R17, R3, c[14], LT;
FMUL32   R21, R7, R7;
FMAX     R18, R18, c[0];
R2A      A1, R17;
FSET     R22, R7, c[14], LT;
FMUL32   R19, R11, R11;
FSET     R17, R11, c[14], LT;
R2A      A2, R22;
R2A      A0, R17;
FMUL32   R17, R21, c[A2 + 93];
FMAD     R22, R20, c[A1 + 85], R17;
FMUL32   R17, R15, c[108];
FMAD     R24, R19, c[A0 + 101], R22;
FMUL32   R17, R17, R16;
FMUL32   R22, R21, c[A2 + 92];
FMAD     R23, R23, R18, R24;
FMAD     R22, R20, c[A1 + 84], R22;
LG2      R23, |R23|;
FMAD     R22, R19, c[A0 + 100], R22;
FMUL32   R23, R23, c[4];
FMAD     R17, R17, R18, R22;
RRO      R22, R23, 1;
LG2      R17, |R17|;
EX2      R22, R22;
FMUL32   R15, R15, c[110];
FMUL32   R21, R21, c[A2 + 94];
FMUL32   R22, R22, c[7];
FMUL32   R15, R15, R16;
FMAD     R16, R20, c[A1 + 86], R21;
FMUL32   R17, R17, c[4];
FMAD     R16, R19, c[A0 + 102], R16;
RRO      R17, R17, 1;
FMAD     R15, R15, R18, R16;
EX2      R16, R17;
LG2      R17, |R15|;
FMUL32   R15, R16, c[7];
FMUL32   R17, R17, c[4];
FMAX     R16, R22, R15;
RRO      R17, R17, 1;
EX2      R17, R17;
FMUL     R6, v[17], R6;
FMUL32   R17, R17, c[7];
FMAD     R5, v[16], R5, R6;
FMAX     R6, R17, c[13];
FMAD     R4, v[18], R4, R5;
FMUL     R5, v[17], R10;
FMAX     R10, R16, R6;
FMUL32   R6, R4, R11;
FMAD     R5, v[16], R9, R5;
RCP      R9, R10;
FMUL     R2, v[17], R2;
FMAD     R5, v[18], R8, R5;
FMUL32   o[4], R15, R9;
FMAD     R1, v[16], R1, R2;
FMUL32   o[5], R22, R9;
FMUL32   o[6], R17, R9;
FMAD     R0, v[18], R0, R1;
FMAD     R1, R7, R5, -R6;
FMUL32   R2, R5, R3;
FMUL32   R6, R0, R7;
FMUL     o[23], v[19], R1;
FMAD     R1, R11, R0, -R2;
FMAD     R2, R3, R4, -R6;
FMUL32   R6, R13, c[41];
FMUL     o[24], v[19], R1;
FMUL     o[25], v[19], R2;
FMAD     R6, R12, c[40], R6;
MOV32    R1, c[43];
FADD32   R2, -R14, c[11];
FMAD     R8, R14, c[42], R6;
FMAX     R2, R2, c[0];
FADD32   R6, -R14, c[10];
FMAD     R1, R1, c[13], R8;
MOV32    R8, c[67];
RCP      R6, R6;
FMUL32   R9, R13, c[33];
FMUL32   R2, R2, R6;
FMAD     R9, R12, c[32], R9;
MOV32    R6, c[35];
FMUL32   R2, R1, R2;
FMAD     R9, R14, c[34], R9;
FMAD     R2, -R2, R8, c[65];
FMUL32   R8, R13, c[37];
FMAD     o[0], R6, c[13], R9;
MOV32    o[32], R2;
FMAD     R6, R12, c[36], R8;
MOV32    o[2], R1;
MOV32    o[33], R2;
FMAD     R1, R14, c[38], R6;
MOV32    o[34], R2;
MOV32    o[35], R2;
MOV32    R2, c[39];
FMUL32   R6, R13, c[45];
MOV32    o[26], R3;
FMAD     R3, R12, c[44], R6;
MOV32    o[27], R7;
FMAD     o[1], R2, c[13], R1;
FMAD     R1, R14, c[46], R3;
MOV32    o[28], R11;
MOV32    o[20], R0;
MOV32    o[21], R4;
MOV32    o[22], R5;
FADD32   o[17], -R12, c[8];
FADD32   o[18], -R13, c[9];
FADD32   o[19], -R14, c[10];
MOV32    R3, c[47];
MOV32    R0, c[14];
MOV32    R2, c[14];
MOV32    o[29], R0;
MOV32    o[30], R2;
FMAD     o[3], R3, c[13], R1;
MOV32    o[31], c[14];
FMUL     R0, v[13], c[377];
FMUL     R1, v[13], c[381];
FMUL     R2, v[13], c[369];
FMAD     R0, v[12], c[376], R0;
FMAD     R1, v[12], c[380], R1;
FMAD     R2, v[12], c[368], R2;
FMAD     R0, v[14], c[378], R0;
FMAD     R1, v[14], c[382], R1;
FMAD     R2, v[14], c[370], R2;
FMAD     o[15], v[15], c[379], R0;
FMAD     o[16], v[15], c[383], R1;
FMAD     o[13], v[15], c[371], R2;
FMUL     R0, v[13], c[373];
FMUL     R1, v[13], c[361];
FMUL     R2, v[13], c[365];
FMAD     R0, v[12], c[372], R0;
FMAD     R1, v[12], c[360], R1;
FMAD     R2, v[12], c[364], R2;
FMAD     R0, v[14], c[374], R0;
FMAD     R1, v[14], c[362], R1;
FMAD     R2, v[14], c[366], R2;
FMAD     o[14], v[15], c[375], R0;
FMAD     o[11], v[15], c[363], R1;
FMAD     o[12], v[15], c[367], R2;
MOV32    R0, c[14];
MOV32    R1, c[14];
MOV32    o[10], c[14];
MOV32    o[8], R0;
MOV32    o[9], R1;
MOV32    R0, c[14];
MOV32    o[7], R0;
END
# 249 instructions, 28 R-regs
# 249 inst, (37 mov, 0 mvi, 0 tex, 12 complex, 200 math)
#    166 64-bit, 83 32-bit, 0 32-bit-const
