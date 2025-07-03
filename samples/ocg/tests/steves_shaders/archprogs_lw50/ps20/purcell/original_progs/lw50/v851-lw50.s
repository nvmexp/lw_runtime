!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     23
.MAX_IBUF    19
.MAX_OBUF    35
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v851-lw40.s -o allprogs-new32//v851-lw50.s
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
#semantic C[27].C[27]
#semantic C[0].C[0]
#semantic C[28].C[28]
#semantic C[16].C[16]
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
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[28] :  : c[28] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
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
FMUL     R0, v[6], c[12];
FADD     R2, -v[4], c[13];
FMUL     R1, v[6], c[12];
R2A      A0, R0;
FADD     R0, -v[5], R2;
R2A      A1, R1;
FMUL     R1, v[7], c[12];
R2A      A2, R1;
FMUL     R2, v[5], c[A2 + 168];
FMUL     R3, v[5], c[A2 + 169];
FMUL     R1, v[5], c[A2 + 170];
FMAD     R2, v[4], c[A1 + 168], R2;
FMAD     R3, v[4], c[A1 + 169], R3;
FMAD     R1, v[4], c[A1 + 170], R1;
FMAD     R2, c[A0 + 168], R0, R2;
FMAD     R3, c[A0 + 169], R0, R3;
FMAD     R1, c[A0 + 170], R0, R1;
FMUL     R6, v[5], c[A2 + 172];
FMUL     R4, v[10], R3;
FMUL     R5, v[5], c[A2 + 173];
FMAD     R6, v[4], c[A1 + 172], R6;
FMAD     R4, v[9], R2, R4;
FMAD     R5, v[4], c[A1 + 173], R5;
FMAD     R6, c[A0 + 172], R0, R6;
FMAD     R4, v[11], R1, R4;
FMAD     R7, c[A0 + 173], R0, R5;
FMUL     R5, v[5], c[A2 + 174];
FMUL     R9, v[5], c[A2 + 176];
FMUL     R8, v[10], R7;
FMAD     R5, v[4], c[A1 + 174], R5;
FMAD     R9, v[4], c[A1 + 176], R9;
FMAD     R8, v[9], R6, R8;
FMAD     R5, c[A0 + 174], R0, R5;
FMAD     R10, c[A0 + 176], R0, R9;
FMUL     R11, v[5], c[A2 + 177];
FMAD     R8, v[11], R5, R8;
FMUL     R9, v[5], c[A2 + 178];
FMAD     R11, v[4], c[A1 + 177], R11;
FMUL32   R12, R8, -c[113];
FMAD     R9, v[4], c[A1 + 178], R9;
FMAD     R11, c[A0 + 177], R0, R11;
FMAD     R13, R4, -c[112], R12;
FMAD     R9, c[A0 + 178], R0, R9;
FMUL     R12, v[10], R11;
FMUL32   R16, R4, R4;
FSET     R14, R4, c[14], LT;
FMAD     R12, v[9], R10, R12;
FMUL32   R17, R8, R8;
R2A      A4, R14;
FMAD     R12, v[11], R9, R12;
FSET     R14, R8, c[14], LT;
FMAD     R13, R12, -c[114], R13;
R2A      A5, R14;
FMUL32   R14, R12, R12;
FMAX     R13, R13, c[0];
FMUL32   R18, R17, c[A5 + 93];
FSET     R15, R12, c[14], LT;
FMAD     R18, R16, c[A4 + 85], R18;
R2A      A3, R15;
FMUL32   R15, R17, c[A5 + 92];
FMAD     R18, R14, c[A3 + 101], R18;
FMAD     R15, R16, c[A4 + 84], R15;
FMAD     R18, R13, c[109], R18;
FMAD     R15, R14, c[A3 + 100], R15;
LG2      R18, |R18|;
FMAD     R15, R13, c[108], R15;
FMUL32   R18, R18, c[4];
LG2      R15, |R15|;
RRO      R18, R18, 1;
FMUL32   R17, R17, c[A5 + 94];
FMUL32   R15, R15, c[4];
EX2      R18, R18;
FMAD     R16, R16, c[A4 + 86], R17;
RRO      R15, R15, 1;
FMUL32   R18, R18, c[7];
FMAD     R14, R14, c[A3 + 102], R16;
EX2      R15, R15;
FMAD     R13, R13, c[110], R14;
FMUL32   R17, R15, c[7];
LG2      R13, |R13|;
FMAX     R15, R18, R17;
FMUL32   R13, R13, c[4];
RRO      R13, R13, 1;
EX2      R14, R13;
FMUL     R13, v[17], R7;
FMUL32   R19, R14, c[7];
FMAD     R13, v[16], R6, R13;
FMAX     R16, R19, c[13];
FMAD     R14, v[18], R5, R13;
FMUL     R13, v[17], R11;
FMAX     R16, R15, R16;
FMUL32   R15, R14, R12;
FMAD     R13, v[16], R10, R13;
RCP      R20, R16;
FMUL     R16, v[17], R3;
FMAD     R13, v[18], R9, R13;
FMUL32   o[4], R17, R20;
FMUL32   o[5], R18, R20;
FMUL32   o[6], R19, R20;
FMAD     R15, R8, R13, -R15;
FMAD     R16, v[16], R2, R16;
FMUL32   R17, R13, R4;
FMUL     o[23], v[19], R15;
FMAD     R15, v[18], R1, R16;
FMUL     R16, v[1], R3;
FMUL     R3, v[5], c[A2 + 171];
FMAD     R17, R12, R15, -R17;
FMAD     R2, v[0], R2, R16;
FMAD     R3, v[4], c[A1 + 171], R3;
FMUL     o[24], v[19], R17;
FMAD     R1, v[2], R1, R2;
FMAD     R2, c[A0 + 171], R0, R3;
FMUL     R7, v[1], R7;
FMUL32   R3, R15, R8;
FMAD     R1, v[3], R2, R1;
FMAD     R2, v[0], R6, R7;
FMAD     R3, R4, R14, -R3;
FMUL     R6, v[5], c[A2 + 175];
FMAD     R2, v[2], R5, R2;
FMUL     o[25], v[19], R3;
FMAD     R3, v[4], c[A1 + 175], R6;
FMUL     R6, v[1], R11;
FMUL     R5, v[5], c[A2 + 179];
FMAD     R3, c[A0 + 175], R0, R3;
FMAD     R6, v[0], R10, R6;
FMAD     R5, v[4], c[A1 + 179], R5;
FMAD     R2, v[3], R3, R2;
FMAD     R3, v[2], R9, R6;
FMAD     R0, c[A0 + 179], R0, R5;
FMUL32   R5, R2, c[41];
MOV32    R6, c[43];
FMAD     R0, v[3], R0, R3;
FMAD     R5, R1, c[40], R5;
MOV32    R3, R6;
MOV32    R6, c[67];
FMAD     R5, R0, c[42], R5;
FMUL32   R7, R2, c[33];
FMAD     R3, R3, c[13], R5;
FMAD     R7, R1, c[32], R7;
MOV32    R5, c[35];
FMAD     R6, -R3, R6, c[64];
FMAD     R7, R0, c[34], R7;
MOV32    o[32], R6;
MOV32    o[33], R6;
FMAD     o[0], R5, c[13], R7;
MOV32    o[34], R6;
MOV32    o[35], R6;
MOV32    o[2], R3;
FMUL32   R5, R2, c[37];
MOV32    R3, c[39];
FMUL32   R6, R2, c[45];
FMAD     R5, R1, c[36], R5;
FMAD     R6, R1, c[44], R6;
FMAD     R5, R0, c[38], R5;
MOV32    o[26], R4;
FMAD     R4, R0, c[46], R6;
FMAD     o[1], R3, c[13], R5;
MOV32    o[27], R8;
MOV32    o[28], R12;
MOV32    o[20], R15;
MOV32    o[21], R14;
MOV32    o[22], R13;
FADD32   o[17], -R1, c[8];
FADD32   o[18], -R2, c[9];
FADD32   o[19], -R0, c[10];
MOV32    R2, c[47];
MOV32    R0, c[14];
MOV32    R1, c[14];
MOV32    o[29], R0;
MOV32    o[30], R1;
FMAD     o[3], R2, c[13], R4;
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
# 203 instructions, 24 R-regs
# 203 inst, (35 mov, 0 mvi, 0 tex, 7 complex, 161 math)
#    148 64-bit, 55 32-bit, 0 32-bit-const
