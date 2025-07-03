!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    18
.MAX_OBUF    35
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v852-lw40.s -o allprogs-new32//v852-lw50.s
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
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
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
#ibuf 13 = v[COL1].z
#ibuf 14 = v[COL1].w
#ibuf 15 = v[UNUSED0].x
#ibuf 16 = v[UNUSED0].y
#ibuf 17 = v[UNUSED0].z
#ibuf 18 = v[UNUSED0].w
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
FMUL     R0, v[7], c[12];
FMUL     R1, v[6], c[12];
R2A      A0, R0;
R2A      A1, R1;
FMUL     R1, v[5], c[A1 + 168];
FMUL     R2, v[5], c[A1 + 169];
FMUL     R0, v[5], c[A1 + 170];
FMAD     R1, v[4], c[A0 + 168], R1;
FMAD     R2, v[4], c[A0 + 169], R2;
FMAD     R0, v[4], c[A0 + 170], R0;
FMUL     R5, v[5], c[A1 + 172];
FMUL     R3, v[9], R2;
FMUL     R4, v[5], c[A1 + 173];
FMAD     R5, v[4], c[A0 + 172], R5;
FMAD     R3, v[8], R1, R3;
FMAD     R6, v[4], c[A0 + 173], R4;
FMUL     R4, v[5], c[A1 + 174];
FMAD     R3, v[10], R0, R3;
FMUL     R7, v[9], R6;
FMAD     R4, v[4], c[A0 + 174], R4;
FMUL     R9, v[5], c[A1 + 176];
FMAD     R7, v[8], R5, R7;
FMUL     R8, v[5], c[A1 + 177];
FMAD     R9, v[4], c[A0 + 176], R9;
FMAD     R7, v[10], R4, R7;
FMAD     R10, v[4], c[A0 + 177], R8;
FMUL     R8, v[5], c[A1 + 178];
FMUL32   R12, R7, -c[113];
FMUL     R11, v[9], R10;
FMAD     R8, v[4], c[A0 + 178], R8;
FMAD     R12, R3, -c[112], R12;
FMAD     R11, v[8], R9, R11;
FMUL32   R14, R3, R3;
FSET     R13, R3, c[14], LT;
FMAD     R11, v[10], R8, R11;
FMUL32   R15, R7, R7;
R2A      A3, R13;
FMAD     R12, R11, -c[114], R12;
FSET     R16, R7, c[14], LT;
FMUL32   R13, R11, R11;
FMAX     R12, R12, c[0];
R2A      A4, R16;
FSET     R16, R11, c[14], LT;
FMUL32   R17, R15, c[A4 + 93];
R2A      A2, R16;
FMAD     R17, R14, c[A3 + 85], R17;
FMUL32   R16, R15, c[A4 + 92];
FMAD     R17, R13, c[A2 + 101], R17;
FMAD     R16, R14, c[A3 + 84], R16;
FMAD     R17, R12, c[109], R17;
FMAD     R16, R13, c[A2 + 100], R16;
LG2      R17, |R17|;
FMAD     R16, R12, c[108], R16;
FMUL32   R17, R17, c[4];
LG2      R16, |R16|;
RRO      R17, R17, 1;
FMUL32   R15, R15, c[A4 + 94];
FMUL32   R16, R16, c[4];
EX2      R17, R17;
FMAD     R14, R14, c[A3 + 86], R15;
RRO      R15, R16, 1;
FMUL32   R17, R17, c[7];
FMAD     R13, R13, c[A2 + 102], R14;
EX2      R14, R15;
FMAD     R12, R12, c[110], R13;
FMUL32   R16, R14, c[7];
LG2      R12, |R12|;
FMAX     R14, R17, R16;
FMUL32   R12, R12, c[4];
RRO      R12, R12, 1;
EX2      R13, R12;
FMUL     R12, v[16], R6;
FMUL32   R18, R13, c[7];
FMAD     R12, v[15], R5, R12;
FMAX     R15, R18, c[13];
FMAD     R13, v[17], R4, R12;
FMUL     R12, v[16], R10;
FMAX     R15, R14, R15;
FMUL32   R14, R13, R11;
FMAD     R12, v[15], R9, R12;
RCP      R19, R15;
FMUL     R15, v[16], R2;
FMAD     R12, v[17], R8, R12;
FMUL32   o[4], R16, R19;
FMUL32   o[5], R17, R19;
FMUL32   o[6], R18, R19;
FMAD     R14, R7, R12, -R14;
FMAD     R15, v[15], R1, R15;
FMUL32   R16, R12, R3;
FMUL     o[23], v[18], R14;
FMAD     R14, v[17], R0, R15;
FMUL     R15, v[1], R2;
FMUL     R2, v[5], c[A1 + 171];
FMAD     R16, R11, R14, -R16;
FMAD     R1, v[0], R1, R15;
FMAD     R2, v[4], c[A0 + 171], R2;
FMUL     o[24], v[18], R16;
FMAD     R0, v[2], R0, R1;
FMUL     R6, v[1], R6;
FMUL32   R1, R14, R7;
FMAD     R0, v[3], R2, R0;
FMAD     R2, v[0], R5, R6;
FMAD     R1, R3, R13, -R1;
FMUL     R5, v[1], R10;
FMAD     R2, v[2], R4, R2;
FMUL     o[25], v[18], R1;
FMAD     R5, v[0], R9, R5;
FMUL     R1, v[5], c[A1 + 175];
FMUL     R4, v[5], c[A1 + 179];
FMAD     R6, v[2], R8, R5;
FMAD     R1, v[4], c[A0 + 175], R1;
FMAD     R5, v[4], c[A0 + 179], R4;
MOV32    R4, c[43];
FMAD     R1, v[3], R1, R2;
FMAD     R2, v[3], R5, R6;
FMUL32   R5, R1, c[41];
MOV32    R6, c[67];
FMAD     R5, R0, c[40], R5;
FMUL32   R8, R1, c[33];
FMAD     R5, R2, c[42], R5;
FMAD     R9, R0, c[32], R8;
MOV32    R8, c[35];
FMAD     R4, R4, c[13], R5;
FMAD     R5, R2, c[34], R9;
FMAD     R6, -R4, R6, c[64];
FMUL32   R9, R1, c[37];
FMAD     o[0], R8, c[13], R5;
MOV32    o[32], R6;
FMAD     R5, R0, c[36], R9;
MOV32    o[2], R4;
MOV32    o[33], R6;
FMAD     R4, R2, c[38], R5;
MOV32    o[34], R6;
MOV32    o[35], R6;
MOV32    R5, c[39];
FMUL32   R6, R1, c[45];
MOV32    o[26], R3;
MOV32    R3, R5;
FMAD     R5, R0, c[44], R6;
MOV32    o[27], R7;
FMAD     o[1], R3, c[13], R4;
FMAD     R3, R2, c[46], R5;
MOV32    o[28], R11;
MOV32    o[20], R14;
MOV32    o[21], R13;
MOV32    o[22], R12;
FADD32   o[17], -R0, c[8];
FADD32   o[18], -R1, c[9];
FADD32   o[19], -R2, c[10];
MOV32    R2, c[47];
MOV32    R0, c[14];
MOV32    R1, c[14];
MOV32    o[29], R0;
MOV32    o[30], R1;
FMAD     o[3], R2, c[13], R3;
MOV32    o[31], c[14];
FMUL     R0, v[12], c[377];
FMUL     R1, v[12], c[381];
FMUL     R2, v[12], c[369];
FMAD     R0, v[11], c[376], R0;
FMAD     R1, v[11], c[380], R1;
FMAD     R2, v[11], c[368], R2;
FMAD     R0, v[13], c[378], R0;
FMAD     R1, v[13], c[382], R1;
FMAD     R2, v[13], c[370], R2;
FMAD     o[15], v[14], c[379], R0;
FMAD     o[16], v[14], c[383], R1;
FMAD     o[13], v[14], c[371], R2;
FMUL     R0, v[12], c[373];
FMUL     R1, v[12], c[361];
FMUL     R2, v[12], c[365];
FMAD     R0, v[11], c[372], R0;
FMAD     R1, v[11], c[360], R1;
FMAD     R2, v[11], c[364], R2;
FMAD     R0, v[13], c[374], R0;
FMAD     R1, v[13], c[362], R1;
FMAD     R2, v[13], c[366], R2;
FMAD     o[14], v[14], c[375], R0;
FMAD     o[11], v[14], c[363], R1;
FMAD     o[12], v[14], c[367], R2;
MOV32    R0, c[14];
MOV32    R1, c[14];
MOV32    o[10], c[14];
MOV32    o[8], R0;
MOV32    o[9], R1;
MOV32    R0, c[14];
MOV32    o[7], R0;
END
# 187 instructions, 20 R-regs
# 187 inst, (34 mov, 0 mvi, 0 tex, 7 complex, 146 math)
#    132 64-bit, 55 32-bit, 0 32-bit-const
