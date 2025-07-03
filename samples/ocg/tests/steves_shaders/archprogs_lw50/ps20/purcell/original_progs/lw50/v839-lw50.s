!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    14
.MAX_OBUF    35
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v839-lw40.s -o allprogs-new32//v839-lw50.s
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
#semantic C[32].C[32]
#semantic C[36].C[36]
#semantic c.c
#semantic C[0].C[0]
#semantic C[30].C[30]
#semantic C[28].C[28]
#semantic C[27].C[27]
#semantic C[31].C[31]
#semantic C[34].C[34]
#semantic C[29].C[29]
#semantic C[2].C[2]
#semantic C[16].C[16]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[3].C[3]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
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
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[32] :  : c[32] : -1 : 0
#var float4 C[36] :  : c[36] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[30] :  : c[30] : -1 : 0
#var float4 C[28] :  : c[28] : -1 : 0
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[31] :  : c[31] : -1 : 0
#var float4 C[34] :  : c[34] : -1 : 0
#var float4 C[29] :  : c[29] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
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
#ibuf 10 = v[NOR].w
#ibuf 11 = v[COL1].x
#ibuf 12 = v[COL1].y
#ibuf 13 = v[COL1].z
#ibuf 14 = v[COL1].w
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
FMUL     R1, v[1], c[169];
FMUL     R0, v[1], c[173];
FMAD     R1, v[0], c[168], R1;
FMAD     R0, v[0], c[172], R0;
FMAD     R2, v[2], c[170], R1;
FMAD     R1, v[2], c[174], R0;
FMUL     R0, v[1], c[177];
FMAD     R3, v[3], c[171], R2;
FMAD     R4, v[3], c[175], R1;
FMAD     R0, v[0], c[176], R0;
FADD32   R7, -R3, c[116];
FADD32   R9, -R4, c[117];
FMAD     R1, v[2], c[178], R0;
FMUL32   R0, R9, R9;
FMAD     R5, v[3], c[179], R1;
MOV32    R2, c[12];
FMAD     R0, R7, R7, R0;
FADD32   R1, -R5, c[118];
FMUL32   R8, R2, c[12];
FMAD     R0, R1, R1, R0;
MOV32    R2, c[12];
RSQ      R6, |R0|;
MOV32    R12, R2;
FMUL32   R2, R0, c[12];
FMUL32   R10, R0, R6;
FMAD     R0, R0, -c[127], R12;
FMUL32   R10, R10, c[125];
FMAX     R0, R0, c[13];
FMUL32   R7, R7, R6;
FMAD     R8, R8, c[124], R10;
FMIN     R0, R0, c[12];
FMUL32   R9, R9, R6;
FMAD     R2, R2, c[126], R8;
FMUL32   R6, R1, R6;
FMUL32   R1, -R9, c[113];
RCP      R2, R2;
FMAD     R1, -R7, c[112], R1;
FMUL32   R13, R2, R0;
FMAD     R0, -R6, c[114], R1;
FMUL32   R8, R13, c[109];
FADD32   R0, R0, -c[122];
FMUL32   R1, R0, c[123];
FMUL     R0, v[5], c[173];
FMAX     R1, R1, c[0];
FMAD     R0, v[4], c[172], R0;
LG2      R2, |R1|;
FMAD     R1, v[6], c[174], R0;
FMUL     R0, v[5], c[169];
FMUL32   R2, R2, c[120];
FMUL32   R10, R1, R9;
FMAD     R0, v[4], c[168], R0;
RRO      R9, R2, 1;
FMUL     R2, v[5], c[177];
FMAD     R0, v[6], c[170], R0;
EX2      R9, R9;
FMAD     R2, v[4], c[176], R2;
FMAD     R7, R0, R7, R10;
FMIN     R14, R9, c[1];
FMAD     R2, v[6], c[178], R2;
FMUL32   R17, R0, R0;
FMUL32   R10, R8, R14;
FMAD     R7, R2, R6, R7;
FSET     R6, R0, c[13], LT;
FMUL32   R18, R1, R1;
FMAX     R15, R7, c[0];
R2A      A1, R6;
FSET     R7, R1, c[13], LT;
FMUL32   R16, R2, R2;
FSET     R6, R2, c[13], LT;
R2A      A2, R7;
FADD32   R8, -R3, c[136];
R2A      A0, R6;
FMUL32   R6, R18, c[A2 + 93];
FADD32   R9, -R4, c[137];
FADD32   R7, -R5, c[138];
FMAD     R11, R17, c[A1 + 85], R6;
FMUL32   R6, R9, R9;
FMAD     R11, R16, c[A0 + 101], R11;
FMAD     R6, R8, R8, R6;
FMAD     R19, R10, R15, R11;
FMAD     R6, R7, R7, R6;
RSQ      R10, |R6|;
FMUL32   R8, R8, R10;
FMUL32   R9, R9, R10;
FMUL32   R7, R7, R10;
FMUL32   R11, R6, R10;
FMUL32   R9, R1, R9;
MOV32    R10, c[145];
FMAD     R12, R6, -c[147], R12;
FMAD     R8, R0, R8, R9;
FMAD     R9, R10, R11, c[144];
FMAX     R10, R12, c[13];
FMAD     R7, R2, R7, R8;
FMAD     R6, R6, c[146], R9;
FMIN     R8, R10, c[12];
FMAX     R7, R7, c[0];
RCP      R9, R6;
FMUL32   R6, R7, c[129];
FMUL32   R8, R9, R8;
FMUL32   R9, R13, c[108];
FMUL32   R10, R18, c[A2 + 92];
FMAD     R6, R6, R8, R19;
FMUL32   R9, R9, R14;
FMAD     R10, R17, c[A1 + 84], R10;
LG2      R11, |R6|;
FMUL32   R6, R7, c[128];
FMAD     R10, R16, c[A0 + 100], R10;
FMUL32   R11, R11, c[4];
FMAD     R9, R9, R15, R10;
RRO      R10, R11, 1;
FMAD     R6, R6, R8, R9;
EX2      R9, R10;
FMUL32   R10, R13, c[110];
LG2      R6, |R6|;
FMUL32   R11, R18, c[A2 + 94];
FMUL32   R10, R10, R14;
FMUL32   R9, R9, c[7];
FMAD     R11, R17, c[A1 + 86], R11;
FMUL32   R6, R6, c[4];
FMUL32   R7, R7, c[130];
FMAD     R11, R16, c[A0 + 102], R11;
RRO      R6, R6, 1;
FMAD     R10, R10, R15, R11;
EX2      R6, R6;
FMAD     R7, R7, R8, R10;
FMUL32   R11, R6, c[7];
LG2      R6, |R7|;
FMAX     R8, R9, R11;
FMUL32   R6, R6, c[4];
RRO      R6, R6, 1;
EX2      R7, R6;
FMUL     R6, v[12], c[173];
FMUL32   R12, R7, c[7];
FMAD     R6, v[11], c[172], R6;
FMAX     R10, R12, c[12];
FMAD     R7, v[13], c[174], R6;
FMUL     R6, v[12], c[177];
FMAX     R10, R8, R10;
FMUL32   R8, R7, R2;
FMAD     R6, v[11], c[176], R6;
RCP      R13, R10;
FMUL     R10, v[12], c[169];
FMAD     R6, v[13], c[178], R6;
FMUL32   o[4], R11, R13;
FMUL32   o[5], R9, R13;
FMUL32   o[6], R12, R13;
FMAD     R8, R1, R6, -R8;
FMAD     R10, v[11], c[168], R10;
FMUL32   R9, R6, R0;
FMUL     o[23], v[14], R8;
FMAD     R8, v[13], c[170], R10;
FMUL32   R11, R4, c[41];
MOV32    R10, c[43];
FMAD     R9, R2, R8, -R9;
FMAD     R11, R3, c[40], R11;
FMUL     o[24], v[14], R9;
FMAD     R11, R5, c[42], R11;
FMUL32   R9, R8, R1;
MOV32    R12, c[67];
FMAD     R10, R10, c[12], R11;
FMAD     R9, R0, R7, -R9;
FMUL32   R11, R4, c[33];
FMAD     R12, -R10, R12, c[64];
FMUL     o[25], v[14], R9;
FMAD     R9, R3, c[32], R11;
MOV32    o[32], R12;
MOV32    o[33], R12;
FMAD     R9, R5, c[34], R9;
MOV32    o[34], R12;
MOV32    o[35], R12;
MOV32    R11, c[35];
FMUL32   R12, R4, c[37];
MOV32    o[2], R10;
MOV32    R10, R11;
FMAD     R12, R3, c[36], R12;
MOV32    R11, c[39];
FMAD     o[0], R10, c[12], R9;
FMAD     R9, R5, c[38], R12;
MOV32    R10, R11;
FMUL32   R12, R4, c[45];
MOV32    R11, c[47];
FMAD     o[1], R10, c[12], R9;
FMAD     R9, R3, c[44], R12;
MOV32    R10, R11;
MOV32    o[26], R0;
FMAD     R0, R5, c[46], R9;
MOV32    o[27], R1;
MOV32    o[28], R2;
FMAD     o[3], R10, c[12], R0;
MOV32    o[20], R8;
MOV32    o[21], R7;
MOV32    o[22], R6;
FADD32   o[17], -R3, c[8];
FADD32   o[18], -R4, c[9];
FADD32   o[19], -R5, c[10];
MOV32    R0, c[13];
MOV32    o[30], c[13];
MOV32    R1, c[13];
MOV32    o[29], R0;
FMUL     R0, v[8], c[377];
MOV32    o[31], R1;
FMUL     R1, v[8], c[381];
FMAD     R0, v[7], c[376], R0;
FMUL     R2, v[8], c[369];
FMAD     R1, v[7], c[380], R1;
FMAD     R0, v[9], c[378], R0;
FMAD     R2, v[7], c[368], R2;
FMAD     R1, v[9], c[382], R1;
FMAD     o[15], v[10], c[379], R0;
FMAD     R0, v[9], c[370], R2;
FMAD     o[16], v[10], c[383], R1;
FMUL     R1, v[8], c[373];
FMAD     o[13], v[10], c[371], R0;
FMUL     R0, v[8], c[361];
FMAD     R1, v[7], c[372], R1;
FMUL     R2, v[8], c[365];
FMAD     R0, v[7], c[360], R0;
FMAD     R1, v[9], c[374], R1;
FMAD     R2, v[7], c[364], R2;
FMAD     R0, v[9], c[362], R0;
FMAD     o[14], v[10], c[375], R1;
FMAD     R1, v[9], c[366], R2;
FMAD     o[11], v[10], c[363], R0;
MOV32    R0, c[13];
FMAD     o[12], v[10], c[367], R1;
MOV32    o[9], c[13];
MOV32    o[8], R0;
MOV32    R0, c[13];
MOV32    R1, c[13];
MOV32    o[10], R0;
MOV32    o[7], R1;
END
# 231 instructions, 20 R-regs
# 231 inst, (38 mov, 0 mvi, 0 tex, 13 complex, 180 math)
#    135 64-bit, 96 32-bit, 0 32-bit-const
