!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    17
.MAX_OBUF    35
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v962-lw40.s -o allprogs-new32//v962-lw50.s
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
#semantic C[27].C[27]
#semantic C[0].C[0]
#semantic C[31].C[31]
#semantic C[1].C[1]
#semantic c.c
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
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[27] :  : c[27] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[31] :  : c[31] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
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
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
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
#ibuf 10 = v[COL0].x
#ibuf 11 = v[COL0].y
#ibuf 12 = v[COL0].z
#ibuf 13 = v[COL0].w
#ibuf 14 = v[FOG].x
#ibuf 15 = v[FOG].y
#ibuf 16 = v[FOG].z
#ibuf 17 = v[FOG].w
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
FADD32   R8, -R3, c[116];
FADD32   R9, -R4, c[117];
FMAD     R0, v[2], c[178], R0;
FMUL32   R1, R9, R9;
FMAD     R5, v[3], c[179], R0;
FMUL     R0, v[5], c[173];
FMAD     R2, R8, R8, R1;
FADD32   R6, -R5, c[118];
FMAD     R1, v[4], c[172], R0;
FMUL     R0, v[5], c[169];
FMAD     R7, R6, R6, R2;
FMAD     R1, v[6], c[174], R1;
FMAD     R0, v[4], c[168], R0;
RSQ      R10, |R7|;
FMUL     R2, v[5], c[177];
FMAD     R0, v[6], c[170], R0;
FMUL32   R8, R8, R10;
FMUL32   R9, R9, R10;
FMUL32   R6, R6, R10;
FMAD     R2, v[4], c[176], R2;
FMUL32   R9, R1, R9;
FMUL32   R10, R7, R10;
FMAD     R2, v[6], c[178], R2;
FMAD     R8, R0, R8, R9;
MOV32    R9, c[125];
MOV32    R11, c[12];
FMAD     R6, R2, R6, R8;
FMAD     R8, R9, R10, c[124];
MOV32    R9, R11;
FMAX     R6, R6, c[0];
FMAD     R8, R7, c[126], R8;
FMAD     R9, R7, -c[127], R9;
FMUL32   R14, R6, c[109];
RCP      R7, R8;
FMAX     R8, R9, c[13];
FMUL32   R10, R0, R0;
FSET     R9, R0, c[13], LT;
FMIN     R8, R8, c[12];
FMUL32   R11, R1, R1;
R2A      A1, R9;
FMUL32   R8, R7, R8;
FSET     R7, R1, c[13], LT;
FMUL32   R9, R2, R2;
FSET     R12, R2, c[13], LT;
R2A      A2, R7;
FMUL     R7, v[8], c[5];
R2A      A0, R12;
FMUL32   R12, R11, c[A2 + 93];
LG2      R7, |R7|;
MOV32    R13, c[4];
FMAD     R12, R10, c[A1 + 85], R12;
RCP      R13, R13;
FMAD     R15, R9, c[A0 + 101], R12;
FMUL32   R7, R7, R13;
RRO      R7, R7, 1;
EX2      R16, R7;
FMUL32   R7, R6, c[108];
FMUL32   R12, R11, c[A2 + 92];
FADD32   R15, R15, R16;
FMAD     R12, R10, c[A1 + 84], R12;
FMAD     R15, R14, R8, R15;
FMUL     R14, v[7], c[5];
FMAD     R12, R9, c[A0 + 100], R12;
LG2      R15, |R15|;
LG2      R14, |R14|;
FMUL32   R15, R15, c[4];
FMUL32   R14, R14, R13;
RRO      R15, R15, 1;
RRO      R14, R14, 1;
EX2      R15, R15;
EX2      R14, R14;
FMUL32   R15, R15, c[7];
FADD32   R12, R12, R14;
FMUL32   R6, R6, c[110];
FMUL32   R11, R11, c[A2 + 94];
FMAD     R7, R7, R8, R12;
FMAD     R10, R10, c[A1 + 86], R11;
LG2      R7, |R7|;
FMUL     R11, v[9], c[5];
FMAD     R9, R9, c[A0 + 102], R10;
FMUL32   R7, R7, c[4];
LG2      R10, |R11|;
RRO      R7, R7, 1;
FMUL32   R10, R10, R13;
EX2      R7, R7;
RRO      R10, R10, 1;
FMUL32   R7, R7, c[7];
EX2      R11, R10;
FMAX     R10, R15, R7;
FADD32   R9, R9, R11;
FMAD     R6, R6, R8, R9;
LG2      R6, |R6|;
FMUL32   R6, R6, c[4];
RRO      R6, R6, 1;
EX2      R8, R6;
FMUL     R6, v[15], c[173];
FMUL32   R11, R8, c[7];
FMAD     R6, v[14], c[172], R6;
FMAX     R9, R11, c[12];
FMAD     R8, v[16], c[174], R6;
FMUL     R6, v[15], c[177];
FMAX     R10, R10, R9;
FMUL32   R9, R8, R2;
FMAD     R6, v[14], c[176], R6;
RCP      R12, R10;
FMUL     R10, v[15], c[169];
FMAD     R6, v[16], c[178], R6;
FMUL32   o[4], R7, R12;
FMUL32   o[5], R15, R12;
FMUL32   o[6], R11, R12;
FMAD     R7, R1, R6, -R9;
FMAD     R10, v[14], c[168], R10;
FMUL32   R9, R6, R0;
FMUL     o[23], v[17], R7;
FMAD     R7, v[16], c[170], R10;
FMUL32   R11, R4, c[41];
MOV32    R10, c[43];
FMAD     R9, R2, R7, -R9;
FMAD     R11, R3, c[40], R11;
FMUL     o[24], v[17], R9;
FMAD     R11, R5, c[42], R11;
FMUL32   R9, R7, R1;
MOV32    R12, c[67];
FMAD     R10, R10, c[12], R11;
FMAD     R9, R0, R8, -R9;
FMUL32   R11, R4, c[33];
FMAD     R12, -R10, R12, c[64];
FMUL     o[25], v[17], R9;
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
MOV32    o[20], R7;
MOV32    o[21], R8;
MOV32    o[22], R6;
FADD32   o[17], -R3, c[8];
FADD32   o[18], -R4, c[9];
FADD32   o[19], -R5, c[10];
MOV32    R0, c[13];
MOV32    o[30], c[13];
MOV32    R1, c[13];
MOV32    o[29], R0;
FMUL     R0, v[11], c[377];
MOV32    o[31], R1;
FMUL     R1, v[11], c[381];
FMAD     R0, v[10], c[376], R0;
FMUL     R2, v[11], c[369];
FMAD     R1, v[10], c[380], R1;
FMAD     R0, v[12], c[378], R0;
FMAD     R2, v[10], c[368], R2;
FMAD     R1, v[12], c[382], R1;
FMAD     o[15], v[13], c[379], R0;
FMAD     R0, v[12], c[370], R2;
FMAD     o[16], v[13], c[383], R1;
FMUL     R1, v[11], c[373];
FMAD     o[13], v[13], c[371], R0;
FMUL     R0, v[11], c[361];
FMAD     R1, v[10], c[372], R1;
FMUL     R2, v[11], c[365];
FMAD     R0, v[10], c[360], R0;
FMAD     R1, v[12], c[374], R1;
FMAD     R2, v[10], c[364], R2;
FMAD     R0, v[12], c[362], R0;
FMAD     o[14], v[13], c[375], R1;
FMAD     R1, v[12], c[366], R2;
FMAD     o[11], v[13], c[363], R0;
MOV32    R0, c[13];
FMAD     o[12], v[13], c[367], R1;
MOV32    o[9], c[13];
MOV32    o[8], R0;
MOV32    R0, c[13];
MOV32    R1, c[13];
MOV32    o[10], R0;
MOV32    o[7], R1;
END
# 205 instructions, 20 R-regs
# 205 inst, (38 mov, 0 mvi, 0 tex, 16 complex, 151 math)
#    126 64-bit, 79 32-bit, 0 32-bit-const
