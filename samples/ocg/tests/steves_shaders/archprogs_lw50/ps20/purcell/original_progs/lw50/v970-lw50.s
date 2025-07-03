!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     15
.MAX_IBUF    15
.MAX_OBUF    35
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v970-lw40.s -o allprogs-new32//v970-lw50.s
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
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
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
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#var float4 C[3] :  : c[3] : -1 : 0
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
#ibuf 10 = v[COL0].z
#ibuf 11 = v[COL0].w
#ibuf 12 = v[FOG].x
#ibuf 13 = v[FOG].y
#ibuf 14 = v[FOG].z
#ibuf 15 = v[FOG].w
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
FMUL     R0, v[4], c[12];
R2A      A0, R0;
FMUL     R0, v[6], c[A0 + 169];
FMUL     R1, v[6], c[A0 + 173];
FMUL     R2, v[6], c[A0 + 177];
FMAD     R0, v[5], c[A0 + 168], R0;
FMAD     R1, v[5], c[A0 + 172], R1;
FMAD     R2, v[5], c[A0 + 176], R2;
FMAD     R0, v[7], c[A0 + 170], R0;
FMAD     R1, v[7], c[A0 + 174], R1;
FMAD     R2, v[7], c[A0 + 178], R2;
FMUL32   R5, R0, R0;
FMUL32   R3, R1, -c[113];
FSET     R4, R0, c[14], LT;
FMUL32   R6, R1, R1;
FMAD     R3, R0, -c[112], R3;
R2A      A2, R4;
FSET     R7, R1, c[14], LT;
FMAD     R3, R2, -c[114], R3;
FMUL32   R4, R2, R2;
R2A      A3, R7;
FMAX     R3, R3, c[0];
FSET     R7, R2, c[14], LT;
FMUL32   R8, R6, c[A3 + 93];
R2A      A1, R7;
FMAD     R8, R5, c[A2 + 85], R8;
FMUL32   R7, R6, c[A3 + 92];
FMAD     R8, R4, c[A1 + 101], R8;
FMAD     R7, R5, c[A2 + 84], R7;
FMAD     R8, R3, c[109], R8;
FMAD     R7, R4, c[A1 + 100], R7;
LG2      R8, |R8|;
FMAD     R7, R3, c[108], R7;
FMUL32   R8, R8, c[4];
LG2      R7, |R7|;
RRO      R8, R8, 1;
FMUL32   R6, R6, c[A3 + 94];
FMUL32   R7, R7, c[4];
EX2      R8, R8;
FMAD     R5, R5, c[A2 + 86], R6;
RRO      R6, R7, 1;
FMUL32   R8, R8, c[7];
FMAD     R4, R4, c[A1 + 102], R5;
EX2      R5, R6;
FMAD     R3, R3, c[110], R4;
FMUL32   R7, R5, c[7];
LG2      R3, |R3|;
FMAX     R5, R8, R7;
FMUL32   R3, R3, c[4];
RRO      R3, R3, 1;
EX2      R4, R3;
FMUL     R3, v[13], c[A0 + 173];
FMUL32   R9, R4, c[7];
FMAD     R3, v[12], c[A0 + 172], R3;
FMAX     R6, R9, c[13];
FMAD     R4, v[14], c[A0 + 174], R3;
FMUL     R3, v[13], c[A0 + 177];
FMAX     R6, R5, R6;
FMUL32   R5, R4, R2;
FMAD     R3, v[12], c[A0 + 176], R3;
RCP      R10, R6;
FMUL     R6, v[13], c[A0 + 169];
FMAD     R3, v[14], c[A0 + 178], R3;
FMUL32   o[4], R7, R10;
FMUL32   o[5], R8, R10;
FMUL32   o[6], R9, R10;
FMAD     R5, R1, R3, -R5;
FMAD     R7, v[12], c[A0 + 168], R6;
FMUL32   R6, R3, R0;
FMUL     o[23], v[15], R5;
FMAD     R5, v[14], c[A0 + 170], R7;
FMUL     R7, v[1], c[A0 + 169];
FMAD     R6, R2, R5, -R6;
FMAD     R8, v[0], c[A0 + 168], R7;
FMUL32   R7, R5, R1;
FMUL     o[24], v[15], R6;
FMAD     R6, v[2], c[A0 + 170], R8;
FMAD     R7, R0, R4, -R7;
FMUL     R8, v[1], c[A0 + 173];
FMAD     R6, v[3], c[A0 + 171], R6;
FMUL     o[25], v[15], R7;
FMAD     R7, v[0], c[A0 + 172], R8;
FMUL     R8, v[1], c[A0 + 177];
FMAD     R7, v[2], c[A0 + 174], R7;
FMAD     R8, v[0], c[A0 + 176], R8;
MOV32    R9, c[43];
FMAD     R7, v[3], c[A0 + 175], R7;
FMAD     R8, v[2], c[A0 + 178], R8;
FMUL32   R10, R7, c[41];
FMAD     R8, v[3], c[A0 + 179], R8;
MOV32    R11, c[67];
FMAD     R10, R6, c[40], R10;
FMUL32   R12, R7, c[33];
FMAD     R10, R8, c[42], R10;
FMAD     R13, R6, c[32], R12;
MOV32    R12, c[35];
FMAD     R9, R9, c[13], R10;
FMAD     R10, R8, c[34], R13;
FMAD     R11, -R9, R11, c[64];
FMUL32   R13, R7, c[37];
FMAD     o[0], R12, c[13], R10;
MOV32    o[32], R11;
FMAD     R10, R6, c[36], R13;
MOV32    o[2], R9;
MOV32    o[33], R11;
FMAD     R9, R8, c[38], R10;
MOV32    o[34], R11;
MOV32    o[35], R11;
MOV32    R10, c[39];
FMUL32   R12, R7, c[45];
MOV32    R11, c[47];
FMAD     R12, R6, c[44], R12;
FMAD     o[1], R10, c[13], R9;
FMAD     R9, R8, c[46], R12;
MOV32    o[26], R0;
MOV32    o[27], R1;
FMAD     o[3], R11, c[13], R9;
MOV32    o[28], R2;
MOV32    o[20], R5;
MOV32    o[21], R4;
MOV32    o[22], R3;
FADD32   o[17], -R6, c[8];
FADD32   o[18], -R7, c[9];
FADD32   o[19], -R8, c[10];
MOV32    R0, c[14];
MOV32    R1, c[14];
MOV32    o[31], c[14];
MOV32    o[29], R0;
MOV32    o[30], R1;
FMUL     R0, v[9], c[377];
FMUL     R1, v[9], c[381];
FMUL     R2, v[9], c[369];
FMAD     R0, v[8], c[376], R0;
FMAD     R1, v[8], c[380], R1;
FMAD     R2, v[8], c[368], R2;
FMAD     R0, v[10], c[378], R0;
FMAD     R1, v[10], c[382], R1;
FMAD     R2, v[10], c[370], R2;
FMAD     o[15], v[11], c[379], R0;
FMAD     o[16], v[11], c[383], R1;
FMAD     o[13], v[11], c[371], R2;
FMUL     R0, v[9], c[373];
FMUL     R1, v[9], c[361];
FMUL     R2, v[9], c[365];
FMAD     R0, v[8], c[372], R0;
FMAD     R1, v[8], c[360], R1;
FMAD     R2, v[8], c[364], R2;
FMAD     R0, v[10], c[374], R0;
FMAD     R1, v[10], c[362], R1;
FMAD     R2, v[10], c[366], R2;
FMAD     o[14], v[11], c[375], R0;
FMAD     o[11], v[11], c[363], R1;
FMAD     o[12], v[11], c[367], R2;
MOV32    R0, c[14];
MOV32    R1, c[14];
MOV32    o[10], c[14];
MOV32    o[8], R0;
MOV32    o[9], R1;
MOV32    R0, c[14];
MOV32    o[7], R0;
END
# 160 instructions, 16 R-regs
# 160 inst, (32 mov, 0 mvi, 0 tex, 7 complex, 121 math)
#    106 64-bit, 54 32-bit, 0 32-bit-const
