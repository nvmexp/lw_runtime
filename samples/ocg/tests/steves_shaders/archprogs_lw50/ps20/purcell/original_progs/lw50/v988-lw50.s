!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     23
.MAX_IBUF    19
.MAX_OBUF    32
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v988-lw40.s -o allprogs-new32//v988-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[0].C[0]
#semantic C[94].C[94]
#semantic C[11].C[11]
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[2].C[2]
#semantic C[16].C[16]
#semantic C[10].C[10]
#semantic C[9].C[9]
#semantic C[8].C[8]
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
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[94] :  : c[94] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 v[FOGC] : $vin.F : F[0] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
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
#ibuf 16 = v[FOG].x
#ibuf 17 = v[FOG].y
#ibuf 18 = v[FOG].z
#ibuf 19 = v[FOG].w
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX1].x
#obuf 7 = o[TEX1].y
#obuf 8 = o[TEX1].z
#obuf 9 = o[TEX1].w
#obuf 10 = o[TEX2].x
#obuf 11 = o[TEX2].y
#obuf 12 = o[TEX2].z
#obuf 13 = o[TEX2].w
#obuf 14 = o[TEX3].x
#obuf 15 = o[TEX3].y
#obuf 16 = o[TEX3].z
#obuf 17 = o[TEX4].x
#obuf 18 = o[TEX4].y
#obuf 19 = o[TEX4].z
#obuf 20 = o[TEX5].x
#obuf 21 = o[TEX5].y
#obuf 22 = o[TEX5].z
#obuf 23 = o[TEX6].x
#obuf 24 = o[TEX6].y
#obuf 25 = o[TEX6].z
#obuf 26 = o[TEX7].x
#obuf 27 = o[TEX7].y
#obuf 28 = o[TEX7].z
#obuf 29 = o[FOGC].x
#obuf 30 = o[FOGC].y
#obuf 31 = o[FOGC].z
#obuf 32 = o[FOGC].w
BB0:
FMUL     R0, v[6], c[4];
FADD     R2, -v[4], c[5];
FMUL     R1, v[6], c[4];
R2A      A0, R0;
FADD     R0, -v[5], R2;
R2A      A1, R1;
FMUL     R1, v[7], c[4];
R2A      A2, R1;
FMUL     R3, v[5], c[A2 + 176];
FMUL     R2, v[5], c[A2 + 177];
FMUL     R1, v[5], c[A2 + 178];
FMAD     R3, v[4], c[A1 + 176], R3;
FMAD     R2, v[4], c[A1 + 177], R2;
FMAD     R1, v[4], c[A1 + 178], R1;
FMAD     R10, c[A0 + 176], R0, R3;
FMAD     R11, c[A0 + 177], R0, R2;
FMAD     R9, c[A0 + 178], R0, R1;
FMUL     R3, v[5], c[A2 + 172];
FMUL     R2, v[10], R11;
FMUL     R1, v[5], c[A2 + 173];
FMAD     R3, v[4], c[A1 + 172], R3;
FMAD     R2, v[9], R10, R2;
FMAD     R1, v[4], c[A1 + 173], R1;
FMAD     R6, c[A0 + 172], R0, R3;
FMAD     R12, v[11], R9, R2;
FMAD     R7, c[A0 + 173], R0, R1;
FMUL     R1, v[5], c[A2 + 174];
FMUL     R3, v[17], R7;
FMAD     R2, v[4], c[A1 + 174], R1;
FMUL     R1, v[10], R7;
FMAD     R3, v[16], R6, R3;
FMAD     R5, c[A0 + 174], R0, R2;
FMAD     R2, v[9], R6, R1;
FMUL     R1, v[17], R11;
FMAD     R14, v[18], R5, R3;
FMAD     R8, v[11], R5, R2;
FMAD     R1, v[16], R10, R1;
FMUL32   R3, R12, R14;
FMUL     R2, v[5], c[A2 + 168];
FMAD     R13, v[18], R9, R1;
FMUL     R1, v[5], c[A2 + 169];
FMAD     R2, v[4], c[A1 + 168], R2;
FMAD     R3, R8, R13, -R3;
FMAD     R1, v[4], c[A1 + 169], R1;
FMAD     R2, c[A0 + 168], R0, R2;
FMUL     R15, v[19], R3;
FMAD     R3, c[A0 + 169], R0, R1;
FMUL     R1, v[5], c[A2 + 170];
FMUL     R4, v[10], R3;
FMAD     R1, v[4], c[A1 + 170], R1;
FMUL     R16, v[17], R3;
FMAD     R4, v[9], R2, R4;
FMAD     R1, c[A0 + 170], R0, R1;
FMAD     R16, v[16], R2, R16;
FMAD     R4, v[11], R1, R4;
FMAD     R16, v[18], R1, R16;
FMUL32   R17, R4, R13;
FMUL32   R18, R8, R16;
FMAD     R17, R12, R16, -R17;
FMAD     R18, R4, R14, -R18;
FMUL     R17, v[19], R17;
FMUL     R18, v[19], R18;
FMUL32   R19, R17, c[37];
FMUL32   R20, R17, c[33];
MOV32    o[24], R17;
FMAD     R17, R15, c[36], R19;
FMAD     R19, R15, c[32], R20;
MOV32    o[23], R15;
FMAD     R15, R18, c[38], R17;
FMAD     R17, R18, c[34], R19;
MOV32    o[25], R18;
FMUL32   o[11], R15, -c[372];
FMUL32   o[7], R17, c[372];
FMUL     R3, v[1], R3;
FMUL     R15, v[1], R7;
FMUL     R7, v[5], c[A2 + 171];
FMAD     R2, v[0], R2, R3;
FMAD     R3, v[0], R6, R15;
FMAD     R6, v[4], c[A1 + 171], R7;
FMAD     R1, v[2], R1, R2;
FMAD     R2, v[2], R5, R3;
FMAD     R3, c[A0 + 171], R0, R6;
FMUL     R6, v[1], R11;
FMUL     R5, v[5], c[A2 + 175];
FMAD     R1, v[3], R3, R1;
FMAD     R3, v[0], R10, R6;
FMAD     R5, v[4], c[A1 + 175], R5;
FMUL     R6, v[5], c[A2 + 179];
FMAD     R3, v[2], R9, R3;
FMAD     R5, c[A0 + 175], R0, R5;
FMAD     R6, v[4], c[A1 + 179], R6;
MOV32    R7, c[39];
FMAD     R2, v[3], R5, R2;
FMAD     R0, c[A0 + 179], R0, R6;
MOV32    R6, R7;
FMUL32   R5, R2, c[37];
FMAD     R0, v[3], R0, R3;
FMUL32   R7, R2, c[45];
FMAD     R3, R1, c[36], R5;
MOV32    R5, c[47];
FMAD     R7, R1, c[44], R7;
FMAD     R3, R0, c[38], R3;
FMAD     R7, R0, c[46], R7;
FMAD     R6, R6, c[5], R3;
FMUL32   R3, R2, c[33];
FMAD     R7, R5, c[5], R7;
MOV32    R5, c[35];
FMAD     R9, R1, c[32], R3;
FMAD     R3, R6, c[379], R7;
FMAD     R9, R0, c[34], R9;
FMUL32   o[12], R3, c[3];
FMUL32   R3, R8, c[33];
FMAD     R9, R5, c[5], R9;
FMUL32   R5, R8, c[37];
FMAD     R3, R4, c[32], R3;
FADD32   R10, R9, R7;
FMAD     R5, R4, c[36], R5;
FMAD     o[14], R12, c[34], R3;
FMUL32   o[8], R10, c[3];
FMAD     o[15], R12, c[38], R5;
FMUL32   R3, R8, c[41];
FMUL32   R5, R14, c[37];
MOV32    o[13], R7;
FMAD     R3, R4, c[40], R3;
FMAD     R5, R16, c[36], R5;
FMUL32   R10, R14, c[33];
FMAD     o[16], R12, c[42], R3;
FMAD     R3, R13, c[38], R5;
FMAD     R5, R16, c[32], R10;
MOV32    o[9], R7;
FMUL32   o[10], R3, -c[372];
FMAD     R3, R13, c[34], R5;
FMUL32   R10, R2, c[41];
MOV32    R5, c[43];
FMUL32   o[6], R3, c[372];
FMAD     R3, R1, c[40], R10;
MOV32    R10, c[67];
FMAD     R3, R0, c[42], R3;
MOV32    o[0], R9;
MOV32    o[1], R6;
FMAD     R3, R5, c[5], R3;
MOV32    o[3], R7;
MOV32    o[26], R4;
FMAD     R4, -R3, R10, c[64];
MOV32    o[2], R3;
MOV32    o[27], R8;
MOV32    o[28], R12;
MOV32    o[20], R16;
MOV32    o[21], R14;
MOV32    o[22], R13;
FADD32   o[17], -R1, c[8];
FADD32   o[18], -R2, c[9];
FADD32   o[19], -R0, c[10];
MOV32    o[29], R4;
MOV32    o[30], R4;
MOV32    o[31], R4;
MOV32    o[32], R4;
FMUL     R0, v[13], c[365];
FMUL     R1, v[13], c[369];
FMAD     R0, v[12], c[364], R0;
FMAD     R1, v[12], c[368], R1;
FMAD     R0, v[14], c[366], R0;
FMAD     R1, v[14], c[370], R1;
FMAD     o[4], v[15], c[367], R0;
FMAD     o[5], v[15], c[371], R1;
END
# 165 instructions, 24 R-regs
# 165 inst, (28 mov, 0 mvi, 0 tex, 0 complex, 137 math)
#    116 64-bit, 49 32-bit, 0 32-bit-const
