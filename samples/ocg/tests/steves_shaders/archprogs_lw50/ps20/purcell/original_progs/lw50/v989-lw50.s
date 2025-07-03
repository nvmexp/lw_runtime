!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     19
.MAX_IBUF    18
.MAX_OBUF    32
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v989-lw40.s -o allprogs-new32//v989-lw50.s
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
#ibuf 13 = v[COL1].z
#ibuf 14 = v[COL1].w
#ibuf 15 = v[FOG].x
#ibuf 16 = v[FOG].y
#ibuf 17 = v[FOG].z
#ibuf 18 = v[FOG].w
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
FMUL     R0, v[7], c[4];
FMUL     R1, v[6], c[4];
R2A      A0, R0;
R2A      A1, R1;
FMUL     R2, v[5], c[A1 + 176];
FMUL     R1, v[5], c[A1 + 177];
FMUL     R0, v[5], c[A1 + 178];
FMAD     R9, v[4], c[A0 + 176], R2;
FMAD     R10, v[4], c[A0 + 177], R1;
FMAD     R8, v[4], c[A0 + 178], R0;
FMUL     R2, v[5], c[A1 + 172];
FMUL     R1, v[9], R10;
FMUL     R0, v[5], c[A1 + 173];
FMAD     R5, v[4], c[A0 + 172], R2;
FMAD     R1, v[8], R9, R1;
FMAD     R6, v[4], c[A0 + 173], R0;
FMUL     R0, v[5], c[A1 + 174];
FMAD     R11, v[10], R8, R1;
FMUL     R1, v[16], R6;
FMAD     R4, v[4], c[A0 + 174], R0;
FMUL     R0, v[9], R6;
FMAD     R2, v[15], R5, R1;
FMUL     R1, v[16], R10;
FMAD     R0, v[8], R5, R0;
FMAD     R13, v[17], R4, R2;
FMAD     R1, v[15], R9, R1;
FMAD     R7, v[10], R4, R0;
FMUL32   R0, R11, R13;
FMAD     R12, v[17], R8, R1;
FMUL     R1, v[5], c[A1 + 168];
FMUL     R2, v[5], c[A1 + 169];
FMAD     R0, R7, R12, -R0;
FMAD     R1, v[4], c[A0 + 168], R1;
FMAD     R2, v[4], c[A0 + 169], R2;
FMUL     R14, v[18], R0;
FMUL     R0, v[5], c[A1 + 170];
FMUL     R3, v[9], R2;
FMUL     R15, v[16], R2;
FMAD     R0, v[4], c[A0 + 170], R0;
FMAD     R3, v[8], R1, R3;
FMAD     R15, v[15], R1, R15;
FMAD     R3, v[10], R0, R3;
FMAD     R15, v[17], R0, R15;
FMUL32   R16, R3, R12;
FMUL32   R17, R7, R15;
FMAD     R16, R11, R15, -R16;
FMAD     R17, R3, R13, -R17;
FMUL     R16, v[18], R16;
FMUL     R17, v[18], R17;
FMUL32   R18, R16, c[37];
FMUL32   R19, R16, c[33];
MOV32    o[24], R16;
FMAD     R16, R14, c[36], R18;
FMAD     R18, R14, c[32], R19;
MOV32    o[23], R14;
FMAD     R14, R17, c[38], R16;
FMAD     R16, R17, c[34], R18;
MOV32    o[25], R17;
FMUL32   o[11], R14, -c[372];
FMUL32   o[7], R16, c[372];
FMUL     R2, v[1], R2;
FMUL     R14, v[1], R6;
FMUL     R6, v[5], c[A1 + 171];
FMAD     R1, v[0], R1, R2;
FMAD     R2, v[0], R5, R14;
FMAD     R5, v[4], c[A0 + 171], R6;
FMAD     R0, v[2], R0, R1;
FMAD     R1, v[2], R4, R2;
FMUL     R2, v[1], R10;
FMAD     R0, v[3], R5, R0;
FMUL     R4, v[5], c[A1 + 175];
FMAD     R5, v[0], R9, R2;
FMUL     R2, v[5], c[A1 + 179];
FMAD     R4, v[4], c[A0 + 175], R4;
FMAD     R5, v[2], R8, R5;
FMAD     R2, v[4], c[A0 + 179], R2;
FMAD     R1, v[3], R4, R1;
MOV32    R4, c[39];
FMAD     R2, v[3], R2, R5;
FMUL32   R5, R1, c[37];
FMUL32   R8, R1, c[45];
FMAD     R6, R0, c[36], R5;
MOV32    R5, c[47];
FMAD     R8, R0, c[44], R8;
FMAD     R6, R2, c[38], R6;
FMAD     R8, R2, c[46], R8;
FMAD     R6, R4, c[5], R6;
FMUL32   R4, R1, c[33];
FMAD     R8, R5, c[5], R8;
MOV32    R5, c[35];
FMAD     R9, R0, c[32], R4;
FMAD     R4, R6, c[379], R8;
FMAD     R9, R2, c[34], R9;
FMUL32   o[12], R4, c[3];
FMUL32   R4, R7, c[33];
FMAD     R9, R5, c[5], R9;
FMUL32   R5, R7, c[37];
FMAD     R4, R3, c[32], R4;
FADD32   R10, R9, R8;
FMAD     R5, R3, c[36], R5;
FMAD     o[14], R11, c[34], R4;
FMUL32   o[8], R10, c[3];
FMAD     o[15], R11, c[38], R5;
FMUL32   R4, R7, c[41];
FMUL32   R5, R13, c[37];
MOV32    o[13], R8;
FMAD     R4, R3, c[40], R4;
FMAD     R5, R15, c[36], R5;
FMUL32   R10, R13, c[33];
FMAD     o[16], R11, c[42], R4;
FMAD     R4, R12, c[38], R5;
FMAD     R5, R15, c[32], R10;
MOV32    o[9], R8;
FMUL32   o[10], R4, -c[372];
FMAD     R4, R12, c[34], R5;
FMUL32   R10, R1, c[41];
MOV32    R5, c[43];
FMUL32   o[6], R4, c[372];
FMAD     R4, R0, c[40], R10;
MOV32    R10, c[67];
FMAD     R4, R2, c[42], R4;
MOV32    o[0], R9;
MOV32    o[1], R6;
FMAD     R4, R5, c[5], R4;
MOV32    o[3], R8;
MOV32    o[26], R3;
FMAD     R3, -R4, R10, c[64];
MOV32    o[2], R4;
MOV32    o[27], R7;
MOV32    o[28], R11;
MOV32    o[20], R15;
MOV32    o[21], R13;
MOV32    o[22], R12;
FADD32   o[17], -R0, c[8];
FADD32   o[18], -R1, c[9];
FADD32   o[19], -R2, c[10];
MOV32    o[29], R3;
MOV32    o[30], R3;
MOV32    o[31], R3;
MOV32    o[32], R3;
FMUL     R0, v[12], c[365];
FMUL     R1, v[12], c[369];
FMAD     R0, v[11], c[364], R0;
FMAD     R1, v[11], c[368], R1;
FMAD     R0, v[13], c[366], R0;
FMAD     R1, v[13], c[370], R1;
FMAD     o[4], v[14], c[367], R0;
FMAD     o[5], v[14], c[371], R1;
END
# 148 instructions, 20 R-regs
# 148 inst, (26 mov, 0 mvi, 0 tex, 0 complex, 122 math)
#    100 64-bit, 48 32-bit, 0 32-bit-const
