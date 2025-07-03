!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    10
.MAX_OBUF    36
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1140-lw40.s -o allprogs-new32//v1140-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[2].C[2]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic c.c
#semantic C[0].C[0]
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
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 c :  : c[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
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
FMUL     R0, v[6], c[4];
FADD     R2, -v[4], c[1];
FMUL     R1, v[6], c[4];
R2A      A0, R0;
FADD     R0, -v[5], R2;
R2A      A1, R1;
FMUL     R1, v[7], c[4];
R2A      A2, R1;
FMUL     R3, v[5], c[A2 + 168];
FMUL     R2, v[5], c[A2 + 169];
FMUL     R1, v[5], c[A2 + 170];
FMAD     R3, v[4], c[A1 + 168], R3;
FMAD     R2, v[4], c[A1 + 169], R2;
FMAD     R1, v[4], c[A1 + 170], R1;
FMAD     R3, c[A0 + 168], R0, R3;
FMAD     R4, c[A0 + 169], R0, R2;
FMAD     R2, c[A0 + 170], R0, R1;
FMUL     R1, v[5], c[A2 + 171];
FMUL     R4, v[1], R4;
FMUL     R5, v[5], c[A2 + 172];
FMAD     R1, v[4], c[A1 + 171], R1;
FMAD     R3, v[0], R3, R4;
FMAD     R4, v[4], c[A1 + 172], R5;
FMAD     R1, c[A0 + 171], R0, R1;
FMAD     R3, v[2], R2, R3;
FMAD     R4, c[A0 + 172], R0, R4;
FMUL     R2, v[5], c[A2 + 173];
FMAD     R1, v[3], R1, R3;
FMUL     R3, v[5], c[A2 + 174];
FMAD     R5, v[4], c[A1 + 173], R2;
FMUL     R2, v[5], c[A2 + 175];
FMAD     R3, v[4], c[A1 + 174], R3;
FMAD     R5, c[A0 + 173], R0, R5;
FMAD     R2, v[4], c[A1 + 175], R2;
FMAD     R3, c[A0 + 174], R0, R3;
FMUL     R5, v[1], R5;
FMAD     R2, c[A0 + 175], R0, R2;
FMUL     R6, v[5], c[A2 + 176];
FMAD     R4, v[0], R4, R5;
FMUL     R5, v[5], c[A2 + 177];
FMAD     R6, v[4], c[A1 + 176], R6;
FMAD     R3, v[2], R3, R4;
FMAD     R4, v[4], c[A1 + 177], R5;
FMAD     R5, c[A0 + 176], R0, R6;
FMAD     R2, v[3], R2, R3;
FMAD     R6, c[A0 + 177], R0, R4;
FMUL     R4, v[5], c[A2 + 178];
FMUL     R3, v[5], c[A2 + 179];
FMUL     R6, v[1], R6;
FMAD     R4, v[4], c[A1 + 178], R4;
FMAD     R3, v[4], c[A1 + 179], R3;
FMAD     R5, v[0], R5, R6;
FMAD     R4, c[A0 + 178], R0, R4;
FMAD     R0, c[A0 + 179], R0, R3;
FMUL32   R3, R2, c[41];
FMAD     R4, v[2], R4, R5;
MOV32    R5, c[43];
FMAD     R3, R1, c[40], R3;
FMAD     R0, v[3], R0, R4;
MOV32    R4, R5;
MOV32    R5, c[67];
FMAD     R3, R0, c[42], R3;
FMUL32   R6, R2, c[33];
FMAD     R3, R4, c[1], R3;
FMAD     R6, R1, c[32], R6;
MOV32    R4, c[35];
FMAD     R5, -R3, R5, c[64];
FMAD     R6, R0, c[34], R6;
MOV32    o[33], R5;
MOV32    o[34], R5;
FMAD     o[0], R4, c[1], R6;
MOV32    o[35], R5;
MOV32    o[36], R5;
MOV32    o[2], R3;
FMUL32   R4, R2, c[37];
MOV32    R3, c[39];
FMUL32   R5, R2, c[45];
FMAD     R4, R1, c[36], R4;
FMAD     R5, R1, c[44], R5;
FMAD     R4, R0, c[38], R4;
FADD32   o[18], -R1, c[8];
FMAD     R1, R0, c[46], R5;
FMAD     o[1], R3, c[1], R4;
FADD32   o[19], -R2, c[9];
FADD32   o[20], -R0, c[10];
MOV32    R2, c[47];
MOV32    o[30], c[0];
MOV32    R0, c[0];
MOV32    R3, R2;
MOV32    R2, c[0];
MOV32    o[31], R0;
FMAD     o[3], R3, c[1], R1;
MOV32    o[32], R2;
MOV32    o[27], c[0];
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    o[24], c[0];
MOV32    o[28], R0;
MOV32    o[29], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    o[21], c[0];
MOV32    o[25], R0;
MOV32    o[26], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
FMUL     R2, v[9], c[360];
MOV32    o[22], R0;
MOV32    o[23], R1;
FMAD     R0, v[10], c[361], R2;
FMUL     R1, v[9], c[364];
MOV32    o[8], c[0];
MOV32    o[16], R0;
FMAD     R1, v[10], c[365], R1;
MOV32    o[14], R0;
MOV32    o[12], R0;
MOV32    o[17], R1;
MOV32    o[15], R1;
MOV32    o[13], R1;
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    R2, c[0];
MOV32    o[9], R0;
MOV32    o[10], R1;
MOV32    o[11], R2;
MOV32    o[4], c[0];
MOV32    R0, c[0];
MOV32    R1, c[0];
MOV32    R2, c[0];
MOV32    o[5], R0;
MOV32    o[6], R1;
MOV32    o[7], R2;
END
# 132 instructions, 8 R-regs
# 132 inst, (55 mov, 0 mvi, 0 tex, 0 complex, 77 math)
#    73 64-bit, 59 32-bit, 0 32-bit-const
