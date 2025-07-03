!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    11
.MAX_OBUF    35
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v841-lw40.s -o allprogs-new32//v841-lw50.s
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
#semantic C[16].C[16]
#semantic C[2].C[2]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#semantic C[0].C[0]
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
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[95] :  : c[95] : -1 : 0
#var float4 C[94] :  : c[94] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[WGT].x
#ibuf 5 = v[WGT].y
#ibuf 6 = v[WGT].z
#ibuf 7 = v[WGT].w
#ibuf 8 = v[NOR].x
#ibuf 9 = v[NOR].y
#ibuf 10 = v[NOR].z
#ibuf 11 = v[NOR].w
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
FMUL     R0, v[1], c[169];
FMUL     R1, v[1], c[173];
FMAD     R0, v[0], c[168], R0;
FMAD     R1, v[0], c[172], R1;
FMUL     R2, v[1], c[177];
FMAD     R0, v[2], c[170], R0;
FMAD     R1, v[2], c[174], R1;
FMAD     R2, v[0], c[176], R2;
FMAD     R0, v[3], c[171], R0;
FMAD     R1, v[3], c[175], R1;
FMAD     R2, v[2], c[178], R2;
MOV32    R3, c[43];
FMUL32   R4, R1, c[41];
FMAD     R2, v[3], c[179], R2;
FMAD     R5, R0, c[40], R4;
MOV32    R4, c[67];
FMUL32   R6, R1, c[33];
FMAD     R5, R2, c[42], R5;
FMAD     R6, R0, c[32], R6;
FMAD     R3, R3, c[0], R5;
MOV32    R5, c[35];
FMAD     R6, R2, c[34], R6;
FMAD     R4, -R3, R4, c[64];
FMUL32   R7, R1, c[37];
MOV32    o[32], R4;
FMAD     o[0], R5, c[0], R6;
FMAD     R5, R0, c[36], R7;
MOV32    o[2], R3;
MOV32    o[33], R4;
FMAD     R3, R2, c[38], R5;
MOV32    o[34], R4;
MOV32    o[35], R4;
MOV32    R4, c[39];
FMUL32   R6, R1, c[45];
MOV32    R5, c[47];
FMAD     R6, R0, c[44], R6;
FMAD     o[1], R4, c[0], R3;
FMAD     R4, R2, c[46], R6;
MOV32    R3, c[1];
MOV32    o[30], c[1];
FMAD     o[3], R5, c[0], R4;
MOV32    o[29], R3;
MOV32    R3, c[1];
MOV32    R4, c[1];
MOV32    o[27], c[1];
MOV32    o[31], R3;
MOV32    o[26], R4;
MOV32    R3, c[1];
MOV32    R4, c[1];
MOV32    o[24], c[1];
MOV32    o[28], R3;
MOV32    o[23], R4;
FADD32   o[17], -R0, c[8];
FADD32   o[18], -R1, c[9];
FADD32   o[19], -R2, c[10];
MOV32    R0, c[1];
MOV32    R1, c[1];
MOV32    o[21], c[1];
MOV32    o[25], R0;
MOV32    o[20], R1;
MOV32    R0, c[1];
FMUL     R1, v[9], c[377];
FMUL     R2, v[9], c[381];
MOV32    o[22], R0;
FMAD     R0, v[8], c[376], R1;
FMAD     R1, v[8], c[380], R2;
FMUL     R2, v[9], c[369];
FMAD     R0, v[10], c[378], R0;
FMAD     R1, v[10], c[382], R1;
FMAD     R2, v[8], c[368], R2;
FMAD     o[15], v[11], c[379], R0;
FMAD     o[16], v[11], c[383], R1;
FMAD     R0, v[10], c[370], R2;
FMUL     R1, v[9], c[373];
FMUL     R2, v[9], c[361];
FMAD     o[13], v[11], c[371], R0;
FMAD     R0, v[8], c[372], R1;
FMAD     R1, v[8], c[360], R2;
FMUL     R2, v[9], c[365];
FMAD     R0, v[10], c[374], R0;
FMAD     R1, v[10], c[362], R1;
FMAD     R2, v[8], c[364], R2;
FMAD     o[14], v[11], c[375], R0;
FMAD     o[11], v[11], c[363], R1;
FMAD     R0, v[10], c[366], R2;
MOV32    R1, c[1];
MOV32    o[9], c[1];
FMAD     o[12], v[11], c[367], R0;
MOV32    o[8], R1;
MOV32    R0, c[1];
MOV      o[4], v[4];
MOV      o[5], v[5];
MOV32    o[10], R0;
MOV      o[6], v[6];
MOV      o[7], v[7];
END
# 95 instructions, 8 R-regs
# 95 inst, (39 mov, 0 mvi, 0 tex, 0 complex, 56 math)
#    53 64-bit, 42 32-bit, 0 32-bit-const
