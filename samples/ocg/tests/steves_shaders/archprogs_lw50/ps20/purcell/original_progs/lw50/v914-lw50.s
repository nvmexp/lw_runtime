!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     11
.MAX_IBUF    16
.MAX_OBUF    24
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v914-lw40.s -o allprogs-new32//v914-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[0].C[0]
#semantic C[2].C[2]
#semantic C[16].C[16]
#semantic C[7].C[7]
#semantic C[6].C[6]
#semantic C[5].C[5]
#semantic C[4].C[4]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#var float4 o[TEX4] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[BCOL0] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[TEX9] : $vin.F : F[0] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX4] : $vin.F : F[0] : -1 : 0
#var float4 v[TEX3] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
#ibuf 0 = v[OPOS].x
#ibuf 1 = v[OPOS].y
#ibuf 2 = v[OPOS].z
#ibuf 3 = v[OPOS].w
#ibuf 4 = v[COL0].x
#ibuf 5 = v[COL0].y
#ibuf 6 = v[COL0].z
#ibuf 7 = v[UNUSED1].x
#ibuf 8 = v[UNUSED1].y
#ibuf 9 = v[UNUSED1].z
#ibuf 10 = v[UNUSED1].w
#ibuf 11 = v[TEX3].x
#ibuf 12 = v[TEX3].y
#ibuf 13 = v[TEX3].z
#ibuf 14 = v[TEX4].x
#ibuf 15 = v[TEX4].y
#ibuf 16 = v[TEX4].z
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[BCOL0].x
#obuf 5 = o[BCOL0].y
#obuf 6 = o[BCOL0].z
#obuf 7 = o[TEX0].x
#obuf 8 = o[TEX0].y
#obuf 9 = o[TEX1].x
#obuf 10 = o[TEX1].y
#obuf 11 = o[TEX1].z
#obuf 12 = o[TEX2].x
#obuf 13 = o[TEX2].y
#obuf 14 = o[TEX2].z
#obuf 15 = o[TEX3].x
#obuf 16 = o[TEX3].y
#obuf 17 = o[TEX3].z
#obuf 18 = o[TEX4].x
#obuf 19 = o[TEX4].y
#obuf 20 = o[TEX4].z
#obuf 21 = o[FOGC].x
#obuf 22 = o[FOGC].y
#obuf 23 = o[FOGC].z
#obuf 24 = o[FOGC].w
BB0:
FMUL     R0, v[1], c[169];
FMUL     R1, v[1], c[173];
FMAD     R0, v[0], c[168], R0;
FMAD     R1, v[0], c[172], R1;
FMAD     R0, v[2], c[170], R0;
FMAD     R1, v[2], c[174], R1;
FMUL     R2, v[1], c[177];
FMAD     R0, v[3], c[171], R0;
FMAD     R1, v[3], c[175], R1;
FMAD     R2, v[0], c[176], R2;
FADD32   R0, -R0, c[8];
FADD32   R1, -R1, c[9];
FMAD     R2, v[2], c[178], R2;
FMUL     R4, v[12], R1;
FMAD     R2, v[3], c[179], R2;
FMUL     R3, v[15], R1;
FMAD     R4, v[11], R0, R4;
FADD32   R2, -R2, c[10];
FMAD     R5, v[14], R0, R3;
FMUL     R3, v[5], R1;
FMAD     R4, v[13], R2, R4;
FMAD     R6, v[16], R2, R5;
FMAD     R5, v[4], R0, R3;
MOV32    R8, c[3];
FMUL32   R3, R6, R6;
FMAD     R5, v[6], R2, R5;
MOV32    R9, c[3];
FMAD     R3, R4, R4, R3;
MOV32    R10, c[3];
FMAD     R7, R5, R5, R3;
FMUL     R3, v[1], c[25];
RSQ      R7, |R7|;
FMAD     R3, v[0], c[24], R3;
FMUL32   R4, R4, R7;
FMUL32   R6, R6, R7;
FMUL32   R5, R5, R7;
FMAD     o[4], R4, R8, c[3];
FMAD     o[5], R6, R9, c[3];
FMAD     o[6], R5, R10, c[3];
FMAD     R3, v[2], c[26], R3;
MOV32    o[18], R0;
MOV32    o[19], R1;
FMAD     R0, v[3], c[27], R3;
MOV32    o[20], R2;
MOV32    R3, c[67];
FMUL     R1, v[12], c[177];
FMUL     R2, v[15], c[177];
FMAD     R3, -R0, R3, c[64];
FMAD     R1, v[11], c[176], R1;
FMAD     R2, v[14], c[176], R2;
MOV32    o[21], R3;
FMAD     o[15], v[13], c[178], R1;
FMAD     o[16], v[16], c[178], R2;
MOV32    o[22], R3;
MOV32    o[23], R3;
MOV32    o[24], R3;
FMUL     R1, v[5], c[177];
FMUL     R2, v[12], c[173];
FMUL     R3, v[15], c[173];
FMAD     R1, v[4], c[176], R1;
FMAD     R2, v[11], c[172], R2;
FMAD     R3, v[14], c[172], R3;
FMAD     o[17], v[6], c[178], R1;
FMAD     o[12], v[13], c[174], R2;
FMAD     o[13], v[16], c[174], R3;
FMUL     R1, v[5], c[173];
FMUL     R2, v[12], c[169];
FMUL     R3, v[15], c[169];
FMAD     R1, v[4], c[172], R1;
FMAD     R2, v[11], c[168], R2;
FMAD     R3, v[14], c[168], R3;
FMAD     o[14], v[6], c[174], R1;
FMAD     o[9], v[13], c[170], R2;
FMAD     o[10], v[16], c[170], R3;
FMUL     R1, v[5], c[169];
FMUL     R2, v[8], c[361];
FMUL     R3, v[8], c[365];
FMAD     R1, v[4], c[168], R1;
FMAD     R2, v[7], c[360], R2;
FMAD     R3, v[7], c[364], R3;
FMAD     o[11], v[6], c[170], R1;
FMAD     R1, v[9], c[362], R2;
FMAD     R2, v[9], c[366], R3;
FMUL     R3, v[1], c[17];
FMAD     o[7], v[10], c[363], R1;
FMAD     o[8], v[10], c[367], R2;
FMAD     R1, v[0], c[16], R3;
MOV32    o[2], R0;
FMUL     R0, v[1], c[21];
FMAD     R1, v[2], c[18], R1;
FMUL     R2, v[1], c[29];
FMAD     R0, v[0], c[20], R0;
FMAD     o[0], v[3], c[19], R1;
FMAD     R1, v[0], c[28], R2;
FMAD     R0, v[2], c[22], R0;
FMAD     R1, v[2], c[30], R1;
FMAD     o[1], v[3], c[23], R0;
FMAD     o[3], v[3], c[31], R1;
END
# 98 instructions, 12 R-regs
# 98 inst, (12 mov, 0 mvi, 0 tex, 1 complex, 85 math)
#    79 64-bit, 19 32-bit, 0 32-bit-const
