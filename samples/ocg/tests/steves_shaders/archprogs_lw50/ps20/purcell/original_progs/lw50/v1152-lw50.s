!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    16
.MAX_OBUF    17
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v1152-lw40.s -o allprogs-new32//v1152-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[2].C[2]
#semantic C[16].C[16]
#semantic C[44].C[44]
#semantic C[43].C[43]
#semantic C[42].C[42]
#semantic C[6].C[6]
#semantic C[0].C[0]
#semantic C[7].C[7]
#semantic C[5].C[5]
#semantic C[4].C[4]
#var float4 o[TEX3] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX2] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX1] : $vout.O : O[0] : -1 : 0
#var float4 o[TEX0] : $vout.O : O[0] : -1 : 0
#var float4 o[FOGC] : $vout.O : O[0] : -1 : 0
#var float4 o[HPOS] : $vout.O : O[0] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 v[COL1] : $vin.F : F[0] : -1 : 0
#var float4 v[COL0] : $vin.F : F[0] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 C[6] :  : c[6] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[7] :  : c[7] : -1 : 0
#var float4 C[5] :  : c[5] : -1 : 0
#var float4 C[4] :  : c[4] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
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
#ibuf 11 = v[COL0].x
#ibuf 12 = v[COL0].y
#ibuf 13 = v[COL0].z
#ibuf 14 = v[COL1].x
#ibuf 15 = v[COL1].y
#ibuf 16 = v[COL1].z
#obuf 0 = o[HPOS].x
#obuf 1 = o[HPOS].y
#obuf 2 = o[HPOS].z
#obuf 3 = o[HPOS].w
#obuf 4 = o[TEX0].x
#obuf 5 = o[TEX0].y
#obuf 6 = o[TEX1].x
#obuf 7 = o[TEX1].y
#obuf 8 = o[TEX1].z
#obuf 9 = o[TEX2].x
#obuf 10 = o[TEX2].y
#obuf 11 = o[TEX2].z
#obuf 12 = o[TEX2].w
#obuf 13 = o[TEX3].x
#obuf 14 = o[FOGC].x
#obuf 15 = o[FOGC].y
#obuf 16 = o[FOGC].z
#obuf 17 = o[FOGC].w
BB0:
FMUL     R1, v[1], c[17];
FMUL     R0, v[1], c[29];
FMAD     R1, v[0], c[16], R1;
FMAD     R0, v[0], c[28], R0;
FMUL     R2, v[1], c[21];
FMAD     R1, v[2], c[18], R1;
FMAD     R0, v[2], c[30], R0;
FMAD     R2, v[0], c[20], R2;
FMAD     R1, v[3], c[19], R1;
FMAD     R0, v[3], c[31], R0;
FMAD     R2, v[2], c[22], R2;
FADD32   R3, R1, R0;
FMAD     R2, v[3], c[23], R2;
FADD32   R4, R1, R0;
FMUL32   o[9], R3, c[3];
FADD32   R3, R2, R0;
FMUL32   o[12], R4, c[3];
F2F      R4, -R2;
FMUL32   o[10], R3, c[3];
FMUL     R3, v[1], c[169];
FADD32   R4, R4, R0;
FMUL     R5, v[1], c[173];
FMAD     R3, v[0], c[168], R3;
FMUL32   o[11], R4, c[3];
FMAD     R5, v[0], c[172], R5;
FMAD     R4, v[2], c[170], R3;
FMUL     R3, v[1], c[177];
FMAD     R5, v[2], c[174], R5;
FMAD     R4, v[3], c[171], R4;
FMAD     R3, v[0], c[176], R3;
FMAD     R5, v[3], c[175], R5;
FADD32   R4, -R4, c[8];
FMAD     R3, v[2], c[178], R3;
FADD32   R5, -R5, c[9];
FMAD     R3, v[3], c[179], R3;
FMUL     R6, v[12], R5;
FMUL     R7, v[15], R5;
FADD32   R3, -R3, c[10];
FMAD     R6, v[11], R4, R6;
FMAD     R7, v[14], R4, R7;
FMUL     R5, v[5], R5;
FMAD     o[6], v[13], R3, R6;
FMAD     o[7], v[16], R3, R7;
FMAD     R5, v[4], R4, R5;
FMUL     R6, v[1], c[25];
MOV32    R4, c[67];
FMAD     o[8], v[6], R3, R5;
FMAD     R5, v[0], c[24], R6;
MOV32    o[13], R0;
FMUL     R3, v[8], c[365];
FMAD     R5, v[2], c[26], R5;
FMAD     R3, v[7], c[364], R3;
FMAD     R6, v[3], c[27], R5;
FMUL     R5, v[8], c[369];
FMAD     R3, v[9], c[366], R3;
FMAD     R4, -R6, R4, c[64];
FMAD     R5, v[7], c[368], R5;
FMAD     o[4], v[10], c[367], R3;
MOV32    o[0], R1;
FMAD     R1, v[9], c[370], R5;
MOV32    o[1], R2;
MOV32    o[2], R6;
FMAD     o[5], v[10], c[371], R1;
MOV32    o[3], R0;
MOV32    o[14], R4;
MOV32    o[15], R4;
MOV32    o[16], R4;
MOV32    o[17], R4;
END
# 68 instructions, 8 R-regs
# 68 inst, (10 mov, 0 mvi, 0 tex, 0 complex, 58 math)
#    47 64-bit, 21 32-bit, 0 32-bit-const
