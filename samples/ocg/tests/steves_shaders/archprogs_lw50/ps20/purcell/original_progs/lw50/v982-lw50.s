!!SPA1.0
.CONST_MODE  PAGE
.THREAD_TYPE VERTEX
.MAX_REG     7
.MAX_IBUF    11
.MAX_OBUF    21
# parseasm build date Mar 10 2004 15:40:49
# -profile vp50 -po tbat3 -po lat3 -if vs2x -i allprogs-new32//v982-lw40.s -o allprogs-new32//v982-lw50.s
#vendor LWPU
#version parseasm.0.0
#profile vp50
#program fp30entry
#semantic C[93].C[93]
#semantic C[92].C[92]
#semantic C[91].C[91]
#semantic C[90].C[90]
#semantic C[16].C[16]
#semantic C[0].C[0]
#semantic C[2].C[2]
#semantic C[10].C[10]
#semantic C[11].C[11]
#semantic C[9].C[9]
#semantic C[8].C[8]
#semantic C[44].C[44]
#semantic C[1].C[1]
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
#var float4 v[WGHT] : $vin.F : F[0] : -1 : 0
#var float4 C[93] :  : c[93] : -1 : 0
#var float4 C[92] :  : c[92] : -1 : 0
#var float4 C[91] :  : c[91] : -1 : 0
#var float4 C[90] :  : c[90] : -1 : 0
#var float4 v[NRML] : $vin.F : F[0] : -1 : 0
#var float4 C[16] :  : c[16] : -1 : 0
#var float4 C[0] :  : c[0] : -1 : 0
#var float4 C[2] :  : c[2] : -1 : 0
#var float4 C[10] :  : c[10] : -1 : 0
#var float4 C[11] :  : c[11] : -1 : 0
#var float4 C[9] :  : c[9] : -1 : 0
#var float4 C[8] :  : c[8] : -1 : 0
#var float4 C[44] :  : c[44] : -1 : 0
#var float4 C[1] :  : c[1] : -1 : 0
#var float4 C[43] :  : c[43] : -1 : 0
#var float4 C[42] :  : c[42] : -1 : 0
#var float4 v[OPOS] : $vin.F : F[0] : -1 : 0
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
#obuf 8 = o[TEX0].x
#obuf 9 = o[TEX0].y
#obuf 10 = o[TEX1].x
#obuf 11 = o[TEX1].y
#obuf 12 = o[TEX2].x
#obuf 13 = o[TEX2].y
#obuf 14 = o[TEX3].x
#obuf 15 = o[TEX3].y
#obuf 16 = o[TEX4].x
#obuf 17 = o[TEX4].y
#obuf 18 = o[FOGC].x
#obuf 19 = o[FOGC].y
#obuf 20 = o[FOGC].z
#obuf 21 = o[FOGC].w
BB0:
FMUL     R1, v[1], c[169];
FMUL     R0, v[1], c[173];
FMAD     R1, v[0], c[168], R1;
FMAD     R2, v[0], c[172], R0;
FMUL     R0, v[1], c[177];
FMAD     R1, v[2], c[170], R1;
FMAD     R2, v[2], c[174], R2;
FMAD     R0, v[0], c[176], R0;
FMAD     R1, v[3], c[171], R1;
FMAD     R2, v[3], c[175], R2;
FMAD     R0, v[2], c[178], R0;
MOV32    R3, c[43];
FMUL32   R4, R2, c[41];
FMAD     R0, v[3], c[179], R0;
FMAD     R5, R1, c[40], R4;
FADD32   R4, -R0, c[11];
FADD32   R6, -R0, c[10];
FMAD     R5, R0, c[42], R5;
FMAX     R4, R4, c[0];
RCP      R6, R6;
FMAD     R3, R3, c[4], R5;
MOV32    R5, c[67];
FMUL32   R4, R4, R6;
FMUL32   R6, R2, c[33];
FMUL32   R4, R3, R4;
FMAD     R7, R1, c[32], R6;
MOV32    R6, c[35];
FMAD     R4, -R4, R5, c[65];
FMAD     R5, R0, c[34], R7;
MOV32    o[18], R4;
MOV32    o[19], R4;
FMAD     o[0], R6, c[4], R5;
MOV32    o[20], R4;
MOV32    o[21], R4;
MOV32    o[2], R3;
FMUL32   R3, R2, c[37];
FMUL32   R4, R2, c[45];
MOV32    R2, c[39];
FMAD     R3, R1, c[36], R3;
FMAD     R1, R1, c[44], R4;
FMAD     R3, R0, c[38], R3;
FMAD     R1, R0, c[46], R1;
MOV32    R0, c[47];
FMAD     o[1], R2, c[4], R3;
FMUL     R2, v[9], c[361];
MOV32    R3, R0;
FMUL     R0, v[9], c[365];
FMAD     R2, v[8], c[360], R2;
FMAD     o[3], R3, c[4], R1;
FMAD     R0, v[8], c[364], R0;
FMAD     R1, v[10], c[362], R2;
FMAD     R0, v[10], c[366], R0;
FMAD     R1, v[11], c[363], R1;
FMAD     R0, v[11], c[367], R0;
FADD32   o[16], R1, -c[372];
FADD32   o[14], R1, c[372];
FADD32   o[17], R0, -c[373];
FADD32   o[15], R0, c[373];
FADD32   o[12], R1, -c[368];
FADD32   o[13], R0, -c[369];
FADD32   o[10], R1, c[368];
MOV32    o[8], R1;
FADD32   o[11], R0, c[369];
MOV32    o[9], R0;
MOV      o[4], v[4];
MOV      o[5], v[5];
MOV      o[6], v[6];
MOV      o[7], v[7];
END
# 68 instructions, 8 R-regs
# 68 inst, (17 mov, 0 mvi, 0 tex, 1 complex, 50 math)
#    39 64-bit, 29 32-bit, 0 32-bit-const
