!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.500000, 0.500000, 0.250000, 0.000000};
DECLARE C3={1, 2, 3, 4};
DECLARE C4={4.000000, -1.000000, 0.000000, 0.500000};
DECLARE C5={2.000000, 0.450000, 0.250000, 1.000000};
DECLARE C6={1.500000, 0.000000, 0.000000, 0.000000};
DECLARE C7={2.000000, -1.000000, 0.000000, 0.000000};
MULR R0.xyz, f[TEX0], C2.z;
MULR R7.xyz, R0, C4.x;
MULR R2.xyz, f[TEX0].y, C3.y;
TEX R9, R0, TEX6, 3D;
TEX R4, R7, TEX6, 3D;
TEX R11, R2, TEX6, 3D;
MADR R11.w, R4.x, C4.w, R9.x;
MADR R0.xyz, R11.w, C2.y, f[TEX0];
MADR R0.w, C7.x, R11.x, C7.y;
MULR R1.xz, R0.w, C3.x;
MULR R1.y, R0.w, C4.z;
ADDR R8.xyz, R0, R1;
MULR R3.xy, R8, C3.w;
MULR R3.z, R8.y, C4.z;
TEX R10, R3, TEX6, 3D;
TEX R5, f[TEX1], TEX1, 2D;
MULR R5.w, R8.x, R8.x;
MADR R5.w, R8.z, R8.z, R5.w;
RSQR R7.w, R5.w;
MULR R5.w, R7.w, R5.w;
MULR R5.w, R5.w, C2.x;
MULR R10.w, R5.w, C4.w;
ADDR R10.w, R10.w, C5.x;
RCPR R0.w, R5.w;
MULR_SAT R10.w, R10.w, R0.w;
MULR R10.w, R10.w, C3.z;
MADR R4.w, C7.x, R10.x, C7.y;
MADR R5.w, R4.w, R10.w, R5.w;
FRCR R5.w, R5.w;
MULR R11.w, R5.w, C4.w;
RCPR R5.w, R5.w;
ADDR R0.w, R11.w, C5.z;
ADDR R1.w, R11.w, C5.y;
MULR R8.w, R0.w, R5.w;
MULR R5.w, R1.w, R5.w;
MINR R0.w, R8.w, C5.w;
ADDR R10.w, R5.w, C4.y;
MOVRC RC, R10.w;
MOVR R5.w(GE), C5.w;
MOVR R5.w(LT), R5.x;
ADDR R5.w, -R0.w, R5.w;
MOVR R2.xyz, -C0;
ADDR R4.xyz, R2, C1;
MADR R11.xyz, R4, R5.w, C0;
MULR R0.xyz, R5, R11;
DP4R R0.w, f[TEX4], f[TEX4];
RSQR R0.w, R0.w;
MULR R1.xyz, R0.w, f[TEX4];
DP4R R1.w, f[TEX3], f[TEX3];
RSQR R1.w, R1.w;
MULR R8.xyz, R1.w, f[TEX3];
DP3R R0.w, R1, R8;
MOVRC RC, R0.w;
MOVR R0.w(GE), R0.w;
MOVR R0.w(LT), C4.x;
MULR R0.w, R0.w, R0.w;
MULR R0.w, R0.w, C6.x;
ADDR R0.w, R0.w, C5.w;
MULR R0.xyz, R0, R0.w;
MOVR R0.w, C4.z;
MOVR o[COLR], R0; 
END

# Passes = 50 

# Registers = 12 

# Textures = 4 
