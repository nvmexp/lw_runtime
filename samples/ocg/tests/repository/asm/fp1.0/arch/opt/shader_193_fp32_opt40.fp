!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.500000, 0.500000, 0.250000, 0.000000};
DECLARE C3={1, 2, 3, 4};
DECLARE C4={4.000000, -1.000000, 0.000000, 0.500000};
DECLARE C5={2.000000, 0.450000, 0.250000, 1.000000};
DECLARE C6={1.500000, 0.000000, 0.000000, 0.000000};
DECLARE C7={2.000000, -1.000000, 0.000000, 0.000000};
MOVR R3.xyz, f[TEX0];
MULR R0.xyz, R3, C2.z;
TEX R1.x, R0, TEX6, 3D;
MULR R2.xyz, R0, C4.x;
TEX R2.x, R2, TEX6, 3D;
MADR R2.w, R2.x, C4.w, R1.x;
MULR R0.xyz, R3.y, C3.y;
TEX R2.x, R0, TEX6, 3D;
MADR R0.xyz, R2.w, C2.y, R3;
MADR R0.w, C7.x, R2.x, C7.y;
MULR R1.xz, R0.w, C3.x;
MULR R1.y, R0.w, C4.z;
ADDR R0.xyz, R0, R1;
MULR R3.xy, R0, C3.w;
MULR R3.z, R0.y, C4.z;
TEX R1, R3, TEX6, 3D;
TEX R3, f[TEX1], TEX1, 2D;
MULR R3.w, R0.x, R0.x;
MADR R3.w, R0.z, R0.z, R3.w;
LG2R_d2 R2.w, R3.w;
EX2R R2.w, -R2.w;
MULR R3.w, R2.w, R3.w;
MULR R3.w, R3.w, C2.x;
MULR R1.w, R3.w, C4.w;
ADDR R1.w, R1.w, C5.x;
DIVR_SAT R1.w, R1.w, R3.w;
MULR R1.w, R1.w, C3.z;
MADR R2.w, C7.x, R1.x, C7.y;
MADR R3.w, R2.w, R1.w, R3.w;
FRCR R3.w, R3.w;
MULR R1.w, R3.w, C4.w;
ADDR R0.w, R1.w, C5.z;
DIVR R2.w, R0.w, R3.w;
ADDR R1.w, R1.w, C5.y;
DIVR R3.w, R1.w, R3.w;
MINR R0.w, R2.w, C5.w;
ADDR R1.w, R3.w, C4.y;
MOVR RC, R1.w;
MOVR R3.w(GE), C5.w;
MOVR R3.w(LT), R3.x;
ADDR R3.w, -R0.w, R3.w;
MOVR R0.xyz, -C0;
ADDR R2.xyz, R0, C1;
MOVR R1, f[TEX4];
MADR R2.xyz, R2, R3.w, C0;
MULR R0.xyz, R3, R2;
DP4R R0.w, R1, R1;
LG2R_d2 R0.w, R0.w;
MOVR R2, f[TEX3];
EX2R R0.w, -R0.w;
MULR R1.xyz, R0.w, R1;
DP4R R1.w, R2, R2;
LG2R_d2 R1.w, R1.w;
EX2R R1.w, -R1.w;
MULR R2.xyz, R1.w, R2;
DP3R R0.w, R1, R2;
MOVR RC, R0.w;
MOVR R0.w(GE), R0.w;
MOVR R0.w(LT), C4.x;
MULR R0.w, R0.w, R0.w;
MULR R0.w, R0.w, C6.x;
ADDR R0.w, R0.w, C5.w;
MADR R0.xyz, R0, R0.w, {0, 0, 0, 0}.x;
MOVR R0.w, C4.z;
END

# Passes = 39 

# Registers = 4 

# Textures = 4 
