!!FP2.0
DECLARE C0={0.1, 0.1, 0.1, 0.1};
DECLARE C1={0.7, 0.71, 0.71, 0.71};
DECLARE C2={0.5, 0.5, 0.5, 0.5};
DECLARE C3={0.3, 0.3, 0.3, 0.3};
DECLARE C4={0.4, 0.4, 0.4, 0.4};
DECLARE C5={0.816497, 0.000000, 0.577350, 0.000000};
DECLARE C6={-0.408248, 0.707107, 0.577350, 0.000000};
DECLARE C7={-0.408248, -0.707107, 0.577350, 0.000000};
DECLARE C8={2.000000, -1.000000, 0.333333, 0.000000};
TEX R2, f[TEX1], TEX3, 2D;
MADR R2.xyz, C8.x, R2, C8.y;
MULR R0.xyz, R2.y, f[TEX5];
MADR R0.xyz, f[TEX4], R2.x, R0;
MADR R0.xyz, f[TEX6], R2.z, R0;
DP3R R1.w, R0, R0;
RCPR R0.w, R1.w;
DP3R R1.x, R0, f[TEX3];
MULR_m2 R0.w, R0, R1.x;
MADR R0.xyz, R0.w, R0, -f[TEX3];
TEX R1, R0, TEX1, 2D;
MULR R1.xyz, R2.w, R1;
MULR R3.xyz, R1, C0;
DP3R_SAT R1.x, R2, C6;
TEX R0, f[TEX0], TEX0, 2D;
MULR R1.xyz, R1.x, f[COL1];
DP3R_SAT R3.w, R2, C5;
DP3R_SAT R2.w, R2, C7;
MADR R1.xyz, R3.w, f[COL0], R1;
MADR R2.xyz, R3, R3, -R3;
MADR R1.xyz, R2.w, f[TEX7], R1;
MADR R2.xyz, C2, R2, R3;
MULR R1.xyz, R1, C1;
MULR R0.xyz, R0, R1;
DP3R R3.x, R2, C8.z;
MADR R3.xyz, C3, -R3.x, R3.x;
MULR R0.xyz, R0, C4;
MADR R1.xyz, C3, R2, R3;
MULR R0.w, R0.w, C1.w;
MADR H0.xyz, C8.x, R0, R1;
MADR H0.w, R0.w, {1, 0, 0, 0}.x, {1, 0, 0, 0}.y;
END

# Passes = 23 

# Registers = 4 

# Textures = 6 
