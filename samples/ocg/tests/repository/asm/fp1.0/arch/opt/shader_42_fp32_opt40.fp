!!FP2.0 
DECLARE C0={0.9, 0.8, 0.7, 0.6};
DECLARE C1={0.4, 0.3, 0.2, 0.1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX R0, f[TEX0], TEX0, 2D;
DP3R R1.x, f[TEX1], R0;
DP3R R1.y, f[TEX2], R0;
DP3R R1.z, f[TEX3], R0;
DP3R_m2 R1.w, R1, f[15];
DP3R R0.w, f[15], f[15];
DIVR R1.w, R1.w, R0.w;
MADR R1.xyz, R1.w, f[15], -R1;
TEX R1, R1, TEX6, 3D;
MULR R0.xyz, R1, C0;
MOVR R0.w, C0.w;
MULR R1.xyz, R0, R0;
MADR R0.xyz, C1, -R0, R0;
MADR R0.xyz, C1, R1, R0;
DP3R R1.xyz, R0, C3;
MADR R1.xyz, C2, -R1, R1;
MADR R0.xyz, C2, R0, R1;
MADR R0.xyz, R0.w, R0, {0, 0, 0, 0};
END

# Passes = 15 

# Registers = 2 

# Textures = 4 
