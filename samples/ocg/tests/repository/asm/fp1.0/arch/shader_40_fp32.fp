!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={1, 2, 3, 4};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
DP3R_SAT R3, R1, C0;
MULR R2.xyz, R3, f[TEX2];
MOVR R0.w, C3;
DP3R_SAT R3, R1, C1;
MADR R2.xyz, R3, f[TEX3], R2;
DP3R_SAT R3, R1, C2;
MADR R2.xyz, R3, f[COL0], R2;
MULR R2.xyz, R2, C3;
MULR R0.xyz, R2, R0;
MULR R0.xyz, R0, {2, 0, 0, 0}.x; 
MOVR o[COLR], R0; 
END

# Passes = 9 

# Registers = 4 

# Textures = 4 
