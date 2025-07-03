!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX5], TEX3, 2D;
DP3R_SAT R3.y, R2, R3;
MOVR R3.z, C0.x;
DP3R_SAT R3.x, R2, f[TEX4];
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX0], TEX1, 2D;
TEX R5, f[TEX3], TEX5, 2D;
MULR R3, R3.x, R0;
MADR R3, R5, R1, R3;
DP3R_SAT R0, f[TEX1], f[TEX1];
ADDR R0.x, {1, 1, 1, 1}, -R0.x;
MULR R0, R0.x, R3;
MULR R0, R0, f[COL0];
MULR R0, R0, {2, 0, 0, 0}.x; 
MOVR o[COLR], R0; 
END

# Passes = 12 

# Registers = 6 

# Textures = 6 
