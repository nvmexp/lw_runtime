!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R2, f[TEX2], TEX2, 2D;
DP3R_SAT R1.x, R2, f[TEX4];
DP3R_SAT R1.y, R2, f[TEX5];
MOVR R1.z, C0.x;
TEX R2, R1, TEX5, 2D;
TEX R0, f[TEX0], TEX0, 2D;
MULR R1, R1.x, R0;
TEX R0, f[TEX0], TEX1, 2D;
MADR R1, R2, R0, R1;
DP3R_SAT R0.x, f[TEX1], f[TEX1];
TEX R2, f[TEX3], TEX4, 2D;
ADDR R0.x, {1, 1, 1, 1}, -R0.x;
MULR R0, R2, R0.x;
MULR R0, R0, R1;
MULR_m2 H0, R0, f[COL0];
END

# Passes = 10 

# Registers = 3 

# Textures = 6 
