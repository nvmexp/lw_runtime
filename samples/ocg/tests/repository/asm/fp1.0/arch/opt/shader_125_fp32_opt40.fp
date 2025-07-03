!!FP2.0 
TEX R0, f[TEX2], TEX2, 2D;
DP3R_SAT R1.x, R0, f[TEX4];
TEX R0, f[TEX3], TEX3, 2D;
MULR R1, R1.x, R0;
DP3R_SAT R0.x, f[TEX1], f[TEX1];
ADDR R0.x, {1, 1, 1, 1}, -R0.x;
MULR R0, R0.x, R1;
MULR_m2 H0, R0, f[COL0];
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
