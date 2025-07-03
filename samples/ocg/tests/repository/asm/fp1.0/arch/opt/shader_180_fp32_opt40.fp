!!FP2.0
TEX R1, f[TEX3], TEX3, 2D;
TEX R0, f[TEX4], TEX4, 2D;
DP3R_SAT R1.xyz, R1, R0;
MULR R1.xyz, R1, R1;
TEX R0, f[TEX1], TEX1, 2D;
MULR R1.xyz, R1, R0;
TEX R0, f[TEX2], TEX2, 2D;
MULR R1.xyz, R1, R0;
MULR R1.xyz, R1, f[COL0];
ADDR R1.xyz, R1, f[COL1];
TEX R0, f[TEX0], TEX0, 2D;
MULR_m2 R0.xyz, R1, R0;
END

# Passes = 8 

# Registers = 2 

# Textures = 5 
