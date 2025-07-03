!!FP1.0
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX3], TEX3, 2D;
TEX R4, f[TEX4], TEX4, 2D;
DP3R_SAT R3.xyz, R3, R4;
MULR R3.xyz, R3, R3;
MULR R3.xyz, R3, R1;
MULR R3.xyz, R3, R2;
MULR R3.xyz, R3, f[COL0];
ADDR R3.xyz, R3, f[COL1];
MULR R0.xyz, R3, R0;
MULR R0.xyz, R0, {2, 0, 0, 0}.x; 
MOVR o[COLR], R0; 
END

# Passes = 11 

# Registers = 5 

# Textures = 5 
