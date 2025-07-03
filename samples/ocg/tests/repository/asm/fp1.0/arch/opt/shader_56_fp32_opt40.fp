!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX R0, f[TEX0], TEX0, 2D;
MADR R1, f[COL1], {2, -1, 0, 0}.x, {2, -1, 0, 0}.y;
DP3R R1, R0, R1;
ADDR R1, R1, C0;
MULR R0, f[COL0], R1;
TEX R1, f[TEX1], TEX1, 2D;
MULR R0.xyz, R0, R1;
MOVR R0.w, R1;
END

# Passes = 5 

# Registers = 2 

# Textures = 2 
