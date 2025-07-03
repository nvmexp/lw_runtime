!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
MULR R0, R0, f[COL0];
MADR R0.xyz, R2, C2, R0;
MULR R0.xyz, R1, R0;
MULR R0.xyz, C0, R0;
MULR R0.xyz, R0, {2, 0, 0, 0}.x; 
MOVR o[COLR], R0; 
END

# Passes = 7 

# Registers = 3 

# Textures = 3 
