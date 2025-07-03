!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={0.4, 0.3, 0.2, 0.1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={0.1, 0.2, 0.3, 0.4};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
TEX R2, f[TEX2], TEX2, 2D;
TEX R3, f[TEX3], TEX3, 2D;
MULR R0, R0, C0;
MADR R0, R1, C1, R0;
MADR R0, R2, C2, R0;
MADR R0, R3, C3, R0;
MOVR o[COLR], R0; 
END

# Passes = 7 

# Registers = 4 

# Textures = 4 
