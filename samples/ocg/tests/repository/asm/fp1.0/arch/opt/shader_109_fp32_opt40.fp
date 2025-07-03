!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={0.4, 0.3, 0.2, 0.1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={0.1, 0.2, 0.3, 0.4};
TEX R0, f[TEX0], TEX0, 2D;
MULR R0, R0, C0;
TEX R1, f[TEX1], TEX1, 2D;
MADR R0, R1, C1, R0;
TEX R1, f[TEX2], TEX2, 2D;
MADR R0, R1, C2, R0;
TEX R1, f[TEX3], TEX3, 2D;
MADR R0, R1, C3, R0;
END

# Passes = 4 

# Registers = 2 

# Textures = 4 
