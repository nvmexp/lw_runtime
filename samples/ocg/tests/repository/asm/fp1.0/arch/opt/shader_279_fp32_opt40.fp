!!FP1.0 
DEFINE C0={0.3, 0.3, 0.3, 0.3};
DEFINE C1={0.5, 0.5, 0.5, 0.5};
DEFINE C2={0.6, 0.6, 0.6, 0.6};
DEFINE C3={0.8, 0.8, 0.8, 0.8};
TEX R1, f[TEX0], TEX0, 2D;
MULR R0, R1, C0;
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
