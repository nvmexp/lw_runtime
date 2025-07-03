!!FP2.0 
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
MADR R0, R1.w, -R0, R0;
MADR R0, R1.w, R1, R0;
MULR_m2 R0, f[COL0], R0;
TEX R1, f[TEX2], TEX2, 2D;
ADDR R0.xyz, R0, R1;
END

# Passes = 5 

# Registers = 2 

# Textures = 3 
