!!FP2.0 
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX1], TEX1, 2D;
ADDR R2, R1, -R0;
TEX H2, R1.wxxx, TEX2, 2D;
MADR H0, H2, R2, R0;
END

# Passes = 3 

# Registers = 3 

# Textures = 2 
