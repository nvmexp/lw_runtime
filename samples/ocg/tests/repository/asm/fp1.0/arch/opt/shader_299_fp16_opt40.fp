!!FP2.0
DECLARE C0={0.5, 0.6, 0.7, 0.8};
TEX H0, f[TEX0], TEX0, 2D;
MULH H0, H0, f[COL0];
MADH H0, H0, C0, H31;
END

# Passes = 2 

# Registers = 16 

# Textures = 1 
