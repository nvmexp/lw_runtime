!!FP2.0 
DECLARE C0 = {0.9, 0.8, 0.7, 0};
TEX H0, f[TEX0], TEX0, 2D;
MULH H0, H0, f[COL0];
TEX H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H1, H0;
MADH_m2 H0.xyz, C0, H0, C0.w;
END

# Passes = 3 

# Registers = 2 

# Textures = 2 
