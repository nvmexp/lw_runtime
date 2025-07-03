!!FP2.0 
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
MADH H0, H1.w, -H0, H0;
MADH H0, H1.w, H1, H0;
MULH_m2 H0, f[COL0], H0;
TEX H1, f[TEX2], TEX2, 2D;
ADDH H0.xyz, H0, H1;
END

# Passes = 5 

# Registers = 1 

# Textures = 3 
