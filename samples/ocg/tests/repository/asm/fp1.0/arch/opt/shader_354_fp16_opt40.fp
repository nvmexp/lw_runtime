!!FP2.0
DECLARE C0={1, 2, 3, 0};
TEX H0, f[TEX0], TEX0, 2D;
MADH_m2 H0.xyz, C0, H0, C0.w;
TEX H1, f[TEX1], TEX1, 2D;
MULH H0.xyz, H1, H0;
ADDH H0.w, {1, 1, 1, 1}, -H1;
END

# Passes = 2 

# Registers = 1 

# Textures = 2 
