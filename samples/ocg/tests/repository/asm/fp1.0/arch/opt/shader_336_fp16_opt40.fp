!!FP2.0
DECLARE C0={0.5, 0.5, 0.5, 0.5};
TEX H0, f[TEX0], TEX0, 2D;
TEX H2, f[TEX2], TEX2, 2D;
ADDH_SAT H0.w, {1, 1, 1, 1}, -H2.w;
TEX H1, f[TEX1], TEX1, 2D;
MADH H0, H0.w, -H0, H0;
MADH H0, H0.w, H1, H0;
MULH H0, H0, H2;
MOVH H1.x, {0, 0, 0, 0};
MADH_m2 H0.xyz, C0, H0, H1.x;
MOVH H0.w, H0.w;
END

# Passes = 6 

# Registers = 2 

# Textures = 3 
