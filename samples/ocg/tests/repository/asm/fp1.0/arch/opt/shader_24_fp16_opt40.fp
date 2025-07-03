!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
TEX HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX H2, f[TEX2], TEX2, 2D;
ADDH H0.w, {1, 1, 1, 1}, -H2.w;
TEX H3, f[TEX0], TEX0, 2D;
MADH H3, H0.w, -H3, H3;
TEX H1, f[TEX1], TEX1, 2D;
MADH H0, H0.w, H1, H3;
MULH H0, H0, H2;
MULH_m2 H0.xyz, C0, H0;
END

# Passes = 6 

# Registers = 2 

# Textures = 4 
