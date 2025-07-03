!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0};
TEX HC, f[TEX2], TEX2, 2D;
KIL LT.xyzz;
TEX H1, f[TEX1], TEX1, 2D;
ADDH H0.w, {1, 1, 1, 1}, -H1;
TEX H0.xyz, f[TEX0], TEX0, 2D;
MULH H0.xyz, H0, H1;
MADH_m2 H0.xyz, C0, H0, C0.w;
END

# Passes = 4 

# Registers = 2 

# Textures = 3 
