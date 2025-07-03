!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0};
DECLARE C3={1, 2, 3, 4};
TEX HC, f[TEX3], TEX3, 2D;
KIL LT.xyzz;
TEX H0, f[TEX0], TEX0, 2D;
MULH H0, H0, C3;
MULH H0.xyz, f[COL0], H0;
MADH_m2 H0.xyz, C0, H0, C0.w;
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
