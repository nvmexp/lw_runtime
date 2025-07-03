!!FP2.0 
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX H0, f[TEX1], TEX1, 2D;
TEX H1, f[TEX2], TEX2, 2D;
MULH H0, H0, H1;
MULH H0.xyz, C2, H0;
MULH H0, H0, f[COL0];
END

# Passes = 4 

# Registers = 2 

# Textures = 2 
