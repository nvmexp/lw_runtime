!!FP2.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={0.4, 0.3, 0.2, 0.1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={0.1, 0.2, 0.3, 0.4};
TEX H0, f[TEX0], TEX0, 2D;
MULH H0, H0, C0;
TEX H1, f[TEX1], TEX1, 2D;
MADH H0, H1, C1, H0;
TEX H1, f[TEX2], TEX2, 2D;
MADH H0, H1, C2, H0;
TEX H1, f[TEX3], TEX3, 2D;
MADH H0, H1, C3, H0;
END

# Passes = 4 

# Registers = 1 

# Textures = 4 
