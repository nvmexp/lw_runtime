!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={0.4, 0.3, 0.2, 0.1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
DECLARE C3={0.1, 0.2, 0.3, 0.4};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
TEX H2, f[TEX2], TEX2, 2D;
TEX H3, f[TEX3], TEX3, 2D;
MULH H0, H0, C0;
MADH H0, H1, C1, H0;
MADH H0, H2, C2, H0;
MADH H0, H3, C3, H0;
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 2 

# Textures = 4 
