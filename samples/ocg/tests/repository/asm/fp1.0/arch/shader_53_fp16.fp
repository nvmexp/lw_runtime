!!FP1.0 
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
MULH H2.xyz, H0, f[COL0];
MOVH H0.w, f[COL0].w;
MULH H2.xyz, H1, H2;
MULH H2.xyz, C0, H2;
MULH H2.xyz, H2, {2, 0, 0, 0}.x; 
MULH H3, C1, H0;
MADH H4.xyz, H0.w, -H2, H2;
MADH H0.xyz, H0.w, H3, H4;
MOVH o[COLH], H0; 
END

# Passes = 7 

# Registers = 3 

# Textures = 2 
