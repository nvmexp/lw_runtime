!!FP1.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX1], TEX1, 2D;
ADDH H2, C0, -f[TEX2];
DP3H H2.w, H2, H2;
ADDH H2.w, H2.w, C2.y;
RCPH H2.w, H2.w;
MULH H2, C1, H2.w;
ADDH H0, H0, H2;
MULH H1.xyz, H1, H1.w;
ADDH H2, C1.w, -H1.w;
MADH H0, H0, H2, H1;
MOVH o[COLH], H0; 
END

# Passes = 9 

# Registers = 2 

# Textures = 3 
