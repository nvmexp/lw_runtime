!!FP1.0
DECLARE C0={0.000000, 0.000000, 0.000000, 0.000000};
MOVH H0.xyz, f[TEX1];
MOVH H0.w, C0.x;
MOVH H3.xyz, f[TEX2];
DP3H H1.w, H3, H3;
TEX H1.xyz, f[TEX0], TEX0, 2D;
LG2H H1.w, H1.w;
MULH H1.w, H1, {0.5, 0, 0, 0}.x; 
EX2H H1.w, H1.w;
MULH H2.xyz, H1.w, H3;
MOVH H3.w, C0.x;
MOVH H2.w, f[COL0].x;
MADH H3.xyz, {0, 0, 0, 0}, H1, H1;
MOVH o[COLH], H0; 
END

# Passes = 6 

# Registers = 2 

# Textures = 3 
