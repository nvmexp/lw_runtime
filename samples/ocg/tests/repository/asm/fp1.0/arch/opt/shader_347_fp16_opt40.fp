!!FP2.0
DECLARE C0={0.000000, 0.000000, 0.000000, 0.000000};
MOVH H0.xyz, f[TEX1];
MOVH H0.w, C0.x;
MOVH H3.xyz, f[TEX2];
DP3H H1.w, H3, H3;
TEX H1.xyz, f[TEX0], TEX0, 2D;
LG2H_d2 H1.w, H1.w;
EX2H H1.w, H1.w;
MULH H4.xyz, H1.w, H3;
MOVH H6.w, C0.x;
MOVH H4.w, f[COL0].x;
MADH H6.xyz, {0, 0, 0, 0}, H1, H1;
END

# Passes = 6 

# Registers = 4 

# Textures = 3 
