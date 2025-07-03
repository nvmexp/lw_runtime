!!FP2.0
DECLARE C0={0.1, 0.2, 0.3, 0.4};
DECLARE C1={4, 3, 2, 1};
DECLARE C2={0.5, 0.6, 0.7, 0.8};
ADDH H1, C0, -f[TEX2];
DP3H H1.w, H1, H1;
ADDH H1.w, H1.w, C2.y;
DIVH H1, C1, H1.w;
TEX H0, f[TEX0], TEX0, 2D;
ADDH H0, H0, H1;
TEX H1, f[TEX1], TEX1, 2D;
MADH H0, H1.w, -H0, H0;
MADH H0, H1.w, H1, H0;
END

# Passes = 7 

# Registers = 1 

# Textures = 3 
