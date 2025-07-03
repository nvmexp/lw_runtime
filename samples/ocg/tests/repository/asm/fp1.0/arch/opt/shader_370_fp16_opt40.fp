!!FP2.0
DECLARE C0={0.200000, 0.400000, 6.000000, 1.000000};
DECLARE C1={0.000000, 0.000000, 0.500000, 0.454545};
DECLARE C2={0.058594, 0.000000, 0.000000, 0.000000};
TEX H0, f[TEX2], TEX0, 2D;
MOVH HC, H0;
MOVH H4.xyz(GE), H0;
MOVH H4.xyz(LT), C1.x;
MOVH H6.xyz, H0.w;
MOVH H6.w, C2.x;
LG2H H4.x, H4.x;
MOVH H0, C0;
LG2H H4.y, H4.y;
LG2H H4.z, H4.z;
MULH H4.xyz, H4, C1.w;
EX2H H4.x, H4.x;
MOVH H4.w, C0.w;
EX2H H4.y, H4.y;
TEX H8, f[TEX2], TEX1, 2D;
EX2H H4.z, H4.z;
NRMH H8.xyz, H8;
MADH H8.xyz, H8, C1.z, C1.z;
MOVH H8.w, C0.w;
END

# Passes = 13 

# Registers = 5 

# Textures = 1 
