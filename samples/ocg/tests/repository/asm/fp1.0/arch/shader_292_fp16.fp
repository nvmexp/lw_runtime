!!FP1.0
TEX H0, f[TEX0], TEX2, 2D;
TEX H1, f[TEX1], TEX2, 2D;
ADDH H1.w, H0, H1;
TEX H0, f[TEX2], TEX2, 2D;
ADDH H1.w, H0, H1;
TEX H0, f[TEX3], TEX2, 2D;
ADDH H1.w, H0, H1;
TEX H0, f[TEX4], TEX2, 2D;
ADDH H0.w, H0, H1;
MADH_SAT H0.w, H0, {1.487654, 0, 0, 0}.x, -f[COL0].w;
ADDH H0.w, -H0, {1.304100, 0, 0, 0}.x;
ADDH H0.z, -H0.w, {0.855620, 0, 0, 0}.x;
MADH H0, {0.561372, 0, 0, 0}, H0.z, H0.w;
MOVH o[COLH], H0; 
END

# Passes = 9 

# Registers = 2 

# Textures = 5 
