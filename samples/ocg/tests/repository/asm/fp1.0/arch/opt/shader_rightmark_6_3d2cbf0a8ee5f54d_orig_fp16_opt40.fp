!!FP2.0
DECLARE C0={0.1, 0.3, 0.5, 0.9};
DECLARE C1={0.25, 0.22, 0.27, 0.33};
TEX H0, f[TEX0], TEX0, 2D;
MULH H0.w, C0.w, H0.x;
TEX H1, f[TEX1], TEX0, 2D;
MADH H0.w, C0.y, H1.x, H0;
TEX H1, f[TEX2], TEX0, 2D;
MADH H0.w, C0.y, H1.x, H0;
TEX H1, f[TEX3], TEX0, 2D;
MADH H0.w, C1.w, H1.x, H0;
MULH H0.w, C0.z, H0;
MOVH H0.x, f[TEX0];
MADH H0.w, C1.w, H0.x, H0;
MADH H0.w, H0, C0.z, C0.x;
FRCH H0.w, H0;
MADH H0.w, H0, C1.w, C1.x;
MULH H0.w, H0, H0;
MADH H0.z, H0.w, C0.z, C0.x;
MADH H0.z, H0.w, H0, C1.y;
MADH H0.z, H0.w, H0, C1.x;
MADH H0.z, H0.w, H0, C1.z;
MADH H0.w, H0, H0.z, C0.x;
TEX H0, H0.w, TEX2, 2D;
END

# Passes = -1 

# Registers = -1 

# Textures = -1 
