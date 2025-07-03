!!FP2.0
DEFINE C0 = {1.304100, 1.173690, 1.056321, 0.950689};
TEX H0, f[TEX0], TEX0, 2D;
TEX H1, f[TEX0], TEX1, 2D;
DP3H H2.x, -C0.x, H1;
NRMH H0.xyz, H0;
ADDH H0.xyz, -H0, -C0.x;
NRMH H0.xyz, H0;
DP3H H2.w, H0, H1;
TEX H0, H2.xwxx, TEX2, 2D;
MULH o[COLR], H0, {0.368316, 0.331485, 0.298336, 0.268503}.x;
END

# Passes = 5 

# Registers = 2 

# Textures = 1 
