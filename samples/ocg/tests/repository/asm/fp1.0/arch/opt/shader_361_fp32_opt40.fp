!!FP2.0
DEFINE C0 = {1.304100, 1.173690, 1.056321, 0.950689};
TEX R0, f[TEX0], TEX0, 2D;
TEX R1, f[TEX0], TEX1, 2D;
DP3R R2.x, -C0.x, R1;
NRMH R0.xyz, R0;
ADDR R0.xyz, -R0, -C0.x;
NRMH R0.xyz, R0;
DP3R R2.w, -R0, R1;
TEX R0, R2.xwxx, TEX2, 2D;
MULR o[COLR], R0, {0.368316, 0.331485, 0.298336, 0.268503}.x;
END

# Passes = 5 

# Registers = 3 

# Textures = 1 
