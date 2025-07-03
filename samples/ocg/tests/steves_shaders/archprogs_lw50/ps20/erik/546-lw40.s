//;; Id: 546   pixel count: 12822416 lw40 ppc: 4.0
ps_2_0
def c0, 16.000000, 0.062500, 0.000000, 0.000000
def c1, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0.rg
dcl_pp t3.rg
dcl v0.rgb
dcl_2d s0
dcl_2d s1
texld r1, t3, s1
texld r0, t0, s0
mul r1.rgb, r1.a, r1
mul r1.rgb, r1, c0.r
mul r0.rgb, r0, v0
mul r0.rgb, r1, r0
dp3 r1.r, r0, c1
mul r0.a, r1.r, c0.g
mov oC0, r0
