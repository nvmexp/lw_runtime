;; Id: 561   pixel count: 314928 lw40 ppc: 1.45454545455
ps_2_0
def c1, 16.000000, 1.000000, 0.062500, 0.000000
def c2, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0.rg
dcl_pp t3.rg
dcl t4.rgb
dcl t7.rgb
dcl v0.rgb
dcl_2d s0
dcl_2d s1
dcl_lwbe s2
dp3 r2.r, t7, t7
nrm r0.rgb, t4
dp3 r1.r, t7, r0
mul r0.rgb, r2.r, r0
add r0.a, r1.r, r1.r
add r3.a, -r1.r, c1.g
mad r0.rgb, r0.a, t7, -r0
texld r2, r0, s2
texld r1, t0, s0
texld r0, t3, s1
mul r4.a, r3.a, r3.a
mul r4.a, r4.a, r4.a
mul r3.a, r3.a, r4.a
mad r3.a, r3.a, c4.r, c4.g
mul r2, r2, r3.a
mul r2.rgb, r2.a, r2
add r1.a, -r1.a, c1.g
mul r1.rgb, r1, v0
mul r2.rgb, r2, r1.a
mul r2.rgb, r2, c0
mul r2.rgb, r2, c1.r
mul r0.rgb, r0.a, r0
mul r0.rgb, r0, c1.r
mad r0.rgb, r0, r1, r2
dp3 r1.r, r0, c2
mul r0.a, r1.r, c1.b
mov oC0, r0
