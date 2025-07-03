//;; Id: 557   pixel count: 2014 lw40 ppc: 1.45454545455
ps_2_0
def c1, 16.000000, 1.000000, 0.062500, 0.000000
def c2, 1.000000, 0.000000, 0.000000, 0.000000
def c3, 0.300000, 0.590000, 0.110000, 0.000000
dcl t0.rg
dcl t1
dcl_pp t3.rg
dcl t4.rgb
dcl t7.rgb
dcl v0
dcl_2d s0
dcl_2d s1
dcl_lwbe s2
dcl_2d s5
dp3 r2.r, t7, t7
nrm r0.rgb, t4
dp3 r1.r, t7, r0
mul r0.rgb, r2.r, r0
add r0.a, r1.r, r1.r
mad r0.rgb, r0.a, t7, -r0
add r4.a, -r1.r, c1.g
mov r1.r, t1.b
mov r1.g, t1.a
texld r3, r0, s2
texld r2, r1, s5
texld r1, t3, s1
texld r0, t0, s0
mul r2.a, r4.a, r4.a
mul r2.a, r2.a, r2.a
mul r2.a, r4.a, r2.a
mad r2.a, r2.a, c4.r, c4.g
mul r3, r3, r2.a
mul r3.rgb, r3.a, r3
mul r2.rgb, r2, r3
mul r2.rgb, r2, c0
mul r2.rgb, r2, c1.r
mul r1.rgb, r1.a, r1
mul r1.rgb, r1, c1.r
mul r0.rgb, r0, v0
mad r0.rgb, r1, r0, r2
mul r0.a, r0.a, v0.a
dp3 r0.r, r0, c3
mul r0.a, r0.a, r0.r
mov r1.gb, c2.brga
mul r0.a, r0.a, c1.b
mov r0.rg, r1.gbra
mov r0.b, r1.g
mov oC0, r0
