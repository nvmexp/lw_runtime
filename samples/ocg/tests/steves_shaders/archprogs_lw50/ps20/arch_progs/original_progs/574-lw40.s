;; Id: 574   pixel count: 1336726 lw40 ppc: 1.77777777778
ps_2_0
def c0, 2.000000, -1.000000, 1.000000, 0.000000
dcl t0.rg
dcl t1.rgb
dcl t2
dcl t3.r
dcl_2d s2
dcl_2d s3
dcl_2d s4
dcl_lwbe s6
texld r2, t0, s3
rcp r0.a, t3.r
mul r1, r0.a, t2
mad r3.rgb, c0.r, r2, c0.g
mul r0.ba, r2.a, r3.abgr
mul r0.rg, r2.a, r3
mad r0, r0, c5, r1
mov r1.rg, r0.abgr
texld r0, r0, s4
texld r1, r1, s2
texld r2, t1, s6
mul r1, r1, c1
mad r0, r0, c4, -r1
mad r2.rgb, c0.r, r2, c0.g
dp3_sat r2.r, r2, r3
add r2.a, -r2.r, c0.b
mul r3.a, r2.a, r2.a
mul r3.a, r3.a, r3.a
mul r2.a, r2.a, r3.a
mad r0, r2.a, r0, r1
mov oC0, r0
