ps_2_0
def c2, 1.000000, 0.000000, 0.000000, 0.000000
def c3, 0.000000, 0.000000, 2.000000, 15.000000
dcl t0.rg
dcl t1.rg
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl t7
dcl_2d s0
dcl_2d s1
dcl_2d s2
texld r0, t1, s1
nrm r1.rgb, t6
mad r2.rgb, r1.b, c3, -r1
nrm r1.rgb, t4
dp3 r2.r, r2, r1
max r2.a, r1.b, c3.r
max r1.a, r2.r, c3.r
pow r0.a, r1.a, c3.a
dp3 r1.r, t5, t5
add r1.a, -r1.r, c2.r
mul r1.rgb, r0, r0.a
max r0.a, r1.a, c3.r
mad r0.rgb, r0, r2.a, r1
mul r0.a, r0.a, r0.a
mul r2.rgb, r0, r0.a
rcp r0.a, t7.a
mul r0.rg, r0.a, t7
mad r0.rg, r0, c0, c0.abgr
texld r0, r0, s0
texld r1, t0, s2
mul r0.rgb, r0, r1.r
mul r0.rgb, r2, r0
mul r0.rgb, r0, c1
mov r0.a, c3.r
mov oC0, r0
