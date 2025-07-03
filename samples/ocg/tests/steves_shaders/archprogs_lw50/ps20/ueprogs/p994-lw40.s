ps_2_0
def c2, 0.050000, -0.025000, 2.000000, -1.000000
def c3, 0.000000, 15.000000, 1.000000, 0.000000
dcl t0.rg
dcl t1.rg
dcl t4.rgb
dcl t5.rgb
dcl t6.rgb
dcl t7
dcl_2d s0
dcl_2d s1
dcl_2d s2
dcl_2d s3
dcl_2d s4
texld r0, t1, s0
dp3 r0.r, t6, t6
rsq r5.a, r0.r
mad r1.a, r0.a, c2.r, c2.g
mul r5.rgb, r5.a, t6
mad r0.a, r0.a, c2.r, c2.g
mad r2.rg, r5, r1.a, t1
mad r1.rg, r5, r0.a, t1
rcp r0.a, t7.a
mul r0.rg, r0.a, t7
mad r0.rg, r0, c0, c0.abgr
texld r4, r2, s0
texld r3, t1, s2
texld r2, r1, s1
texld r0, r0, s3
texld r1, t0, s4
mad r6.rgb, c2.b, r4, c2.a
nrm r4.rgb, r6
dp3 r5.r, r4, r5
mul r5.rgb, r4, r5.r
add r5.rgb, r5, r5
mad r6.rgb, t6, -r5.a, r5
nrm r5.rgb, t4
dp3 r6.r, r6, r5
max r1.a, r6.r, c3.r
dp3 r4.r, r4, r5
pow r0.a, r1.a, c3.g
max r1.a, r4.r, c3.r
dp3 r4.r, t5, t5
add r2.a, -r4.r, c3.b
mul r3.rgb, r3, r0.a
max r0.a, r2.a, c3.r
mad r2.rgb, r2, r1.a, r3
mul r0.a, r0.a, r0.a
mul r2.rgb, r2, r0.a
mul r0.rgb, r0, r1.r
mul r0.rgb, r2, r0
mul r0.rgb, r0, c1
mov r0.a, c3.r
mov oC0, r0
