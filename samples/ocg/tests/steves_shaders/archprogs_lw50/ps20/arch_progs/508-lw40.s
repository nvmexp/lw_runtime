//;; Id: 508   pixel count: 28953496 lw40 ppc: 2.66666666667
ps_2_0
def c0, 0.200000, 1.000000, 0.000000, 0.000000
dcl_pp t0.rg
dcl_pp t1.rg
dcl_pp t2.rg
dcl_pp t3.rg
dcl_pp t4.rg
dcl v0
dcl_2d s0
texld r3, t0, s0
texld r4, t1, s0
texld r2, t2, s0
texld r1, t3, s0
texld r0, t4, s0
add r3.a, r3.a, r4.a
add r2.a, r2.a, r3.a
add r1.a, r1.a, r2.a
add r0.a, r0.a, r1.a
mad_sat r0.a, r0.a, c0.r, -v0.a
mad r0, r0.a, c1, -r0.a
add r0, r0, c0.g
mov oC0, r0
