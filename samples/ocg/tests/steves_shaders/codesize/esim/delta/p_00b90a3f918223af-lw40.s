ps_2_0
dcl t0.rg
dcl_2d s0
dcl_2d s1
texld_pp r1, t0, s1
texld_pp r0, t0, s0
mad_sat_pp r1.xyz, r1, c0.x, c0.y
log_pp r1.x, r1.x
log_pp r1.y, r1.y
log_pp r1.z, r1.z
mul_pp r1.xyz, r1, c0.z
exp_pp r1.x, r1.x
exp_pp r1.y, r1.y
exp_pp r1.z, r1.z
mul_pp r1.xyz, r1, c4
dp3_pp r2.x, r0, c1
mad_pp r2.xyz, r2.x, c2, -r0
mad_pp r0.xyz, c2.w, r2, r0
mad_pp r2.xyz, r0, c3, r1
mul_pp r0.xyz, r0, c3
mad_pp r0.xyz, r0, -r1, r2
mad_sat_pp r0.xyz, r0, c5.x, c5.y
log_pp r0.x, r0.x
log_pp r0.y, r0.y
log_pp r0.z, r0.z
mul_pp r0.xyz, r0, c5.z
exp_pp r0.zw, r0.z
exp_pp r0.x, r0.x
exp_pp r0.y, r0.y
mov_pp oC0, r0
