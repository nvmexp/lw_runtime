ps_3_0

def c10, 0.00000000, 1.00000000, -0.01000000, 0.25000000 ; 0x000000 0x3f800000 0xbc23d70a 0x3e800000
def c11, 1.44269502, 0.00000000, 0.00000000, 0.00000000 ; 0x3fb8aa3b 0x000000 0x000000 0x000000
dcl_texcoord2 v0.rgb
dcl_texcoord3 v1.rg
dcl_texcoord4 v2.rgb
dcl_volume s0
add r1.xyz, v2, -c8
nrm r0.xyz, r1
add r0.w, -v2.y, c5.x
rcp r1.x, r0.y
cmp r1.w, r0.w, c10.x, c10.y
mul r1.y, r0.w, r1.x
add r1.z, r0_abs.y, c10.z
mov r0.w, c8.y
add r0.w, -r0.w, c5.x
cmp r1.y, r1.z, r1.y, c10.x
mul r1.x, r1.x, r0.w
cmp r0.w, r0.w, c10.x, c10.y
cmp r1.z, r1.z, r1.x, c10.x
mul r1.w, r1.w, r1.y
mul r0.w, r0.w, r1.z
mad r1.xyz, r1.w, -r0, v2
mad r0.xyz, r0.w, r0, c8
add r1.xyz, r1, -r0
dp3 r0.w, r1, r1
rsq r0.w, r0.w
mul r1.xyz, r1, c10.w
rcp r0.w, r0.w
mul_pp r0.w, r0.w, c10.w
add r3.xyz, r0, r1
rcp r4.w, c6.x
mov r0.z, c9.x
mul r2.xyz, r0.z, c7
mul_pp r2.w, r0.w, -c4.x
mad r0.xyz, r3, r4.w, r2
texld_pp r0, r0, s0
rcp r3.w, c5.x
mul_sat_pp r0.w, r3.y, r3.w
add_pp r0.z, -r0.w, c10.y
mul_pp r0.w, r0.x, r0.x
add r3.xyz, r1, r3
mul_pp r0.w, r0.z, r0.w
mul_pp r4.z, r0.x, r0.w
mul_sat_pp r0.w, r3.w, r3.y
add_pp r1.w, -r0.w, c10.y
mad r0.xyz, r3, r4.w, r2
texld_pp r0, r0, s0
mul_pp r0.w, r0.x, r0.x
mul_pp r0.z, r2.w, r4.z
mul_pp r0.w, r1.w, r0.w
mul_pp r0.z, r0.z, c11.x
mul_pp r0.w, r0.x, r0.w
exp_pp r1.w, r0.z
mul_pp r0.w, r2.w, r0.w
add r3.xyz, r1, r3
mul_pp r0.w, r0.w, c11.x
exp_pp r4.y, r0.w
mul_sat_pp r0.w, r3.w, r3.y
add_pp r4.z, -r0.w, c10.y
mad r0.xyz, r3, r4.w, r2
texld_pp r0, r0, s0
mul_pp r0.w, r0.x, r0.x
mul_pp r1.w, r1.w, r4.y
mul_pp r0.w, r4.z, r0.w
add r3.xyz, r1, r3
mul_pp r0.w, r0.x, r0.w
add r1.xyz, r1, r3
mul_pp r0.w, r2.w, r0.w
mul_pp r4.z, r0.w, c11.x
mul_sat_pp r0.w, r3.w, r3.y
mad r0.xyz, r3, r4.w, r2
add_pp r3.z, -r0.w, c10.y
texld_pp r0, r0, s0
mul_pp r0.w, r0.x, r0.x
exp_pp r0.z, r4.z
mul_pp r0.w, r3.z, r0.w
mul_pp r1.w, r1.w, r0.z
mul_pp r0.w, r0.x, r0.w
mad r0.xyz, r1, r4.w, r2
mul_pp r1.z, r2.w, r0.w
mul_sat_pp r0.w, r3.w, r1.y
mul_pp r1.z, r1.z, c11.x
exp_pp r1.y, r1.z
add_pp r1.z, -r0.w, c10.y
texld_pp r0, r0, s0
mul_pp r0.w, r0.x, r0.x
mul_pp r1.w, r1.w, r1.y
mul_pp r0.w, r1.z, r0.w
mul_pp r0.z, r0.x, r0.w
mov_sat r0.w, v1.y
mul_pp r1.z, r2.w, r0.z
mul r0, r0.w, c2.xyz
mul_pp r1.z, r1.z, c11.x
mad r0, c1.xyz, v1.x, r0
exp_pp r1.z, r1.z
mul r0, r0, c0.xyz
mul_pp r1.w, r1.w, r1.z
mad r0, r0, v0.xyz, -c3.xyz
mad oC0, r1.w, r0, c3.xyz
;Auto options added
;#PASM_OPTS: -srcalpha 1 -fog 0 -signtex 0 -texrange 3
