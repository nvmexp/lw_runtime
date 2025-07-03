vs_1_1
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
m4x4 oPos, v0, c[0]
add r2.xyz, c[7].xyz, -v0.xyz
dp3 r2.w, r2.xyz, r2.xyz
rsq r2.w, r2.w
mul r2.xyz, r2.xyz, r2.w
dp3 r3.x, r2, v1
add r3.x, r3.x, r3.x
mad r3.xyz, r3.x, v1.xyz, -r2.xyz
dp3 oT0.x, r3, c[4]
dp3 oT0.y, r3, c[5]
dp3 oT0.z, r3, c[6]
dp3 oT1.xyz, r2.xyz, v1.xyz
mov oT1.w, v0.w
