vs_1_1
dcl_position0 v0
dcl_normal0 v1
dcl_texcoord0 v2
dcl_blendweight0 v3
mov a0.x, v3.y
dp4 r0.x, c[a0.x+0], v0
dp4 r0.y, c[a0.x+1], v0
dp4 r0.z, c[a0.x+2], v0
dp3 r2.x, c[a0.x+0], v1
dp3 r2.y, c[a0.x+1], v1
dp3 r2.z, c[a0.x+2], v1
mov a0.x, v3.w
dp4 r1.x, c[a0.x+0], v0
dp4 r1.y, c[a0.x+1], v0
dp4 r1.z, c[a0.x+2], v0
dp3 r3.x, c[a0.x+0], v1
dp3 r3.y, c[a0.x+1], v1
dp3 r3.z, c[a0.x+2], v1
mul r0.xyz, r0.xyz, v3.x
mad r0.xyz, r1.xyz, v3.z, r0.xyz
mov r0.w, v0.w
m4x4 oPos, r0, c[0]
mul r2.xyz, r2.xyz, v3.x
mad r1.xyz, r3.xyz, v3.z, r2.xyz
dp3 r1.w, r1.xyz, r1.xyz
rsq r1.w, r1.w
mul r1.xyz, r1.xyz, r1.w
mov oT0.xy, v2.xy
add r2.xyz, c[4], -r0
dp3 r2.w, r2.xyz, r2.xyz
rsq r2.w, r2.w
mul r2.xyz, r2.xyz, r2.w
dp3 r2.x, r2.xyz, r1.xyz
max r2.x, r2.x, c[4].w
mul r2, r2.x, c[5]
add oD0, r2, c[6]
