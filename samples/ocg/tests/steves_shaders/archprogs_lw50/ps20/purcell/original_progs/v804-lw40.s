vs_1_1
dcl_position0 v0
dcl_tangent0 v1
dcl_binormal0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_blendweight0 v5
mov a0.x, v5.y
dp4 r0.x, c[a0.x+0], v0
dp4 r0.y, c[a0.x+1], v0
dp4 r0.z, c[a0.x+2], v0
dp3 r1.x, c[a0.x+0], v3
dp3 r1.y, c[a0.x+1], v3
dp3 r1.z, c[a0.x+2], v3
mov a0.x, v5.w
dp4 r2.x, c[a0.x+0], v0
dp4 r2.y, c[a0.x+1], v0
dp4 r2.z, c[a0.x+2], v0
dp3 r3.x, c[a0.x+0], v3
dp3 r3.y, c[a0.x+1], v3
dp3 r3.z, c[a0.x+2], v3
mul r0.xyz, r0.xyz, v5.x
mad r0.xyz, r2.xyz, v5.z, r0.xyz
mov r0.w, v0.w
m4x4 oPos, r0, c[0]
mul r1.xyz, r1.xyz, v5.x
mad r1.xyz, r3.xyz, v5.z, r1.xyz
dp3 r1.w, r1.xyz, r1.xyz
rsq r1.w, r1.w
mul r1.xyz, r1.xyz, r1.w
mov oT0.xy, v4.xy
add r2.xyz, c[13].xyz, -r0.xyz
dp3 r2.w, r2.xyz, r2.xyz
rsq r2.w, r2.w
mul r2.xyz, r2.xyz, r2.w
dp3 r3.x, r2, r1
add r3.x, r3.x, r3.x
mad r3.xyz, r3.x, r1.xyz, -r2.xyz
dp3 oT1.x, r3, c[4]
dp3 oT1.y, r3, c[5]
dp3 oT1.z, r3, c[6]
dp3 oT2.x, r2.xyz, r1.xyz
dp3 oT3.x, r1, c[4]
dp3 oT3.y, r1, c[5]
dp3 oT3.z, r1, c[6]
add r3.xyz, c[7].xyz, -r0.xyz
dp3 r3.w, r3.xyz, r3.xyz
rsq r3.w, r3.w
mul r3.xyz, r3.xyz, r3.w
dp3 r7.x, r3.xyz, r1.xyz
add r7.x, r7.x, c[14].x
mul r7.x, r7.x, c[14].z
max r7.x, r7.x, c[7].w
dp3 r3.xy, r3.xyz, r1.xyz
mov r3.w, c[9].w
lit r3, r3
mul r4.xyz, r3.z, c[10].xyz
add r4.xyz, r4.xyz, c[9].xyz
mad oD0.xyz, r4.xyz, r7.x, c[8].xyz
