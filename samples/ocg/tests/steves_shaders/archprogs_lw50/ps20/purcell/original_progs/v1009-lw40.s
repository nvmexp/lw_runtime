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
dp3 r1.x, c[a0.x+0], v1
dp3 r1.y, c[a0.x+1], v1
dp3 r1.z, c[a0.x+2], v1
dp3 r2.x, c[a0.x+0], v2
dp3 r2.y, c[a0.x+1], v2
dp3 r2.z, c[a0.x+2], v2
dp3 r3.x, c[a0.x+0], v3
dp3 r3.y, c[a0.x+1], v3
dp3 r3.z, c[a0.x+2], v3
mov a0.x, v5.w
dp4 r4.x, c[a0.x+0], v0
dp4 r4.y, c[a0.x+1], v0
dp4 r4.z, c[a0.x+2], v0
dp3 r5.x, c[a0.x+0], v1
dp3 r5.y, c[a0.x+1], v1
dp3 r5.z, c[a0.x+2], v1
dp3 r6.x, c[a0.x+0], v2
dp3 r6.y, c[a0.x+1], v2
dp3 r6.z, c[a0.x+2], v2
dp3 r7.x, c[a0.x+0], v3
dp3 r7.y, c[a0.x+1], v3
dp3 r7.z, c[a0.x+2], v3
mul r0.xyz, r0.xyz, v5.x
mad r0.xyz, r4.xyz, v5.z, r0.xyz
mov r0.w, v0.w
m4x4 oPos, r0, c[0]
mul r1.xyz, r1.xyz, v5.x
mad r1.xyz, r5.xyz, v5.z, r1.xyz
dp3 r1.w, r1.xyz, r1.xyz
rsq r1.w, r1.w
mul r1.xyz, r1.xyz, r1.w
mul r2.xyz, r2.xyz, v5.x
mad r2.xyz, r6.xyz, v5.z, r2.xyz
dp3 r2.w, r2.xyz, r2.xyz
rsq r2.w, r2.w
mul r2.xyz, r2.xyz, r2.w
mul r3.xyz, r3.xyz, v5.x
mad r3.xyz, r7.xyz, v5.z, r3.xyz
dp3 r3.w, r3.xyz, r3.xyz
rsq r3.w, r3.w
mul r3.xyz, r3.xyz, r3.w
mov oT0.xy, v4
mov oT1.xy, v4
add r0.xyz, c[11].xyz, -v0.xyz
dp3 r6.x, r0.xyz, r0.xyz
rsq r6.x, r6.x
mul r0.xyz, r0.xyz, r6.x
dp3 r6.x, r0.xyz, r1.xyz
dp3 r6.y, r0.xyz, r2.xyz
dp3 r6.z, r0.xyz, r3.xyz
add r7.xyz, c[12].xyz, -v0.xyz
dp3 r5.x, r7.xyz, r7.xyz
rsq r5.x, r5.x
mul r7.xyz, r7.xyz, r5.x
dp3 r4.x, r7.xyz, r1.xyz
dp3 r4.y, r7.xyz, r2.xyz
dp3 r4.z, r7.xyz, r3.xyz
add r4.xyz, r4.xyz, r6.xyz
dp3 r5.x, r4.xyz, r4.xyz
rsq r1.x, r5.x
mul r4.xyz, r4.xyz, r1.x
mov oT2.xyz, r6.xyz
mov oT3.xyz, r4.xyz
dp3 r7.xy, r0.xyz, r3.xyz
mov r7.w, c[11].w
lit r1, r7
mov r4.xyz, c[13].xyz
mad oD0.xyz, r1.z, c[14].xyz, r4.xyz
