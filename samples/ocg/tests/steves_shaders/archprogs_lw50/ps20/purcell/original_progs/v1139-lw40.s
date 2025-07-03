vs_1_1
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v7
mad r2, v2, c[3].z, c[3].w
mov a0.x, r2.z
mul r3, v1.x, c[a0.x+0]
mul r4, v1.x, c[a0.x+1]
mul r5, v1.x, c[a0.x+2]
mov a0.x, r2.y
mad r3, v1.y, c[a0.x+0], r3
mad r4, v1.y, c[a0.x+1], r4
mad r5, v1.y, c[a0.x+2], r5
add r2.w, v1.x, v1.y
add r2.w, c[0].y, -r2.w
mov a0.x, r2.x
mad r3, r2.w, c[a0.x+0], r3
mad r4, r2.w, c[a0.x+1], r4
mad r5, r2.w, c[a0.x+2], r5
dp4 r0.x, v0, r3
dp4 r0.y, v0, r4
dp4 r0.z, v0, r5
mov r0.w, c[0].y
dp3 r1.x, v3, r3
dp3 r1.y, v3, r4
dp3 r1.z, v3, r5
dp4 r2.x, r0, c[8]
dp4 r2.y, r0, c[9]
dp4 r2.z, r0, c[10]
dp4 r2.w, r0, c[11]
mov oPos, r2
mad oFog, -r2.z, c[16].w, c[16].x
dp3 r2.x, r1, c[90]
dp3 r2.y, r1, c[91]
dp3 r2.z, r1, c[92]
dp3 r2.w, r2, r2
rsq r2.w, r2.w
mul r2, r2, r2.w
dp4 r1.x, r0, c[90]
dp4 r1.y, r0, c[91]
dp4 r1.z, r0, c[92]
mov r0.xyz, -r1
dp3 r0.w, r0, r0
rsq r0.w, r0.w
mul r0, r0, r0.w
dp3 r2.x, r0, r2
add r2.x, r2.x, c[95]
mov oT0, r2
mov oT1, v7
