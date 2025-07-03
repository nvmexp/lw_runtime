vs_1_1
dcl_position0 v0
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v7
mad r2, v2, c[3].z, c[3].w
mov a0.x, r2.z
dp4 r0.x, v0, c[a0.x+0]
dp4 r0.y, v0, c[a0.x+1]
dp4 r0.z, v0, c[a0.x+2]
mov r0.w, c[0].y
dp3 r1.x, v3, c[a0.x+0]
dp3 r1.y, v3, c[a0.x+1]
dp3 r1.z, v3, c[a0.x+2]
dp4 r2.x, r0, c[8]
dp4 r2.y, r0, c[9]
dp4 r2.z, r0, c[10]
dp4 r2.w, r0, c[11]
mov oPos, r2
mad oFog, -r2.z, c[16].w, c[16].x
mul r3.xyz, r1.xyz, r1.xyz
slt r4.xyz, r1.xyz, c[0].x
mov a0.x, r4.x
mul r2.xyz, r3.x, c[a0.x+21]
mov a0.x, r4.y
mad r2.xyz, r3.y, c[a0.x+23], r2
mov a0.x, r4.z
mad r2.xyz, r3.z, c[a0.x+25], r2
mov a0.x, c[3].x
add r3, c[a0.x+2], -r0
dp3 r4, r3, r3
rsq r5, r4.x
mul r3, r3, r5.x
dst r6, r4, r5
dp3 r4, r6, c[a0.x+4]
rcp r3.w, r4.w
dp3 r4.x, r1, r3
dp3 r5.x, c[a0.x+1], -r3
add r4.y, r5.x, -c[a0.x+3].z
mul r4.y, r4.y, c[a0.x+3].w
mov r4.w, c[a0.x+3].x
lit r5, r4
min r5.z, r5.z, c[0].y
mul r4, c[a0.x+0], r3.w
mul r6.x, r5.y, r5.z
mad r2.xyz, r4, r6.x, r2
mov a0.x, c[3].y
add r3, c[a0.x+2], -r0
dp3 r4, r3, r3
rsq r5, r4.x
mul r3, r3, r5.x
dst r6, r4, r5
dp3 r4, r6, c[a0.x+4]
rcp r3.w, r4.w
dp3 r4.x, r1, r3
dp3 r5.x, c[a0.x+1], -r3
add r4.y, r5.x, -c[a0.x+3].z
mul r4.y, r4.y, c[a0.x+3].w
mov r4.w, c[a0.x+3].x
lit r5, r4
min r5.z, r5.z, c[0].y
mul r4, c[a0.x+0], r3.w
mul r6.x, r5.y, r5.z
mad r2.xyz, r4, r6.x, r2
mul r2.xyz, c[90].w, r2
dp3 r3, r1, c[90]
max r3, c[0].x, r3
mul r2.xyz, r3, r2
mov r2.w, c[1].x
lit r4.z, r2.zzzw
lit r5.z, r2.yyyw
mov r4.y, r5.z
lit r5.z, r2.xxxw
mov r4.x, r5.z
mul oD0.xyz, r4.xyz, c[1].w
mov oD0.w, c[0].y
mov oT0, v7
