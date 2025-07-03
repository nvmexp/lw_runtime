vs_1_1
dcl_position0 v0
dcl_texcoord0 v7
dp4 r0.x, v0, c[42]
dp4 r0.y, v0, c[43]
dp4 r0.z, v0, c[44]
mov r0.w, c[0].y
dp4 r1.x, r0, c[8]
dp4 r1.y, r0, c[9]
dp4 r1.z, r0, c[10]
dp4 r1.w, r0, c[11]
mov oPos, r1
add r2, r0, -c[90]
dp3 r3, r2, c[91]
mul r3, r3, c[0].w
mad r2, -r3, c[91], r2
dp3 r2.w, r2, r2
rsq r2.w, r2.w
mul r2, r2, r2.w
mul r5.xyz, r2.xyz, r2.xyz
slt r6.xyz, r2.xyz, c[0].x
mov a0.x, r6.x
mul r3.xyz, r5.x, c[a0.x+21]
mov a0.x, r6.y
mad r3.xyz, r5.y, c[a0.x+23], r3
mov a0.x, r6.z
mad r3.xyz, r5.z, c[a0.x+25], r3
mov r3.w, c[1].x
lit r4.z, r3.zzzw
lit r5.z, r3.yyyw
mov r4.y, r5.z
lit r5.z, r3.xxxw
mov r4.x, r5.z
mul r4.xyz, r4.xyz, c[1].w
max r4.w, r4.x, r4.y
max r4.w, r4.w, r4.z
max r4.w, r4.w, c[0].y
rcp r4.w, r4.w
mul oD0.xyz, r4.w, r4.xyz
mov oD0.w, c[0].y
mad oFog, -r1.z, c[16].w, c[16].x
mov oT0, v7
dp4 oT1.x, c[92], r0
dp4 oT1.y, c[93], r0
dp4 oT2.x, c[94], r0
dp4 oT2.y, c[95], r0
