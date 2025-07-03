vs_2_0
def c[1], 765.005859, 0.816497, -0.408248, 0.577350
def c[3], 0.707107, -0.707107, 0.000000, 0.000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_normal0 v2
dcl_texcoord0 v3
dcl_tangent0 v5
mul r0.w, v1.z, c[1].x
mova a0.w, r0.w
mov r0.w, c[0].y
dp4 r0.x, v0, c[a0.w+42]
dp4 r0.y, v0, c[a0.w+43]
dp4 r0.z, v0, c[a0.w+44]
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 r1.y, r0, c[10]
dp4 oPos.w, r0, c[11]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
add r2.xyz, -r0, c[29]
add oT3.xyz, -r0, c[2]
dp3 r1.z, r2, r2
rsq r0.w, r1.z
add r8.xyz, -r0, c[34]
dst r1.xy, r1.z, r0.w
mul r9.xyz, r2, r0.w
dp3 r0.x, c[31], r1
rcp r0.w, r0.x
dp3 r0.x, -r9, c[28]
dp3 r1.x, v2, c[a0.w+42]
dp3 r6.x, v5, c[a0.w+42]
dp3 r1.y, v2, c[a0.w+43]
dp3 r1.z, v2, c[a0.w+44]
dp3 r6.y, v5, c[a0.w+43]
dp3 r6.z, v5, c[a0.w+44]
add r1.w, r0.x, -c[30].z
mul r0.xyz, r1.zxyw, r6.yzxw
mul r1.w, r1.w, c[30].w
mad r0.xyz, r1.yzxw, r6.zxyw, -r0
max r2.w, r1.w, c[0].x
mul r0.xyz, r0, v5.w
pow r1.w, r2.w, c[30].x
dp3 r2.y, c[1].yzzw, r0
dp3 r2.x, c[1].yzzw, r6
dp3 r2.z, c[1].yzzw, r1
min r1.w, r1.w, c[0].y
slt r3.xyz, r2, c[0].x
mova a0.xyz, r3
mul r4.xyz, r2, r2
mul r3.xyz, r1.w, c[27]
mul r5.xyz, r4.y, c[a0.y+23]
mul r3.xyz, r0.w, r3
mad r5.xyz, r4.x, c[a0.x+21], r5
mad r7.xyz, r4.z, c[a0.z+25], r5
dp3 r5.x, c[1].w, r6
mov oT4.x, r6.x
dp3 r5.z, c[1].w, r1
mov oT4.z, r1.x
dp3 r5.y, c[1].w, r0
dp3 r4.z, r5, r9
mul r4.xy, r6.yzzw, c[3]
mov oT5.x, r6.y
mov oT6.x, r6.z
add r6.x, r4.y, r4.x
mul r4.xy, r1.yzzw, c[3]
mov oT5.z, r1.y
mov oT6.z, r1.z
add r6.z, r4.y, r4.x
dp3 r1.z, r8, r8
mul r1.xy, r0.yzzw, c[3]
rsq r1.w, r1.z
add r6.y, r1.y, r1.x
mul r8.xyz, r8, r1.w
dp3 r4.y, r6, r9
dp3 r1.x, -r8, c[33]
dp3 r4.x, r2, r9
add r0.w, r1.x, -c[35].z
max r4.xyz, r4, c[0].x
mul r0.w, r0.w, c[35].w
dst r1.xy, r1.z, r1.w
max r1.w, r0.w, c[0].x
pow r0.w, r1.w, c[35].x
dp3 r1.x, c[36], r1
min r1.w, r0.w, c[0].y
rcp r0.w, r1.x
mul r1.xyz, r1.w, c[32]
mad r7.xyz, r3, r4.x, r7
mul r1.xyz, r0.w, r1
dp3 r2.x, r2, r8
dp3 r2.y, r6, r8
dp3 r2.z, r5, r8
slt r8.xyz, r6, c[0].x
mul r6.xyz, r6, r6
mova a0.xyw, r8.xyz
max r2.xyz, r2, c[0].x
mul r8.xyz, r6.y, c[a0.y+23]
mad r8.xyz, r6.x, c[a0.x+21], r8
slt r9.xyz, r5, c[0].x
mul r5.xyz, r5, r5
mova a0.xyz, r9
mad oD0.xyz, r1, r2.x, r7
mul r7.xyz, r5.y, c[a0.y+23]
mad r6.xyz, r6.z, c[a0.w+25], r8
mad r7.xyz, r5.x, c[a0.x+21], r7
mad r6.xyz, r3, r4.y, r6
mad r5.xyz, r5.z, c[a0.z+25], r7
mad oD1.xyz, r1, r2.y, r6
mad r3.xyz, r3, r4.z, r5
mad oT7.xyz, r1, r2.z, r3
mul r1.xy, v3, c[90]
add r0.w, r1.y, r1.x
mul r1.xy, v3, c[91]
mov oT0.x, r0.w
add r1.z, r1.y, r1.x
mov oT0.y, r1.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r1.z
mov oT2.y, r1.z
mov oT4.y, r0.x
mov oT5.y, r0.y
mov oT6.y, r0.z
mov oD0.w, c[0].x
mov oD1.w, c[0].x
