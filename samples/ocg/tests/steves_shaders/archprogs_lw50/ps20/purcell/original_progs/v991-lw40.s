vs_2_0
def c[3], 765.005859, 1.000000, 0.816497, -0.408248
def c[4], 0.707107, -0.707107, 0.577350, 0.000000
dcl_position0 v0
dcl_blendindices0 v1
dcl_normal0 v2
dcl_texcoord0 v3
dcl_tangent0 v5
mov r0.w, c[3].y
mul r1.w, v1.z, c[3].x
mova a0.w, r1.w
dp4 r0.x, v0, c[a0.w+42]
dp4 r0.y, v0, c[a0.w+43]
dp4 r0.z, v0, c[a0.w+44]
dp4 oPos.x, r0, c[8]
dp4 oPos.y, r0, c[9]
dp4 oPos.w, r0, c[11]
dp4 r1.y, r0, c[10]
mad oFog, -r1.y, c[16].w, c[16].x
mov oPos.z, r1.y
add oT3.xyz, -r0, c[2]
add r1.xyz, -r0, c[29]
dp3 r6.z, r1, r1
rsq r0.w, r6.z
dst r6.xy, r6.z, r0.w
mul r1.xyz, r1, r0.w
dp3 r6.x, c[31], r6
rcp r0.w, r6.x
dp3 r6.x, -r1, c[28]
add r1.w, r6.x, -c[30].z
mul r1.w, r1.w, c[30].w
max r2.w, r1.w, c[0].x
pow r1.w, r2.w, c[30].x
min r1.w, r1.w, c[0].y
mul r6.xyz, r1.w, c[27]
mul r5.xyz, r0.w, r6
dp3 r3.y, v5, c[a0.w+43]
dp3 r3.z, v5, c[a0.w+44]
mul r6.xy, r3.yzzw, c[4]
add r4.x, r6.y, r6.x
dp3 r3.x, v5, c[a0.w+42]
dp3 r6.x, v2, c[a0.w+42]
dp3 r6.y, v2, c[a0.w+43]
dp3 r6.z, v2, c[a0.w+44]
mul r0.xyz, r3.yzxw, r6.zxyw
mad r0.xyz, r6.yzxw, r3.zxyw, -r0
mul r8.xyz, r0, v5.w
mul r0.xy, r8.yzzw, c[4]
add r4.y, r0.y, r0.x
mul r0.xy, r6.yzzw, c[4]
add r4.z, r0.y, r0.x
dp3 r0.y, r4, r1
dp3 r2.x, c[3].zw, r3
dp3 r2.y, c[3].zw, r8
dp3 r2.z, c[3].zw, r6
dp3 r0.x, r2, r1
dp3 r7.x, c[4].z, r3
mov oT4.xyz, r3
dp3 r7.y, c[4].z, r8
mov oT5.xyz, r8
dp3 r7.z, c[4].z, r6
dp3 r0.z, r7, r1
dp3 r0.w, r6, r1
max r1, r0, c[4].w
slt r0.xyz, r2, c[4].w
mova a0.xyz, r0
mul r8.xyz, r2, r2
dp3 r0.x, r2, -c[33]
mul r2.xyz, r8.y, c[a0.y+23]
mad r2.xyz, r8.x, c[a0.x+21], r2
mad r8.xyz, r8.z, c[a0.z+25], r2
mad r8.xyz, r5, r1.x, r8
dp3 r0.y, r4, -c[33]
dp3 r0.z, r7, -c[33]
dp3 r0.w, r6, -c[33]
max r0, r0, c[4].w
mad r3.xyz, r0.x, c[32], r8
slt r8.xyz, r4, c[4].w
mul r4.xyz, r4, r4
mova a0.xyz, r8
mul r8.xyz, r4.y, c[a0.y+23]
mad r8.xyz, r4.x, c[a0.x+21], r8
mad r4.xyz, r4.z, c[a0.z+25], r8
mad r4.xyz, r5, r1.y, r4
mad r2.xyz, r0.y, c[32], r4
add r4.xyz, r3, r2
slt r8.xyz, r7, c[4].w
mul r7.xyz, r7, r7
mova a0.xyz, r8
mul r8.xyz, r7.y, c[a0.y+23]
mad r8.xyz, r7.x, c[a0.x+21], r8
mad r7.xyz, r7.z, c[a0.z+25], r8
mad r1.xyz, r5, r1.z, r7
mad r0.xyz, r0.z, c[32], r1
add r1.xyz, r4, r0
mul r1.xyz, r1, c[4].z
rcp r4.x, r1.x
rcp r4.y, r1.y
rcp r4.z, r1.z
slt r1.xyz, r6, c[4].w
mova a0.xyz, r1
mul r1.xyz, r6, r6
mov oT6.xyz, r6
mul r6.xyz, r1.y, c[a0.y+23]
mad r6.xyz, r1.x, c[a0.x+21], r6
mad r1.xyz, r1.z, c[a0.z+25], r6
mad r1.xyz, r5, r1.w, r1
mad r1.xyz, r0.w, c[32], r1
log r1.x, r1.x
log r1.y, r1.y
log r1.z, r1.z
mul r1.xyz, r1, c[1].x
exp r1.x, r1.x
exp r1.y, r1.y
exp r1.z, r1.z
mul r1.xyz, r1, c[1].w
mul r1.xyz, r4, r1
mul oD0.xyz, r3, r1
mul oD1.xyz, r2, r1
mul oT7.xyz, r0, r1
dp4 oT0.x, v3, c[90]
dp4 oT0.y, v3, c[91]
dp4 oT1.x, v3, c[92]
dp4 oT1.y, v3, c[93]
dp4 oT2.x, v3, c[94]
dp4 oT2.y, v3, c[95]
mov oD0.w, c[4].w
