vs_2_0
def c[3], 765.005859, 1.000000, 0.816497, -0.408248
def c[4], 0.707107, -0.707107, 0.577350, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_tangent0 v6
add r0.w, -v1.x, c[3].y
add r3.w, r0.w, -v1.y
mul r1.xyz, v2.zyxw, c[3].x
mova a0.xyz, r1
mul r0, v1.y, c[a0.y+42]
mad r0, c[a0.x+42], v1.x, r0
mad r0, c[a0.z+42], r3.w, r0
dp4 r3.x, v0, r0
mul r1, v1.y, c[a0.y+43]
mad r1, c[a0.x+43], v1.x, r1
mad r2, c[a0.z+43], r3.w, r1
dp4 r3.y, v0, r2
mul r1, v1.y, c[a0.y+44]
mad r1, c[a0.x+44], v1.x, r1
mad r1, c[a0.z+44], r3.w, r1
dp4 r3.z, v0, r1
mov r3.w, c[3].y
dp4 oPos.x, r3, c[8]
dp4 oPos.y, r3, c[9]
dp4 oPos.w, r3, c[11]
dp4 r6.y, r3, c[10]
mad oFog, -r6.y, c[16].w, c[16].x
mov oPos.z, r6.y
add oT3.xyz, -r3, c[2]
add r4.xyz, -r3, c[29]
dp3 r6.z, r4, r4
rsq r0.w, r6.z
mul r3.xyz, r4, r0.w
dst r6.xy, r6.z, r0.w
dp3 r6.x, c[31], r6
rcp r0.w, r6.x
dp3 r6.x, -r3, c[28]
add r1.w, r6.x, -c[30].z
mul r1.w, r1.w, c[30].w
max r2.w, r1.w, c[0].x
pow r1.w, r2.w, c[30].x
min r1.w, r1.w, c[0].y
mul r6.xyz, r1.w, c[27]
mul r5.xyz, r0.w, r6
dp3 r8.y, v6, r2
dp3 r6.y, v3, r2
dp3 r8.z, v6, r1
dp3 r6.z, v3, r1
mul r1.xy, r8.yzzw, c[4]
add r1.x, r1.y, r1.x
dp3 r8.x, v6, r0
dp3 r6.x, v3, r0
mul r4.xyz, r8.yzxw, r6.zxyw
mad r4.xyz, r6.yzxw, r8.zxyw, -r4
mul r2.xyz, r4, v6.w
mul r4.xy, r2.yzzw, c[4]
add r1.y, r4.y, r4.x
mul r4.xy, r6.yzzw, c[4]
add r1.z, r4.y, r4.x
dp3 r0.y, r1, r3
dp3 r7.x, c[3].zw, r8
dp3 r7.z, c[3].zw, r6
dp3 r7.y, c[3].zw, r2
dp3 r0.x, r7, r3
dp3 r4.x, c[4].z, r8
mov oT4.xyz, r8
dp3 r4.z, c[4].z, r6
dp3 r4.y, c[4].z, r2
mov oT5.xyz, r2
dp3 r0.z, r4, r3
dp3 r0.w, r6, r3
max r0, r0, c[4].w
slt r2.xyz, r7, c[4].w
mul r7.xyz, r7, r7
mova a0.xyz, r2
mul r2.xyz, r7.y, c[a0.y+23]
mad r2.xyz, r7.x, c[a0.x+21], r2
mad r7.xyz, r7.z, c[a0.z+25], r2
mad r3.xyz, r5, r0.x, r7
slt r7.xyz, r1, c[4].w
mul r1.xyz, r1, r1
mova a0.xyz, r7
mul r7.xyz, r1.y, c[a0.y+23]
mad r7.xyz, r1.x, c[a0.x+21], r7
mad r1.xyz, r1.z, c[a0.z+25], r7
mad r2.xyz, r5, r0.y, r1
add r1.xyz, r3, r2
slt r7.xyz, r4, c[4].w
mul r4.xyz, r4, r4
mova a0.xyz, r7
mul r7.xyz, r4.y, c[a0.y+23]
mad r7.xyz, r4.x, c[a0.x+21], r7
mad r4.xyz, r4.z, c[a0.z+25], r7
mad r0.xyz, r5, r0.z, r4
add r1.xyz, r1, r0
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
mad r1.xyz, r5, r0.w, r1
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
dp4 oT0.x, v4, c[90]
dp4 oT0.y, v4, c[91]
dp4 oT1.x, v4, c[92]
dp4 oT1.y, v4, c[93]
dp4 oT2.x, v4, c[94]
dp4 oT2.y, v4, c[95]
mov oD0.w, c[4].w
