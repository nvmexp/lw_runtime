vs_2_0
def c[1], 765.005859, 0.816497, -0.408248, 0.577350
def c[3], 0.707107, -0.707107, 0.000000, 0.000000
dcl_position0 v0
dcl_blendweight0 v1
dcl_blendindices0 v2
dcl_normal0 v3
dcl_texcoord0 v4
dcl_tangent0 v6
mul r0.xy, v2.zyzw, c[1].x
mova a0.xy, r0
mul r0, v1.y, c[a0.y+42]
mad r2, c[a0.x+42], v1.x, r0
mov r3.w, c[0].y
dp4 r3.x, v0, r2
mul r1, v1.y, c[a0.y+43]
mul r0, v1.y, c[a0.y+44]
mad r1, c[a0.x+43], v1.x, r1
mad r0, c[a0.x+44], v1.x, r0
dp4 r3.y, v0, r1
dp4 r3.z, v0, r0
dp4 oPos.x, r3, c[8]
dp4 oPos.y, r3, c[9]
dp4 oPos.w, r3, c[11]
dp4 r5.y, r3, c[10]
mad oFog, -r5.y, c[16].w, c[16].x
add r4.xyz, -r3, c[29]
mov oPos.z, r5.y
dp3 r5.z, r4, r4
add oT3.xyz, -r3, c[2]
rsq r0.w, r5.z
dst r5.xy, r5.z, r0.w
add r8.xyz, -r3, c[34]
dp3 r3.x, c[31], r5
mul r9.xyz, r4, r0.w
rcp r0.w, r3.x
dp3 r6.x, v3, r2
dp3 r10.x, v6, r2
dp3 r6.y, v3, r1
dp3 r10.y, v6, r1
dp3 r6.z, v3, r0
dp3 r10.z, v6, r0
dp3 r1.x, -r9, c[28]
mul r0.xyz, r6.zxyw, r10.yzxw
add r1.w, r1.x, -c[30].z
mad r0.xyz, r6.yzxw, r10.zxyw, -r0
mul r1.w, r1.w, c[30].w
mul r1.xyz, r0, v6.w
max r2.w, r1.w, c[0].x
dp3 r2.y, c[1].yzzw, r1
dp3 r2.x, c[1].yzzw, r10
dp3 r2.z, c[1].yzzw, r6
pow r1.w, r2.w, c[30].x
slt r0.xyz, r2, c[0].x
mova a0.xyz, r0
mul r0.xyz, r2, r2
min r1.w, r1.w, c[0].y
mul r4.xyz, r0.y, c[a0.y+23]
mul r3.xyz, r1.w, c[27]
mad r4.xyz, r0.x, c[a0.x+21], r4
mul r3.xyz, r0.w, r3
mad r7.xyz, r0.z, c[a0.z+25], r4
dp3 r5.x, c[1].w, r10
mov oT4.x, r10.x
dp3 r5.z, c[1].w, r6
dp3 r5.y, c[1].w, r1
mov oT4.z, r6.x
dp3 r4.z, r5, r9
mul r0.xy, r10.yzzw, c[3]
mov oT5.x, r10.y
mov oT6.x, r10.z
add r6.x, r0.y, r0.x
mul r4.xy, r6.yzzw, c[3]
mov oT5.z, r6.y
mov oT6.z, r6.z
mul r0.xy, r1.yzzw, c[3]
add r6.z, r4.y, r4.x
add r6.y, r0.y, r0.x
dp3 r4.y, r6, r9
dp3 r0.z, r8, r8
dp3 r4.x, r2, r9
rsq r0.w, r0.z
dst r0.xy, r0.z, r0.w
max r4.xyz, r4, c[0].x
dp4 r1.w, c[36], r0
mad r7.xyz, r3, r4.x, r7
rcp r1.w, r1.w
mul r8.xyz, r8, r0.w
mul r0.xyz, r1.w, c[32]
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
mad oD0.xyz, r0, r2.x, r7
mul r7.xyz, r5.y, c[a0.y+23]
mad r6.xyz, r6.z, c[a0.w+25], r8
mad r7.xyz, r5.x, c[a0.x+21], r7
mad r6.xyz, r3, r4.y, r6
mad r5.xyz, r5.z, c[a0.z+25], r7
mad oD1.xyz, r0, r2.y, r6
mad r3.xyz, r3, r4.z, r5
mad oT7.xyz, r0, r2.z, r3
mul r0.xy, v4, c[90]
add r0.w, r0.y, r0.x
mul r0.xy, v4, c[91]
mov oT0.x, r0.w
add r0.z, r0.y, r0.x
mov oT0.y, r0.z
mov oT1.x, r0.w
mov oT2.x, r0.w
mov oT1.y, r0.z
mov oT2.y, r0.z
mov oT4.y, r1.x
mov oT5.y, r1.y
mov oT6.y, r1.z
mov oD0.w, c[0].x
mov oD1.w, c[0].x
