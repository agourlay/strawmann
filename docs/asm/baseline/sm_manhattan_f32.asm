cmp    rdx,0x20
jae    1003205 <sm_manhattan_f32+0x25>
xorps  xmm1,xmm1
xor    ecx,ecx
xorps  xmm4,xmm4
xorps  xmm0,xmm0
xorps  xmm3,xmm3
xorps  xmm6,xmm6
xorps  xmm7,xmm7
xorps  xmm5,xmm5
xorps  xmm2,xmm2
jmp    1003302 <sm_manhattan_f32+0x122>
xorps  xmm2,xmm2
xor    eax,eax
movaps xmm8,XMMWORD PTR [rip+0xffffffffffffd07e]        # 1000290 <__anon_4651+0x20>
xorps  xmm5,xmm5
xorps  xmm7,xmm7
xorps  xmm6,xmm6
xorps  xmm3,xmm3
xorps  xmm0,xmm0
xorps  xmm4,xmm4
xorps  xmm1,xmm1
nop    WORD PTR [rax+rax*1+0x0]
movups xmm9,XMMWORD PTR [rdi+rax*4]
movups xmm10,XMMWORD PTR [rdi+rax*4+0x10]
movups xmm11,XMMWORD PTR [rdi+rax*4+0x20]
movups xmm12,XMMWORD PTR [rdi+rax*4+0x30]
movups xmm13,XMMWORD PTR [rsi+rax*4]
subps  xmm9,xmm13
movups xmm13,XMMWORD PTR [rsi+rax*4+0x10]
subps  xmm10,xmm13
movups xmm13,XMMWORD PTR [rsi+rax*4+0x20]
subps  xmm11,xmm13
movups xmm13,XMMWORD PTR [rsi+rax*4+0x30]
subps  xmm12,xmm13
andps  xmm9,xmm8
addps  xmm7,xmm9
andps  xmm10,xmm8
addps  xmm6,xmm10
andps  xmm11,xmm8
addps  xmm3,xmm11
andps  xmm12,xmm8
addps  xmm0,xmm12
movups xmm9,XMMWORD PTR [rdi+rax*4+0x40]
movups xmm10,XMMWORD PTR [rsi+rax*4+0x40]
subps  xmm9,xmm10
andps  xmm9,xmm8
addps  xmm4,xmm9
movups xmm9,XMMWORD PTR [rdi+rax*4+0x50]
movups xmm10,XMMWORD PTR [rsi+rax*4+0x50]
subps  xmm9,xmm10
andps  xmm9,xmm8
addps  xmm1,xmm9
movups xmm9,XMMWORD PTR [rdi+rax*4+0x60]
movups xmm10,XMMWORD PTR [rsi+rax*4+0x60]
subps  xmm9,xmm10
andps  xmm9,xmm8
addps  xmm5,xmm9
movups xmm9,XMMWORD PTR [rdi+rax*4+0x70]
movups xmm10,XMMWORD PTR [rsi+rax*4+0x70]
subps  xmm9,xmm10
andps  xmm9,xmm8
addps  xmm2,xmm9
lea    rcx,[rax+0x20]
add    rax,0x40
cmp    rax,rdx
mov    rax,rcx
jbe    1003230 <sm_manhattan_f32+0x50>
push   rbp
mov    rbp,rsp
mov    rax,rcx
or     rax,0x4
cmp    rax,rdx
jbe    1003317 <sm_manhattan_f32+0x137>
mov    rax,rcx
jmp    1003346 <sm_manhattan_f32+0x166>
movaps xmm8,XMMWORD PTR [rip+0xffffffffffffcf71]        # 1000290 <__anon_4651+0x20>
nop
movups xmm9,XMMWORD PTR [rdi+rcx*4]
movups xmm10,XMMWORD PTR [rsi+rcx*4]
subps  xmm9,xmm10
andps  xmm9,xmm8
addps  xmm7,xmm9
lea    rax,[rcx+0x4]
add    rcx,0x8
cmp    rcx,rdx
mov    rcx,rax
jbe    1003320 <sm_manhattan_f32+0x140>
addps  xmm4,xmm7
addps  xmm1,xmm6
addps  xmm3,xmm5
addps  xmm3,xmm4
addps  xmm0,xmm2
addps  xmm0,xmm1
addps  xmm0,xmm3
movaps xmm2,xmm0
unpckhpd xmm2,xmm0
addps  xmm2,xmm0
movaps xmm1,xmm2
shufps xmm1,xmm2,0x55
addss  xmm1,xmm2
cmp    rdx,rax
jbe    1003393 <sm_manhattan_f32+0x1b3>
mov    r8d,edx
sub    r8d,eax
lea    rcx,[rax+0x1]
test   r8b,0x1
jne    100339f <sm_manhattan_f32+0x1bf>
cmp    rdx,rcx
jne    10033bf <sm_manhattan_f32+0x1df>
xorps  xmm0,XMMWORD PTR [rip+0xffffffffffffcf0f]        # 10002a0 <__anon_4651+0x30>
pop    rbp
ret
movaps xmm0,xmm1
xorps  xmm0,XMMWORD PTR [rip+0xffffffffffffcf03]        # 10002a0 <__anon_4651+0x30>
pop    rbp
ret
movss  xmm0,DWORD PTR [rdi+rax*4]
subss  xmm0,DWORD PTR [rsi+rax*4]
andps  xmm0,XMMWORD PTR [rip+0xffffffffffffcee0]        # 1000290 <__anon_4651+0x20>
addss  xmm0,xmm1
movaps xmm1,xmm0
mov    rax,rcx
cmp    rdx,rcx
je     100338a <sm_manhattan_f32+0x1aa>
movaps xmm2,XMMWORD PTR [rip+0xffffffffffffceca]        # 1000290 <__anon_4651+0x20>
cs nop WORD PTR [rax+rax*1+0x0]
movss  xmm3,DWORD PTR [rdi+rax*4]
movss  xmm0,DWORD PTR [rdi+rax*4+0x4]
subss  xmm3,DWORD PTR [rsi+rax*4]
andps  xmm3,xmm2
addss  xmm3,xmm1
subss  xmm0,DWORD PTR [rsi+rax*4+0x4]
andps  xmm0,xmm2
addss  xmm0,xmm3
add    rax,0x2
movaps xmm1,xmm0
cmp    rdx,rax
jne    10033d0 <sm_manhattan_f32+0x1f0>
jmp    100338a <sm_manhattan_f32+0x1aa>
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax+0x0]
