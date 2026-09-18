cmp    rdx,0x20
jae    1003435 <sm_euclid_f32+0x25>
xorps  xmm1,xmm1
xor    ecx,ecx
xorps  xmm4,xmm4
xorps  xmm0,xmm0
xorps  xmm2,xmm2
xorps  xmm6,xmm6
xorps  xmm7,xmm7
xorps  xmm5,xmm5
xorps  xmm3,xmm3
jmp    1003522 <sm_euclid_f32+0x112>
xorps  xmm3,xmm3
xor    eax,eax
xorps  xmm5,xmm5
xorps  xmm7,xmm7
xorps  xmm6,xmm6
xorps  xmm2,xmm2
xorps  xmm0,xmm0
xorps  xmm4,xmm4
xorps  xmm1,xmm1
nop
movups xmm8,XMMWORD PTR [rdi+rax*4]
movups xmm9,XMMWORD PTR [rdi+rax*4+0x10]
movups xmm10,XMMWORD PTR [rdi+rax*4+0x20]
movups xmm11,XMMWORD PTR [rdi+rax*4+0x30]
movups xmm12,XMMWORD PTR [rsi+rax*4]
subps  xmm8,xmm12
movups xmm12,XMMWORD PTR [rsi+rax*4+0x10]
subps  xmm9,xmm12
movups xmm12,XMMWORD PTR [rsi+rax*4+0x20]
subps  xmm10,xmm12
movups xmm12,XMMWORD PTR [rsi+rax*4+0x30]
subps  xmm11,xmm12
mulps  xmm8,xmm8
addps  xmm7,xmm8
mulps  xmm9,xmm9
addps  xmm6,xmm9
mulps  xmm10,xmm10
addps  xmm2,xmm10
mulps  xmm11,xmm11
addps  xmm0,xmm11
movups xmm8,XMMWORD PTR [rdi+rax*4+0x40]
movups xmm9,XMMWORD PTR [rsi+rax*4+0x40]
subps  xmm8,xmm9
mulps  xmm8,xmm8
addps  xmm4,xmm8
movups xmm8,XMMWORD PTR [rdi+rax*4+0x50]
movups xmm9,XMMWORD PTR [rsi+rax*4+0x50]
subps  xmm8,xmm9
mulps  xmm8,xmm8
addps  xmm1,xmm8
movups xmm8,XMMWORD PTR [rdi+rax*4+0x60]
movups xmm9,XMMWORD PTR [rsi+rax*4+0x60]
subps  xmm8,xmm9
mulps  xmm8,xmm8
addps  xmm5,xmm8
movups xmm8,XMMWORD PTR [rdi+rax*4+0x70]
movups xmm9,XMMWORD PTR [rsi+rax*4+0x70]
subps  xmm8,xmm9
mulps  xmm8,xmm8
addps  xmm3,xmm8
lea    rcx,[rax+0x20]
add    rax,0x40
cmp    rax,rdx
mov    rax,rcx
jbe    1003450 <sm_euclid_f32+0x40>
mov    rax,rcx
or     rax,0x4
cmp    rax,rdx
jbe    1003540 <sm_euclid_f32+0x130>
mov    rax,rcx
jmp    1003566 <sm_euclid_f32+0x156>
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax]
movups xmm8,XMMWORD PTR [rdi+rcx*4]
movups xmm9,XMMWORD PTR [rsi+rcx*4]
subps  xmm8,xmm9
mulps  xmm8,xmm8
addps  xmm7,xmm8
lea    rax,[rcx+0x4]
add    rcx,0x8
cmp    rcx,rdx
mov    rcx,rax
jbe    1003540 <sm_euclid_f32+0x130>
addps  xmm4,xmm7
addps  xmm1,xmm6
addps  xmm2,xmm5
addps  xmm2,xmm4
addps  xmm0,xmm3
addps  xmm0,xmm1
addps  xmm0,xmm2
movaps xmm1,xmm0
unpckhpd xmm1,xmm0
addps  xmm1,xmm0
movaps xmm0,xmm1
shufps xmm0,xmm1,0x55
addss  xmm0,xmm1
mov    rcx,rax
sub    rcx,rdx
jae    1003637 <sm_euclid_f32+0x227>
mov    r8d,edx
sub    r8d,eax
and    r8d,0x3
je     10035cc <sm_euclid_f32+0x1bc>
nop    DWORD PTR [rax+rax*1+0x0]
movss  xmm1,DWORD PTR [rdi+rax*4]
subss  xmm1,DWORD PTR [rsi+rax*4]
mulss  xmm1,xmm1
addss  xmm0,xmm1
add    rax,0x1
add    r8,0xffffffffffffffff
jne    10035b0 <sm_euclid_f32+0x1a0>
cmp    rcx,0xfffffffffffffffc
ja     1003637 <sm_euclid_f32+0x227>
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax+0x0]
movss  xmm1,DWORD PTR [rdi+rax*4]
movss  xmm2,DWORD PTR [rdi+rax*4+0x4]
subss  xmm1,DWORD PTR [rsi+rax*4]
mulss  xmm1,xmm1
addss  xmm1,xmm0
subss  xmm2,DWORD PTR [rsi+rax*4+0x4]
mulss  xmm2,xmm2
movss  xmm3,DWORD PTR [rdi+rax*4+0x8]
subss  xmm3,DWORD PTR [rsi+rax*4+0x8]
mulss  xmm3,xmm3
addss  xmm3,xmm2
addss  xmm3,xmm1
movss  xmm0,DWORD PTR [rdi+rax*4+0xc]
subss  xmm0,DWORD PTR [rsi+rax*4+0xc]
mulss  xmm0,xmm0
addss  xmm0,xmm3
add    rax,0x4
cmp    rdx,rax
jne    10035e0 <sm_euclid_f32+0x1d0>
push   rbp
mov    rbp,rsp
xorps  xmm0,XMMWORD PTR [rip+0xffffffffffffcc5e]        # 10002a0 <__anon_4651+0x30>
pop    rbp
ret
cs nop WORD PTR [rax+rax*1+0x0]
xchg   ax,ax
