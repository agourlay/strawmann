push   rbp
mov    rbp,rsp
and    rsp,0xffffffffffffffe0
sub    rsp,0x160
cmp    rdx,0x80
jae    10027f7 <sm_dot_u8u8+0x77>
vpxor  xmm0,xmm0,xmm0
vmovdqa YMMWORD PTR [rsp+0x60],ymm0
xor    r8d,r8d
vmovdqa YMMWORD PTR [rsp+0x80],ymm0
vmovdqa YMMWORD PTR [rsp+0x40],ymm0
vpxor  xmm15,xmm15,xmm15
vpxor  xmm11,xmm11,xmm11
vpxor  xmm8,xmm8,xmm8
vpxor  xmm3,xmm3,xmm3
vpxor  xmm9,xmm9,xmm9
vpxor  xmm5,xmm5,xmm5
vpxor  xmm13,xmm13,xmm13
vmovdqa YMMWORD PTR [rsp+0x20],ymm0
vpxor  xmm14,xmm14,xmm14
vpxor  xmm7,xmm7,xmm7
vpxor  xmm1,xmm1,xmm1
vpxor  xmm12,xmm12,xmm12
vpxor  xmm10,xmm10,xmm10
jmp    1002a1f <sm_dot_u8u8+0x29f>
vpxor  xmm7,xmm7,xmm7
xor    eax,eax
vpxor  xmm1,xmm1,xmm1
vpxor  xmm12,xmm12,xmm12
vpxor  xmm10,xmm10,xmm10
vpxor  xmm5,xmm5,xmm5
vpxor  xmm13,xmm13,xmm13
vpxor  xmm0,xmm0,xmm0
vmovdqa YMMWORD PTR [rsp+0x20],ymm0
vpxor  xmm14,xmm14,xmm14
vpxor  xmm11,xmm11,xmm11
vpxor  xmm8,xmm8,xmm8
vpxor  xmm3,xmm3,xmm3
vpxor  xmm9,xmm9,xmm9
vmovdqa YMMWORD PTR [rsp+0x60],ymm0
vmovdqa YMMWORD PTR [rsp+0x80],ymm0
vmovdqa YMMWORD PTR [rsp+0x40],ymm0
vpxor  xmm15,xmm15,xmm15
vmovdqa YMMWORD PTR [rsp],ymm1
vmovdqa YMMWORD PTR [rsp+0xc0],ymm10
vmovdqa YMMWORD PTR [rsp+0xa0],ymm12
vpmovzxbd ymm0,QWORD PTR [rdi+rax*1+0x18]
vpmovzxbd ymm1,QWORD PTR [rdi+rax*1+0x10]
vpmovzxbd ymm4,QWORD PTR [rdi+rax*1]
vpmovzxbd ymm12,QWORD PTR [rdi+rax*1+0x8]
vpmovzxbd ymm6,QWORD PTR [rsi+rax*1+0x18]
{vex} vpdpwssd ymm9,ymm6,ymm0
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x10]
vpmovzxbd ymm6,QWORD PTR [rsi+rax*1]
{vex} vpdpwssd ymm3,ymm0,ymm1
{vex} vpdpwssd ymm11,ymm6,ymm4
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x8]
vpmovzxbd ymm1,QWORD PTR [rdi+rax*1+0x38]
vpmovzxbd ymm4,QWORD PTR [rdi+rax*1+0x30]
vpmovzxbd ymm6,QWORD PTR [rdi+rax*1+0x20]
vmovdqa ymm10,ymm7
vpmovzxbd ymm7,QWORD PTR [rdi+rax*1+0x28]
{vex} vpdpwssd ymm8,ymm0,ymm12
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x38]
{vex} vpdpwssd ymm15,ymm0,ymm1
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x30]
vpmovzxbd ymm1,QWORD PTR [rsi+rax*1+0x20]
vmovdqa ymm12,YMMWORD PTR [rsp+0x40]
{vex} vpdpwssd ymm12,ymm0,ymm4
vmovdqa YMMWORD PTR [rsp+0x40],ymm12
vmovdqa ymm0,YMMWORD PTR [rsp+0x60]
{vex} vpdpwssd ymm0,ymm1,ymm6
vmovdqa YMMWORD PTR [rsp+0x60],ymm0
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x28]
vpmovzxbd ymm1,QWORD PTR [rdi+rax*1+0x58]
vpmovzxbd ymm4,QWORD PTR [rdi+rax*1+0x50]
vpmovzxbd ymm12,QWORD PTR [rdi+rax*1+0x40]
vpmovzxbd ymm6,QWORD PTR [rdi+rax*1+0x48]
vmovdqa ymm2,ymm15
vmovdqa ymm15,ymm14
vmovdqa ymm14,YMMWORD PTR [rsp+0x80]
{vex} vpdpwssd ymm14,ymm0,ymm7
vmovdqa YMMWORD PTR [rsp+0x80],ymm14
vmovdqa ymm14,ymm15
vmovdqa ymm15,ymm2
vmovdqa ymm7,ymm10
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x58]
{vex} vpdpwssd ymm14,ymm0,ymm1
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x50]
vpmovzxbd ymm1,QWORD PTR [rsi+rax*1+0x40]
vmovdqa ymm10,YMMWORD PTR [rsp+0x20]
{vex} vpdpwssd ymm10,ymm0,ymm4
vmovdqa YMMWORD PTR [rsp+0x20],ymm10
{vex} vpdpwssd ymm5,ymm1,ymm12
vmovdqa ymm12,YMMWORD PTR [rsp+0xa0]
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x48]
vpmovzxbd ymm1,QWORD PTR [rdi+rax*1+0x78]
vpmovzxbd ymm4,QWORD PTR [rdi+rax*1+0x70]
{vex} vpdpwssd ymm13,ymm0,ymm6
vmovdqa ymm10,YMMWORD PTR [rsp+0xc0]
vpmovzxbd ymm0,QWORD PTR [rsi+rax*1+0x78]
{vex} vpdpwssd ymm10,ymm0,ymm1
vpmovzxbd ymm0,QWORD PTR [rdi+rax*1+0x60]
vpmovzxbd ymm1,QWORD PTR [rsi+rax*1+0x70]
{vex} vpdpwssd ymm12,ymm1,ymm4
vpmovzxbd ymm1,QWORD PTR [rsi+rax*1+0x60]
{vex} vpdpwssd ymm7,ymm1,ymm0
vpmovzxbd ymm0,QWORD PTR [rdi+rax*1+0x68]
vpmovzxbd ymm1,QWORD PTR [rsi+rax*1+0x68]
vmovdqa ymm2,YMMWORD PTR [rsp]
{vex} vpdpwssd ymm2,ymm1,ymm0
vmovdqa YMMWORD PTR [rsp],ymm2
vmovdqa ymm1,YMMWORD PTR [rsp]
lea    r8,[rax+0x80]
add    rax,0x100
cmp    rax,rdx
mov    rax,r8
jbe    1002850 <sm_dot_u8u8+0xd0>
vmovdqa YMMWORD PTR [rsp+0xe0],ymm14
vmovdqa YMMWORD PTR [rsp+0x100],ymm15
vmovdqa YMMWORD PTR [rsp+0xc0],ymm10
vmovdqa YMMWORD PTR [rsp+0x120],ymm13
vmovdqa ymm15,YMMWORD PTR [rsp+0x20]
vmovdqa ymm14,YMMWORD PTR [rsp+0x40]
vmovdqa YMMWORD PTR [rsp+0xa0],ymm5
vmovdqa ymm10,YMMWORD PTR [rsp+0x60]
mov    rax,r8
or     rax,0x20
cmp    rax,rdx
vmovdqa YMMWORD PTR [rsp],ymm1
jbe    1002a79 <sm_dot_u8u8+0x2f9>
mov    rcx,r8
vmovdqa ymm13,ymm15
jmp    1002ada <sm_dot_u8u8+0x35a>
vmovdqa ymm13,ymm15
xchg   ax,ax
vpmovzxbd ymm0,QWORD PTR [rdi+r8*1+0x18]
vpmovzxbd ymm1,QWORD PTR [rdi+r8*1+0x10]
vpmovzxbd ymm4,QWORD PTR [rdi+r8*1]
vpmovzxbd ymm5,QWORD PTR [rdi+r8*1+0x8]
vpmovzxbd ymm6,QWORD PTR [rsi+r8*1+0x18]
{vex} vpdpwssd ymm9,ymm6,ymm0
vpmovzxbd ymm0,QWORD PTR [rsi+r8*1+0x10]
vpmovzxbd ymm6,QWORD PTR [rsi+r8*1]
{vex} vpdpwssd ymm3,ymm0,ymm1
{vex} vpdpwssd ymm11,ymm6,ymm4
vpmovzxbd ymm0,QWORD PTR [rsi+r8*1+0x8]
{vex} vpdpwssd ymm8,ymm0,ymm5
lea    rcx,[r8+0x20]
add    r8,0x40
cmp    r8,rdx
mov    r8,rcx
jbe    1002a80 <sm_dot_u8u8+0x300>
vmovdqa ymm0,YMMWORD PTR [rsp+0x120]
vpaddd ymm0,ymm0,YMMWORD PTR [rsp+0x80]
vpaddd ymm0,ymm0,YMMWORD PTR [rsp]
vpaddd ymm0,ymm8,ymm0
vmovdqa ymm1,YMMWORD PTR [rsp+0xe0]
vpaddd ymm1,ymm1,YMMWORD PTR [rsp+0x100]
vpaddd ymm1,ymm1,YMMWORD PTR [rsp+0xc0]
vpaddd ymm1,ymm9,ymm1
vpaddd ymm0,ymm0,ymm1
vpaddd ymm1,ymm10,YMMWORD PTR [rsp+0xa0]
vpaddd ymm1,ymm1,ymm7
vpaddd ymm1,ymm11,ymm1
vpaddd ymm2,ymm13,ymm14
vpaddd ymm2,ymm12,ymm2
vpaddd ymm2,ymm2,ymm3
vpaddd ymm1,ymm1,ymm2
vpaddd ymm0,ymm1,ymm0
vextracti128 xmm1,ymm0,0x1
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0xee
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0x55
vpaddd xmm0,xmm0,xmm1
vmovd  eax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    1002c4c <sm_dot_u8u8+0x4cc>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1002b99 <sm_dot_u8u8+0x419>
cs nop WORD PTR [rax+rax*1+0x0]
movzx  r10d,BYTE PTR [rdi+rcx*1]
movzx  r11d,BYTE PTR [rsi+rcx*1]
imul   r11d,r10d
add    eax,r11d
inc    rcx
dec    r9
jne    1002b80 <sm_dot_u8u8+0x400>
cmp    r8,0xfffffffffffffff8
ja     1002c4c <sm_dot_u8u8+0x4cc>
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  r8d,BYTE PTR [rdi+rcx*1]
movzx  r9d,BYTE PTR [rsi+rcx*1]
imul   r9d,r8d
add    r9d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x1]
movzx  r8d,BYTE PTR [rsi+rcx*1+0x1]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x2]
movzx  r10d,BYTE PTR [rsi+rcx*1+0x2]
imul   r10d,eax
add    r10d,r8d
add    r10d,r9d
movzx  eax,BYTE PTR [rdi+rcx*1+0x3]
movzx  r8d,BYTE PTR [rsi+rcx*1+0x3]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x4]
movzx  r9d,BYTE PTR [rsi+rcx*1+0x4]
imul   r9d,eax
add    r9d,r8d
movzx  eax,BYTE PTR [rdi+rcx*1+0x5]
movzx  r8d,BYTE PTR [rsi+rcx*1+0x5]
imul   r8d,eax
add    r8d,r9d
add    r8d,r10d
movzx  eax,BYTE PTR [rdi+rcx*1+0x6]
movzx  r9d,BYTE PTR [rsi+rcx*1+0x6]
imul   r9d,eax
movzx  r10d,BYTE PTR [rdi+rcx*1+0x7]
movzx  eax,BYTE PTR [rsi+rcx*1+0x7]
imul   eax,r10d
add    eax,r9d
add    eax,r8d
add    rcx,0x8
cmp    rdx,rcx
jne    1002bb0 <sm_dot_u8u8+0x430>
mov    rsp,rbp
pop    rbp
vzeroupper
ret
data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
