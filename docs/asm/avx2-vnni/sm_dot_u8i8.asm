push   rbp
mov    rbp,rsp
cmp    rdx,0x80
jae    1002c81 <sm_dot_u8i8+0x21>
vpxor  xmm1,xmm1,xmm1
xor    eax,eax
vpxor  xmm2,xmm2,xmm2
vpxor  xmm3,xmm3,xmm3
vpxor  xmm0,xmm0,xmm0
jmp    1002cfc <sm_dot_u8i8+0x9c>
vpxor  xmm0,xmm0,xmm0
xor    ecx,ecx
vpxor  xmm3,xmm3,xmm3
vpxor  xmm2,xmm2,xmm2
vpxor  xmm1,xmm1,xmm1
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vmovups ymm4,YMMWORD PTR [rdi+rcx*1]
vmovups ymm5,YMMWORD PTR [rdi+rcx*1+0x20]
vmovups ymm6,YMMWORD PTR [rdi+rcx*1+0x40]
vmovups ymm7,YMMWORD PTR [rdi+rcx*1+0x60]
vmovups ymm8,YMMWORD PTR [rsi+rcx*1]
vmovups ymm9,YMMWORD PTR [rsi+rcx*1+0x20]
vmovups ymm10,YMMWORD PTR [rsi+rcx*1+0x40]
vmovups ymm11,YMMWORD PTR [rsi+rcx*1+0x60]
vpdpbusd ymm2,ymm4,ymm8
vpdpbusd ymm1,ymm5,ymm9
vpdpbusd ymm3,ymm6,ymm10
vpdpbusd ymm0,ymm7,ymm11
lea    rax,[rcx+0x80]
add    rcx,0x100
cmp    rcx,rdx
mov    rcx,rax
jbe    1002ca0 <sm_dot_u8i8+0x40>
mov    rcx,rax
or     rcx,0x20
cmp    rcx,rdx
jbe    1002d10 <sm_dot_u8i8+0xb0>
mov    rcx,rax
jmp    1002d30 <sm_dot_u8i8+0xd0>
nop    DWORD PTR [rax]
vmovups ymm4,YMMWORD PTR [rdi+rax*1]
vmovups ymm5,YMMWORD PTR [rsi+rax*1]
vpdpbusd ymm2,ymm4,ymm5
lea    rcx,[rax+0x20]
add    rax,0x40
cmp    rax,rdx
mov    rax,rcx
jbe    1002d10 <sm_dot_u8i8+0xb0>
vpaddd ymm1,ymm3,ymm1
vpaddd ymm0,ymm1,ymm0
vpaddd ymm0,ymm0,ymm2
vextracti128 xmm1,ymm0,0x1
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0xee
vpaddd xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0x55
vpaddd xmm0,xmm0,xmm1
vmovd  eax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    1002e4c <sm_dot_u8i8+0x1ec>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1002d99 <sm_dot_u8i8+0x139>
data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  r10d,BYTE PTR [rdi+rcx*1]
movsx  r11d,BYTE PTR [rsi+rcx*1]
imul   r11d,r10d
add    eax,r11d
inc    rcx
dec    r9
jne    1002d80 <sm_dot_u8i8+0x120>
cmp    r8,0xfffffffffffffff8
ja     1002e4c <sm_dot_u8i8+0x1ec>
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
movzx  r8d,BYTE PTR [rdi+rcx*1]
movsx  r9d,BYTE PTR [rsi+rcx*1]
imul   r9d,r8d
add    r9d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x1]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x1]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x2]
movsx  r10d,BYTE PTR [rsi+rcx*1+0x2]
imul   r10d,eax
add    r10d,r8d
add    r10d,r9d
movzx  eax,BYTE PTR [rdi+rcx*1+0x3]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x3]
imul   r8d,eax
movzx  eax,BYTE PTR [rdi+rcx*1+0x4]
movsx  r9d,BYTE PTR [rsi+rcx*1+0x4]
imul   r9d,eax
add    r9d,r8d
movzx  eax,BYTE PTR [rdi+rcx*1+0x5]
movsx  r8d,BYTE PTR [rsi+rcx*1+0x5]
imul   r8d,eax
add    r8d,r9d
add    r8d,r10d
movzx  eax,BYTE PTR [rdi+rcx*1+0x6]
movsx  r9d,BYTE PTR [rsi+rcx*1+0x6]
imul   r9d,eax
movzx  r10d,BYTE PTR [rdi+rcx*1+0x7]
movsx  eax,BYTE PTR [rsi+rcx*1+0x7]
imul   eax,r10d
add    eax,r9d
add    eax,r8d
add    rcx,0x8
cmp    rdx,rcx
jne    1002db0 <sm_dot_u8i8+0x150>
pop    rbp
vzeroupper
ret
data16 data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
