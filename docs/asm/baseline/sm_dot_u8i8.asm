push   rbp
mov    rbp,rsp
sub    rsp,0x10
cmp    rdx,0x40
jae    1002caf <sm_dot_u8i8+0x5f>
pxor   xmm0,xmm0
movdqa XMMWORD PTR [rbp-0x30],xmm0
xor    ecx,ecx
movdqa XMMWORD PTR [rbp-0x40],xmm0
movdqa XMMWORD PTR [rbp-0x70],xmm0
movdqa XMMWORD PTR [rbp-0x50],xmm0
pxor   xmm5,xmm5
pxor   xmm11,xmm11
pxor   xmm2,xmm2
pxor   xmm10,xmm10
pxor   xmm14,xmm14
movdqa XMMWORD PTR [rbp-0x60],xmm0
pxor   xmm15,xmm15
xorps  xmm3,xmm3
pxor   xmm12,xmm12
pxor   xmm9,xmm9
pxor   xmm1,xmm1
jmp    1002fe1 <sm_dot_u8i8+0x391>
pxor   xmm4,xmm4
xor    eax,eax
xorps  xmm3,xmm3
pxor   xmm12,xmm12
pxor   xmm9,xmm9
pxor   xmm1,xmm1
pxor   xmm14,xmm14
pxor   xmm0,xmm0
movdqa XMMWORD PTR [rbp-0x60],xmm0
pxor   xmm15,xmm15
pxor   xmm5,xmm5
pxor   xmm11,xmm11
pxor   xmm2,xmm2
pxor   xmm10,xmm10
pxor   xmm6,xmm6
movdqa XMMWORD PTR [rbp-0x30],xmm6
movdqa XMMWORD PTR [rbp-0x40],xmm6
movdqa XMMWORD PTR [rbp-0x70],xmm6
movdqa XMMWORD PTR [rbp-0x50],xmm6
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax]
movdqa XMMWORD PTR [rbp-0x10],xmm12
movdqa XMMWORD PTR [rbp-0x80],xmm1
movaps XMMWORD PTR [rbp-0x20],xmm3
movdqu xmm6,XMMWORD PTR [rdi+rax*1]
movdqa xmm8,xmm6
punpcklbw xmm6,xmm4
movdqa xmm7,xmm6
punpcklwd xmm6,xmm4
movdqu xmm12,XMMWORD PTR [rsi+rax*1]
punpcklbw xmm1,xmm12
psraw  xmm1,0x8
movdqa xmm13,xmm1
punpcklwd xmm13,xmm13
pmaddwd xmm13,xmm6
movdqu xmm6,XMMWORD PTR [rdi+rax*1+0x10]
punpckhbw xmm8,xmm4
paddd  xmm5,xmm13
movdqa xmm13,xmm8
punpckhwd xmm13,xmm4
punpcklwd xmm8,xmm4
punpckhwd xmm7,xmm4
punpckhwd xmm1,xmm1
pmaddwd xmm1,xmm7
movdqu xmm7,XMMWORD PTR [rsi+rax*1+0x10]
paddd  xmm11,xmm1
punpckhbw xmm1,xmm12
psraw  xmm1,0x8
movdqa xmm12,xmm1
punpcklwd xmm12,xmm12
pmaddwd xmm12,xmm8
paddd  xmm2,xmm12
punpckhwd xmm1,xmm1
pmaddwd xmm1,xmm13
paddd  xmm10,xmm1
movdqa xmm12,xmm6
punpcklbw xmm6,xmm4
movdqa xmm1,xmm6
punpcklwd xmm6,xmm4
punpcklbw xmm13,xmm7
psraw  xmm13,0x8
movdqa xmm3,xmm0
movdqa xmm0,xmm9
movdqa xmm9,xmm15
movdqa xmm15,xmm14
movdqa xmm14,xmm13
punpcklwd xmm14,xmm14
pmaddwd xmm14,xmm6
movdqu xmm8,XMMWORD PTR [rdi+rax*1+0x20]
movdqa xmm6,XMMWORD PTR [rbp-0x30]
paddd  xmm6,xmm14
movdqa XMMWORD PTR [rbp-0x30],xmm6
movdqa xmm14,xmm15
movdqa xmm15,xmm9
movdqa xmm9,xmm0
movdqa xmm0,xmm3
movdqu xmm6,XMMWORD PTR [rsi+rax*1+0x20]
punpckhbw xmm12,xmm4
punpckhwd xmm1,xmm4
punpckhwd xmm13,xmm13
pmaddwd xmm13,xmm1
movdqa xmm1,xmm12
punpckhwd xmm1,xmm4
punpcklwd xmm12,xmm4
movdqa xmm3,XMMWORD PTR [rbp-0x40]
paddd  xmm3,xmm13
movdqa XMMWORD PTR [rbp-0x40],xmm3
punpckhbw xmm7,xmm7
psraw  xmm7,0x8
movdqa xmm13,xmm7
punpcklwd xmm13,xmm13
pmaddwd xmm13,xmm12
movdqa xmm3,XMMWORD PTR [rbp-0x70]
paddd  xmm3,xmm13
movdqa XMMWORD PTR [rbp-0x70],xmm3
punpckhwd xmm7,xmm7
pmaddwd xmm7,xmm1
movdqa xmm1,XMMWORD PTR [rbp-0x50]
paddd  xmm1,xmm7
movdqa XMMWORD PTR [rbp-0x50],xmm1
movdqa xmm12,xmm8
punpcklbw xmm8,xmm4
movdqa xmm1,xmm8
punpcklwd xmm8,xmm4
punpcklbw xmm13,xmm6
psraw  xmm13,0x8
movdqa xmm7,xmm13
punpcklwd xmm7,xmm7
pmaddwd xmm7,xmm8
movdqu xmm8,XMMWORD PTR [rdi+rax*1+0x30]
paddd  xmm14,xmm7
movdqu xmm7,XMMWORD PTR [rsi+rax*1+0x30]
punpckhbw xmm12,xmm4
punpckhwd xmm1,xmm4
punpckhwd xmm13,xmm13
pmaddwd xmm13,xmm1
movdqa xmm1,xmm12
punpckhwd xmm1,xmm4
punpcklwd xmm12,xmm4
movdqa xmm3,XMMWORD PTR [rbp-0x60]
paddd  xmm3,xmm13
movdqa XMMWORD PTR [rbp-0x60],xmm3
punpckhbw xmm6,xmm6
psraw  xmm6,0x8
movdqa xmm13,xmm6
punpcklwd xmm13,xmm13
pmaddwd xmm13,xmm12
paddd  xmm0,xmm13
punpckhwd xmm6,xmm6
pmaddwd xmm6,xmm1
paddd  xmm15,xmm6
movdqa xmm6,xmm8
punpcklbw xmm8,xmm4
movdqa xmm1,xmm8
punpcklwd xmm8,xmm4
punpcklbw xmm12,xmm7
psraw  xmm12,0x8
movdqa xmm13,xmm12
punpcklwd xmm13,xmm13
pmaddwd xmm13,xmm8
movdqa xmm3,XMMWORD PTR [rbp-0x20]
paddd  xmm3,xmm13
movdqa XMMWORD PTR [rbp-0x20],xmm3
movaps xmm3,XMMWORD PTR [rbp-0x20]
punpckhbw xmm6,xmm4
punpckhwd xmm1,xmm4
punpckhwd xmm12,xmm12
pmaddwd xmm12,xmm1
movdqa xmm1,xmm6
punpcklwd xmm6,xmm4
movdqa xmm8,XMMWORD PTR [rbp-0x10]
paddd  xmm8,xmm12
movdqa XMMWORD PTR [rbp-0x10],xmm8
movdqa xmm12,XMMWORD PTR [rbp-0x10]
punpckhbw xmm7,xmm7
psraw  xmm7,0x8
movdqa xmm8,xmm7
punpcklwd xmm8,xmm8
pmaddwd xmm8,xmm6
paddd  xmm9,xmm8
punpckhwd xmm1,xmm4
punpckhwd xmm7,xmm7
pmaddwd xmm7,xmm1
movdqa xmm1,XMMWORD PTR [rbp-0x80]
paddd  xmm1,xmm7
lea    rcx,[rax+0x40]
sub    rax,0xffffffffffffff80
cmp    rax,rdx
mov    rax,rcx
jbe    1002d10 <sm_dot_u8i8+0xc0>
movdqa XMMWORD PTR [rbp-0x10],xmm12
movaps XMMWORD PTR [rbp-0x20],xmm3
movdqa xmm3,XMMWORD PTR [rbp-0x30]
movdqa XMMWORD PTR [rbp-0x90],xmm14
mov    rax,rcx
or     rax,0x10
cmp    rax,rdx
movdqa XMMWORD PTR [rbp-0x80],xmm1
jbe    1003012 <sm_dot_u8i8+0x3c2>
mov    r8,rcx
jmp    10030be <sm_dot_u8i8+0x46e>
pxor   xmm4,xmm4
cs nop WORD PTR [rax+rax*1+0x0]
movdqu xmm1,XMMWORD PTR [rdi+rcx*1]
movdqu xmm8,XMMWORD PTR [rsi+rcx*1]
movdqa xmm7,xmm1
punpckhbw xmm7,xmm4
movdqa xmm6,xmm7
punpckhwd xmm6,xmm4
punpcklwd xmm7,xmm4
punpcklbw xmm1,xmm4
movdqa xmm12,xmm1
punpckhwd xmm12,xmm4
punpcklwd xmm1,xmm4
punpcklbw xmm13,xmm8
psraw  xmm13,0x8
movdqa xmm14,xmm13
punpcklwd xmm14,xmm14
pmaddwd xmm14,xmm1
paddd  xmm5,xmm14
punpckhwd xmm13,xmm13
pmaddwd xmm13,xmm12
paddd  xmm11,xmm13
punpckhbw xmm1,xmm8
psraw  xmm1,0x8
movdqa xmm8,xmm1
punpcklwd xmm8,xmm8
pmaddwd xmm8,xmm7
paddd  xmm2,xmm8
punpckhwd xmm1,xmm1
pmaddwd xmm1,xmm6
paddd  xmm10,xmm1
lea    r8,[rcx+0x10]
add    rcx,0x20
cmp    rcx,rdx
mov    rcx,r8
jbe    1003020 <sm_dot_u8i8+0x3d0>
movdqa xmm1,XMMWORD PTR [rbp-0x60]
paddd  xmm1,XMMWORD PTR [rbp-0x40]
paddd  xmm1,XMMWORD PTR [rbp-0x10]
paddd  xmm1,xmm11
paddd  xmm15,XMMWORD PTR [rbp-0x50]
paddd  xmm15,XMMWORD PTR [rbp-0x80]
paddd  xmm15,xmm10
paddd  xmm15,xmm1
movdqa xmm1,XMMWORD PTR [rbp-0x90]
paddd  xmm1,xmm3
paddd  xmm1,XMMWORD PTR [rbp-0x20]
paddd  xmm1,xmm5
paddd  xmm0,XMMWORD PTR [rbp-0x70]
paddd  xmm0,xmm9
paddd  xmm0,xmm2
paddd  xmm0,xmm1
paddd  xmm0,xmm15
pshufd xmm1,xmm0,0xee
paddd  xmm1,xmm0
pshufd xmm0,xmm1,0x55
paddd  xmm0,xmm1
movd   eax,xmm0
mov    rcx,r8
sub    rcx,rdx
jae    10031d1 <sm_dot_u8i8+0x581>
mov    r9d,edx
sub    r9d,r8d
and    r9d,0x3
je     100316b <sm_dot_u8i8+0x51b>
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax+0x0]
movzx  r10d,BYTE PTR [rdi+r8*1]
movsx  r11d,BYTE PTR [rsi+r8*1]
imul   r11d,r10d
add    eax,r11d
add    r8,0x1
add    r9,0xffffffffffffffff
jne    1003150 <sm_dot_u8i8+0x500>
cmp    rcx,0xfffffffffffffffc
ja     10031d1 <sm_dot_u8i8+0x581>
cs nop WORD PTR [rax+rax*1+0x0]
nop    DWORD PTR [rax+rax*1+0x0]
movzx  ecx,BYTE PTR [rdi+r8*1]
movsx  r9d,BYTE PTR [rsi+r8*1]
imul   r9d,ecx
add    r9d,eax
movzx  eax,BYTE PTR [rdi+r8*1+0x1]
movsx  ecx,BYTE PTR [rsi+r8*1+0x1]
imul   ecx,eax
movzx  eax,BYTE PTR [rdi+r8*1+0x2]
movsx  r10d,BYTE PTR [rsi+r8*1+0x2]
imul   r10d,eax
add    r10d,ecx
add    r10d,r9d
movzx  ecx,BYTE PTR [rdi+r8*1+0x3]
movsx  eax,BYTE PTR [rsi+r8*1+0x3]
imul   eax,ecx
add    eax,r10d
add    r8,0x4
cmp    rdx,r8
jne    1003180 <sm_dot_u8i8+0x530>
add    rsp,0x10
pop    rbp
ret
nop    WORD PTR [rax+rax*1+0x0]
