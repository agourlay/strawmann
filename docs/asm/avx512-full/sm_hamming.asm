push   rbp
mov    rbp,rsp
cmp    rdx,0x20
jae    1001bf1 <sm_hamming+0x21>
vpxor  xmm1,xmm1,xmm1
xor    eax,eax
vpxor  xmm2,xmm2,xmm2
vpxor  xmm3,xmm3,xmm3
vpxor  xmm0,xmm0,xmm0
jmp    1001c8e <sm_hamming+0xbe>
vpxor  xmm0,xmm0,xmm0
xor    ecx,ecx
vpxor  xmm3,xmm3,xmm3
vpxor  xmm2,xmm2,xmm2
vpxor  xmm1,xmm1,xmm1
data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
vmovdqu64 zmm4,ZMMWORD PTR [rsi+rcx*8]
vmovdqu64 zmm5,ZMMWORD PTR [rsi+rcx*8+0x40]
vmovdqu64 zmm6,ZMMWORD PTR [rsi+rcx*8+0x80]
vmovdqu64 zmm7,ZMMWORD PTR [rsi+rcx*8+0xc0]
vpxorq zmm4,zmm4,ZMMWORD PTR [rdi+rcx*8]
vpopcntq zmm4,zmm4
vpaddq zmm2,zmm4,zmm2
vpxorq zmm4,zmm5,ZMMWORD PTR [rdi+rcx*8+0x40]
vpopcntq zmm4,zmm4
vpaddq zmm1,zmm4,zmm1
vpxorq zmm4,zmm6,ZMMWORD PTR [rdi+rcx*8+0x80]
vpopcntq zmm4,zmm4
vpxorq zmm5,zmm7,ZMMWORD PTR [rdi+rcx*8+0xc0]
vpaddq zmm3,zmm4,zmm3
vpopcntq zmm4,zmm5
vpaddq zmm0,zmm4,zmm0
lea    rax,[rcx+0x20]
add    rcx,0x40
cmp    rcx,rdx
mov    rcx,rax
jbe    1001c10 <sm_hamming+0x40>
mov    rcx,rax
or     rcx,0x8
cmp    rcx,rdx
jbe    1001ca0 <sm_hamming+0xd0>
mov    rcx,rax
jmp    1001cca <sm_hamming+0xfa>
nop
vmovdqu64 zmm4,ZMMWORD PTR [rsi+rax*8]
vpxorq zmm4,zmm4,ZMMWORD PTR [rdi+rax*8]
vpopcntq zmm4,zmm4
vpaddq zmm2,zmm4,zmm2
lea    rcx,[rax+0x8]
add    rax,0x10
cmp    rax,rdx
mov    rax,rcx
jbe    1001ca0 <sm_hamming+0xd0>
vpaddq zmm1,zmm3,zmm1
vpaddq zmm0,zmm1,zmm0
vpaddq zmm0,zmm0,zmm2
vextracti64x4 ymm1,zmm0,0x1
vpaddq zmm0,zmm0,zmm1
vextracti128 xmm1,ymm0,0x1
vpaddq xmm0,xmm0,xmm1
vpshufd xmm1,xmm0,0xee
vpaddq xmm0,xmm0,xmm1
vmovq  rax,xmm0
mov    r8,rcx
sub    r8,rdx
jae    1001df5 <sm_hamming+0x225>
mov    r9d,edx
sub    r9d,ecx
and    r9d,0x7
je     1001d38 <sm_hamming+0x168>
nop    DWORD PTR [rax+0x0]
mov    r10,QWORD PTR [rsi+rcx*8]
xor    r10,QWORD PTR [rdi+rcx*8]
popcnt r10,r10
add    rax,r10
inc    rcx
dec    r9
jne    1001d20 <sm_hamming+0x150>
cmp    r8,0xfffffffffffffff8
ja     1001df5 <sm_hamming+0x225>
data16 data16 data16 data16 cs nop WORD PTR [rax+rax*1+0x0]
mov    r8,QWORD PTR [rsi+rcx*8]
mov    r9,QWORD PTR [rsi+rcx*8+0x8]
xor    r8,QWORD PTR [rdi+rcx*8]
popcnt r8,r8
add    r8,rax
xor    r9,QWORD PTR [rdi+rcx*8+0x8]
xor    eax,eax
popcnt rax,r9
mov    r9,QWORD PTR [rsi+rcx*8+0x10]
xor    r9,QWORD PTR [rdi+rcx*8+0x10]
popcnt r9,r9
add    r9,rax
add    r9,r8
mov    rax,QWORD PTR [rsi+rcx*8+0x18]
xor    rax,QWORD PTR [rdi+rcx*8+0x18]
popcnt rax,rax
mov    r8,QWORD PTR [rsi+rcx*8+0x20]
xor    r8,QWORD PTR [rdi+rcx*8+0x20]
popcnt r8,r8
add    r8,rax
mov    rax,QWORD PTR [rsi+rcx*8+0x28]
xor    rax,QWORD PTR [rdi+rcx*8+0x28]
xor    r10d,r10d
popcnt r10,rax
add    r10,r8
add    r10,r9
mov    rax,QWORD PTR [rsi+rcx*8+0x30]
xor    rax,QWORD PTR [rdi+rcx*8+0x30]
mov    r8,QWORD PTR [rsi+rcx*8+0x38]
xor    r8,QWORD PTR [rdi+rcx*8+0x38]
xor    r9d,r9d
popcnt r9,rax
xor    eax,eax
popcnt rax,r8
add    rax,r9
add    rax,r10
add    rcx,0x8
cmp    rdx,rcx
jne    1001d50 <sm_hamming+0x180>
pop    rbp
vzeroupper
ret
nop    WORD PTR [rax+rax*1+0x0]
