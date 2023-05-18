/*
 * Copyright (c) 2003, 2007-11 Matteo Frigo
 * Copyright (c) 2003, 2007-11 Massachusetts Institute of Technology
 * Copyright (c) 2023 Tactical Computing Laboratories, LLC
 * 
 * RISC-V V support implemented by Romain Dolbeau. (c) 2019 Romain Dolbeau
 * Modified to support RVV spec v1.0 by Zheng Shuo. (c) 2022 Zheng Shuo
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */

#ifndef defined(__riscv_xlen) && __riscv_xlen == 64

#if defined(FFTW_LDOUBLE) || defined(FFTW_QUAD)
#error "RISC-V V vector instructions only works in single or double precision"
#endif

#ifdef FFTW_SINGLE
 #  define DS(d,s) s /* single-precision option */
 #  define TYPE(name) __riscv_ ## name ## _f32m1
 #  define TYPEUINT(name) __riscv_ ## name ## _u32m1
 #  define TYPEINTERPRETF2U(name) __riscv_ ## name ## _f32m1_u32m1
 #  define TYPEINTERPRETU2F(name) __riscv_ ## name ## _u32m1_f32m1
 #  define TYPEMEM(name) __riscv_ ## name ## e32_v_f32m1
 #  define VETYPE e32
 #else
 #  define DS(d,s) d /* double-precision option */
 #  define TYPE(name) __riscv_ ## name ## _f64m1
 #  define TYPEUINT(name) __riscv_ ## name ## _u64m1
 #  define TYPEINTERPRETF2U(name) __riscv_ ## name ## _f64m1_u64m1
 #  define TYPEINTERPRETU2F(name) __riscv_ ## name ## _u64m1_f64m1
 #  define TYPEMEM(name) __riscv_ ## name ## e64_v_f64m1
 #  define VETYPE e64
 #endif

#define ZERO DS(0.0,0.0f)

#include <stdint.h>
typedef DS(double*, float*) V;
typedef DS(uint64_t*, uint32_t*) Vuint;

// Get vector length from register
#define VL(VLEN,vetype) VL2(VLEN,vetype)
#define VL2(VLEN,vetype)                                               \
    __asm__ volatile(                                               \
        "vsetvli %0, zero, " #vetype ", m1, ta, ma"                  \
        :                                                           \
        :"m"(VLEN)                                                  \
    );

// Assembly for arithmetic operations
#define VOP(op, etype, vetype, eshift) VOP2(op,etype,vetype,eshift) 
#define VOP2(op, etype, vetype, eshift)                             \
    void v##op##_##etype(unsigned int n, const etype* x,            \
                         const etype* y, etype* z)                  \
    {                                                               \
       __asm__ volatile(                                            \
            "1:"                                                    \
            "\n\t vsetvli t0, %0, " #vetype ", ta, ma"              \
            "\n\t vl" #vetype  " v0, (%1)"                          \
                "\n\t sub %0, %0, t0"                               \
                "\n\t slli t0, t0, " #eshift                        \
                "\n\t add %1, %1, t0"                               \
            "\n\t vl" #vetype  " v1, (%2)"                          \
                "\n\t add %2, %2, t0"                               \
            "\n\t v" #op " v2, v0, v1"                              \
            "\n\t vs" #vetype  " v2, (%3)"                          \
                "\n\t add %3, %3, t0"                               \
                "\n\t bnez %0, 1b"                                  \
                "\n\t ret"                                          \
            :                                                       \
            :"r"(n), "m"(x), "m"(y), "m"(z)                         \
            :"t0"                                                   \
            );                                                      \
    }

// Assembly for VNEG
#define VOP_NEG(etype, vetype, eshift) VOP_NEG2(etype, vetype, eshift)
#define VOP_NEG2(etype, vetype, eshift)                              \
    void vfneg_##etype(unsigned int n, etype* x)                    \
    {                                                               \
       __asm__ volatile(                                            \
            "vneg:"                                                 \
            "\n\t vsetvli t0, %0, " #vetype ", ta, ma"              \
            "\n\t vl" #vetype  " v0, (%1)"                          \
                "\n\t sub %0, %0, t0"                               \
                "\n\t slli t0, t0, " #eshift                        \
            "\n\t vfneg v1, v0"                                     \
            "\n\t vs" #vetype  " v1, (%1)"                          \
            "\n\t add %1, %1, t0"                                   \
            "\n\t bnez %0, vneg"                                    \
            "\n\t ret"                                              \
            :                                                       \
            :"r"(n), "m"(x)                                         \
            :"t0"                                                   \
            );                                                      \
    }

#define VOP_ALL(op)                                                 \
    VOP(op,V,VETYPE,DS(3,2));                                       \

// Generate prototypes
VOP_ALL(fadd);         
VOP_ALL(fsub);         
VOP_ALL(fmul);          
VOP_ALL(fdiv);      
VOP_ALL(fmacc);      
VOP_ALL(fmsac);      
VOP_ALL(fnmsac);      
VOP_NEG(V,VETYPE,DS(3,2))                                      

// Basic arithmetic functions
#define VADD(n,x,y,z) (vfadd_V)(n,x,y,z)
#define VSUB(n,x,y,z) (vfsub_V)(n,x,y,z)
#define VMUL(n,x,y,z) (vfmul_V)(n,x,y,z)
#define VDIV(n,x,y,z) (vfdiv_V)(n,x,y,z)
#define VNEG(n,x)     (vfneg_V)(n,x)

// Generates a vector (-1, 0, -1, 0...) of length vl
#define VPARTSPLIT(partr, vetype) VPARTSPLIT2(partr,vetype)
#define VPARTSPLIT2(partr, vetype)                                  \
    __asm__ volatile ( 		                                        \
	"vid.v v0" 	                                                    \
    "vand.vi v0, v0, 1"                                             \
    "vsub.vi v0, v0, 1"                                             \
    "vs" #vetype ".v v0, (%0)"                                      \
	: 			                                                    \
	: "m"(partr)  		                                            \
    );	

static inline V VDUPL(const V x) {
    Vuint VLEN, partr;
	VL(VLEN,VETYPE);
    VPARTSPLIT(partr,VETYPE); // (-1, 0, -1, 0...)
    V xl = TYPEINTERPRETU2F(vreinterpret_v)(TYPEUINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), partr, 2**VLEN)); // set odd elements to 0
    VADD(*VLEN,TYPE(vfslide1up_vf)(xl, ZERO, 2**VLEN),xl,xl);
    return xl;
    //return VADD(TYPE(vfslide1up_vf)(xl, ZERO, 2*VLEN), xl);
}

static inline V VDUPH(const V x) {
	Vuint VLEN, partr, parti;
	VL(VLEN,ARCH);
    VPARTSPLIT(VLEN); // (-1, 0, -1, 0, ...)
    Vuint parti = TYPEUINT(vnot_v)(partr, 2*VLEN); // (0, -1, 0, -1, ...)
    V xh = TYPEINTERPRETU2F(vreinterpret_v)(TYPEUINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), parti, 2*VLEN)); // set even elements to 0
    return VADD(TYPE(vfslide1down_vf)(xh, ZERO, 2*VLEN), xh);
}

#define DVK(var, val, vl) V var = TYPE(vfmv_v_f)(val, 2*vl)

static inline V FLIP_RI(const V x) {
	Vuint VLEN;
	VL(VLEN,ARCH);
    Vuint partr = VPARTSPLIT(VLEN); // (-1, 0, -1, 0, ...)
    V xl = TYPEINTERPRETU2F(vreinterpret_v)(TYPEUINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), partr, 2*VLEN)); // set odd elements to 0
    Vuint parti = TYPEUINT(vnot_v)(partr, 2*VL); // (0, -1, 0, -1, ...)
    V xh = TYPEINTERPRETU2F(vreinterpret_v)(TYPEUINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), parti, 2*VLEN)); // set even elements to 0
    return VADD(TYPE(vfslide1down_vf)(xh, ZERO, 2*VLEN), TYPE(vfslide1up_vf)(xl, ZERO, 2*VLEN));
}

static inline V VCONJ(const V x) {
	Vuint VLEN;
	VL(VLEN,ARCH);
    Vuint partr = VPARTSPLIT(VLEN); // (-1, 0, -1, 0, ...)
    V xl = TYPEINTERPRETU2F(vreinterpret_v)(TYPEUINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), partr, 2*VLEN)); // set odd elements to 0
    Vuint parti = TYPEUINT(vnot_v)(partr, 2*VL); // (0, -1, 0, -1, ...)
    V xh = TYPEINTERPRETU2F(vreinterpret_v)(TYPEUINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), parti, 2*VLEN)); // set even elements to 0
    return VADD(xl, VNEG(VLEN,xh));
}

static inline V VBYI(V x) {
	Vuint VLEN;
	VL(VLEN,ARCH);
    Vuint partr = VPARTSPLIT(VLEN); // (-1, 0, -1, 0, ...)
    V xl = TYPEINTERPRETU2F(vreinterpret_v)(TYPEUINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(x), partr, 2*VLEN)); // set odd elements to 0
    Vuint parti = TYPEUINT(vnot_v)(partr, 2*VLEN); // (0, -1, 0, -1, ...)
    V xh = TYPEINTERPRETU2F(vreinterpret_v)(TYPEUINT(vand_vv)(TYPEINTERPRETF2U(vreinterpret_v)(VNEG(VLENx)), parti, 2*VLEN)); // set elements to negative, then set even elements to 0
    return VADD(TYPE(vfslide1down_vf)(xh, ZERO, 2*VLEN), TYPE(vfslide1up_vf)(xl, ZERO, 2*VLEN));
}

#define LDK(x) x
#define VFMA(a, b, c) TYPE(vfmacc_vv)(c, a, b, 2*VL)
#define VFMS(a, b, c) TYPE(vfmsac_vv)(c, a, b, 2*VL)
#define VFNMS(a, b, c) TYPE(vfnmsac_vv)(c, a, b, 2*VL)
#define VFMAI(b, c) VADD(c, VBYI(b))
#define VFNMSI(b, c) VSUB(c, VBYI(b))
#define VFMACONJ(b, c) VADD(VCONJ(b), c)
#define VFMSCONJ(b, c) VSUB(VCONJ(b), c)
#define VFNMSCONJ(b, c) VSUB(c, VCONJ(b))

static inline V VZMUL(V tx, V sr) {
    Vuint VLEN;
	VL(VLEN,ARCH);
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    tr = VMUL(sr, tr);
    sr = VBYI(sr);
    return VFMA(ti, sr, tr);
}

static inline V VZMULJ(V tx, V sr) {
    Vuint VLEN;
	VL(VLEN,ARCH);
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    tr = VMUL(sr, tr);
    sr = VBYI(sr);
    return VFNMS(ti, sr, tr);
}

static inline V VZMULI(V tx, V sr) {
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    ti = VMUL(ti, sr);
    sr = VBYI(sr);
    return VFMS(tr, sr, ti);
}

static inline V VZMULIJ(V tx, V sr) {
    Vuint VLEN;
	VL(VLEN,ARCH);
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    ti = VMUL(ti, sr);
    sr = VBYI(sr);
    return VFMA(tr, sr, ti);
}


#else
#error "RISC-V V vector only works for 64 bits"
#endif

