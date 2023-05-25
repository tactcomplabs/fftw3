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
 #  define ESHIFT 2
 #else
 #  define DS(d,s) d /* double-precision option */
 #  define TYPE(name) __riscv_ ## name ## _f64m1
 #  define TYPEUINT(name) __riscv_ ## name ## _u64m1
 #  define TYPEINTERPRETF2U(name) __riscv_ ## name ## _f64m1_u64m1
 #  define TYPEINTERPRETU2F(name) __riscv_ ## name ## _u64m1_f64m1
 #  define TYPEMEM(name) __riscv_ ## name ## e64_v_f64m1
 #  define VETYPE e64
 #  define ESHIFT 3
 #endif

#define ZERO DS(0.0,0.0f)

typedef DS(vfloat64m1_t, vfloat32m1_t) V;
typedef DS(vuint64m1_t, vuint32m1_t) Vuint;

// Get vector length (# elements) from vsetvli
#define VL(vetype) VL2(vetype)
#define VL2(vetype) ({                                              \
    Vuint res;                                                      \
    __asm__ volatile(                                               \
        "vsetvli %0, zero, " #vetype ", m1, ta, ma"                 \
        :"=m" (res)                                                 \
        :                                                           \
    );                                                              \
    res;                                                            \
})

// Assembly for basic arithmetic operations
#define VOP(op, etype, vetype, eshift) VOP2(op,etype,vetype,eshift) 
#define VOP2(op, etype, vetype, eshift)                             \
    etype v##op(Vuint n, const etype* x, const etype* y)            \
    {                                                               \
       etype res;                                                   \
       __asm__ volatile(                                            \
            "1:"                                                    \
            "\n\t vsetvli t0, %1," #vetype ", ta, ma"               \
            "\n\t vl" #vetype  " v0, (%2)"                          \
                "\n\t sub %1, %1, t0"                               \
                "\n\t slli t0, t0, " #eshift                        \
                "\n\t add %2, %2, t0"                               \
            "\n\t vl" #vetype  " v1, (%3)"                          \
                "\n\t add %3, %3, t0"                               \
            "\n\t" #op " v2, v0, v1"                                \
            "\n\t vs" #vetype  " v2, (%0)"                          \
                "\n\t add %0, %0, t0"                               \
                "\n\t bnez %1, 1b"                                  \
                "\n\t ret"                                          \
            :"=m"(res)                                              \
            :"r"(n), "m"(x), "m"(y)                                 \
            :"t0"                                                   \
        );                                                          \
        return res;                                                 \
    }

// Assembly for fused arithmetic instructions
#define VFOP(op, etype, vetype, eshift) VFOP2(op,etype,vetype,eshift) 
#define VFOP2(op, etype, vetype, eshift)                            \
    etype v##op(Vuint n, const etype* x, const etype* y, etype* z)  \
    {                                                               \
       __asm__ volatile(                                            \
            "1:"                                                    \
            "\n\t vsetvli t0, %1," #vetype ", ta, ma"               \
            "\n\t vl" #vetype  " v0, (%2)"                          \
                "\n\t sub %1, %1, t0"                               \
                "\n\t slli t0, t0, " #eshift                        \
                "\n\t add %2, %2, t0"                               \
            "\n\t vl" #vetype  " v1, (%3)"                          \
                "\n\t add %3, %3, t0"                               \
            "\n\t" #op " v2, v0, v1"                                \
            "\n\t vs" #vetype  " v2, (%0)"                          \
                "\n\t add %0, %0, t0"                               \
                "\n\t bnez %1, 1b"                                  \
                "\n\t ret"                                          \
            :"=m"(z)                                                \
            :"r"(n), "m"(x), "m"(y)                                 \
            :"t0"                                                   \
        );                                                          \
        return z;                                                   \
    }                                                                      

// Assembly for unary operations
#define VOP_UN(op, etype, vetype, eshift) VOP_UN2(op, etype, vetype, eshift)
#define VOP_UN2(op,etype, vetype, eshift)                               \
    etype v##op(Vuint n, const etype* x)                                \
    {                                                                   \
       etype res;                                                       \
       __asm__ volatile(                                                \
            "1:"                                                        \
            "\n\t vsetvli t0, %1," #vetype ", ta, ma"                   \
            "\n\t vl" #vetype  " v0, (%2)"                              \
                "\n\t sub %1, %1, t0"                                   \
                "\n\t slli t0, t0, " #eshift                            \
                "\n\t add %2, %2, t0"                                   \
            "\n\t" #op " v1, v0"                                        \
            "\n\t vs" #vetype  " v1, (%0)"                              \
                "\n\t add %2, %2, t0"                                   \
            "\n\t bnez %1, 1b"                                          \
            "\n\t ret"                                                  \
            :"=m"(res)                                                  \
            :"r"(n), "m"(x)                                             \
            :"t0"                                                       \
            );                                                          \
        return res;                                                     \
    }

// Generate prototypes
VOP(vfadd, V, VETYPE, ESHIFT)         
VOP(vfsub, V, VETYPE, ESHIFT)         
VOP(vfmul, V, VETYPE, ESHIFT)          
VOP(vfdiv, V, VETYPE, ESHIFT)

VOP(vand, Vuint, VETYPE, ESHIFT)
VOP(vor, Vuint, VETYPE, ESHIFT)

VFOP(vfmacc, V, VETYPE, ESHIFT)
VFOP(vfmsac, V, VETYPE, ESHIFT)
VFOP(vfnmsac, V, VETYPE, ESHIFT)

VOP_UN(vfneg,V,VETYPE,ESHIFT)                                      
VOP_UN(vnot,Vuint,VETYPE,ESHIFT)                                      

// Basic arithmetic functions
#define VADD(n,x,y)     (vvfadd)(n,x,y)
#define VSUB(n,x,y)     (vvfsub)(n,x,y)
#define VMUL(n,x,y)     (vvfmul)(n,x,y)
#define VDIV(n,x,y)     (vvfdiv)(n,x,y)
#define VNEG(n,x)       (vvfneg)(n,x)

// Basic logical functions
#define VAND(n,x,y)     (vvand)(n,x,y)
#define VOR(n,x,y)      (vvand)(n,x,y)
#define VNOT(n,x)       (vvnot)(n,x)

// Fused arithmetic functions
#define VFMA(n,a,b,c) (vvfmacc)(n,c,a,b)
#define VFMS(n,a,b,c) (vvfmsac)(n,c,a,b)
#define VFNMS(n,a,b,c) (vvfnmsac)(n,c,a,b)

// Generates a vector (-1, 0, -1, 0...) of length vl
#define VPARTSPLIT(vetype) VPARTSPLIT2(vetype)
#define VPARTSPLIT2(vetype) ({                                      \
    Vuint res;                                                      \
    __asm__ volatile ( 		                                        \
	    "vid.v v0" 	                                                \
        "vand.vi v0, v0, 1"                                         \
        "vsub.vi v0, v0, 1"                                         \
        "vs" #vetype ".v v0, (%0)"                                  \
	    :"=m"(partr) 	                                            \
	    :            		                                        \
    );                                                              \
    res;                                                            \
})

// Type punning (may need replacing)
// Rely on function signature to reinterpret pointers?
#define FTU(x) (*(Vuint*)&x)
#define UTF(x) (*(V*)&x)

#define SLIDEUP1(x) ({                                              \
    V res;                                                          \
    __asm__ volatile(                                               \
        "vfslide1up.vf %0, %1, zero \n\t"                           \
        :"=m"(res)                                                  \
        :"m"(x)                                                     \
    );                                                              \   
    res;                                                            \
})

#define SLIDEDOWN1(x) ({                                            \
    V res;                                                          \
    __asm__ volatile(                                               \
        "vfslide1down.vf %0, %1, zero \n\t"                         \
        :"=m"(res)                                                  \
        :"m"(x)                                                     \
    );                                                              \   
    res;                                                            \
})

// a+bi => a+ai for all elements
static inline V VDUPL(const V x) {
    Vuint partr;
	Vuint VLEN = VL(VETYPE);
    Vuint partr = VPARTSPLIT(VETYPE); // (-1, 0, -1, 0...)
    V xl = UTF(VAND(VLEN,FTU(x), partr)); // Mask off odd indices
    return VADD(VLEN,SLIDEUP1(xl),xl);
}

// a+bi => b+bi for all elements
static inline V VDUPH(const V x) {
	Vuint VLEN = VL(VETYPE);
    Vuint partr = VPARTSPLIT(VETYPE); // (-1, 0, -1, 0, ...)
    Vuint parti = VNOT(VLEN,partr); // (0, -1, 0, -1, ...)
    V xh = UTF(VAND(VLEN,FTU(x),parti)); // Mask off even indices
    return VADD(VLEN,SLIDEDOWN1(xh),xh);
}

#define DVK(var, val, vl) V var = TYPE(vfmv_v_f)(val, 2*vl)

// a+bi => b+ai for all elements
static inline V FLIP_RI(const V x) {
	Vuint VLEN = VL(VETYPE);
    Vuint partr = VPARTSPLIT(VETYPE); // (-1, 0, -1, 0, ...)
    V xl = UTF(VAND(VLEN,FTU(x),partr));
    Vuint parti = VNOT(VLEN, partr); // (0, -1, 0, -1, ...)
    V xh = UTF(VAND(VLEN,FTU(x), parti)); // set even elements to 0
    return VADD(VLEN, SLIDEDOWN1(xh), SLIDEUP1(xl));
}

// a+bi => a-bi for all elements
static inline V VCONJ(const V x) {
	Vuint VLEN = VL(VETYPE);
    Vuint partr = VPARTSPLIT(VETYPE); // (-1, 0, -1, 0, ...)
    V xl = UTF(VAND(VLEN,FTU(x),partr));
    Vuint parti = VNOT(VLEN, partr); // (0, -1, 0, -1, ...)
    V xh = UTF(VAND(VLEN,FTU(x), parti)); // set even elements to 0
    return VADD(VLEN, xl, VNEG(VLEN,xh));
}

// Same as FLIP_RI, argument isn't const qualified
static inline V VBYI(V x) {
	Vuint VLEN = VL(VETYPE);
    Vuint partr = VPARTSPLIT(VETYPE); // (-1, 0, -1, 0, ...)
    V xl = UTF(VAND(VLEN,FTU(x),partr));
    Vuint parti = VNOT(VLEN, partr); // (0, -1, 0, -1, ...)
    V xh = UTF(VAND(VLEN,FTU(x), parti)); // set even elements to 0
    return VADD(VLEN, SLIDEDOWN1(xh), SLIDEUP1(xl));
}

// Hybrid instructions
#define VFMAI(b, c) VADD(c, VBYI(b))
#define VFNMSI(b, c) VSUB(c, VBYI(b))
#define VFMACONJ(b, c) VADD(VCONJ(b), c)
#define VFMSCONJ(b, c) VSUB(VCONJ(b), c)
#define VFNMSCONJ(b, c) VSUB(c, VCONJ(b))

static inline V VZMUL(V tx, V sr) {
	Vuint VLEN = VL(VETYPE);
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    tr = VMUL(VLEN, sr, tr);
    sr = VBYI(sr);
    return VFMA(VLEN, ti, sr, tr);
}

static inline V VZMULJ(V tx, V sr) {
	Vuint VLEN = VL(VETYPE);
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    tr = VMUL(VLEN, sr, tr);
    sr = VBYI(sr);
    return VFNMS(VLEN, ti, sr, tr);
}

static inline V VZMULI(V tx, V sr) {
    Vuint VLEN = VL(VETYPE);
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    ti = VMUL(VLEN, ti, sr);
    sr = VBYI(sr);
    return VFMS(VLEN, tr, sr, ti);
}

static inline V VZMULIJ(V tx, V sr) {
	Vuint VLEN = VL(VETYPE);
    V tr = VDUPL(tx);
    V ti = VDUPH(tx);
    ti = VMUL(VLEN, ti, sr);
    sr = VBYI(sr);
    return VFMA(VLEN, tr, sr, ti);
}


#else
#error "RISC-V V vector only works for 64 bits"
#endif

