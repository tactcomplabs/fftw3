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

#if !defined(__riscv_xlen) || __riscv_xlen != 64
#error "RISCV-V vector extension targets RV64"
#endif

#if defined(FFTW_LDOUBLE) || defined(FFTW_QUAD)
#error "RISC-V V vector instructions only works in single or double precision"
#endif

#ifdef FFTW_SINGLE
 #  define DS(d, s) s /* single-precision option */
 #  define SEW 32
 #  define ESHIFT 2   /* FIXME: Add 1 for complex elems? */
#else
 #  define DS(d, s) d /* double-precision option */
 #  define SEW 64
 #  define ESHIFT 3
#endif

#define ZERO DS(0.0,0.0f)
#define str(x) xstr(x)
#define xstr(x) #x

#include <stdint.h> // uint32/64_t
#include <stddef.h> // ptrdiff_t
#include <string.h> // memset
#include <stdlib.h> // calloc

// Scalar types
typedef DS(uint64_t, uint32_t) Suint;
typedef ptrdiff_t INT; // from kernel/ifftw.h
typedef DS(double, float) R;

// Vector types
typedef Suint* Vuint;
typedef R* V;

// Assembly for basic arithmetic operations
#define VOP(op, etype)                                                          \
    void v##op(const etype x, const etype y, etype z, const Suint nElem) {      \
        const Suint n = 2 * nElem;                                              \
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"                        \
            "\n\t vle" str(SEW)  ".v v0, (%2)"                                  \
                "\n\t sub %1, %1, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %2, %2, t0"                                           \
            "\n\t vle" str(SEW)  ".v v1, (%3)"                                  \
                "\n\t add %3, %3, t0"                                           \
            "\n\t" #op ".vv v2, v0, v1"                                         \
            "\n\t vse" str(SEW)  ".v v2, (%0)"                                  \
                "\n\t add %0, %0, t0"                                           \
                "\n\t bnez %1, 1b"                                              \
                "\n\t ret"                                                      \
            :"=m"(z)                                                            \
            :"r"(n), "m"(x), "m"(y)                                             \
            :"t0"                                                               \
        );                                                                      \
    }

// Assembly for fused arithmetic instructions
#define VFOP(op, etype)                                                         \
    void v##op(const etype x, const etype y, etype z, const Suint nElem) {      \
        const Suint n = 2 * nElem;                                              \
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"                        \
            "\n\t vle" str(SEW)  ".v v0, (%2)"                                  \
                "\n\t sub %1, %1, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %2, %2, t0"                                           \
            "\n\t vle" str(SEW)  ".v v1, (%3)"                                  \
                "\n\t add %3, %3, t0"                                           \
            "\n\t" #op ".vv v2, v0, v1"                                         \
            "\n\t vse" str(SEW)  ".v v2, (%0)"                                  \
                "\n\t add %0, %0, t0"                                           \
                "\n\t bnez %1, 1b"                                              \
                "\n\t ret"                                                      \
            :"=m"(z)                                                            \
            :"r"(n), "m"(x), "m"(y)                                             \
            :"t0"                                                               \
        );                                                                      \
    }                                                                      

// Assembly for unary operations
#define VOP_UN(op, etype)                                                       \
    void v##op(const etype x, etype y, const Suint nElem) {                     \
        const Suint n = 2 * nElem;                                              \
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"                        \
            "\n\t vle" str(SEW)  ".v v0, (%2)"                                  \
                "\n\t sub %1, %1, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %2, %2, t0"                                           \
            "\n\t" #op ".v v1, v0"                                              \
            "\n\t vse" str(SEW)  ".v v1, (%0)"                                  \
                "\n\t add %2, %2, t0"                                           \
            "\n\t bnez %1, 1b"                                                  \
            "\n\t ret"                                                          \
            :"=m"(y)                                                            \
            :"r"(n), "m"(x)                                                     \
            :"t0"                                                               \
        );                                                                      \
    }

// Generate prototypes
VOP(vfadd, V)         
VOP(vfsub, V)         
VOP(vfmul, V)          
VOP(vfdiv, V)

VOP(vand, Vuint)
VOP(vor, Vuint)

VFOP(vfmacc, V)
VFOP(vfmsac, V)
VFOP(vfnmsac, V)

VOP_UN(vfneg, V)                                      
VOP_UN(vnot, Vuint)                                      

// Macro wrappers for arithmetic functions
#define VADD(x, y, z, n)     (vvfadd)(x, y, z, n)
#define VSUB(x, y, z, n)     (vvfsub)(x, y, z, n)
#define VMUL(x, y, z, n)     (vvfmul)(x, y, z, n)
#define VDIV(x, y, z, n)     (vvfdiv)(x, y, z, n)
#define VNEG(x, y, n)        (vvfneg)(x, y, n)

// Macro wrappers for logical functions
#define VAND(x, y, z, n)     (vvand)(x, y, z, n)
#define VOR(x, y, z, n)      (vvand)(x, y, z, n)
#define VNOT(x, y, n)        (vvnot)(x, y, n)

// Macro wrappers for fused arithmetic functions
#define VFMA(a, b, c, n)     (vvfmacc)(c, a, b, n)
#define VFMS(a, b, c, n)     (vvfmsac)(c, a, b, n)
#define VFNMS(a, b, c, n)    (vvfnmsac)(c, a, b, n)

// z[i] = -z[i] + a[i] * b[i] if i is even
// z[i] = z[i] + a[i] * b[i] if i is odd
static inline void VSUBADD(const V x, const V y, V z, const Suint nElem) {
    const Suint n = 2 * nElem;
    __asm__ volatile (
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"    /* # of elems left */               \
        "\n\t vid.v v0"                                 /* 1, 2, 3, 4 ... */                \
        "\n\t vand.vi v0, v0, 1"                        /* 0, 1, 0, 1 ... */                \
        "\n\t vnot.v v0, v0, 1"                         /* 1, 0, 1, 0 ... */                \
        "\n\t vle" str(SEW) ".v v1, (%2)"               /* load x into v1 */                \
        "\n\t vle" str(SEW) ".v v2, (%3)"               /* load y into v2 */                \
        "\n\t vle" str(SEW) ".v v3, (%0)"               /* load z into v3 */                \
            "\n\t sub %1, %1, t0"                       /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)            /* convert t0 to bytes */           \
        "\n\t vneg.v v3, v3, v0.t"                      /* negate even indices of z */      \
        "\n\t vmul.vv v2, v1, v2"                       /* x[i] * y[i] */                   \
        "\n\t vadd.vv v3, v2, v3"                       /* accumulate product */            \
        "\n\t vse" str(SEW) ".v v3, (%0)"               /* store result */                  \
            "\n\t add %0, %0, t0"                       /* bump pointer */                  \
            "\n\t add %2, %2, t0"                       /* bump pointer */                  \
            "\n\t add %3, %3, t0"                       /* bump pointer */                  \
            "\n\t bnez %1, 1b"                          /* loop back? */                    \
            "\n\t ret"                                                                      \
        :"=m"(z)
        :"r"(n), "m"(x), "m"(y)               
    );
}

// z[i] = z[i] + a[i] * b[i] if i is even
// z[i] = -z[i] + a[i] * b[i] if i is odd
static inline void VADDSUB(const V x, const V y, V z, const Suint nElem) {
    const Suint n = 2 * nElem;
    __asm__ volatile (
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"    /* # of elems left */               \
        "\n\t vid.v v0"                                 /* 1, 2, 3, 4 ... */                \
        "\n\t vand.vi v0, v0, 1"                        /* 0, 1, 0, 1 ... */                \
        "\n\t vle" str(SEW) ".v v1, (%2)"               /* load x into v1 */                \
        "\n\t vle" str(SEW) ".v v2, (%3)"               /* load y into v2 */                \
        "\n\t vle" str(SEW) ".v v3, (%0)"               /* load z into v3 */                \
            "\n\t sub %1, %1, t0"                       /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)            /* convert t0 to bytes */           \
        "\n\t vneg.v v3, v3, v0.t"                      /* negate even indices of z */      \
        "\n\t vmul.vv v2, v1, v2"                       /* x[i] * y[i] */                   \
        "\n\t vadd.vv v3, v2, v3"                       /* accumulate product */            \
        "\n\t vse" str(SEW) ".v v3, (%0)"               /* store result */                  \
            "\n\t add %0, %0, t0"                       /* bump pointer */                  \
            "\n\t add %2, %2, t0"                       /* bump pointer */                  \
            "\n\t add %3, %3, t0"                       /* bump pointer */                  \
            "\n\t bnez %1, 1b"                          /* loop back? */                    \
            "\n\t ret"                                                                      \
        :"=m"(z)
        :"r"(n), "m"(x), "m"(y)               
    );
}

// a+bi => a+ai for all elements
static inline void VDUPL(const V src, V res, const Suint nElem) {
    const Suint n = 2 * nElem;                                                             
    __asm__ volatile(                                                                       \
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"  /* # of elems left */                 \
        "\n\t vlseg2e" str(SEW)  ".v v0, (%2)"        /* load Re/Im into v0/v1 */           \
            "\n\t slli t0, t0, 1"                     /* adjust t0 for double load */       \
            "\n\t sub %1, %1, t0"                     /* decrement elems left by t0 */      \
            "\n\t slli t0, t0, " str(ESHIFT)          /* convert # elems to bytes */        \
            "\n\t add %2, %2, t0"                     /* bump pointer */                    \
        "\n\t vmv.v.v v1, v0"                         /* replace Im with Re */              \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"         /* store back into memory */          \
            "\n\t add %0, %0, t0"                     /* bump res pointer */                \
        "\n\t bnez %1, 1b"                            /* loop back? */                      \
        "\n\t ret"                                                                          \
        :"=m"(res)                                                                          \
        :"r"(n), "m"(src)                                                                   \
        :"t0"                                                                               \
    );                                                                                      \
}

// a+bi => b+bi for all elements
static inline void VDUPH(const V src, V res, const Suint nElem) {
    const Suint n = 2 * nElem;                                                              \
    __asm__ volatile(                                                                       \
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"  /* # of elems left */                 \
        "\n\t vlseg2e" str(SEW) ".v v0, (%2)"         /* load Re/Im into v0/v1 */           \
            "\n\t slli t0, t0, 1"                     /* adjust t0 for double load */       \
            "\n\t sub %1, %1, t0"                     /* decrement elems left by t0 */      \
            "\n\t slli t0, t0, " str(ESHIFT)          /* convert t0 to bytes */             \
            "\n\t add %2, %2, t0"                     /* bump pointer */                    \
        "\n\t vmv.v.v v0, v1"                         /* replace Re with Im */              \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"         /* store back into memory */          \
            "\n\t add %0, %0, t0"                     /* bump res pointer */                \
        "\n\t bnez %1, 1b"                            /* loop back? */                      \
        "\n\t ret"                                                                          \
        :"=m"(res)                                                                          \
        :"r"(n), "m"(src)                                                                   \
        :"t0"                                                                               \
    );                                                                                      \
}

// a+bi => b+ai for all elements
static inline void FLIP_RI(const V src, V res, const Suint nElem) {
    const Suint n = 2 * nElem;
    const Suint SEWB = SEW / 8; // SEW in bytes
    __asm__ volatile(                                                                       \
            "\n\t slli t2, %3, 1"                       /* stride in bytes */               \
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"    /* # of elems left */               \
        "\n\t vlse" str(SEW) ".v v1, (%2), t2"          /* load Re into v1 */               \
        "\n\t add t1, %2, %3"                           /* advance one elem */              \
        "\n\t vlse" str(SEW) ".v v0, t1, t2"            /* load Im into v0 */               \
            "\n\t slli t0, t0, 1"                       /* adjust t0 for double load */     \
            "\n\t sub %1, %1, t0"                       /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)            /* convert t0 to bytes */           \
            "\n\t add %2, %2, t0"                       /* bump pointer */                  \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"           /* store back into memory */        \
            "\n\t add %0, %0, t0"                       /* bump res pointer */              \
        "\n\t bnez %1, 1b"                              /* loop back? */                    \
        "\n\t ret"                                                                          \
        :"=m"(res)                                                                          \
        :"r"(n), "m"(src), "r"(SEWB)                                                        \
        :"t0","t1","t2"                                                                     \
    );
}

// a+bi => a-bi for all elements
static inline void VCONJ(const V src, V res, const Suint nElem) {
    const Suint n = 2 * nElem;                                                              \
    __asm__ volatile(                                                                       \
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"  /* # of elems left */                 \
        "\n\t vlseg2e" str(SEW)  ".v v0, (%2)"        /* load Re/Im into v0/v1 */           \
            "\n\t sub %1, %1, t0"                     /* decrement elems left by t0 */      \
            "\n\t slli t0, t0, " str(ESHIFT)          /* convert t0 to bytes */             \
            "\n\t add %2, %2, t0"                     /* bump pointer */                    \
        "\n\t vneg.v v1, v1"                          /* conjugate Im */                    \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"         /* store back into memory */          \
            "\n\t add %2, %2, t0"                     /* bump pointer */                    \
        "\n\t bnez %1, 1b"                            /* loop back? */                      \
        "\n\t ret"                                                                          \
        :"=m"(res)                                                                          \
        :"r"(n), "m"(src)                                                                   \
        :"t0"                                                                               \
    );                                                                                      \
}

// a+bi => -b+ai for all elements
static inline void VBYI(const V src, V res, const Suint nElem) {
    const Suint n = 2 * nElem;
    const Suint SEWB = SEW / 8; // SEW in bytes
    __asm__ volatile(                                                                       \
            "\n\t slli t2, %3, 1"                       /* stride in bytes */               \
        "1:"                                                                                \
        "\n\t vsetvli t0, %1, e" str(SEW) ", ta, ma"    /* # of elems left */               \
        "\n\t vlse" str(SEW) ".v v1, (%2), t2"          /* load Re into v1 */               \
            "\n\t add t1, %2, %3"                       /* advance one element*/            \
        "\n\t vlse" str(SEW) ".v v0, t1, t2"            /* load Im into v0 */               \
            "\n\t slli t0, t0, 1"                       /* adjust t0 for 2x load */         \
            "\n\t sub %1, %1, t0"                       /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)            /* convert t0 to bytes */           \
            "\n\t add %2, %2, t0"                       /* bump pointer */                  \
        "\n\t vneg.v.v v0, v0"                          /* conjugate Im */                  \
        "\n\t vsseg2e" str(SEW) ".v v0, (%0)"           /* store back into memory */        \
            "\n\t add %0, %0, t0"                       /* bump res pointer */              \
        "\n\t bnez %1, 1b"                              /* loop back? */                    \
        "\n\t ret"                                                                          \
        :"=m"(res)                                                                          \
        :"r"(n), "m"(src), "r"(SEWB)                                                        \
        :"t0","t1","t2"                                                                     \
    );
}

// Hybrid instructions
#define LDK(x) x
#define VFMAI(b, c, n)     VADD(c, VBYI(b, n), n)
#define VFNMSI(b, c, n)    VSUB(c, VBYI(b, n), n)
#define VFMACONJ(b, c, n)  VADD(c, VCONJ(b, n), n)
#define VFMSCONJ(b, c, n)  VSUB(VCONJ(b, n), c, n)
#define VFNMSCONJ(b, c, n) VSUB(c, VCONJ(b, n), n)

// (a+bi) * (c+di)
static inline void VZMUL(V tx, V sr, V res, const Suint nElem) {
    const Suint n = 2 * nElem;
    R txRe[n];
    R txIm[n];
    memset(&txRe, 0, n * SEW);
    memset(&txIm, 0, n * SEW);

    VDUPL(tx, &txRe[0], nElem);
    VDUPH(tx, &txIm[0], nElem);
    FLIP_RI(sr, res, nElem);

    VMUL(&txIm[0], res, res, nElem);
    VSUBADD(&txRe[0], sr, res, nElem);

}

// conj(a+bi) * (c+di)
static inline void VZMULJ(V tx, V sr, V res, const Suint nElem) {
    const Suint n = 2 * nElem;
    R txRe[n];
    R txIm[n];
    memset(&txRe, 0, n * SEW);
    memset(&txIm, 0, n * SEW);

    VDUPL(tx, &txRe[0], nElem);
    VDUPH(tx, &txIm[0], nElem);
    FLIP_RI(sr, res, nElem);

    VMUL(&txIm[0], res, res, nElem);
    VADDSUB(&txRe[0], sr, res, nElem);

}

// (a+bi) * (c+di) -> conj -> flip R/I
static inline void VZMULI(V tx, V sr, V res, const Suint nElem) {
    const Suint n = 2 * nElem;
    R txRe[n];
    memset(&txRe, 0, n * SEW);

    VDUPL(tx, &txRe[0], nElem);
    VDUPH(tx, res, nElem);

    VMUL(res, sr, res, nElem);
    VBYI(sr, sr, nElem);
    VFMS(&txRe[0], sr, res, nElem);
}

// (a+bi) * (c+di) -> flip R/I -> conj
static inline void VZMULIJ(V tx, V sr, V res, const Suint nElem) {
    const Suint n = 2 * nElem;
    R txRe[n];
    R txIm[n];
    memset(&txRe, 0, n * SEW);
    memset(&txIm, 0, n * SEW);

    VDUPL(tx, &txRe[0], nElem);
    VDUPH(tx, &txIm[0], nElem);
    FLIP_RI(sr, res, nElem);
    VMUL(res, &txRe[0], res, nElem);
    VADDSUB(sr, &txIm[0], res, nElem);
}

// static inline void ST(R *x, V v, INT ovs, const R *aligned_like) {
//     (void)aligned_like; // suppress unused var warning

//     Vuint idx = TYPEUINT(vid_v)(VL); // (0, 1, 2, 3, ...)
//     Vuint idx1 = TYPEUINT(vsll_vx)(idx, 1, VL); // (0, 2, 4, 6, ...)

//     V vl = TYPE(vrgather_vv)(v, idx1, VL);
//     TYPEMEM(vss)(x, sizeof(R)*ovs, vl, ovs ? VL : 1); // if ovs=0, store the first element

//     Vuint idx2 = TYPEUINT(vadd_vx)(idx1, 1, VL); // (1, 3, 5, 7, ...)

//     V vh = TYPE(vrgather_vv)(v, idx2, VL);
//     TYPEMEM(vss)(x+1, sizeof(R)*ovs, vh, ovs ? VL : 1); // if ovs=0, store the first element
// }

// #define USE_VTW1
// #define VLEN32 ({ Suint vlen; __asm__ volatile ("vsetvli %0, zero, e32, m1, ta, ma" : r(vlen)); vlen; })
// #define VLEN64 ({ Suint vlen; __asm__ volatile ("vsetvli %0, zero, e64, m1, ta, ma" : r(vlen)); vlen; })
// #include "vtw.h"
// #undef USE_VTW1

