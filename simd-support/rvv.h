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
#define VOP(op, etype)                                                   \
    void v##op(etype x, etype y, etype z, uintptr_t nElem) {             \
        uintptr_t n = 2 * nElem;                                         \
        __asm__ volatile(                                                \
            "1:"                                                         \
            "\n\t vsetvli t0, %3, e" str(SEW) ", m1, ta, ma"             \
            "\n\t vle" str(SEW)  ".v v0, (%0)"                           \
            "\n\t vle" str(SEW)  ".v v1, (%1)"                           \
            "\n\t" #op ".vv v2, v0, v1"                                  \
            "\n\t vse" str(SEW)  ".v v2, (%2)"                           \
                "\n\t sub %3, %3, t0"                                    \
                "\n\t slli t0, t0, " str(ESHIFT)                         \
                "\n\t add %0, %0, t0"                                    \
                "\n\t add %1, %1, t0"                                    \
                "\n\t add %2, %2, t0"                                    \
                "\n\t bnez %3, 1b"                                       \
            :                                                            \
            :"r"(x), "r"(y), "r"(z), "r"(n)                              \
            :"t0","memory"                                               \
        );                                                               \
    }                                                                    \

// Assembly for fused arithmetic instructions
#define VFOP(op, etype)                                                  \
    void v##op(etype x, etype y, etype z, uintptr_t nElem) {             \
        uintptr_t n = 2 * nElem;                                         \
        __asm__ volatile(                                                \
            "1:"                                                         \
            "\n\t vsetvli t0, %3, e" str(SEW) ", m1, ta, ma"             \
            "\n\t vle" str(SEW) ".v v0, (%0)"                           \
            "\n\t vle" str(SEW) ".v v1, (%1)"                           \
            "\n\t vle" str(SEW) ".v v2, (%2)"                            \
            "\n\t" #op ".vv v2, v0, v1"                                  \
            "\n\t vse" str(SEW)  ".v v2, (%2)"                           \
                "\n\t sub %3, %3, t0"                                    \
                "\n\t slli t0, t0, " str(ESHIFT)                         \
                "\n\t add %0, %0, t0"                                    \
                "\n\t add %1, %1, t0"                                    \
                "\n\t add %2, %2, t0"                                    \
                "\n\t bnez %3, 1b"                                       \
            :                                                            \
            :"r"(x), "r"(y), "r"(z), "r"(n)                              \
            :"t0","memory"                                               \
        );                                                               \
    }                                                                    \

// Assembly for unary operations
#define VOP_UN(op, etype)                                                \
    void v##op(etype x, etype y, uintptr_t nElem) {                      \
        uintptr_t n = 2 * nElem;                                         \
        __asm__ volatile(                                                \
            "1:"                                                         \
            "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"             \
            "\n\t vle" str(SEW)  ".v v0, (%0)"                           \
            "\n\t" #op ".v v1, v0"                                       \
            "\n\t vse" str(SEW)  ".v v1, (%1)"                           \
                "\n\t sub %2, %2, t0"                                    \
                "\n\t slli t0, t0, " str(ESHIFT)                         \
                "\n\t add %0, %0, t0"                                    \
                "\n\t add %1, %1, t0"                                    \
            "\n\t bnez %2, 1b"                                           \
            :                                                            \
            :"r"(x), "r"(y), "r"(n)                                      \
            :"t0","memory"                                               \
        );                                                               \
    }                                                                    \

// Generate prototypes
VOP(vfadd, V)
VOP(vfsub, V)
VOP(vfmul, V)

VFOP(vfmacc, V)
VFOP(vfmsac, V)
VFOP(vfnmsub, V)

VOP_UN(vfneg, V)

// Macro wrappers for arithmetic functions
#define VADD(x, y, z, n)     (vvfadd)(x, y, z, n)
#define VSUB(x, y, z, n)     (vvfsub)(x, y, z, n)
#define VMUL(x, y, z, n)     (vvfmul)(x, y, z, n)
#define VNEG(x, y, n)        (vvfneg)(x, y, n)

// Macro wrappers for fused arithmetic functions
#define VFMA(a, b, c, n)     (vvfmacc)(a, b, c, n)
#define VFMS(a, b, c, n)     (vvfmsac)(a, b, c, n)
#define VFNMS(a, b, c, n)    (vvfnmsub)(a, b, c, n)

// y[i] = x[i] - y[i] if i is odd
// y[i] = x[i] + y[i] if i is even
static inline void VSUBADD(V x, V y, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    __asm__ volatile (
        "\n\t 1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"    /* # of elems left */
        "\n\t vlseg2e" str(SEW) ".v v0, (%0)"
        "\n\t vlseg2e" str(SEW) ".v v2, (%1)"
        "\n\t vfsub.vv v2, v0, v2"                          /* accumulate product */
        "\n\t vfadd.vv v3, v1, v3"                          /* accumulate product */
        "\n\t vsseg2e" str(SEW) ".v v2, (%1)"           
            "\n\t slli t0, t0, 1"                        
            "\n\t sub %2, %2, t0"
            "\n\t slli t0, t0, " str(ESHIFT)                /* # elems in bytes */
            "\n\t add %0, %0, t0"                           /* bump pointer */
            "\n\t add %1, %1, t0"                           /* bump pointer */
        "\n\t bgt %2, t0, 1b"                               /* loop back? */
        "\n\t srai %2, %2, 1"                               /* half load for final pass */
        "\n\t bgtz %2, 1b"                                  /* make final pass */
        :
        :"r"(x), "r"(y), "r"(n)
        :"t0","memory"
    );
}

// y[i] = x[i] + y[i] if i is odd
// y[i] = x[i] - y[i] if i is even
static inline void VADDSUB(V x, V y, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    __asm__ volatile (
        "\n\t 1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"    /* # of elems left */
        "\n\t vlseg2e" str(SEW) ".v v0, (%0)"
        "\n\t vlseg2e" str(SEW) ".v v2, (%1)"
        "\n\t vfadd.vv v2, v0, v2"                          /* accumulate product */
        "\n\t vfsub.vv v3, v1, v3"                          /* accumulate product */
        "\n\t vsseg2e" str(SEW) ".v v2, (%1)"           
            "\n\t slli t0, t0, 1"                        
            "\n\t sub %2, %2, t0"
            "\n\t slli t0, t0, " str(ESHIFT)                /* # elems in bytes */
            "\n\t add %0, %0, t0"                           /* bump pointer */
            "\n\t add %1, %1, t0"                           /* bump pointer */
        "\n\t bgt %2, t0, 1b"                               /* loop back? */
        "\n\t srai %2, %2, 1"                               /* half load for final pass */
        "\n\t bgtz %2, 1b"                                  /* make final pass */
        :
        :"r"(x), "r"(y), "r"(n)
        :"t0","memory"
    );
}

// a+bi => a+ai for all elements
static inline void VDUPL(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    uintptr_t stride = SEW/4; // stride in bytes
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* # of elems left */
        "\n\t vlse" str(SEW) ".v v0, (%0), %3"            /* load Re into v0 */
        "\n\t vmv.v.v v1, v0"                             /* copy Re into v1 */
            "\n\t slli t0, t0, 1"                         /* account for 2x store */
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store into res */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bgt %2, t0, 1b"                             /* loop back? */
        "\n\t srai %2, %2, 1"                             /* half load for final pass */
        "\n\t bgtz %2, 1b"                                /* make final pass */
        :
        :"r"(src),"r"(res),"r"(n),"r"(stride)
        :"t0","t1","memory"
    );
}

// a+bi => b+bi for all elements
static inline void VDUPH(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    uintptr_t stride = SEW/4;
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* # of elems left */
        "\n\t vlse" str(SEW) ".v v0, (%0), %3"            /* load Im into v0 */
        "\n\t vmv.v.v v1, v0"                             /* copy Im into v1 */
            "\n\t slli t0, t0, 1"                         /* account for 2x store */
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store into res */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bgt %2, t0, 1b"                             /* loop back? */
        "\n\t srai %2, %2, 1"                             /* half load for final pass */
        "\n\t bgtz %2, 1b"                                /* make final pass */
        :
        :"r"(src+1),"r"(res),"r"(n),"r"(stride)
        :"t0","t1","memory"
    );
}


// a+bi => b+ai for all elements
static inline void FLIP_RI(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* # of elems left */
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"           
        "\n\t vmv.v.v v2, v1"                             /* swap v0/v1 */
        "\n\t vmv.v.v v1, v0"                             
        "\n\t vmv.v.v v0, v2"                             
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"         
            "\n\t slli t0, t0, 1"                         /* account for 2x load */
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bgt %2, t0, 1b"                             /* loop back? */
        "\n\t srai %2, %2, 1"                             /* half load for final pass */
        "\n\t bgtz %2, 1b"                                /* make final pass */
        :
        :"r"(src),"r"(res),"r"(n)
        :"t0","t1","memory"
    );
}

// a+bi => a-bi for all elements
static inline void VCONJ(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* # of elems left */
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"            /* load Re/Im into v0/v1 */
        "\n\t vfneg.v v1, v1"                             /* conjugate Im */
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store back into memory */
            "\n\t slli t0, t0, 1"                         /* account for 2x load */
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bgt %2, t0, 1b"                             /* loop back? */
        "\n\t srai %2, %2, 1"                             /* half load for final pass */
        "\n\t bgtz %2, 1b"                                /* make final pass */
        :
        :"r"(src),"r"(res),"r"(n)
        :"t0","t1","memory"
    );
}

// a+bi => -b+ai for all elements
static inline void VBYI(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"        
        "\n\t vfneg.v v2, v1"                             /* conjugate Im */
        "\n\t vmv.v.v v1, v0"                             /* swap Re/Im */
        "\n\t vmv.v.v v0, v2"
            "\n\t slli t0, t0, 1"                         /* account for 2x load */
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */
            "\n\t slli t1, t0," str(ESHIFT)               /* convert # elems to bytes */
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store into res */
            "\n\t add %0, %0, t1"                         /* bump pointer */
            "\n\t add %1, %1, t1"                         /* bump pointer */
        "\n\t bgt %2, t0, 1b"                             /* loop back? */
        "\n\t srai %2, %2, 1"                             /* half load for final pass */
        "\n\t bgtz %2, 1b"                                /* make final pass */
        :
        :"r"(src),"r"(res),"r"(n)
        :"t0","t1","memory"
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
static inline void VZMUL(V tx, V sr, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    R txRe[n];
    R srFlip[n]; // prevent modifying sr
    V tr = &txRe[0];
    V srf = &srFlip[0];

    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %1, e" str(SEW) ", m1, ta, ma"
        "\n\t vmv.v.i v0, 0"
        "\n\t vse" str(SEW) ".v v0, (%0)"
            "\n\t sub %1, %1, t0"
            "\n\t slli t0, t0, " str(ESHIFT)
            "\n\t add %0, %0, t0"
        "\n\t bnez %1, 1b"
        :
        :"r"(tr), "r"(n)
        :"t0", "memory"
    );

    VDUPL(tx, tr, nElem);
    VDUPH(tx, res, nElem);
    VMUL(tr, sr, tr, nElem);
    FLIP_RI(sr, srf, nElem);
    VMUL(res, srf, res, nElem);
    VSUBADD(tr, res, nElem);
}

// conj(a+bi) * (c+di)
static inline void VZMULJ(V tx, V sr, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    R txRe[n];
    R srFlip[n];
    V tr = &txRe[0];
    V srf = &srFlip[0];

    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %1, e" str(SEW) ", m1, ta, ma"
        "\n\t vmv.v.i v0, 0"
        "\n\t vse" str(SEW) ".v v0, (%0)"
            "\n\t sub %1, %1, t0"
            "\n\t slli t0, t0, " str(ESHIFT)
            "\n\t add %0, %0, t0"
        "\n\t bnez %1, 1b"
        :
        :"r"(tr), "r"(n)
        :"t0", "memory"
    );

    VDUPL(tx, tr, nElem);
    VDUPH(tx, res, nElem);
    VMUL(tr, sr, tr, nElem);
    VBYI(sr, srf, nElem);
    VFNMS(srf, tr, res, nElem); // -(res*sr)+tr
}

// (a+bi) * (c+di) -> conj -> flip R/I
static inline void VZMULI(V tx, V sr, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    R txRe[n];
    R srFlip[n];
    V tr = &txRe[0];
    V srf = &srFlip[0];

    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %1, e" str(SEW) ", m1, ta, ma"
        "\n\t vmv.v.i v0, 0"
        "\n\t vse" str(SEW) ".v v0, (%0)"
            "\n\t sub %1, %1, t0"
            "\n\t slli t0, t0, " str(ESHIFT)
            "\n\t add %0, %0, t0"
        "\n\t bnez %1, 1b"
        :
        :"r"(tr),"r"(n)
        :"t0", "memory"
    );

    VDUPL(tx, tr, nElem);
    VDUPH(tx, res, nElem);
    VMUL(res, sr, res, nElem);
    VBYI(sr, srf, nElem);
    VFMS(tr, srf, res, nElem);
}

// (b+ai) * (c+di)
static inline void VZMULIJ(V tx, V sr, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    R txIm[n];
    R srFlip[n];
    V ti = &txIm[0];
    V srf = &srFlip[0];

    __asm__ volatile(
        "1:"
        "\n\t vsetvli t0, %1, e" str(SEW) ", m1, ta, ma"
        "\n\t vmv.v.i v0, 0"
        "\n\t vse" str(SEW) ".v v0, (%0)"
            "\n\t sub %1, %1, t0"
            "\n\t slli t0, t0, " str(ESHIFT)
            "\n\t add %0, %0, t0"
        "\n\t bnez %1, 1b"
        :
        :"r"(ti),"r"(n)
        :"t0", "memory"
    );

    VDUPL(tx, res, nElem);
    VDUPH(tx, ti, nElem);
    VMUL(ti, sr, ti, nElem);
    FLIP_RI(sr, srf, nElem);
    VMUL(res, srf, res, nElem);
    VSUBADD(ti, res, nElem);
}

// Loads data from x into new vector
// Assumes that it can fit into single vector
static inline V LDA(R* x, INT ivs, R* aligned_like, uintptr_t nElem) {
    (void) aligned_like; // suppress unused var warning
    V res = calloc(nElem*2, sizeof(R));
    memcpy(res, x, 2*nElem*sizeof(R));
    return res;
}

// Stores data from v into x
static inline void STA(R* x, V v, INT ovs, R* aligned_like, uintptr_t nElem) {
    (void) aligned_like; // suppress unused var warning
    memcpy(v, x, 2*nElem*sizeof(R));
}

// static inline void ST(R *x, V v, INT ovs, R *aligned_like) {
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
// #define VLEN32 ({ uintptr_t vlen; __asm__ volatile ("vsetvli %0, zero, e32, m1, ta, ma" : r(vlen)); vlen; })
// #define VLEN64 ({ uintptr_t vlen; __asm__ volatile ("vsetvli %0, zero, e64, m1, ta, ma" : r(vlen)); vlen; })
// #include "vtw.h"
// #undef USE_VTW1

