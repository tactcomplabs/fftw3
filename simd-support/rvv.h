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
    void v##op(etype x, etype y, etype z, uintptr_t nElem) {                    \
        uintptr_t n = 2 * nElem;                                                \
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %3, e" str(SEW) ", m1, ta, ma"                    \
            "\n\t vle" str(SEW)  ".v v0, (%0)"                                  \
                "\n\t sub %3, %3, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %0, %0, t0"                                           \
            "\n\t vle" str(SEW)  ".v v1, (%1)"                                  \
                "\n\t add %1, %1, t0"                                           \
            "\n\t" #op ".vv v2, v0, v1"                                         \
            "\n\t vse" str(SEW)  ".v v2, (%2)"                                  \
                "\n\t add %2, %2, t0"                                           \
                "\n\t bnez %3, 1b"                                              \
            :"+r"(x), "+r"(y), "+r"(z), "+r"(n)                                 \
            :                                                                   \
            :"t0","memory"                                                      \
        );                                                                      \
    }

// Assembly for fused arithmetic instructions
#define VFOP(op, etype)                                                         \
    void v##op(etype x, etype y, etype z, uintptr_t nElem) {                    \
        uintptr_t n = 2 * nElem;                                                \
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %3, e" str(SEW) ", m1, ta, ma"                    \
            "\n\t vle" str(SEW)  ".v v0, (%0)"                                  \
                "\n\t sub %3, %3, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %0, %0, t0"                                           \
            "\n\t vle" str(SEW)  ".v v1, (%1)"                                  \
                "\n\t add %1, %1, t0"                                           \
            "\n\t" #op ".vv v2, v0, v1"                                         \
            "\n\t vse" str(SEW)  ".v v2, (%2)"                                  \
                "\n\t add %2, %2, t0"                                           \
                "\n\t bnez %3, 1b"                                              \
            :"+r"(x), "+r"(y), "+r"(z), "+r"(n)                                 \
            :                                                                   \
            :"t0","memory"                                                      \
        );                                                                      \
    }                                                                      

// Assembly for unary operations
#define VOP_UN(op, etype)                                                       \
    void v##op(etype x, etype y, uintptr_t nElem) {                             \
        uintptr_t n = 2 * nElem;                                                \
        __asm__ volatile(                                                       \
            "1:"                                                                \
            "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"                    \
            "\n\t vle" str(SEW)  ".v v0, (%0)"                                  \
                "\n\t sub %2, %2, t0"                                           \
                "\n\t slli t0, t0, " str(ESHIFT)                                \
                "\n\t add %0, %0, t0"                                           \
            "\n\t" #op ".v v1, v0"                                              \
            "\n\t vse" str(SEW)  ".v v1, (%1)"                                  \
                "\n\t add %1, %1, t0"                                           \
            "\n\t bnez %2, 1b"                                                  \
            :"+r"(x), "+r"(y), "+r"(n)                                          \
            :                                                                   \
            :"t0","memory"                                                      \
        );                                                                      \
    }

// Generate prototypes
VOP(vfadd, V)         
VOP(vfsub, V)         
VOP(vfmul, V)          

VFOP(vfmacc, V)
VFOP(vfmsac, V)
VFOP(vfnmsac, V)

VOP_UN(vfneg, V)                                      

// Macro wrappers for arithmetic functions
#define VADD(x, y, z, n)     (vvfadd)(x, y, z, n)
#define VSUB(x, y, z, n)     (vvfsub)(x, y, z, n)
#define VMUL(x, y, z, n)     (vvfmul)(x, y, z, n)
#define VNEG(x, y, n)        (vvfneg)(x, y, n)

// Macro wrappers for fused arithmetic functions
#define VFMA(a, b, c, n)     (vvfmacc)(c, a, b, n)
#define VFMS(a, b, c, n)     (vvfmsac)(c, a, b, n)
#define VFNMS(a, b, c, n)    (vvfnmsac)(c, a, b, n)

// z[i] = -z[i] + a[i] * b[i] if i is even
// z[i] = z[i] + a[i] * b[i] if i is odd
static inline void VSUBADD(V x, V y, V z, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    __asm__ volatile ( 
        "1:"                                                                                    \
        "\n\t vsetvli t0, %3, e" str(SEW) ", m1, ta, ma"    /* # of elems left */               \
        "\n\t vid.v v0"                                     /* 1, 2, 3, 4 ... */                \
        "\n\t vand.vi v0, v0, 1"                            /* 0, 1, 0, 1 ... */                \
        "\n\t vrsub.vi v0, v0, 1"                           /* 1, 0, 1, 0 ... */                \
        "\n\t vle" str(SEW) ".v v1, (%0)"                   /* load x into v1 */                \
        "\n\t vle" str(SEW) ".v v2, (%1)"                   /* load y into v2 */                \
        "\n\t vle" str(SEW) ".v v3, (%2)"                   /* load z into v3 */                \
            "\n\t sub %3, %3, t0"                           /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)                /* convert t0 to bytes */           \
        "\n\t vfneg.v v3, v3, v0.t"                         /* negate even indices of z */      \
        "\n\t vfmul.vv v1, v1, v2"                          /* x[i] * y[i] */                   \
        "\n\t vfadd.vv v3, v3, v2"                          /* accumulate product */            \
        "\n\t vse" str(SEW) ".v v3, (%2)"                   /* store result */                  \
            "\n\t add %0, %0, t0"                           /* bump pointer */                  \
            "\n\t add %1, %1, t0"                           /* bump pointer */                  \
            "\n\t add %2, %2, t0"                           /* bump pointer */                  \
            "\n\t bnez %3, 1b"                              /* loop back? */                    \
        :"+r"(x), "+r"(y), "+r"(z), "+r"(n)                                                     \
        :                                                                                       \
        :"t0","memory"                                                                          \
    );
}

// z[i] = z[i] + a[i] * b[i] if i is even
// z[i] = -z[i] + a[i] * b[i] if i is odd
static inline void VADDSUB(V x, V y, V z, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    __asm__ volatile (
        "1:"                                                                                    \
        "\n\t vsetvli t0, %3, e" str(SEW) ", m1, ta, ma"    /* # of elems left */               \
        "\n\t vid.v v0"                                     /* 1, 2, 3, 4 ... */                \
        "\n\t vand.vi v0, v0, 1"                            /* 0, 1, 0, 1 ... */                \
        "\n\t vle" str(SEW) ".v v1, (%0)"                   /* load x into v1 */                \
        "\n\t vle" str(SEW) ".v v2, (%1)"                   /* load y into v2 */                \
        "\n\t vle" str(SEW) ".v v3, (%2)"                   /* load z into v3 */                \
            "\n\t sub %3, %3, t0"                           /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)                /* convert t0 to bytes */           \
        "\n\t vfneg.v v3, v3, v0.t"                         /* negate odd indices of z */       \
        "\n\t vfmul.vv v1, v1, v2"                          /* x[i] * y[i] */                   \
        "\n\t vfadd.vv v3, v3, v2"                          /* accumulate product */            \
        "\n\t vse" str(SEW) ".v v3, (%2)"                   /* store result */                  \
            "\n\t add %0, %0, t0"                           /* bump pointer */                  \
            "\n\t add %1, %1, t0"                           /* bump pointer */                  \
            "\n\t add %2, %2, t0"                           /* bump pointer */                  \
            "\n\t bnez %3, 1b"                              /* loop back? */                    \
        :"+r"(x), "+r"(y), "+r"(z), "+r"(n)                                                     \
        :                                                                                       \
        :"t0","memory"                                                                          \
    );
}

// a+bi => a+ai for all elements
static inline void VDUPL(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;                                                             
    __asm__ volatile(                                                                           \
        "1:"                                                                                    \
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* # of elems left */                 \
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"            /* load Re/Im into v0/v1 */           \
            "\n\t slli t0, t0, 1"                         /* adjust t0 for double load */       \
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */      \
            "\n\t slli t0, t0, " str(ESHIFT)              /* convert # elems to bytes */        \
            "\n\t add %0, %0, t0"                         /* bump pointer */                    \
        "\n\t vmv.v.v v1, v0"                             /* replace Im with Re */              \
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store back into memory */          \
            "\n\t add %1, %1, t0"                         /* bump res pointer */                \
        "\n\t bnez %2, 1b"                                /* loop back? */                      \
        :"+r"(src),"+r"(res),"+r"(n)                                                            \
        :                                                                                       \
        :"t0","memory"                                                                          \
    );                                                                                      
}

// a+bi => b+bi for all elements
static inline void VDUPH(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;                                                             
    __asm__ volatile(                                                                       \
        "1:"                                                                                \
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma" /* # of elems left */              \
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"           /* load Re/Im into v0/v1 */        \
            "\n\t slli t0, t0, 1"                        /* adjust t0 for double load */    \
            "\n\t sub %2, %2, t0"                        /* decrement elems left by t0 */   \
            "\n\t slli t0, t0, " str(ESHIFT)             /* convert # elems to bytes */     \
            "\n\t add %0, %0, t0"                        /* bump pointer */                 \
        "\n\t vmv.v.v v0, v1"                            /* replace Re with Im */           \
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"            /* store back into memory */       \
            "\n\t add %1, %1, t0"                        /* bump res pointer */             \
        "\n\t bnez %2, 1b"                               /* loop back? */                   \
        :"+r"(src),"+r"(res),"+r"(n)                                                        \
        :                                                                                   \
        :"t0","memory"                                                                      \
    ); 
}


// a+bi => b+ai for all elements
static inline void FLIP_RI(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    uintptr_t SEWB = SEW / 8; // SEW in bytes
    __asm__ volatile(                                                                         \
        "1:"                                                                                  \
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* # of elems left */               \
        "\n\t vlse" str(SEW) ".v v1, (%0), %4"            /* load Re into v1 */               \
        "\n\t add t1, %0, %3"                             /* advance one elem */              \
        "\n\t vlse" str(SEW) ".v v0, (t1), %4"            /* load Im into v0 */               \
            "\n\t slli t0, t0, 1"                         /* adjust t0 for double load */     \
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)              /* convert t0 to bytes */           \
            "\n\t add %0, %0, t0"                         /* bump pointer */                  \
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store back into memory */        \
            "\n\t add %1, %1, t0"                         /* bump res pointer */              \
        "\n\t bnez %2, 1b"                                /* loop back? */                    \
        :"+r"(src),"+r"(res),"+r"(n)                                                          \
        :"r"(SEWB), "r"(2*SEWB)                                                               \
        :"t0","t1","memory"                                                              \
    );
}

// a+bi => a-bi for all elements
static inline void VCONJ(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;                                                                  \
    __asm__ volatile(                                                                         \
        "1:"                                                                                  \
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* # of elems left */               \
        "\n\t vlseg2e" str(SEW)  ".v v0, (%0)"            /* load Re/Im into v0/v1 */         \
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)              /* convert t0 to bytes */           \
            "\n\t add %0, %0, t0"                         /* bump pointer */                  \
        "\n\t vneg.v v1, v1"                              /* conjugate Im */                  \
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store back into memory */        \
            "\n\t add %1, %1, t0"                         /* bump pointer */                  \
        "\n\t bnez %2, 1b"                                /* loop back? */                    \
        :"+r"(src),"+r"(res),"+r"(n)                                                          \
        :                                                                                     \
        :"t0","memory"                                                                        \
    );                                                                                        
}

// a+bi => -b+ai for all elements
static inline void VBYI(V src, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    uintptr_t SEWB = SEW / 8; // SEW in bytes
    __asm__ volatile(                                                                         \
        "1:"                                                                                  \
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"  /* # of elems left */               \
        "\n\t vlse" str(SEW) ".v v1, (%0), %4"            /* load Re into v1 */               \
        "\n\t add t1, %0, %3"                             /* advance one elem */              \
        "\n\t vlse" str(SEW) ".v v0, (t1), %4"            /* load Im into v0 */               \
            "\n\t slli t0, t0, 1"                         /* adjust t0 for double load */     \
            "\n\t sub %2, %2, t0"                         /* decrement elems left by t0 */    \
            "\n\t slli t0, t0, " str(ESHIFT)              /* convert t0 to bytes */           \
            "\n\t add %0, %0, t0"                         /* bump pointer */                  \
        "\n\t vneg.v v0, v0"                              /* conjugate Im */                  \
        "\n\t vsseg2e" str(SEW) ".v v0, (%1)"             /* store back into memory */        \
            "\n\t add %1, %1, t0"                         /* bump res pointer */              \
        "\n\t bnez %2, 1b"                                /* loop back? */                    \
        :"+r"(src),"+r"(res),"+r"(n)                                                          \
        :"r"(SEWB),"r"(2*SEWB)                                                                \
        :"t0","t1","memory"                                                                   \
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
    R txIm[n];
    V rePtr = &txRe[0];
    V imPtr = &txIm[0];
    __asm__ volatile(
        "1:"                        
        "\n\t vsetvli t0, %2, e" str(SEW) ", m1, ta, ma"
        "\n\t vmv.v.i v0, 0"
        "\n\t vse" str(SEW) ".v v0, (%0)"
        "\n\t vse" str(SEW) ".v v0, (%1)"
            "\n\t sub %2, %2, t0"
            "\n\t slli t0, t0, " str(ESHIFT)
            "\n\t add %0, %0, t0"
            "\n\t add %1, %1, t0"
        "\n\t bnez %2, 1b"
        :
        :"r"(rePtr), "r"(imPtr), "r"(n)
        :"t0", "memory"
    );
  
    VDUPL(tx, &txRe[0], nElem);
    VDUPH(tx, &txIm[0], nElem);
    FLIP_RI(sr, res, nElem);

    VMUL(&txIm[0], res, res, nElem);
    VSUBADD(&txRe[0], sr, res, nElem);

}

// conj(a+bi) * (c+di)
static inline void VZMULJ(V tx, V sr, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
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
static inline void VZMULI(V tx, V sr, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
    R txRe[n];
    memset(&txRe, 0, n * SEW);

    VDUPL(tx, &txRe[0], nElem);
    VDUPH(tx, res, nElem);

    VMUL(res, sr, res, nElem);
    VBYI(sr, sr, nElem);
    VFMS(&txRe[0], sr, res, nElem);
}

// (a+bi) * (c+di) -> flip R/I -> conj
static inline void VZMULIJ(V tx, V sr, V res, uintptr_t nElem) {
    uintptr_t n = 2 * nElem;
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



