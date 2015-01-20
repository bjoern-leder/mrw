
/*******************************************************************************
*
* File chexp.c
*
* Copyright (C) 2009, 2010, 2011 Filippo Palombi, Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the SU(3) exponential function and its first and second
* derivatives using the Cayley-Hamilton representation
*
* The externally accessible functions are
*
*   void ch2mat(complex_dble *p,su3_alg_dble *X,su3_dble *u)
*     Computes u=p[0]+p[1]*X+p[2]*X^2 given the Cayley-Hamilton coefficients
*     p[0],p[1],p[2] and the matrix X
*
*   void chexp_drv0(su3_alg_dble *X,ch_drv0_t *s);
*     Assigns the Cayley-Hamilton coefficients of the exponential function
*     exp(X) to the elements of s, assuming the norm of X is not be larger
*     than 1 (an error occurs if this condition is violated)
*
*   void chexp_drv1(su3_alg_dble *X,ch_drv1_t *s);
*     Assigns the Cayley-Hamilton coefficients of the exponential function
*     exp(X) and their first derivatives to the elements of s, assuming the
*     the norm of X is not larger than 1 (an error occurs if this condition
*     is violated)
*
*   void chexp_drv2(su3_alg_dble *X,ch_drv2_t *s);
*     Assigns the Cayley-Hamilton coefficients of the exponential function
*     exp(X) and their first and second derivatives to the elements of s,
*     assuming the norm of X is not larger than 1 (an error occurs if this
*     condition is violated)
*
*   void expXsu3(double eps,su3_alg_dble *X,su3_dble *u)
*     Replaces u by exp(eps*X)*u, where "exp" is the SU(3) exponential
*     function
*
* Notes:
*
* The programs are based on the notes
*
*   M. Luescher: "SU(3) matrix functions", August 2009
*
* The output is delivered in structures of the type
* 
*   typedef struct
*   {
*      double t,d;
*      complex_dble p[3];
*   } ch_drv0_t;
* 
*   typedef struct
*   {
*      double t,d;
*      complex_dble p[3];
*      complex_dble pt[3],pd[3];
*   } ch_drv1_t;
* 
*   typedef struct
*   {
*      double t,d;
*      complex_dble p[3];
*      complex_dble pt[3],pd[3];
*      complex_dble ptt[3],ptd[3],pdd[3];
*   } ch_drv2_t;
* 
* defined in wflow.h. Their elements are the Cayley-Hamilton coefficients
* of the exponential function and their derivatives with respect to the
* parameters t and d (see the notes cited above).
*
* The programs in this module do not perform any communications and can be 
* called locally. All errors are registered by the error_loc() function. If
* SSE instructions are used, it is assumed that the output structures are
* aligned to a 16 byte boundary.
*
*******************************************************************************/

#define CHEXP_C

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "utils.h"
#include "su3.h"
#include "su3fcts.h"

static void mapX2v(su3_alg_dble *X);
static void eval_td(su3_alg_dble *X);
static void ch_init(void);

static int N,init_flag=0;
static double *c,t,d;
static su3_vector_dble v1,v2,v3,w ALIGNED16;
static su3_dble umat1,umat2 ALIGNED16;
static su3_alg_dble Y ALIGNED16;
static ch_drv0_t ALIGNED16 s;
static const ch_drv0_t sp0 ALIGNED16 ={0.0};
static const ch_drv1_t sp1 ALIGNED16 ={0.0};
static const ch_drv2_t sp2 ALIGNED16 ={0.0};


static void eval_td(su3_alg_dble *X)
{
   t=3.0*((*X).c1*(*X).c1+(*X).c2*(*X).c2-(*X).c1*(*X).c2)+
          (*X).c3*(*X).c3+(*X).c4*(*X).c4+(*X).c5*(*X).c5+
          (*X).c6*(*X).c6+(*X).c7*(*X).c7+(*X).c8*(*X).c8;

   mapX2v(X);
   _vector_cross_prod(w,v2,v3);
   d=_vector_prod_im(v1,w);

   error_loc(fabs(d)>(1.000001*(1.000002-t)),1,"eval_td [chexp.c]",
             "The norm of X is larger than 1");
}


static void ch_init(void)
{
   int k;
   double fctr;

   N=7;
   fctr=1.0;
   
   while (fctr>DBL_EPSILON)
   {
      N++;
      fctr/=(double)(N-7);
   }

   N+=(N%2);

   c=amalloc((N+1)*sizeof(*c),4);

   if (error_loc(c==NULL,1,"ch_init [chexp.c]",
		 "Unable to allocate auxiliary array")==0)
   {
      c[0]=1.0;
      for (k=0;k<N;k++)
         c[k+1]=c[k]/(double)(k+1);

      init_flag=1;
   }
   else
      init_flag=2;
}


void expXsu3(double eps,su3_alg_dble *X,su3_dble *u)
{
   int k,n;
   double nfrb;
   su3_dble *u1,*u2,*u3;

   nfrb=4.0*(3.0*((*X).c1*(*X).c1+(*X).c2*(*X).c2-(*X).c1*(*X).c2)+
  	          (*X).c3*(*X).c3+(*X).c4*(*X).c4+(*X).c5*(*X).c5+
	          (*X).c6*(*X).c6+(*X).c7*(*X).c7+(*X).c8*(*X).c8);

   nfrb*=eps*eps;
   n=0;

   while (nfrb>3.0)
   {
      nfrb*=0.25;
      eps*=0.5;
      n++;
   }

   Y.c1=eps*(*X).c1;
   Y.c2=eps*(*X).c2;
   Y.c3=eps*(*X).c3;
   Y.c4=eps*(*X).c4;
   Y.c5=eps*(*X).c5;
   Y.c6=eps*(*X).c6;
   Y.c7=eps*(*X).c7;
   Y.c8=eps*(*X).c8;

   u1=&umat1;
   u2=&umat2;
   
   chexp_drv0(&Y,&s);
   ch2mat(s.p,&Y,u2);

   for (k=0;k<n;k++)
   {
      u3=u1;
      u1=u2;
      u2=u3;
      su3xsu3(u1,u1,u2);
   }

   su3xsu3(u2,u,u);
}   

#if (defined x64)
#include "sse2.h"

static void mapX2v(su3_alg_dble *X)
{
   __asm__ __volatile__("movsd %3, %%xmm0 \n\t"
                        "movsd %4, %%xmm1 \n\t"
                        "movsd %4, %%xmm2 \n\t"
                        "movsd %3, %%xmm3 \n\t"
			"subsd %%xmm0, %%xmm2 \n\t"
                        "subsd %%xmm1, %%xmm3 \n\t"
			"subsd %%xmm0, %%xmm2 \n\t"
                        "subsd %%xmm1, %%xmm3 \n\t"
                        "addsd %%xmm0, %%xmm1 \n\t"
                        "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                        "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                        "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                        "movapd %%xmm2, %0 \n\t"
                        "movapd %%xmm3, %1 \n\t"
                        "movapd %%xmm1, %2"                        
			:
			"=m" (v2.c2),
			"=m" (v3.c3),
			"=m" (v1.c1)
			:
			"m" ((*X).c1),
			"m" ((*X).c2)
			:
			"xmm0", "xmm1", "xmm2", "xmm3");

   __asm__ __volatile__("movapd %3, %%xmm8 \n\t"
			"movapd %5, %%xmm9 \n\t"
			"movapd %7, %%xmm10 \n\t"
			"movapd %%xmm8, %0 \n\t"
			"movapd %%xmm9, %1 \n\t"
			"movapd %%xmm10, %2"
			:
			"=m" (v1.c2),
			"=m" (v1.c3),
			"=m" (v2.c3)
			:
			"m" ((*X).c3),
			"m" ((*X).c4),
			"m" ((*X).c5),
			"m" ((*X).c6),
			"m" ((*X).c7),
			"m" ((*X).c8)
			:
			"xmm8", "xmm9", "xmm10");

   __asm__ __volatile__("mulpd %3, %%xmm8 \n\t"
			"mulpd %3, %%xmm9 \n\t"
			"mulpd %3, %%xmm10 \n\t"
			"movapd %%xmm8, %0 \n\t"
			"movapd %%xmm9, %1 \n\t"
			"movapd %%xmm10, %2"
			:
			"=m" (v2.c1),
			"=m" (v3.c1),
			"=m" (v3.c2)
			:
			"m" (_sse_sgn1_dble)
			:
			"xmm8", "xmm9", "xmm10");
}


void ch2mat(complex_dble *p,su3_alg_dble *X,su3_dble *u)
{
   __asm__ __volatile__("movsd %3, %%xmm0 \n\t"
                        "movsd %4, %%xmm1 \n\t"
                        "movsd %4, %%xmm2 \n\t"
                        "movsd %3, %%xmm3 \n\t"
			"subsd %%xmm0, %%xmm2 \n\t"
                        "subsd %%xmm1, %%xmm3 \n\t"
			"subsd %%xmm0, %%xmm2 \n\t"
                        "subsd %%xmm1, %%xmm3 \n\t"
                        "addsd %%xmm0, %%xmm1 \n\t"
                        "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                        "shufpd $0x1, %%xmm3, %%xmm3 \n\t"
                        "shufpd $0x1, %%xmm1, %%xmm1 \n\t"
                        "movapd %%xmm2, %0 \n\t"
                        "movapd %%xmm3, %1 \n\t"
                        "movapd %%xmm1, %2"                        
			:
			"=m" (umat1.c22),
			"=m" (umat1.c33),
			"=m" (umat1.c11)
			:
			"m" ((*X).c1),
			"m" ((*X).c2)
			:
			"xmm0", "xmm1", "xmm2", "xmm3");

   __asm__ __volatile__("movapd %3, %%xmm8 \n\t"
			"movapd %5, %%xmm9 \n\t"
			"movapd %7, %%xmm10 \n\t"
			"movapd %%xmm8, %0 \n\t"
			"movapd %%xmm9, %1 \n\t"
			"movapd %%xmm10, %2"
			:
			"=m" (umat1.c12),
			"=m" (umat1.c13),
			"=m" (umat1.c23)
			:
			"m" ((*X).c3),
			"m" ((*X).c4),
			"m" ((*X).c5),
			"m" ((*X).c6),
			"m" ((*X).c7),
			"m" ((*X).c8)
			:
			"xmm8", "xmm9", "xmm10");

   __asm__ __volatile__("mulpd %3, %%xmm8 \n\t"
			"mulpd %3, %%xmm9 \n\t"
			"mulpd %3, %%xmm10 \n\t"
			"movapd %%xmm8, %0 \n\t"
			"movapd %%xmm9, %1 \n\t"
			"movapd %%xmm10, %2"
			:
			"=m" (umat1.c21),
			"=m" (umat1.c31),
			"=m" (umat1.c32)
			:
			"m" (_sse_sgn1_dble)
			:
			"xmm8", "xmm9", "xmm10");

   cm3x3_lc1(p+1,&umat1,u);
   su3xsu3(&umat1,u,u);

   __asm__ __volatile__("movapd %3, %%xmm0 \n\t"
                        "movapd %4, %%xmm1 \n\t"
                        "movapd %5, %%xmm2 \n\t"
			"addpd %6, %%xmm0 \n\t"
			"addpd %6, %%xmm1 \n\t"
			"addpd %6, %%xmm2 \n\t"
			"movapd %%xmm0, %0 \n\t"
			"movapd %%xmm1, %1 \n\t"
			"movapd %%xmm2, %2"                        
			:
			"=m" ((*u).c11),
			"=m" ((*u).c22),
			"=m" ((*u).c33)                        
			:
			"m" ((*u).c11),
			"m" ((*u).c22),
			"m" ((*u).c33),                        
			"m" (p[0])
			:
			"xmm0", "xmm1", "xmm2");   
}

void chexp_drv0(su3_alg_dble *X,ch_drv0_t *s)
{
   int n;

   if (init_flag==0)
      ch_init();

   if (init_flag==2)
   {
      (*s)=sp0;
      (*s).p[0].re=1.0;
      return;
   }

   eval_td(X);
   (*s).t=t;
   (*s).d=d;

   __asm__ __volatile__("movddup %0, %%xmm6 \n\t"
			"movddup %1, %%xmm7 \n\t"
                        "movsd %2, %%xmm0 \n\t"
                        "xorpd %%xmm1, %%xmm1 \n\t"
                        "xorpd %%xmm2, %%xmm2 \n\t"
                        "mulpd %3, %%xmm6 \n\t"
                        "mulpd %3, %%xmm7 \n\t"
                        "shufpd $0x0, %%xmm6, %%xmm6 \n\t"
                        "shufpd $0x1, %%xmm7, %%xmm7"
			:
			:
			"m" ((*s).t),
			"m" ((*s).d),
			"m" (c[N-6]),
			"m" (_sse_sgn1_dble)
			:
			"xmm0", "xmm1", "xmm2", "xmm6",
                        "xmm7");

   for (n=(N-7);n>0;n-=2)
   {
      __asm__ __volatile__("movapd %%xmm2, %%xmm4 \n\t"
			   "mulpd %%xmm6, %%xmm2 \n\t"
			   "shufpd $0x1, %%xmm4, %%xmm4 \n\t"
			   "addpd %%xmm0, %%xmm2\n\t"	
                           "mulpd %%xmm7, %%xmm4 \n\t"
                           "movapd %%xmm1, %%xmm0 \n\t"
			   "addsd %0, %%xmm4 \n\t"                           
			   "shufpd $0x1, %%xmm0, %%xmm0 \n\t"
			   "mulpd %%xmm6, %%xmm1 \n\t"
			   "mulpd %%xmm7, %%xmm0 \n\t"
			   "addpd %%xmm4, %%xmm1 \n\t"	
			   "addsd %1, %%xmm0"
			   :
			   :
			   "m" (c[n]),                           
			   "m" (c[n-1])
			   :
			   "xmm0", "xmm1", "xmm2", "xmm4");      
   }
   
   __asm__ __volatile__ ("movapd %%xmm0, %0 \n\t"
			 "movapd %%xmm1, %1 \n\t"
			 "movapd %%xmm2, %2"
			 :
			 "=m" ((*s).p[0]),
			 "=m" ((*s).p[1]),
			 "=m" ((*s).p[2]));
}


void chexp_drv1(su3_alg_dble *X,ch_drv1_t *s)
{
   int n;

   if (init_flag==0)
      ch_init();

   if (init_flag==2)
   {
      (*s)=sp1;
      (*s).p[0].re=1.0;
      return;
   }

   eval_td(X);
   (*s).t=t;
   (*s).d=d;

   __asm__ __volatile__("movddup %0, %%xmm14 \n\t"
			"movddup %1, %%xmm15 \n\t"
			"movsd %2, %%xmm0 \n\t"
			"xorpd %%xmm1, %%xmm1 \n\t"
			"xorpd %%xmm2, %%xmm2 \n\t"
			"mulpd %3, %%xmm14 \n\t"
			"mulpd %3, %%xmm15 \n\t"
			"xorpd %%xmm3, %%xmm3 \n\t"
			"xorpd %%xmm4, %%xmm4 \n\t"
			"xorpd %%xmm5, %%xmm5 \n\t"
                        "shufpd $0x0, %%xmm14, %%xmm14 \n\t"
                        "shufpd $0x1, %%xmm15, %%xmm15"                        
			:
			:
			"m" ((*s).t),
			"m" ((*s).d),
			"m" (c[N-3]),
			"m" (_sse_sgn1_dble)
			:
                        "xmm0", "xmm1", "xmm2", "xmm3",
			"xmm4", "xmm5", "xmm14", "xmm15");

   for (n=N-4;n>=0;n--)
   {
      __asm__ __volatile__("movapd %%xmm2, %%xmm6 \n\t"
                           "movapd %%xmm5, %%xmm7 \n\t"
                           "movapd %%xmm2, %%xmm8 \n\t"
                           "movapd %%xmm5, %%xmm9 \n\t"
                           "shufpd $0x1, %%xmm6, %%xmm6 \n\t"
                           "shufpd $0x1, %%xmm7, %%xmm7 \n\t"
                           "mulpd %0, %%xmm2 \n\t"                           
                           "mulpd %%xmm14, %%xmm8 \n\t"
                           "mulpd %%xmm14, %%xmm9 \n\t"
                           "mulpd %%xmm15, %%xmm6 \n\t"
                           "mulpd %%xmm15, %%xmm7 \n\t"
                           "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
                           "addsd %1, %%xmm6"
			   :
			   :
                           "m" (_sse_sgn1_dble),
			   "m" (c[n])
			   :
			   "xmm2", "xmm6", "xmm7", "xmm8",
                           "xmm9");

      __asm__ __volatile__("addpd %%xmm0, %%xmm8 \n\t"
                           "addpd %%xmm2, %%xmm7 \n\t"
                           "addpd %%xmm3, %%xmm9 \n\t"                           
                           "movapd %%xmm1, %%xmm2 \n\t"
                           "movapd %%xmm6, %%xmm0 \n\t"
                           "movapd %%xmm8, %%xmm1 \n\t"
                           "movapd %%xmm4, %%xmm5 \n\t"
                           "movapd %%xmm7, %%xmm3 \n\t"
                           "movapd %%xmm9, %%xmm4"
			   :
			   :
			   :
			   "xmm0", "xmm1", "xmm2", "xmm3",
                           "xmm4", "xmm5", "xmm7", "xmm8",
                           "xmm9");
   }

   __asm__ __volatile__ ("movapd %%xmm0, %0 \n\t"
			 "movapd %%xmm1, %1 \n\t"
			 "movapd %%xmm2, %2 \n\t"
			 "movapd %%xmm3, %3 \n\t"
			 "movapd %%xmm4, %4 \n\t"
			 "movapd %%xmm5, %5"
			 :
			 "=m" ((*s).p[0]),
			 "=m" ((*s).p[1]),
			 "=m" ((*s).p[2]),
			 "=m" ((*s).pd[0]),
			 "=m" ((*s).pd[1]),
			 "=m" ((*s).pd[2]));

   (*s).pt[0].re=-d*(*s).pd[2].re;
   (*s).pt[0].im=-d*(*s).pd[2].im;
   (*s).pt[1].re= (*s).pd[0].im-t*(*s).pd[2].im;
   (*s).pt[1].im=-(*s).pd[0].re+t*(*s).pd[2].re;
   (*s).pt[2].re= (*s).pd[1].im;
   (*s).pt[2].im=-(*s).pd[1].re;
}


void chexp_drv2(su3_alg_dble *X,ch_drv2_t *s)
{
   int n;

   if (init_flag==0)
      ch_init();

   if (init_flag==2)
   {
      (*s)=sp2;
      (*s).p[0].re=1.0;
      return;
   }

   eval_td(X);
   (*s).t=t;
   (*s).d=d;

   __asm__ __volatile__("movddup %0, %%xmm14 \n\t"
			"movddup %1, %%xmm15 \n\t"
			"movsd %2, %%xmm0 \n\t"
			"xorpd %%xmm1, %%xmm1 \n\t"
			"xorpd %%xmm2, %%xmm2 \n\t"
			"mulpd %3, %%xmm14 \n\t"
			"xorpd %%xmm3, %%xmm3 \n\t"
			"xorpd %%xmm4, %%xmm4 \n\t"
			"xorpd %%xmm5, %%xmm5 \n\t"
                        "shufpd $0x1, %%xmm14, %%xmm14 \n\t"
			"xorpd %%xmm6, %%xmm6 \n\t"
			"xorpd %%xmm7, %%xmm7 \n\t"
			"xorpd %%xmm8, %%xmm8"
			:
			:
			"m" ((*s).d),
			"m" ((*s).t),
			"m" (c[N]),
			"m" (_sse_sgn1_dble)
			:
                        "xmm0", "xmm1", "xmm2", "xmm3",
			"xmm4", "xmm5", "xmm6", "xmm7",
                        "xmm8", "xmm14", "xmm15");

   for (n=N-1;n>=0;n--)
   {
      __asm__ __volatile__("movapd %%xmm2, %%xmm9 \n\t"
			   "movapd %%xmm0, %%xmm10 \n\t"
			   "shufpd $0x1, %%xmm9, %%xmm9 \n\t"
			   "movapd %%xmm2, %%xmm11 \n\t"
			   "mulpd %%xmm14, %%xmm9 \n\t"
			   "mulpd %%xmm15, %%xmm11 \n\t"
			   "addsd %0, %%xmm9 \n\t"
			   "subpd %%xmm11, %%xmm10"	
			   :
			   :
			   "m" (c[n])
			   :
			   "xmm9", "xmm10", "xmm11");

      __asm__ __volatile__("movapd %%xmm5, %%xmm11 \n\t"
			   "shufpd $0x1, %%xmm2, %%xmm2 \n\t"
			   "movapd %%xmm3, %%xmm12 \n\t"
			   "shufpd $0x1, %%xmm11, %%xmm11 \n\t"
			   "mulpd %0, %%xmm2 \n\t"
			   "movapd %%xmm5, %%xmm13 \n\t"
			   "mulpd %%xmm14, %%xmm11 \n\t"
			   "mulpd %%xmm15, %%xmm13 \n\t"
                           "addpd %%xmm5, %%xmm5 \n\t"
			   "subpd %%xmm2, %%xmm11 \n\t"
			   "subpd %%xmm13, %%xmm12"
			   :
			   :
			   "m" (_sse_sgn1_dble)
			   :
			   "xmm2", "xmm5", "xmm11", "xmm12",
                           "xmm13");

      __asm__ __volatile__("movapd %%xmm1, %%xmm2 \n\t"
			   "movapd %%xmm9, %%xmm0 \n\t"
			   "movapd %%xmm10, %%xmm1"
			   :
			   :
			   :
			   "xmm0", "xmm1", "xmm2");

      __asm__ __volatile__("movapd %%xmm8, %%xmm9 \n\t"
			   "shufpd $0x1, %%xmm5, %%xmm5 \n\t"
			   "movapd %%xmm6, %%xmm10 \n\t"
			   "shufpd $0x1, %%xmm9, %%xmm9 \n\t"
			   "mulpd %0, %%xmm5 \n\t"
			   "movapd %%xmm8, %%xmm13 \n\t"
			   "mulpd %%xmm14, %%xmm9 \n\t"
			   "mulpd %%xmm15, %%xmm13 \n\t"
			   "subpd %%xmm5, %%xmm9 \n\t"
			   "subpd %%xmm13, %%xmm10"
			   :
			   :
			   "m" (_sse_sgn1_dble)
			   :
			   "xmm5", "xmm9", "xmm10", "xmm13");

      __asm__ __volatile__("movapd %%xmm4, %%xmm5 \n\t"
			   "movapd %%xmm11, %%xmm3 \n\t"
			   "movapd %%xmm12, %%xmm4 \n\t"
			   "movapd %%xmm7, %%xmm8 \n\t"
			   "movapd %%xmm9, %%xmm6 \n\t"
			   "movapd %%xmm10, %%xmm7"
			   :
			   :
			   :
			   "xmm3", "xmm4", "xmm5", "xmm6",
                           "xmm7", "xmm8");
   }

   __asm__ __volatile__ ("movapd %%xmm0, %0 \n\t"
			 "movapd %%xmm1, %1 \n\t"
			 "movapd %%xmm2, %2 \n\t"
			 "movapd %%xmm3, %3 \n\t"
			 "movapd %%xmm4, %4 \n\t"
			 "movapd %%xmm5, %5 \n\t"
			 "movapd %%xmm6, %6 \n\t"
			 "movapd %%xmm7, %7 \n\t"
			 "movapd %%xmm8, %8"
			 :
			 "=m" ((*s).p[0]),
			 "=m" ((*s).p[1]),
			 "=m" ((*s).p[2]),
			 "=m" ((*s).pd[0]),
			 "=m" ((*s).pd[1]),
			 "=m" ((*s).pd[2]),
			 "=m" ((*s).pdd[0]),
			 "=m" ((*s).pdd[1]),
			 "=m" ((*s).pdd[2]));

   (*s).pt[0].re=-d*(*s).pd[2].re;
   (*s).pt[0].im=-d*(*s).pd[2].im;
   (*s).pt[1].re= (*s).pd[0].im-t*(*s).pd[2].im;
   (*s).pt[1].im=-(*s).pd[0].re+t*(*s).pd[2].re;
   (*s).pt[2].re= (*s).pd[1].im;
   (*s).pt[2].im=-(*s).pd[1].re;

   (*s).ptd[0].re=-(*s).pd[2].re-d*(*s).pdd[2].re;
   (*s).ptd[0].im=-(*s).pd[2].im-d*(*s).pdd[2].im;
   (*s).ptd[1].re= (*s).pdd[0].im-t*(*s).pdd[2].im;
   (*s).ptd[1].im=-(*s).pdd[0].re+t*(*s).pdd[2].re;
   (*s).ptd[2].re= (*s).pdd[1].im;
   (*s).ptd[2].im=-(*s).pdd[1].re;

   (*s).ptt[0].re=-d*(*s).pdd[1].im;
   (*s).ptt[0].im= d*(*s).pdd[1].re;
   (*s).ptt[1].re=-2.0*(*s).pd[2].im+t*(*s).pdd[1].re-d*(*s).pdd[2].im;
   (*s).ptt[1].im= 2.0*(*s).pd[2].re+t*(*s).pdd[1].im+d*(*s).pdd[2].re;
   (*s).ptt[2].re=-(*s).pdd[0].re+t*(*s).pdd[2].re;
   (*s).ptt[2].im=-(*s).pdd[0].im+t*(*s).pdd[2].im;
}

#else

static void mapX2v(su3_alg_dble *X)
{
   v1.c1.re=0.0;
   v1.c1.im=(*X).c1+(*X).c2;
   v1.c2.re=(*X).c3;
   v1.c2.im=(*X).c4;
   v1.c3.re=(*X).c5;
   v1.c3.im=(*X).c6;

   v2.c1.re=-(*X).c3;
   v2.c1.im=(*X).c4;
   v2.c2.re=0.0;
   v2.c2.im=(*X).c2-2.0*(*X).c1;
   v2.c3.re=(*X).c7;
   v2.c3.im=(*X).c8;

   v3.c1.re=-(*X).c5;
   v3.c1.im=(*X).c6;
   v3.c2.re=-(*X).c7;
   v3.c2.im=(*X).c8;
   v3.c3.re=0.0;
   v3.c3.im=(*X).c1-2.0*(*X).c2;   
}


void ch2mat(complex_dble *p,su3_alg_dble *X,su3_dble *u)
{
   complex_dble z;
   
   mapX2v(X);
   
   (*u).c11.re=p[0].re-p[1].im*v1.c1.im;
   (*u).c11.im=p[0].im+p[1].re*v1.c1.im;
   (*u).c12.re=p[1].re*v1.c2.re-p[1].im*v1.c2.im;
   (*u).c12.im=p[1].re*v1.c2.im+p[1].im*v1.c2.re;
   (*u).c13.re=p[1].re*v1.c3.re-p[1].im*v1.c3.im;
   (*u).c13.im=p[1].re*v1.c3.im+p[1].im*v1.c3.re;    

   (*u).c21.re=p[1].re*v2.c1.re-p[1].im*v2.c1.im;
   (*u).c21.im=p[1].re*v2.c1.im+p[1].im*v2.c1.re;    
   (*u).c22.re=p[0].re-p[1].im*v2.c2.im;
   (*u).c22.im=p[0].im+p[1].re*v2.c2.im;
   (*u).c23.re=p[1].re*v2.c3.re-p[1].im*v2.c3.im;
   (*u).c23.im=p[1].re*v2.c3.im+p[1].im*v2.c3.re;    
   
   (*u).c31.re=p[1].re*v3.c1.re-p[1].im*v3.c1.im;
   (*u).c31.im=p[1].re*v3.c1.im+p[1].im*v3.c1.re;    
   (*u).c32.re=p[1].re*v3.c2.re-p[1].im*v3.c2.im;
   (*u).c32.im=p[1].re*v3.c2.im+p[1].im*v3.c2.re;    
   (*u).c33.re=p[0].re-p[1].im*v3.c3.im;
   (*u).c33.im=p[0].im+p[1].re*v3.c3.im;

   z.re=_vector_prod_re(v1,v1);
   (*u).c11.re-=p[2].re*z.re;
   (*u).c11.im-=p[2].im*z.re;

   z.re=_vector_prod_re(v2,v2);
   (*u).c22.re-=p[2].re*z.re;
   (*u).c22.im-=p[2].im*z.re;
   
   z.re=_vector_prod_re(v3,v3);
   (*u).c33.re-=p[2].re*z.re;
   (*u).c33.im-=p[2].im*z.re;
   
   z.re=_vector_prod_re(v1,v2);
   z.im=_vector_prod_im(v1,v2);   
   (*u).c12.re-=p[2].re*z.re+p[2].im*z.im;
   (*u).c12.im-=p[2].im*z.re-p[2].re*z.im;
   (*u).c21.re-=p[2].re*z.re-p[2].im*z.im;
   (*u).c21.im-=p[2].im*z.re+p[2].re*z.im;   

   z.re=_vector_prod_re(v1,v3);
   z.im=_vector_prod_im(v1,v3);   
   (*u).c13.re-=p[2].re*z.re+p[2].im*z.im;
   (*u).c13.im-=p[2].im*z.re-p[2].re*z.im;
   (*u).c31.re-=p[2].re*z.re-p[2].im*z.im;
   (*u).c31.im-=p[2].im*z.re+p[2].re*z.im;   

   z.re=_vector_prod_re(v2,v3);
   z.im=_vector_prod_im(v2,v3);   
   (*u).c23.re-=p[2].re*z.re+p[2].im*z.im;
   (*u).c23.im-=p[2].im*z.re-p[2].re*z.im;
   (*u).c32.re-=p[2].re*z.re-p[2].im*z.im;
   (*u).c32.im-=p[2].im*z.re+p[2].re*z.im;   
}


void chexp_drv0(su3_alg_dble *X,ch_drv0_t *s)
{
   int n;
   complex_dble q0,q1,q2;

   if (init_flag==0)
      ch_init();

   if (init_flag==2)
   {
      (*s)=sp0;
      (*s).p[0].re=1.0;
      return;
   }

   eval_td(X);
   (*s)=sp0;
   (*s).t=t;
   (*s).d=d;
   (*s).p[0].re=c[N-6];
   
   for (n=(N-7);n>=0;n--)
   {
      q0=(*s).p[0];
      q1=(*s).p[1];
      q2=(*s).p[2];
      
      (*s).p[0].re=c[n]+d*q2.im;
      (*s).p[0].im=-d*q2.re;
      (*s).p[1].re=q0.re-t*q2.re;
      (*s).p[1].im=q0.im-t*q2.im;
      (*s).p[2].re=q1.re;
      (*s).p[2].im=q1.im;
   }
}


void chexp_drv1(su3_alg_dble *X,ch_drv1_t *s)
{
   int n;
   complex_dble q0,q1,q2;
   complex_dble q0d,q1d,q2d;

   if (init_flag==0)
      ch_init();

   if (init_flag==2)
   {
      (*s)=sp1;
      (*s).p[0].re=1.0;
      return;
   }

   eval_td(X);   
   (*s)=sp1;
   (*s).t=t;
   (*s).d=d;
   (*s).p[0].re=c[N-3];
   
   for (n=(N-4);n>=0;n--)
   {
      q0=(*s).p[0];
      q1=(*s).p[1];
      q2=(*s).p[2];

      (*s).p[0].re=c[n]+d*q2.im;
      (*s).p[0].im=-d*q2.re;
      (*s).p[1].re=q0.re-t*q2.re;
      (*s).p[1].im=q0.im-t*q2.im;
      (*s).p[2].re=q1.re;
      (*s).p[2].im=q1.im;

      q0d=(*s).pd[0];
      q1d=(*s).pd[1];
      q2d=(*s).pd[2];
      
      (*s).pd[0].re= q2.im+d*q2d.im;
      (*s).pd[0].im=-q2.re-d*q2d.re;
      (*s).pd[1].re=q0d.re-t*q2d.re;
      (*s).pd[1].im=q0d.im-t*q2d.im;
      (*s).pd[2].re=q1d.re;
      (*s).pd[2].im=q1d.im;
   }

   (*s).pt[0].re=-d*(*s).pd[2].re;
   (*s).pt[0].im=-d*(*s).pd[2].im;
   (*s).pt[1].re= (*s).pd[0].im-t*(*s).pd[2].im;
   (*s).pt[1].im=-(*s).pd[0].re+t*(*s).pd[2].re;
   (*s).pt[2].re= (*s).pd[1].im;
   (*s).pt[2].im=-(*s).pd[1].re;
}


void chexp_drv2(su3_alg_dble *X,ch_drv2_t *s)
{
   int n;
   complex_dble q0,q1,q2;
   complex_dble q0d,q1d,q2d;
   complex_dble q0dd,q1dd,q2dd;

   if (init_flag==0)
      ch_init();

   if (init_flag==2)
   {
      (*s)=sp2;
      (*s).p[0].re=1.0;
      return;
   }

   eval_td(X);   
   (*s)=sp2;
   (*s).t=t;
   (*s).d=d;
   (*s).p[0].re=c[N];
   
   for (n=(N-1);n>=0;n--)
   {
      q0=(*s).p[0];
      q1=(*s).p[1];
      q2=(*s).p[2];

      (*s).p[0].re=c[n]+d*q2.im;
      (*s).p[0].im=-d*q2.re;
      (*s).p[1].re=q0.re-t*q2.re;
      (*s).p[1].im=q0.im-t*q2.im;
      (*s).p[2].re=q1.re;
      (*s).p[2].im=q1.im;

      q0d=(*s).pd[0];
      q1d=(*s).pd[1];
      q2d=(*s).pd[2];
      
      (*s).pd[0].re= q2.im+d*q2d.im;
      (*s).pd[0].im=-q2.re-d*q2d.re;
      (*s).pd[1].re=q0d.re-t*q2d.re;
      (*s).pd[1].im=q0d.im-t*q2d.im;
      (*s).pd[2].re=q1d.re;
      (*s).pd[2].im=q1d.im;

      q0dd=(*s).pdd[0];
      q1dd=(*s).pdd[1];
      q2dd=(*s).pdd[2];
      
      (*s).pdd[0].re= 2.0*q2d.im+d*q2dd.im;
      (*s).pdd[0].im=-2.0*q2d.re-d*q2dd.re;
      (*s).pdd[1].re=q0dd.re-t*q2dd.re;
      (*s).pdd[1].im=q0dd.im-t*q2dd.im;
      (*s).pdd[2].re=q1dd.re;
      (*s).pdd[2].im=q1dd.im;
   }

   (*s).pt[0].re=-d*(*s).pd[2].re;
   (*s).pt[0].im=-d*(*s).pd[2].im;
   (*s).pt[1].re= (*s).pd[0].im-t*(*s).pd[2].im;
   (*s).pt[1].im=-(*s).pd[0].re+t*(*s).pd[2].re;
   (*s).pt[2].re= (*s).pd[1].im;
   (*s).pt[2].im=-(*s).pd[1].re;

   (*s).ptd[0].re=-(*s).pd[2].re-d*(*s).pdd[2].re;
   (*s).ptd[0].im=-(*s).pd[2].im-d*(*s).pdd[2].im;
   (*s).ptd[1].re= (*s).pdd[0].im-t*(*s).pdd[2].im;
   (*s).ptd[1].im=-(*s).pdd[0].re+t*(*s).pdd[2].re;
   (*s).ptd[2].re= (*s).pdd[1].im;
   (*s).ptd[2].im=-(*s).pdd[1].re;

   (*s).ptt[0].re=-d*(*s).pdd[1].im;
   (*s).ptt[0].im= d*(*s).pdd[1].re;
   (*s).ptt[1].re=-2.0*(*s).pd[2].im+t*(*s).pdd[1].re-d*(*s).pdd[2].im;
   (*s).ptt[1].im= 2.0*(*s).pd[2].re+t*(*s).pdd[1].im+d*(*s).pdd[2].re;
   (*s).ptt[2].re=-(*s).pdd[0].re+t*(*s).pdd[2].re;
   (*s).ptt[2].im=-(*s).pdd[0].im+t*(*s).pdd[2].im;
}

#endif

