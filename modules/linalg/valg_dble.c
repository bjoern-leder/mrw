
/*******************************************************************************
*
* File valg_dble.c
*
* Copyright (C) 2007, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Generic linear algebra routines for double-precision complex fields
*
* The externally accessible functions are
*
*   complex_dble vprod_dble(int n,int icom,complex_dble *v,complex_dble *w)
*     Computes the scalar product of the n-vectors v and w. 
*
*   double vnorm_square_dble(int n,int icom,complex_dble *v)
*     Computes the square of the norm of the n-vector v.
*
*   void mulc_vadd_dble(int n,complex_dble *v,complex_dble *w,complex_dble z)
*     Replaces the n-vector v by v+z*w.
*
*   void vproject_dble(int n,int icom,complex_dble *v,complex_dble *w)
*     Replaces the n-vector v by v-(w,v)*w.
*
*   void vscale_dble(int n,double r,complex_dble *v)
*     Replaces the n-vector v by r*v.
*
*   double vnormalize_dble(int n,int icom,complex_dble *v)
*     Normalizes the n-vector v to unity and returns the norm of the
*     input vector.
*
*   void vrotate_dble(int n,int nv,complex_dble **pv,complex_dble *a)
*     Replaces the n-vectors vk=pv[k], k=0,..,nv-1, by the linear
*     combinations sum_{j=0}^{nv-1} vj*a[n*j+k].
*
* Notes:
*
* All these programs operate on complex n-vectors whose base addresses are
* passed through the arguments. The length n of the arrays is specified by
* the parameter n. Scalar products are globally summed if the parameter
* icom is equal to 1. In this case the calculated values are guaranteed to
* be exactly the same on all processes.
*
* The programs perform no communications except in the case of the scalar
* products if these are globally summed.
*
*******************************************************************************/

#define VALG_DBLE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "linalg.h"
#include "global.h"

#define MAX_LEVELS 8
#define BLK_LENGTH 32

static int nrot=0,ifail=0;
static int cnt[MAX_LEVELS];
static double smx[MAX_LEVELS],smy[MAX_LEVELS];
static complex_dble *psi;


static void alloc_wrotate(int n)
{
   if (nrot>0)
      afree(psi);
   
   psi=amalloc(n*sizeof(*psi),ALIGN);

   if (psi==NULL)
   {
      error_loc(1,1,"alloc_wrotate [valg_dble.c]",
                "Unable to allocate workspace");
      nrot=0;
      ifail=1;      
   }
   else
      nrot=n;
}


complex_dble vprod_dble(int n,int icom,complex_dble *v,complex_dble *w)
{
   int k;
   complex_dble s,t;
   complex_dble *vm,*vb;

   for (k=0;k<MAX_LEVELS;k++)
   {
      cnt[k]=0;
      smx[k]=0.0;
      smy[k]=0.0;
   }

   vm=v+n;
   
   for (vb=v;vb<vm;)
   {
      vb+=BLK_LENGTH;
      if (vb>vm)
         vb=vm;
      s.re=0.0;
      s.im=0.0;

      for (;v<vb;v++)
      {
         s.re+=((*v).re*(*w).re+(*v).im*(*w).im);
         s.im+=((*v).re*(*w).im-(*v).im*(*w).re);
         w+=1;
      }

      cnt[0]+=1;
      smx[0]+=s.re;
      smy[0]+=s.im;

      for (k=1;(cnt[k-1]>=BLK_LENGTH)&&(k<MAX_LEVELS);k++)
      {
         cnt[k]+=1;
         smx[k]+=smx[k-1];
         smy[k]+=smy[k-1];

         cnt[k-1]=0;
         smx[k-1]=0.0;
         smy[k-1]=0.0;
      }
   }

   s.re=0.0;
   s.im=0.0;

   for (k=0;k<MAX_LEVELS;k++)
   {
      s.re+=smx[k];
      s.im+=smy[k];
   }

   if ((icom!=1)||(NPROC==1))
      return s;
   else
   {
      MPI_Reduce(&s.re,&t.re,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&t.re,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      return t;
   }
}

 
double vnorm_square_dble(int n,int icom,complex_dble *v)
{
   int k;
   double s,t;
   complex_dble *vm,*vb;

   for (k=0;k<MAX_LEVELS;k++)
   {
      cnt[k]=0;
      smx[k]=0.0;
   }

   vm=v+n;

   for (vb=v;vb<vm;)
   {
      vb+=BLK_LENGTH;
      if (vb>vm)
         vb=vm;
      s=0.0;

      for (;v<vb;v++)
         s+=((*v).re*(*v).re+(*v).im*(*v).im);

      cnt[0]+=1;
      smx[0]+=s;

      for (k=1;(cnt[k-1]>=BLK_LENGTH)&&(k<MAX_LEVELS);k++)
      {
         cnt[k]+=1;
         smx[k]+=smx[k-1];

         cnt[k-1]=0;
         smx[k-1]=0.0;
      }
   }

   s=0.0;

   for (k=0;k<MAX_LEVELS;k++)
      s+=smx[k];

   if ((icom!=1)||(NPROC==1))
      return s;
   else
   {
      MPI_Reduce(&s,&t,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&t,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      return t;    
   }
}


void mulc_vadd_dble(int n,complex_dble *v,complex_dble *w,complex_dble z)
{
   complex_dble *vm;

   vm=v+n;
   
   for (;v<vm;v++)
   {
      (*v).re+=(z.re*(*w).re-z.im*(*w).im);
      (*v).im+=(z.re*(*w).im+z.im*(*w).re);
      w+=1;      
   }
}


void vproject_dble(int n,int icom,complex_dble *v,complex_dble *w)
{
   complex_dble z;

   z=vprod_dble(n,icom,w,v);
   z.re=-z.re;
   z.im=-z.im;
   mulc_vadd_dble(n,v,w,z);
}


void vscale_dble(int n,double r,complex_dble *v)
{
   complex_dble *vm;
   
   vm=v+n;
   
   for (;v<vm;v++)
   {
      (*v).re*=r;
      (*v).im*=r;      
   }
}


double vnormalize_dble(int n,int icom,complex_dble *v)
{
   double r;

   r=vnorm_square_dble(n,icom,v);
   r=sqrt(r);

   if (r==0.0)
   {
      error_loc(r==0.0,1,"vnormalize_dble [valg_dble.c]",
                "Vector field has vanishing norm");
      return 0.0;
   }

   vscale_dble(n,1.0/r,v);
   
   return r;
}


void vrotate_dble(int n,int nv,complex_dble **pv,complex_dble *a)
{
   int i,k,j;
   complex_dble s,*z,*vj;

   if ((nv>nrot)&&(ifail==0))
      alloc_wrotate(nv);

   if ((nv>0)&&(ifail==0))
   {
      for (i=0;i<n;i++)
      {
         for (k=0;k<nv;k++)  
         {
            z=a+k;
            s.re=0.0;
            s.im=0.0;
     
            for (j=0;j<nv;j++)
            {
               vj=pv[j]+i;
               s.re+=((*z).re*(*vj).re-(*z).im*(*vj).im);
               s.im+=((*z).re*(*vj).im+(*z).im*(*vj).re);
               z+=nv;               
            }

            psi[k].re=s.re;
            psi[k].im=s.im;
         }

         for (k=0;k<nv;k++)
         {
            pv[k][i].re=psi[k].re;
            pv[k][i].im=psi[k].im;
         }
      }
   }
}
