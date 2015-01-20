
/*******************************************************************************
*
* File ym_action.c
*
* Copyright (C) 2010, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the Yang-Mills action using the symmetric field tensor
*
* The externally accessible functions are
*
*   double ym_action(void)
*     Returns the Yang-Mills action (w/o prefactor 1/g0^2) of the
*     double-precision gauge field, using a symmetric expression for 
*     the gauge-field tensor.
*
*   double ym_action_slices(double *asl)
*     Computes the sum asl[t] of the Yang-Mills action density (w/o
*     prefactor 1/g0^2) of the double-precision gauge field at time
*     t=0,1,...,N0-1 (where N0=NPROC0*L0). The program returns the
*     total action.
*
* Notes:
*
* The Yang-Mills action density s(x) is defined by
*
*  s(x)=w(x_0)*(1/4)*sum_{mu,nu} [F_{mu,nu}^a(x)]^2
*
* where
*
*  F_{mu,nu}^a(x)=-2*tr{F_{mu,nu}(x)*T^a}, a=1,..,8,
*
* are the SU(3) components of the symmetric field tensor returned by the
* program ftensor() [ftensor.c]. The weight w(x_0) is equal to 1/2 at time
* 0 and N0-1 and equal to 1 elsewhere.
*
* The programs in this module perform global operations and must be called
* simultaneously on all MPI processes.
*
*******************************************************************************/

#define YM_ACTION_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "linalg.h"
#include "tcharge.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define MAX_LEVELS 12
#define BLK_LENGTH 8

static int cnt[L0][MAX_LEVELS];
static double smx[L0][MAX_LEVELS],asl0[N0];
static u3_alg_dble **ft;


static double prodXX(u3_alg_dble *X)
{
   double sm;
   
   sm=(-2.0/3.0)*((*X).c1+(*X).c2+(*X).c3)*((*X).c1+(*X).c2+(*X).c3)+
      2.0*((*X).c1*(*X).c1+(*X).c2*(*X).c2+(*X).c3*(*X).c3)+
      4.0*((*X).c4*(*X).c4+(*X).c5*(*X).c5+(*X).c6*(*X).c6+
           (*X).c7*(*X).c7+(*X).c8*(*X).c8+(*X).c9*(*X).c9);

   return sm;
}


static double density(int ix)
{
   int t;
   double sm;
   
   sm=prodXX(ft[0]+ix)+prodXX(ft[1]+ix)+prodXX(ft[2]+ix)+
      prodXX(ft[3]+ix)+prodXX(ft[4]+ix)+prodXX(ft[5]+ix);

   t=global_time(ix);

   if ((t==0)||(t==(N0-1)))
      sm*=0.5;
   
   return 0.5*sm;
}


double ym_action(void)
{
   int n,ix,*cnt0;
   double s,*smx0;
   
   ft=ftensor();
   cnt0=cnt[0];
   smx0=smx[0];
   
   for (n=0;n<MAX_LEVELS;n++)
   {
      cnt0[n]=0;
      smx0[n]=0.0;
   }
   
   for (ix=0;ix<VOLUME;ix++)
   {
      cnt0[0]+=1;
      smx0[0]+=density(ix);

      for (n=1;(cnt0[n-1]>=BLK_LENGTH)&&(n<MAX_LEVELS);n++)
      {
         cnt0[n]+=1;
         smx0[n]+=smx0[n-1];

         cnt0[n-1]=0;
         smx0[n-1]=0.0;
      }
   }
   
   for (n=1;n<MAX_LEVELS;n++)
      smx0[0]+=smx0[n];

   MPI_Reduce(smx0,&s,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Bcast(&s,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   return s;
}


double ym_action_slices(double *asl)
{
   int n,t,t0,ix;
   double s;
   
   ft=ftensor();
   
   for (t=0;t<L0;t++)
   {
      for (n=0;n<MAX_LEVELS;n++)
      {
         cnt[t][n]=0;
         smx[t][n]=0.0;
      }
   }

   t0=cpr[0]*L0;

   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix)-t0;
      smx[t][0]+=density(ix);
      cnt[t][0]+=1;

      for (n=1;(cnt[t][n-1]>=BLK_LENGTH)&&(n<MAX_LEVELS);n++)
      {
         cnt[t][n]+=1;
         smx[t][n]+=smx[t][n-1];

         cnt[t][n-1]=0;
         smx[t][n-1]=0.0;
      }      
   }

   for (t=0;t<N0;t++)
      asl0[t]=0.0;

   for (t=0;t<L0;t++)
   {
      for (n=1;n<MAX_LEVELS;n++)
         smx[t][0]+=smx[t][n];

      asl0[t+t0]=smx[t][0];
   }
   
   MPI_Reduce(asl0,asl,N0,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Bcast(asl,N0,MPI_DOUBLE,0,MPI_COMM_WORLD);

   s=0.0;

   for (t=0;t<N0;t++)
      s+=asl[t];

   return s;
}
