
/*******************************************************************************
*
* File tcharge.c
*
* Copyright (C) 2010, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Computation of the topological charge
*
* The externally accessible functions are
*
*   double tcharge(void)
*     Returns the "field-theoretic" topological charge Q of the global
*     double-precision gauge field, using a symmetric expression for the
*     gauge-field tensor.
*
*   double tcharge_slices(double *qsl)
*     Computes the sum qsl[t] of the "field-theoretic" topological charge 
*     density of the double-precision gauge field at time t=0,1,...,N0-1
*     (where N0=NPROC0*L0). The program returns the total charge.
*
* Notes:
*
* The topological charge density q(x) is defined by
*
*  q(x)=w(x0)*(8*Pi^2)^(-1)*{F_{01}^a(x)*F_{23}^a(x)+
*                            F_{02}^a(x)*F_{31}^a(x)+
*                            F_{03}^a(x)*F_{12}^a(x)}
*
* where
*
*  F_{mu,nu}^a(x)=-2*tr{F_{mu,nu}(x)*T^a}, a=1,..,8,
*
* are the SU(3) components of the symmetric field tensor returned by the
* program ftensor() [ftensor.c]. The weight w(x_0) is equal to 1/2 at time
* 0 and N0-1 and equal to 1 elsewhere.
*
* The programs in this module perform global communications and must be
* called simultaneously on all processes.
*
*******************************************************************************/

#define TCHARGE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "lattice.h"
#include "tcharge.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define MAX_LEVELS 8
#define BLK_LENGTH 8

static int cnt[L0][MAX_LEVELS];
static double smx[L0][MAX_LEVELS],qsl0[N0];
static u3_alg_dble **ft;


static double prodXY(u3_alg_dble *X,u3_alg_dble *Y)
{
   double sm;
   
   sm=(-2.0/3.0)*((*X).c1+(*X).c2+(*X).c3)*((*Y).c1+(*Y).c2+(*Y).c3)+
      2.0*((*X).c1*(*Y).c1+(*X).c2*(*Y).c2+(*X).c3*(*Y).c3)+
      4.0*((*X).c4*(*Y).c4+(*X).c5*(*Y).c5+(*X).c6*(*Y).c6+
           (*X).c7*(*Y).c7+(*X).c8*(*Y).c8+(*X).c9*(*Y).c9);

   return sm;
}


static double density(int ix)
{
   int t;
   double sm;
   
   sm=prodXY(ft[0]+ix,ft[3]+ix)+
      prodXY(ft[1]+ix,ft[4]+ix)+
      prodXY(ft[2]+ix,ft[5]+ix);

   t=global_time(ix);

   if ((t==0)||(t==(N0-1)))
      sm*=0.5;
   
   return sm;
}


double tcharge(void)
{
   int n,ix,*cnt0;
   double pi,Q,*smx0;
   
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

   MPI_Reduce(smx0,&Q,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Bcast(&Q,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   pi=4.0*atan(1.0);

   return Q/(8.0*pi*pi);
}


double tcharge_slices(double *qsl)
{
   int n,t,t0,ix;
   double pi,fact,Q;
   
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
      qsl0[t]=0.0;

   pi=4.0*atan(1.0);
   fact=1.0/(8.0*pi*pi);

   for (t=0;t<L0;t++)
   {
      for (n=1;n<MAX_LEVELS;n++)
         smx[t][0]+=smx[t][n];

      qsl0[t+t0]=fact*smx[t][0];
   }
   
   MPI_Reduce(qsl0,qsl,N0,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Bcast(qsl,N0,MPI_DOUBLE,0,MPI_COMM_WORLD);

   Q=0.0;

   for (t=0;t<N0;t++)
      Q+=qsl[t];

   return Q;
}
