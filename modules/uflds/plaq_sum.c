
/*******************************************************************************
*
* File plaq_sum.c
*
* Copyright (C) 2005, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Calculation of plaquette sums
*
* The externally accessible functions are
*
*   double plaq_sum_dble(int icom)
*     Returns the sum of Re(tr{U(p)}) over all unoriented plaquettes p,
*     where U(p) is the product of the double-precision link variables
*     around p. The sum runs over all plaquettes with lower-left corner
*     in the local lattice if icom!=1 and over all plaquettes in the
*     global lattice if icom=1.
*
*   double plaq_wsum_dble(int icom)
*     Returns the sum of w(p)*Re(tr{U(p)}) over all unoriented plaquettes
*     p, where w(p)=0 for all time-like p at time NPROC0*L0-1, w(p)=1/2 
*     for all space-like plaquettes at time 0 and NPROC0*L0-1 and w(p)=1
*     in all other cases. The sum runs over all plaquettes with lower-left
*     corner in the local lattice if icom!=1 and over all plaquettes in
*     the global lattice if icom=1.
*
*   double plaq_action_slices(double *asl)
*     Computes the sum asl[t] of the Wilson plaquette action density
*     (w/o prefactor 1/g0^2) of the double-precision gauge field at time
*     t=0,1,..,NPROC0*L0-1 (see the notes). The program returns the total
*     action.
*
* Notes:
*
* If icom=1, the values returned by plaq_sum_dble() and plaq_wsum_dble()
* are guaranteed to be the same on all MPI processes.
*
* The Wilson plaquette action density is defined in a way analogous to
* the Yang-Mills action density computed by the programs in the module
* tcharge/ym_action.c. Explicitly, for 0<x0<(N0-1) the density s(x) is
* given by
*
*   s(x)=2*Retr{6-0.5*[U_{01}(x)+U_{01}(x-0)+
*                      U_{02}(x)+U_{02}(x-0)+
*                      U_{03}(x)+U_{03}(x-0)]-U_{12}(x)-U_{23}(x)-U_{31}(x)}
*
* where U_{mu nu}(x) denotes the product of the link variables around the
* (mu,nu)-plaquette with lower-left corner x. At the boundary time-slices
*
*   s(x)=Retr{6-U_{01}(x)-U_{02}(x)-U_{03}(x)-
*               U_{12}(x)-U_{23}(x)-U_{31}(x)}      if x0=0 
*
*   s(x)=Retr{6-U_{01}(x-0)-U_{02}(x-0)-U_{03}(x-0)-
*               U_{12}(x)-U_{23}(x)-U_{31}(x)}      if x0=NPROC0*L0-1
*
* independently of the boundary conditions on the gauge field.
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define PLAQ_SUM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "flags.h"
#include "su3fcts.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define MAX_LEVELS 8
#define BLK_LENGTH 8

static int cnt[L0][MAX_LEVELS];
static double smE[L0][MAX_LEVELS],smB[L0][MAX_LEVELS];
static double aslE[N0],aslB[N0];
static su3_dble *udb;
static su3_dble wd1,wd2 ALIGNED16;


static double plaq_dble(int n,int ix)
{
   int ip[4];
   double sm;

   plaq_uidx(n,ix,ip);

   su3xsu3(udb+ip[0],udb+ip[1],&wd1);
   su3dagxsu3dag(udb+ip[3],udb+ip[2],&wd2);
   cm3x3_retr(&wd1,&wd2,&sm);
   
   return sm;
}


static double local_plaq_sum_dble(int iw)
{
   int n,ix,t,*cnt0;
   double pa,*smx0;

   udb=udfld();
   cnt0=cnt[0];
   smx0=smE[0];
   
   for (n=0;n<MAX_LEVELS;n++)
   {
      cnt0[n]=0;
      smx0[n]=0.0;
   }
   
   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix);
      pa=0.0;

      for (n=0;n<6;n++)
      {
         if (iw==0)
            pa+=plaq_dble(n,ix);
         else
         {
            if (((t>0)&&(t<(N0-1)))||((t==0)&&(n<3)))
               pa+=plaq_dble(n,ix);
            else if (n>=3)
               pa+=0.5*plaq_dble(n,ix);               
         }
      }

      cnt0[0]+=1;
      smx0[0]+=pa;

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
   
   return smx0[0];
}


double plaq_sum_dble(int icom)
{
   double p,pa;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   p=local_plaq_sum_dble(0);

   if (icom==1)
   {
      MPI_Reduce(&p,&pa,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&pa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      pa=p;
   
   return pa;
}


double plaq_wsum_dble(int icom)
{
   double p,pa;

   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();

   p=local_plaq_sum_dble(1);

   if (icom==1)
   {
      MPI_Reduce(&p,&pa,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
      MPI_Bcast(&pa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   }
   else
      pa=p;
   
   return pa;
}


double plaq_action_slices(double *asl)
{
   int n,ix,t,t0;
   double sE,sB,A;
   
   if (query_flags(UDBUF_UP2DATE)!=1)
      copy_bnd_ud();
   
   for (t=0;t<L0;t++)
   {
      for (n=0;n<MAX_LEVELS;n++)
      {
         cnt[t][n]=0;
         smE[t][n]=0.0;
         smB[t][n]=0.0;
      }
   }

   t0=cpr[0]*L0;
   udb=udfld();

   for (ix=0;ix<VOLUME;ix++)
   {
      t=global_time(ix);            
      sE=0.0;
      sB=0.0;

      if (t<(N0-1))
      {
         for (n=0;n<3;n++)
            sE+=(3.0-plaq_dble(n,ix));
      }
      
      for (n=3;n<6;n++)
         sB+=(3.0-plaq_dble(n,ix));      

      t-=t0;            
      smE[t][0]+=sE;
      smB[t][0]+=sB;
      cnt[t][0]+=1;

      for (n=1;(cnt[t][n-1]>=BLK_LENGTH)&&(n<MAX_LEVELS);n++)
      {
         cnt[t][n]+=1;
         smE[t][n]+=smE[t][n-1];
         smB[t][n]+=smB[t][n-1];
         
         cnt[t][n-1]=0;
         smE[t][n-1]=0.0;
         smB[t][n-1]=0.0;
      }      
   }

   for (t=0;t<L0;t++)
   {
      for (n=1;n<MAX_LEVELS;n++)
      {
         smE[t][0]+=smE[t][n];
         smB[t][0]+=smB[t][n];
      }
   }
   
   for (t=0;t<N0;t++)
      asl[t]=0.0;

   for (t=0;t<L0;t++)
      asl[t+t0]=smE[t][0];

   MPI_Reduce(asl,aslE,N0,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);   
   MPI_Bcast(aslE,N0,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   for (t=0;t<N0;t++)
      asl[t]=0.0;
   
   for (t=0;t<L0;t++)
      asl[t+t0]=smB[t][0];

   MPI_Reduce(asl,aslB,N0,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   MPI_Bcast(aslB,N0,MPI_DOUBLE,0,MPI_COMM_WORLD);

   asl[0]=aslE[0]+aslB[0];
   asl[N0-1]=aslE[N0-2]+aslB[N0-1];
   
   for (t=1;t<(N0-1);t++)
      asl[t]=aslE[t-1]+aslE[t]+2.0*aslB[t];

   A=0.0;

   for (t=0;t<N0;t++)
      A+=asl[t];

   return A;
}
