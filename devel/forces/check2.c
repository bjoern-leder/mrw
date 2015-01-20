
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Gauge action of constant Abelian background fields
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "forces.h"
#include "global.h"

static double mt[4][4];
static su3_dble ud0={{0.0}};


static double Amt(void)
{
   int mu,nu;
   double beta,c0,c1,cG;
   double smt0,smt1,sms0,sms1,pi;
   double xl[4],phi,s0,s1;
   lat_parms_t lat;

   lat=lat_parms();
   beta=lat.beta;
   c0=lat.c0;
   c1=lat.c1;
   cG=lat.cG;

   xl[0]=(double)(NPROC0*L0);
   xl[1]=(double)(NPROC1*L1);
   xl[2]=(double)(NPROC2*L2);
   xl[3]=(double)(NPROC3*L3);

   pi=4.0*atan(1.0);
   smt0=0.0;
   smt1=0.0;
   sms0=0.0;
   sms1=0.0;

   for (mu=1;mu<4;mu++)
   {
      for (nu=0;nu<mu;nu++)
      {
         phi=2.0*pi*mt[mu][nu]/(xl[mu]*xl[nu]);

         s0=3.0-2.0*cos(phi)-cos(2.0*phi);
         s1=3.0-2.0*cos(2.0*phi)-cos(4.0*phi);

         if (nu==0)
         {
            smt0+=s0;
            smt1+=s1;
         }
         else
         {
            sms0+=s0;
            sms1+=s1;
         }
      }
   }

   s0=(xl[0]-1.0)*smt0+(xl[0]-2.0+cG)*sms0;
   s1=(xl[0]-2.0)*smt1+(xl[0]-1.0)*smt1+2.0*(xl[0]-2.0+cG)*sms1;

   return (beta/3.0)*(c0*s0+c1*s1)*xl[1]*xl[2]*xl[3];
}


static void choose_mt(void)
{
   int mu,nu;
   double r[6];

   ranlxd(r,6);
   MPI_Bcast(r,6,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   mt[0][1]=(double)((int)(3.0*r[0])-1);
   mt[0][2]=(double)((int)(3.0*r[1])-1);
   mt[0][3]=(double)((int)(3.0*r[2])-1);
   mt[1][2]=(double)((int)(3.0*r[3])-1);
   mt[1][3]=(double)((int)(3.0*r[4])-1);
   mt[2][3]=(double)((int)(3.0*r[5])-1);   

   for (mu=0;mu<4;mu++)
   {
      mt[mu][mu]=0.0;
            
      for (nu=0;nu<mu;nu++)
         mt[mu][nu]=-mt[nu][mu];
   }
}


static void set_ud(void)
{
   int np[4],bo[4];
   int x0,x1,x2,x3;
   int ix,mu,nu;
   double pi,inp[4],xt[4];
   double phi,sml,smh;
   su3_dble *udb,*u;

   np[0]=NPROC0*L0;
   np[1]=NPROC1*L1;
   np[2]=NPROC2*L2;
   np[3]=NPROC3*L3;   
   
   bo[0]=cpr[0]*L0;
   bo[1]=cpr[1]*L1;
   bo[2]=cpr[2]*L2;
   bo[3]=cpr[3]*L3;

   pi=4.0*atan(1.0);

   inp[0]=1.0/(double)(np[0]);
   inp[1]=1.0/(double)(np[1]);
   inp[2]=1.0/(double)(np[2]);
   inp[3]=1.0/(double)(np[3]);   

   udb=udfld();
   
   for (x0=0;x0<L0;x0++)
   {
      for (x1=0;x1<L1;x1++)
      {
         for (x2=0;x2<L2;x2++)
         {
            for (x3=0;x3<L3;x3++)
            {
               ix=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*x0];

               if (ix>=(VOLUME/2))
               {
                  xt[0]=(double)(bo[0]+x0);
                  xt[1]=(double)(bo[1]+x1);
                  xt[2]=(double)(bo[2]+x2);
                  xt[3]=(double)(bo[3]+x3);                  
                  
                  u=udb+8*(ix-(VOLUME/2));

                  for (mu=0;mu<4;mu++)
                  {
                     smh=0.0;
                     sml=0.0;

                     for (nu=(mu+1);nu<4;nu++)
                        smh+=inp[nu]*mt[mu][nu]*xt[nu];

                     for (nu=0;nu<mu;nu++)
                        sml+=inp[nu]*mt[mu][nu]*xt[nu];

                     (*u)=ud0;
                     phi=inp[mu]*sml;
                     if (xt[mu]==(double)(np[mu]-1))
                        phi+=smh;
                     phi*=(-2.0*pi);
                     (*u).c11.re=cos(phi);
                     (*u).c11.im=sin(phi);
                     (*u).c22.re=(*u).c11.re;
                     (*u).c22.im=(*u).c11.im;
                     (*u).c33.re=cos(-2.0*phi);
                     (*u).c33.im=sin(-2.0*phi); 
                     u+=1;

                     (*u)=ud0;
                     phi=inp[mu]*sml;
                     if (xt[mu]==0.0)
                        phi+=smh;
                     phi*=(-2.0*pi);
                     (*u).c11.re=cos(phi);
                     (*u).c11.im=sin(phi);
                     (*u).c22.re=(*u).c11.re;
                     (*u).c22.im=(*u).c11.im;
                     (*u).c33.re=cos(-2.0*phi);
                     (*u).c33.im=sin(-2.0*phi); 
                     u+=1;
                  }
               }
            }
         }
      }
   }

   openbcd();
   set_flags(UPDATED_UD);
}


int main(int argc,char *argv[])
{
   int my_rank,i;
   double A1,A2,d,dmax;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      printf("\n");
      printf("Gauge action of constant Abelian background fields\n");
      printf("--------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,123);
   geometry();
   set_lat_parms(5.75,1.667,0.0,0.0,0.0,1.234,0.789,1.113);
   dmax=0.0;
   
   for (i=0;i<10;i++)
   {
      choose_mt();
      set_ud();

      A1=Amt();
      A2=action0(1);

      if (my_rank==0)
         printf("Field no = %2d, A1 = %12.6e, A2 = %12.6e\n",i+1,A1,A2);

      d=fabs(A1-A2)/A1;
      if (d>dmax)
         dmax=d;
   }

   error_chk();

   if (my_rank==0)
   {
      printf("\n");
      printf("Maximal relative deviation = %.1e\n\n",dmax);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
