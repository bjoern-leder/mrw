
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2009, 2010, 2011, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Topological charge of constant abelian background fields
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "su3fcts.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "tcharge.h"
#include "global.h"

static double mt[4][4];
static su3_dble ud0={{0.0}};


static double Qmt(void)
{
   int i,mu,nu,ro,si;
   double xl[4],pi,sm,phi,tr;
   double ft1,ft2,ft3,fs1,fs2,fs3;

   xl[0]=(double)(NPROC0*L0);
   xl[1]=(double)(NPROC1*L1);
   xl[2]=(double)(NPROC2*L2);
   xl[3]=(double)(NPROC3*L3);

   pi=4.0*atan(1.0);
   sm=0.0;
   
   mu=0;
   nu=1;
   ro=2;
   si=3;
   
   for (i=0;i<3;i++)
   {
      phi=2.0*pi*mt[mu][nu]/(xl[mu]*xl[nu]);
         
      ft1=sin(phi);
      ft2=ft1;
      ft3=-sin(2.0*phi);

      tr=(ft1+ft2+ft3)/3.0;
         
      ft1-=tr;
      ft2-=tr;
      ft3-=tr;
      
      phi=2.0*pi*mt[ro][si]/(xl[ro]*xl[si]);
         
      fs1=sin(phi);
      fs2=fs1;
      fs3=-sin(2.0*phi);

      tr=(fs1+fs2+fs3)/3.0;

      fs1-=tr;
      fs2-=tr;
      fs3-=tr;      

      sm+=(ft1*fs1+ft2*fs2+ft3*fs3);

      nu=nu+1;
      ro=(ro+1)%4+(ro==3);
      si=(si+1)%4+(si==3);      
   }

   return sm*(xl[0]-1.0)*xl[1]*xl[2]*xl[3]/(4.0*pi*pi);
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
}


int main(int argc,char *argv[])
{
   int my_rank,i;
   double Q1,Q2,d,dmax;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      printf("\n");
      printf("Topological charge of constant abelian background fields\n");
      printf("--------------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,123);
   geometry();
   dmax=0.0;

   for (i=0;i<10;i++)
   {
      choose_mt();
      set_ud();
      Q1=Qmt();      
      Q2=tcharge();

      if (my_rank==0)
         printf("Field no = %2d, Q1 = % 8.4e, Q2 = % 8.4e\n",i+1,Q1,Q2);

      d=fabs(Q1-Q2);
      if (d>dmax)
         dmax=d;
   }

   error_chk();

   if (my_rank==0)
   {
      printf("\n");
      printf("Maximal absolute deviation = %.1e\n\n",dmax);
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
