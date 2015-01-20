
/*******************************************************************************
*
* File check9.c
*
* Copyright (C) 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Statistical consistency of mrw2eo
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
#include "mdflds.h"
#include "sflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "global.h"
#include "mrw.h"


static void expc(complex_dble* z, double x,int n)
{
   double c;
   int i;
   
   for (i=0;i<n;i++)
   {
      z[i].re=exp(z[i].re-x);
      c=cos(z[i].im);
      z[i].im=sin(z[i].im)*z[i].re;
      z[i].re*=c;
   }
}


static double minr(complex_dble* z, int n)
{
   double mre;
   int i;
   
   mre=z[0].re;
   for (i=1;i<n;i++)
   {
      if (z[i].re<mre)
         mre=z[i].re;
   }

   return mre;
}


int main(int argc,char *argv[])
{
   int my_rank,irw,isp,ispp[2],status[6],mnkv;
   int bs[4],Ns,nmx,nkv,nmr,ncy,ninv,n,nsrc;
   double kappa,m0,dm,mu0,mu,res;
   double d,lnr0m,lnr1m;
   double sqnp0[2],sqnp1[2],sqne0,sqne1;
   complex_dble lnr0,lnr1,lnw10[2],lnw11[2],dr,r,*z0,*z1,s0,s1;
   solver_parms_t sp;
   mrw_masses_t ms;
   FILE *flog=NULL,*fin=NULL;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check9.log","w",stdout);
      fin=freopen("check9.in","r",stdin);
      
      printf("\n");
      printf("Statistical consistency of mrw2eo\n");
      printf("-----------------------------------\n\n");
      
      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   mnkv=0;
   
   for (isp=0;isp<3;isp++)
   {
      read_solver_parms(isp);
      sp=solver_parms(isp);

      if (sp.nkv>mnkv)
         mnkv=sp.nkv;
   }
   
   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,0,1,1);

   if (my_rank==0)
   {
      find_section("Deflation subspace");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("Ns","%d",&Ns);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);   
   set_dfl_parms(bs,Ns);

   if (my_rank==0)
   {
      find_section("Deflation subspace generation");
      read_line("kappa","%lf",&kappa);
      read_line("mu","%lf",&mu);
      read_line("ninv","%d",&ninv);
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mu,ninv,nmr,ncy);
   
   if (my_rank==0)
   {
      find_section("Deflation projection");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);
      fclose(fin);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);     
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);  
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);      
   set_dfl_pro_parms(nkv,nmx,res);

   set_lat_parms(6.0,1.0,0.0,0.0,0.0,1.234,1.0,1.34);

   print_solver_parms(status,status+1);
   print_sap_parms(0);
   print_dfl_parms(0);
   
   start_ranlux(0,1245);
   geometry();

   mnkv=2*mnkv+2;
   if (mnkv<(Ns+2))
      mnkv=Ns+2;
   if (mnkv<5)
      mnkv=5;
   
   alloc_ws(mnkv);
   alloc_wsd(7);
   alloc_wv(2*nkv+2);
   alloc_wvd(4);
   
   nsrc=24;
   
   z0=malloc(2*nsrc*sizeof(complex_dble));
   z1=z0+nsrc;

   for (irw=0;irw<1;irw++)
   {
      dm=1.0e-2;

      for (isp=0;isp<3;isp++)
      {
         ispp[0]=isp;
         ispp[1]=isp;
         if (isp==0)
         {
            m0=1.0877;
            mu0=0.5;
         }
         else if (isp==1)
         {
            dm/=100.0;
            m0=0.0877;
            mu0=0.1;
         }
         else
         {
            dm/=100.0;
            m0=-0.0123;
            mu0=0.05;
         }
      
         random_ud();

         if (isp==2)
         {
            dfl_modes(status);
            error_root(status[0]<0,1,"main [check3.c]",
                        "dfl_modes failed");
         }      

         if (irw==0)
         {
            for (n=0;n<nsrc;n++)
            {
               ms.m1=m0;
               ms.d1=-dm;
               ms.mu1=mu0;
               ms.m2=m0+dm;
               ms.d2=dm;
               ms.mu2=mu0;
               z0[n]=mrw2eo(ms,1,ispp,lnw10,sqnp0,&sqne0,status);
               
               ms.m1=m0;
               ms.d1=dm;
               ms.mu1=mu0-dm;
               ms.m2=m0+dm;
               ms.d2=-dm;
               ms.mu2=mu0+dm;
               z1[n]=mrw2eo(ms,1,ispp,lnw11,sqnp1,&sqne1,status);
            }
            
            lnr0m=minr(z0,nsrc);
            expc(z0,lnr0m,nsrc);
            lnr1m=minr(z1,nsrc);
            expc(z1,lnr1m,nsrc);
            
            lnr0.re=0.0;
            lnr0.im=0.0;
            lnr1=lnr0;
            for (n=0;n<nsrc;n++)
            {
               lnr0.re+=z0[n].re;
               lnr0.im+=z0[n].im;
               lnr1.re+=z1[n].re;
               lnr1.im+=z1[n].im;
            }
            lnr0.re/=(double)nsrc;
            lnr0.im/=(double)nsrc;
            lnr1.re/=(double)nsrc;
            lnr1.im/=(double)nsrc;

            s0.re=0.0;
            s0.im=0.0;
            s1=s0;
            for (n=0;n<nsrc;n++)
            {
               d=z0[n].re-lnr0.re;
               s0.re+=d*d;
               d=z0[n].im-lnr0.im;
               s0.im+=d*d;

               d=z1[n].re-lnr1.re;
               s1.re+=d*d;
               d=z1[n].im-lnr1.im;
               s1.im+=d*d;
            }
            s0.re/=(double)(nsrc*(nsrc-1));
            s0.im/=(double)(nsrc*(nsrc-1));
            s1.re/=(double)(nsrc*(nsrc-1));
            s1.im/=(double)(nsrc*(nsrc-1));
         }

         r.re=lnr0.re*lnr1.re-lnr0.im*lnr1.im;
         r.im=lnr0.re*lnr1.im+lnr0.im*lnr1.re;
         dr.re=sqrt(lnr0.re*lnr0.re*s1.re+lnr1.re*lnr1.re*s0.re+lnr0.im*lnr0.im*s1.im+lnr1.im*lnr1.im*s0.im);
         dr.im=sqrt(lnr0.re*lnr0.re*s1.im+lnr1.re*lnr1.re*s0.im+lnr0.im*lnr0.im*s1.re+lnr1.im*lnr1.im*s0.re);
         r.re*=exp(lnr0m+lnr1m);
         r.im*=exp(lnr0m+lnr1m);
         dr.re*=exp(lnr0m+lnr1m);
         dr.im*=exp(lnr0m+lnr1m);
         r.re=fabs(1.0-r.re);
         r.im=fabs(r.im);

         if (my_rank==0)
         {
            if (irw==0)
               printf("W_2 * W_2tm: ");
            
            if ((isp==0)||(isp==1))
               printf("status = %d\n",status[0]);
            else if (isp==2)
               printf("status = (%d,%d,%d)\n",
                        status[0],status[1],status[2]);

            printf("|1-exp(lnr1+lnr0)| = %.1e + i%.1e (sigma: %.1e + i%.1e)\n\n",r.re,r.im,dr.re,dr.im);
         }      
      
         error_chk();
      }
   }
      
   if (my_rank==0)
   {
      printf("(should be zero within +/- 2*sigma)\n\n");
      fclose(flog);
   }

   MPI_Finalize();    
   exit(0);
}
