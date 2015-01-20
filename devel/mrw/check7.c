
/*******************************************************************************
*
* File check7.c
*
* Copyright (C) 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Direct check of mrw2eo
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


int main(int argc,char *argv[])
{
   int my_rank,irw,isp,ispp[2],status[6],mnkv;
   int bs[4],Ns,nmx,nkv,nmr,ncy,ninv;
   double kappa,m0,dm,mu0,mu,res,mres;
   double sqne,sqnp[2];
   complex_dble lnw1[2],lnr,dr,drmx;
   solver_parms_t sp;
   mrw_masses_t ms;
   FILE *flog=NULL,*fin=NULL;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check7.log","w",stdout);
      fin=freopen("check7.in","r",stdin);
      
      printf("\n");
      printf("Direct check of mrw2eo\n");
      printf("------------------------\n\n");
      
      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   mnkv=0;
   
   mres=0.0;
   for (isp=0;isp<3;isp++)
   {
      read_solver_parms(isp);
      sp=solver_parms(isp);

      if (sp.res>mres)
         mres=sp.res;
      
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
   drmx.re=0.0;
   drmx.im=0.0;
    
   for (irw=0;irw<3;irw++)
   {
      dm=1.0e-2;
      
      for (isp=0;isp<2;isp++)
      {
         ispp[0]=isp;
         ispp[1]=isp;
         if (isp==0)
         {
            m0=1.0877;
            mu0=0.1;
         }
         else if (isp==1)
         {
            m0=0.0877;
            mu0=0.01;
         }
         else
         {
            m0=-0.0123;
            mu0=0.001;
         }
      
         random_ud();

         if (isp==2)
         {
            dfl_modes(status);
            error_root(status[0]<0,1,"main [check7.c]",
                        "dfl_modes failed");
         }      
         
         if (irw==0)
         {
            ms.m1=m0;
            ms.d1=dm;
            ms.mu1=mu0;
            ms.m2=m0;
            ms.d2=dm;
            ms.mu2=mu0;

            lnr=mrw2eo(ms,1,ispp,lnw1,sqnp,&sqne,status);            
            dr.re=fabs(lnr.re-(2.0*mu0*dm+dm*dm)*sqnp[0]);
            dr.re+=fabs(lnw1[0].re-lnw1[1].re);
            dr.re+=fabs(sqnp[0]-sqnp[1]);
            dr.im=fabs(lnr.im);
            dr.im+=fabs(lnw1[0].im-lnw1[1].im);
         }
         else
         {
            ms.m1=m0;
            ms.d1=dm;
            ms.mu1=mu0;
            ms.m2=m0;
            ms.d2=-dm;
            ms.mu2=mu0;

            lnr=mrw2eo(ms,1,ispp,lnw1,sqnp,&sqne,status);            
            dr.re=fabs(lnr.re+dm*dm*sqnp[0]);
            dr.re+=fabs(lnw1[0].re+lnw1[1].re);
            dr.re+=fabs(sqnp[0]-sqnp[1]);
            dr.im=fabs(lnr.im-2.0*lnw1[0].im);
            dr.im+=fabs(lnw1[0].im+lnw1[1].im);
         }
         
         if (dr.re>drmx.re)
            drmx.re=dr.re;
         if (dr.im>drmx.im)
            drmx.im=dr.im;

         if (my_rank==0)
         {
            if (irw==0)
               printf("mrw2eo(d2=d1): ");
            else if (irw==1)
               printf("mrw2eo(d2=-d1): ");
            
            if ((isp==0)||(isp==1))
               printf("status = %d\n",status[0]);
            else if (isp==2)
               printf("status = (%d,%d,%d)\n",
                        status[0],status[1],status[2]);

            printf("diff = %.1e + i%.1e\n\n",dr.re,dr.im);
         }      
      
         error_chk();
      }
   }
   
   if (my_rank==0)
   {
      printf("\n");
      printf("max diff = %.1e + i%.1e\n",drmx.re,drmx.im);
      printf("(should be smaller than %.1e)\n\n",mres*sqrt((double)(VOLUME/2*NPROC*24)));
      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
