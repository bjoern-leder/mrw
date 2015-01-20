
/*******************************************************************************
*
* File check5.c
*
* Copyright (C) 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Equivalenz of mrw1 and rwtm1, mrw3 and rwtm2, and direct test of mrw3
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
#include "update.h"
#include "mrw.h"


int main(int argc,char *argv[])
{
   int my_rank,isp,ispp[2],irw,status[9],mnkv;
   int bs[4],Ns,nmx,nkv,nmr,ncy,ninv;
   double kappa,mu,m0,mu0,dm,res,mres;
   double ds,dsmx;
   double sqn0,sqn1,lnr0,lnr1,dr,drmx,sqnp[2];
   complex_dble lnrw1[2];
   solver_parms_t sp;
   mrw_masses_t ms;
   FILE *flog=NULL,*fin=NULL;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check5.log","w",stdout);
      fin=freopen("check5.in","r",stdin);
      
      printf("\n");
      printf("Equivalenz of mrw1 and rwtm1, mrw3 and rwtm2, and direct test of mrw3\n");
      printf("-----------------------------------------------------------------------\n\n");
      
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
   drmx=0.0;
   dsmx=0.0;    
    

   for (irw=0;irw<4;irw++)
   {
      dm=1.0e-1;
      
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
            dm/=10.0;
            m0=0.0877;
            mu0=0.1;
         }
         else
         {
            dm/=10.0;
            m0=-0.0123;
            mu0=0.05;
         }
      
         random_ud();

         if (isp==2)
         {
            dfl_modes(status);
            error_root(status[0]<0,1,"main [check5.c]",
                        "dfl_modes failed");
         }      
         
         if (irw==0)
         {
            ms.m1=m0;
            ms.d1=dm;
            ms.mu1=0.0;
            ms.m2=0.0;
            ms.d2=0.0;
            ms.mu2=0.0;
   
            start_ranlux(0,8910+isp);
            mrw1(ms,1,isp,&lnr0,&sqn0,status);
            lnr0*=dm*dm;
            
            start_ranlux(0,8910+isp);
            lnr1=rwtm1(dm,isp,&sqn1,status);

            dr=fabs(1-lnr0/lnr1);
            ds=fabs(1.0-sqn1/sqn0);
         }
         else if (irw==1)
         {
            ms.m1=m0;
            ms.d1=dm*dm;
            ms.mu1=0.0;
            ms.m2=m0;
            ms.d2=-ms.d1;
            ms.mu2=sqrt(2.0)*dm;
   
            start_ranlux(0,8910+isp);
            lnr0=mrw3(ms,ispp,lnrw1,sqnp,&sqn0,status);
            
            start_ranlux(0,8910+isp);
            lnr1=rwtm2(dm,isp,&sqn1,status);

            dr=fabs(1-lnr0/lnr1);
            ds=fabs(1.0-sqn1/sqn0);
         }
         else if (irw==2)
         {
            ms.m1=m0;
            ms.d1=dm*dm;
            ms.mu1=0.0;
            ms.m2=m0;
            ms.d2=-ms.d1;
            ms.mu2=sqrt(2.0)*dm;
   
            start_ranlux(0,8910+isp);
            lnr0=mrw3(ms,ispp,lnrw1,sqnp,&sqn0,status);
            
            ms.m2=m0;
            ms.d2=dm*dm;
            ms.mu2=0.0;
            ms.m1=m0;
            ms.d1=-ms.d1;
            ms.mu1=sqrt(2.0)*dm;
  
            start_ranlux(0,8910+isp);
            lnr1=mrw3(ms,ispp,lnrw1,sqnp,&sqn1,status);
            
            dr=fabs(1-lnr0/lnr1);
            ds=fabs(1.0-sqn1/sqn0);
         }
         else
         {
            ms.m1=m0;
            ms.d1=(mu0+dm)*(mu0+dm)-mu0*mu0;
            ms.mu1=mu0;
            ms.m2=m0;
            ms.d2=-ms.d1;
            ms.mu2=mu0+dm;
   
            start_ranlux(0,8910+isp);
            lnr0=mrw3(ms,ispp,lnrw1,sqnp,&sqn0,status);
            
            dr=fabs(lnr0);
            ds=0.0;
         }
            

         if (dr>drmx)
            drmx=dr;
         if (ds>dsmx)
            dsmx=ds;

         if (my_rank==0)
         {            
            if ((isp==0)||(isp==1))
               printf("status = %d\n",status[0]);
            else if (isp==2)
               printf("status = (%d,%d,%d)\n",
                        status[0],status[1],status[2]);
            if (irw==0)
               printf("mrw1 vs. tmrw1: ");
            else if (irw==1)
               printf("mrw3 vs. tmrw2: ");
            else if (irw==2)
               printf("mrw3(m1=m2) vs. mrw3(m1=m2,D1<->D2): ");
            else
               printf("mrw3(m1=m2,d2=-d1,mu2=mu1+d1): ");
            
            if (irw<3)
               printf("diff = %.1e, |1-sqn1/sqn0| = %.1e\n\n",dr,ds);
            else
               printf("diff = %.1e\n\n",dr);
         }      
      
         error_chk();
      }
   }
   
   if (my_rank==0)
   {
      printf("\n");
      printf("max diff = %.1e, |1-sqn1/sqn0| = %.1e\n",drmx,dsmx);
      printf("(should be smaller than %.1e)\n\n",mres*sqrt((double)(VOLUME*NPROC*24)));
      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
