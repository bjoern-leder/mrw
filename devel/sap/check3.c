
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check and performance of the SAP+GCR solver
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "dirac.h"
#include "sap.h"
#include "global.h"

int my_rank,id,first,last,step;
int bs[4],nmr,ncy,nkv,nmx,eoflg;
double kappa,csw,cF,mu;
double m0,res;
char cnfg_dir[NAME_SIZE],cnfg_file[NAME_SIZE],nbase[NAME_SIZE];


int main(int argc,char *argv[])
{
   int isolv,nsize,icnfg,status;
   double rho,nrm,del;
   double wt1,wt2,wdt;
   spinor_dble **psd;
   lat_parms_t lat;
   sap_parms_t sap;
   sw_parms_t sw;
   tm_parms_t tm;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);

      printf("\n");
      printf("Check and performance of the SAP+GCR solver\n");
      printf("-------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      find_section("Configurations");
      read_line("name","%s",nbase);
      read_line("cnfg_dir","%s",cnfg_dir);
      read_line("first","%d",&first);
      read_line("last","%d",&last);  
      read_line("step","%d",&step);  

      find_section("Lattice parameters");
      read_line("kappa","%lf",&kappa);
      read_line("csw","%lf",&csw);
      read_line("cF","%lf",&cF);
      read_line("mu","%lf",&mu);
      read_line("eoflg","%d",&eoflg);
      
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);

      find_section("GCR");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);
      read_line("res","%lf",&res);      
      
      fclose(fin);
   }
   
   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&eoflg,1,MPI_INT,0,MPI_COMM_WORLD);
   
   MPI_Bcast(&bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   lat=set_lat_parms(6.0,1.0,kappa,0.0,0.0,csw,1.0,cF);
   sap=set_sap_parms(bs,0,nmr,ncy);
   m0=lat.m0u;
   sw=set_sw_parms(m0);
   tm=set_tm_parms(eoflg);

   start_ranlux(0,1234);
   geometry();
   alloc_ws(2*nkv+1);
   alloc_wsd(5);
   psd=reserve_wsd(3);

   if (my_rank==0)
   {
      printf("kappa = %.6f\n",lat.kappa_u);
      printf("csw = %.6f\n",sw.csw);
      printf("cF = %.6f\n",sw.cF);
      printf("mu = %.6f\n",mu);
      printf("eoflg = %d\n\n",tm.eoflg);

      printf("bs = (%d,%d,%d,%d)\n",sap.bs[0],sap.bs[1],sap.bs[2],sap.bs[3]);
      printf("nmr = %d\n",sap.nmr);
      printf("ncy = %d\n\n",sap.ncy);

      printf("nkv = %d\n",nkv);
      printf("nmx = %d\n",nmx);      
      printf("res = %.2e\n\n",res);

      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);
      fflush(flog);
   }

   error_root(((last-first)%step)!=0,1,"main [check3.c]",
              "last-first is not a multiple of step");

   nsize=name_size("%s/%sn%d",cnfg_dir,nbase,last);
   error_root(nsize>=NAME_SIZE,1,"main [check3.c]",
              "cnfg_dir name is too long");

   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file);

      if (my_rank==0)
      {
         printf("Configuration no %d\n",icnfg);
         fflush(flog);
      } 
         
      random_sd(VOLUME,psd[0],1.0);
      bnd_sd2zero(ALL_PTS,psd[0]);
      nrm=sqrt(norm_square_dble(VOLUME,1,psd[0]));

      for (isolv=0;isolv<2;isolv++)
      {
         assign_sd2sd(VOLUME,psd[0],psd[2]);         
         set_sap_parms(bs,isolv,nmr,ncy);

         rho=sap_gcr(nkv,nmx,res,mu,psd[0],psd[1],&status);
      
         error_chk();
         mulr_spinor_add_dble(VOLUME,psd[2],psd[0],-1.0);
         del=norm_square_dble(VOLUME,1,psd[2]);
         error_root(del!=0.0,1,"main [check3.c]",
                    "Source field is not preserved");

         Dw_dble(mu,psd[1],psd[2]);
         mulr_spinor_add_dble(VOLUME,psd[2],psd[0],-1.0);
         del=sqrt(norm_square_dble(VOLUME,1,psd[2]));
      
         if (my_rank==0)
         {
            printf("isolv = %d:\n",isolv);
            printf("status = %d\n",status);
            printf("rho   = %.2e, res   = %.2e\n",rho,res);
            printf("check = %.2e, check = %.2e\n",del,del/nrm);
         }

         assign_sd2sd(VOLUME,psd[0],psd[2]);
         
         MPI_Barrier(MPI_COMM_WORLD);
         wt1=MPI_Wtime();              

         rho=sap_gcr(nkv,nmx,res,mu,psd[2],psd[2],&status);

         MPI_Barrier(MPI_COMM_WORLD);
         wt2=MPI_Wtime();
         wdt=wt2-wt1;

         if (my_rank==0)
         {
            printf("time = %.2e sec (total)\n",wdt);
            if (status>0)
               printf("     = %.2e usec (per point and GCR iteration)",
                      (1.0e6*wdt)/((double)(status)*(double)(VOLUME)));
            printf("\n\n");
            fflush(flog);
         }

         mulr_spinor_add_dble(VOLUME,psd[2],psd[1],-1.0);
         del=norm_square_dble(VOLUME,1,psd[2]);
         error_root(del!=0.0,1,"main [check3.c]",
                    "Incorrect result when the input and "
                    "output fields coincide");
      }
   }

   if (my_rank==0)
      fclose(flog);
   
   MPI_Finalize();    
   exit(0);
}
