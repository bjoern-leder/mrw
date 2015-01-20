
/*******************************************************************************
*
* File check6.c
*
* Copyright (C) 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Spectral range of the hermitian Dirac operator
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "archive.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "sap.h"
#include "dfl.h"
#include "ratfcts.h"
#include "update.h"
#include "global.h"

#define MAX(n,m) \
   if ((n)<(m)) \
      (n)=(m)

static int my_rank;
static double ar[36];


static void read_lat_parms(void)
{
   double kappa,csw,cF;

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      read_line("kappa","%lf",&kappa);
      read_line("csw","%lf",&csw);
      read_line("cF","%lf",&cF);   
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   set_lat_parms(0.0,1.0,kappa,0.0,0.0,csw,1.0,cF);
   set_sw_parms(sea_quark_mass(0));
}


static void read_sap_parms(void)
{
   int bs[4];

   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,1,4,5);
}


static void read_dfl_parms(void)
{
   int bs[4],Ns;
   int ninv,nmr,ncy,nkv,nmx;
   double kappa,mu,res;

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
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,res);
}


static void read_solver(void)
{
   int isap,idfl;
   solver_parms_t sp;

   isap=0;
   idfl=0;
   read_solver_parms(0);
   sp=solver_parms(0);

   if (sp.solver==SAP_GCR)
      isap=1;
   else if (sp.solver==DFL_SAP_GCR)
   {
      isap=1;
      idfl=1;
   }
      
   if (isap)
      read_sap_parms();

   if (idfl)
      read_dfl_parms();
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpp;

   dp=dfl_parms();
   dpp=dfl_pro_parms();

   MAX(*nws,dp.Ns+2);
   MAX(*nwv,2*dpp.nkv+2);
   MAX(*nwvd,4);
}


static void wsize(int *nws,int *nwsd,int *nwv,int *nwvd)
{
   int nsd;
   solver_parms_t sp;

   (*nws)=0;
   (*nwsd)=0;
   (*nwv)=0;
   (*nwvd)=0;

   nsd=2;
   sp=solver_parms(0);

   if (sp.solver==SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);
      MAX(*nwsd,nsd+2);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+2);      
      MAX(*nwsd,nsd+4);
      dfl_wsize(nws,nwv,nwvd);
   }
   else
      error_root(1,1,"wsize [check6.c]",
                 "Unknown or unsupported solver");   
}


static double power1(int pmx,int *status)
{
   int k,l,stat[6];
   double r;
   spinor_dble **wsd;
   solver_parms_t sp;
   sap_parms_t sap;

   sw_term(NO_PTS);
   sp=solver_parms(0);
   sap=sap_parms();
   set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);   
   
   if (sp.solver==SAP_GCR)
      status[0]=0;
   else
   {
      for (l=0;l<3;l++)
         status[l]=0;
   }

   wsd=reserve_wsd(2);
   random_sd(VOLUME/2,wsd[0],1.0);
   bnd_sd2zero(EVEN_PTS,wsd[0]);
   r=normalize_dble(VOLUME/2,1,wsd[0]);   
   
   for (k=0;k<pmx;k++)
   {
      if (sp.solver==SAP_GCR)
      {
         mulg5_dble(VOLUME/2,wsd[0]);
         set_sd2zero(VOLUME/2,wsd[0]+(VOLUME/2));         
         sap_gcr(sp.nkv,sp.nmx,sp.res,0.0,wsd[0],wsd[1],stat);
         mulg5_dble(VOLUME/2,wsd[1]);
         set_sd2zero(VOLUME/2,wsd[1]+(VOLUME/2));
         sap_gcr(sp.nkv,sp.nmx,sp.res,0.0,wsd[1],wsd[0],stat+1);

         error_root((stat[0]<0)||(stat[1]<0),1,"power2 [check6.c]",
                    "SAP_GCR solver failed (status = %d;%d)",
                    stat[0],stat[1]);

         for (l=0;l<2;l++)
         {
            if (status[0]<stat[l])
               status[0]=stat[l];
         }
      }
      else
      {
         mulg5_dble(VOLUME/2,wsd[0]);
         set_sd2zero(VOLUME/2,wsd[0]+(VOLUME/2));                  
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,0.0,wsd[0],wsd[1],stat);
         mulg5_dble(VOLUME/2,wsd[1]);
         set_sd2zero(VOLUME/2,wsd[1]+(VOLUME/2));                  
         dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,0.0,wsd[1],wsd[0],stat+4);

         error_root((stat[0]<0)||(stat[1]<0)||(stat[3]<0)||(stat[4]<0),1,
                    "power2 [check6.c]","DFL_SAP_GCR solver failed "
                    "(status = %d,%d,%d;%d,%d,%d)",
                    stat[0],stat[1],stat[2],stat[3],
                    stat[4],stat[5]);
      
         for (l=0;l<2;l++)
         {
            if (status[l]<stat[l])
               status[l]=stat[l];

            if (status[l]<stat[l+3])
               status[l]=stat[l+3];            
         }

         status[2]+=stat[2];
         status[2]+=stat[5];
      }

      r=normalize_dble(VOLUME/2,1,wsd[0]);
   }
   
   release_wsd();
   
   return 1.0/sqrt(r);
}


static double power2(int pmx)
{
   int k;
   double r;
   spinor_dble **wsd;

   sw_term(ODD_PTS);
   
   wsd=reserve_wsd(2);
   random_sd(VOLUME/2,wsd[0],1.0);
   bnd_sd2zero(EVEN_PTS,wsd[0]);   
   r=normalize_dble(VOLUME/2,1,wsd[0]);   

   for (k=0;k<pmx;k++)
   {
      Dwhat_dble(0.0,wsd[0],wsd[1]);
      mulg5_dble(VOLUME/2,wsd[1]);
      Dwhat_dble(0.0,wsd[1],wsd[0]);
      mulg5_dble(VOLUME/2,wsd[0]);

      r=normalize_dble(VOLUME/2,1,wsd[0]);
   }

   release_wsd();
   
   return sqrt(r);
}


int main(int argc,char *argv[])
{
   int first,last,step;
   int nc,nsize,icnfg;
   int isap,idfl,pmx,n,status[3];
   int nws,nwsd,nwv,nwvd;
   double ra,ramin,ramax,raavg;
   double rb,rbmin,rbmax,rbavg;
   double A,eps,delta,Ne,d1,d2;
   dfl_parms_t dfl;
   char cnfg_dir[NAME_SIZE],cnfg_file[NAME_SIZE];
   char nbase[NAME_SIZE];   
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check6.log","w",stdout);
      fin=freopen("check6.in","r",stdin);

      printf("\n");
      printf("Spectral range of the hermitian Dirac operator\n");
      printf("----------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      find_section("Configurations");
      read_line("cnfg_dir","%s",cnfg_dir);
      read_line("name","%s",nbase);
      read_line("first","%d",&first);
      read_line("last","%d",&last);  
      read_line("step","%d",&step);

      find_section("Power method");
      read_line("pmx","%d",&pmx);
   }
   
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&pmx,1,MPI_INT,0,MPI_COMM_WORLD);
   
   read_lat_parms();
   read_solver();

   if (my_rank==0)
   {
      fclose(fin);

      print_lat_parms();
      print_solver_parms(&isap,&idfl);
      if (isap)
         print_sap_parms(0);
      if (idfl)
         print_dfl_parms(0);
   }

   dfl=dfl_parms();
   wsize(&nws,&nwsd,&nwv,&nwvd);
   alloc_ws(nws);
   alloc_wsd(nwsd);
   alloc_wv(nwv);
   alloc_wvd(nwvd);   
   
   if (my_rank==0)
   {
      printf("Configurations %sn%d -> %sn%d in steps of %d\n\n",
             nbase,first,nbase,last,step);      
      fflush(flog);
   }

   start_ranlux(0,1234);
   geometry();
   
   error_root(((last-first)%step)!=0,1,"main [check6.c]",
              "last-first is not a multiple of step");
   check_dir_root(cnfg_dir);   

   nsize=name_size("%s/%sn%d",cnfg_dir,nbase,last);
   error_root(nsize>=NAME_SIZE,1,"main [check6.c]",
              "Configuration file name is too long");

   ramin=0.0;
   ramax=0.0;
   raavg=0.0;
   
   rbmin=0.0;
   rbmax=0.0;
   rbavg=0.0;
   
   for (icnfg=first;icnfg<=last;icnfg+=step)
   {
      sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,icnfg);
      import_cnfg(cnfg_file);

      if (my_rank==0)
      {
         printf("Configuration no %d\n",icnfg);
         fflush(flog);
      }

      if (dfl.Ns)
      {
         dfl_modes(status);
         error_root(status[0]<0,1,"main [check6.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status[0]);
      }

      ra=power1(pmx,status);
      rb=power2(pmx);
      
      if (icnfg==first)
      {
         ramin=ra;
         ramax=ra;
         raavg=ra;

         rbmin=rb;
         rbmax=rb;
         rbavg=rb;
      }
      else
      {
         if (ra<ramin)
            ramin=ra;
         if (ra>ramax)
            ramax=ra;
         raavg+=ra;

         if (rb<rbmin)
            rbmin=rb;
         if (rb>rbmax)
            rbmax=rb;
         rbavg+=rb;
      }
      
      if (my_rank==0)
      {
         printf("ra = %.2e, rb = %.2e\n",ra,rb);

         if (idfl)
            printf("status = %d,%d,%d\n\n",
                   status[0],status[1],status[2]);
         else
            printf("status = %d\n\n",status[0]);
         
         fflush(flog);
      }
   }

   if (my_rank==0)
   {
      nc=(last-first)/step+1;
      
      printf("Summary\n");
      printf("-------\n\n");

      printf("Considered %d configurations in the range %d -> %d\n\n",
             nc,first,last);

      printf("The three figures quoted in each case are the minimal,\n");
      printf("maximal and average values\n\n");

      printf("Spectral gap ra    = %.2e, %.2e, %.2e\n",
             ramin,ramax,raavg/(double)(nc));
      printf("Spectral radius rb = %.2e, %.2e, %.2e\n\n",
             rbmin,rbmax,rbavg/(double)(nc));

      ra=0.90*ramin;
      rb=1.03*rbmax;
      eps=ra/rb;
      eps=eps*eps;
      Ne=0.5*(double)(NPROC0*L0-2)*(double)(NPROC1*NPROC2*NPROC3*L1*L2*L3);

      printf("Zolotarev rational approximation:\n");
      printf("Spectral range = [%.2e,%.2e]\n",ra,rb);
      printf("    n      delta    12*Ne*delta     12*Ne*delta^2\n");

      for (n=6;n<=18;n+=2)
      {
         zolotarev(n,eps,&A,ar,&delta);
         d1=12.0*Ne*delta;
         d2=d1*delta;

         printf("   %2d     %.1e      %.1e         %.1e\n",n,delta,d1,d2);
         
         if ((d1<1.0e-2)&&(d2<1.0e-4))
            break;
      }

      printf("\n");
   }

   error_chk();
   
   if (my_rank==0)
      fclose(flog);
   
   MPI_Finalize();    
   exit(0);
}
