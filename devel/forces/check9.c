
/*******************************************************************************
*
* File check9.c
*
* Copyright (C) 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of force3() and action3()
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
#include "sw_term.h"
#include "dfl.h"
#include "forces.h"
#include "global.h"


static void rot_ud(double t)
{
   su3_dble *u,*um;
   su3_alg_dble *mom;
   mdflds_t *mdfs;

   u=udfld();
   um=u+4*VOLUME;   

   mdfs=mdflds();
   mom=(*mdfs).mom;
   
   for (;u<um;u++)
   {
      expXsu3(t,mom,u);
      mom+=1;
   }

   set_flags(UPDATED_UD);
}


static int is_not_zero(su3_alg_dble *frc)
{
   int ie;

   ie=((*frc).c1!=0.0);
   ie|=((*frc).c2!=0.0);
   ie|=((*frc).c3!=0.0);
   ie|=((*frc).c4!=0.0);
   ie|=((*frc).c5!=0.0);
   ie|=((*frc).c6!=0.0);
   ie|=((*frc).c7!=0.0);
   ie|=((*frc).c8!=0.0);   

   return ie;
}


static void check_bnd_frc(void)
{
   int l,nlks,*lks,ie;
   su3_alg_dble *fdb;
   mdflds_t *mdfs;

   mdfs=mdflds();
   fdb=(*mdfs).frc;

   lks=bnd_lks(&nlks);
   ie=0;

   for (l=0;l<nlks;l++)
      ie|=is_not_zero(fdb+lks[l]);

   error(ie!=0,1,"check_bnd_frc [check9.c]",
         "Non-zero force components on the time-like links at x0=N0-1");
}


static double dSdt(int *irat,int ipf,int isw,int isp,int *status)
{
   mdflds_t *mdfs;

   set_frc2zero();   
   force3(irat,ipf,isw,isp,1.2345,status);
   check_bnd_frc();
   mdfs=mdflds();

   return scalar_prod_alg(4*VOLUME,0,(*mdfs).mom,(*mdfs).frc);
}   


int main(int argc,char *argv[])
{
   int my_rank,irat[3],isw,isp,status[6],mnkv;
   int bs[4],Ns,nmx,nkv,isolv,nmr,ncy,ninv;
   int isap,idfl;
   double kappa,mu,res;
   double eps,act0,act1,dact,dsdt;
   double dev_act[2],dev_frc,sig_loss,rdmy;
   rat_parms_t rp;
   solver_parms_t sp;
   FILE *flog=NULL,*fin=NULL;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check9.log","w",stdout);
      fin=freopen("check9.in","r",stdin);
      
      printf("\n");
      printf("Check of force3() and action3()\n");
      printf("-------------------------------\n\n");
      
      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   set_lat_parms(6.0,1.0,0.0,0.0,0.0,1.234,1.0,1.34);
   set_sw_parms(-0.0123);
   read_rat_parms(0);
   
   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("isolv","%d",&isolv);
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&isolv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,isolv,nmr,ncy);

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

   set_hmc_parms(0,NULL,1,0,NULL,1,1.0);
   mnkv=0;
      
   for (isp=0;isp<3;isp++)
   {
      read_solver_parms(isp);
      sp=solver_parms(isp);

      if (sp.nkv>mnkv)
         mnkv=sp.nkv;
   }

   if (my_rank==0)
      fclose(fin);

   print_rat_parms();
   print_solver_parms(&isap,&idfl);
   print_sap_parms(1);
   print_dfl_parms(0);
   
   start_ranlux(0,1245);
   geometry();
   rp=rat_parms(0);
   irat[0]=0;

   mnkv=2*mnkv+2;
   if (mnkv<(Ns+2))
      mnkv=Ns+2;
   if (mnkv<5)
      mnkv=5;
   
   alloc_ws(mnkv);

   if (2*rp.degree>4)
      alloc_wsd(2*rp.degree+3);
   else
      alloc_wsd(7);
   
   alloc_wv(2*nkv+2);
   alloc_wvd(4);

   for (isw=0;isw<2;isw++)
   {
      for (isp=0;isp<3;isp++)
      {
         if (isp==0)
         {
            irat[1]=0;
            irat[2]=rp.degree/3;
            eps=1.0e-4;   
         }
         else if (isp==1)
         {
            irat[1]=rp.degree/3+1;
            irat[2]=(2*rp.degree)/3;
            eps=2.0e-4;
         }
         else
         {
            irat[1]=(2*rp.degree)/3+1;
            irat[2]=rp.degree-1;
            eps=3.0e-4;
         }

         random_ud();
         random_mom();

         if (isp==2)
         {
            dfl_modes(status);
            error_root(status[0]<0,1,"main [check9.c]",
                       "dfl_modes failed");
         }

         status[0]=0;
         status[1]=0;
         
         act0=setpf3(irat,0,isw,isp,0,status);
         error_root((status[0]<0)||(status[1]<0),1,
                    "main [check9.c]","setpf3 failed "
                    "(irat=(%d,%d,%d), isp=%d)",irat[0],irat[1],irat[2],isp); 

         act1=action3(irat,0,isw,isp,0,status);
         error_root((status[0]<0)||(status[1]<0),1,
                    "main [check9.c]","action3 failed "
                    "(irat=(%d,%d,%d), isp=%d)",irat[0],irat[1],irat[2],isp);
         
         rdmy=fabs(act1-act0);
         MPI_Reduce(&rdmy,dev_act,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
         rdmy=act1-act0;
         MPI_Reduce(&rdmy,dev_act+1,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
         MPI_Bcast(dev_act,2,MPI_DOUBLE,0,MPI_COMM_WORLD); 

         rot_ud(eps);         
         dsdt=dSdt(irat,0,isw,isp,status);
      
         if (my_rank==0)
         {
            printf("Solver number %d, poles %d,..,%d, isw %d\n",
                   isp,irat[1],irat[2],isw);
         
            if (isp==0)
               printf("Status = %d\n",status[0]);
            else if (isp==1)
               printf("Status = %d,%d\n",status[0],status[1]);
            else
               printf("Status = (%d,%d,%d),(%d,%d,%d)\n",
                      status[0],status[1],status[2],status[3],
                      status[4],status[5]);

            printf("Absolute action difference |setpf3-action3| = %.1e,",
                   fabs(dev_act[1]));
            printf(" %.1e (local)\n",dev_act[0]);
            fflush(flog);
         }
      
         rot_ud(eps);
         act0=2.0*action3(irat,0,isw,isp,0,status)/3.0;
         rot_ud(-eps);

         rot_ud(-eps);
         act1=2.0*action3(irat,0,isw,isp,0,status)/3.0;
         rot_ud(eps);

         rot_ud(2.0*eps);
         act0-=action3(irat,0,isw,isp,0,status)/12.0;
         rot_ud(-2.0*eps);

         rot_ud(-2.0*eps);
         act1-=action3(irat,0,isw,isp,0,status)/12.0;
         rot_ud(2.0*eps);

         dact=1.2345*(act0-act1)/eps;
         dev_frc=dsdt-dact;
         sig_loss=-log10(fabs(1.0-act0/act1));
      
         rdmy=dsdt;
         MPI_Reduce(&rdmy,&dsdt,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
         MPI_Bcast(&dsdt,1,MPI_DOUBLE,0,MPI_COMM_WORLD); 

         rdmy=dev_frc;
         MPI_Reduce(&rdmy,&dev_frc,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
         MPI_Bcast(&dev_frc,1,MPI_DOUBLE,0,MPI_COMM_WORLD);       

         rdmy=sig_loss;
         MPI_Reduce(&rdmy,&sig_loss,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
         MPI_Bcast(&sig_loss,1,MPI_DOUBLE,0,MPI_COMM_WORLD);      
      
         error_chk();
      
         if (my_rank==0)
         {
            printf("Relative deviation of dS/dt = %.2e ",fabs(dev_frc/dsdt));
            printf("[significance loss = %d digits]\n\n",(int)(sig_loss));
            fflush(flog);
         }
      }
   }
   
   if (my_rank==0)
      fclose(flog);
   
   MPI_Finalize();    
   exit(0);
}
