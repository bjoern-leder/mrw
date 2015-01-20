
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005, 2008, 2009, 2010, 
*               2011, 2012                  Martin Luescher, Filippo Palombi 
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of force0() and action0()
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
#include "linalg.h"
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

   error(ie!=0,1,"check_bnd_frc [check3.c]",
         "Non-zero force components on the time-like links at x0=N0-1");
}


static double dSdt(double c)
{
   mdflds_t *mdfs;

   force0(c);   
   check_bnd_frc();
   mdfs=mdflds();

   return scalar_prod_alg(4*VOLUME,0,(*mdfs).mom,(*mdfs).frc);
}   


int main(int argc,char *argv[])
{
   int my_rank,ia,ie;
   double c,eps,act0,act1,dact,dsdt;
   double dev_frc,sig_loss,rdmy;
   double phi[2],phi_prime[2];
   lat_parms_t lat;
   sf_parms_t sf;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      
      printf("\n");
      printf("Check of force0() and action0()\n");
      printf("-------------------------------\n\n");
      
      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }
   
   start_ranlux(0,1234);
   geometry();
   c=0.789;

   for (ia=0;ia<3;ia++)
   {
      if (ia==0)
      {
         set_lat_parms(5.75,1.0,0.0,0.0,0.0,1.234,0.789,1.113);
         lat=lat_parms();
         
         if (my_rank==0)
         {
            printf("Plaquette action:\n");
            printf("lat.beta=%.2f, lat.c0=%.4e, lat.c1=%.4e, lat.cG=%.4e\n\n",
                   lat.beta,lat.c0,lat.c1,lat.cG);
         }
      }
      else if (ia==1)
      {
         set_lat_parms(5.75,1.66667,0.0,0.0,0.0,1.234,0.789,1.113);
         lat=lat_parms();
         
         if (my_rank==0)
         {
            printf("Symanzik action:\n");
            printf("lat.beta=%.2f, lat.c0=%.4e, lat.c1=%.4e, lat.cG=%.4e\n\n",
                   lat.beta,lat.c0,lat.c1,lat.cG);
         }
      }
      else
      {
         set_lat_parms(5.75,1.66667,0.0,0.0,0.0,1.234,1.412,1.113);
         lat=lat_parms();
         
         phi[0]=0.1;
         phi[1]=0.2;
         phi_prime[0]=-0.4;
         phi_prime[1]=0.5;
         
         set_sf_parms(phi,phi_prime);
         sf=sf_parms();
         
         if (my_rank==0)
         {
            printf("Symanzik action with Schroedinger functional b.c.:\n");
            printf("lat.beta=%.2f, lat.c0=%.4e, lat.c1=%.4e, lat.cG=%.4e\n",
                   lat.beta,lat.c0,lat.c1,lat.cG);            
            printf("phi  = % .3e,% .3e,% .3e\n",
                   sf.phi[0],sf.phi[1],sf.phi[2]);
            printf("phi' = % .3e,% .3e,% .3e\n\n",
                   sf.phi_prime[0],sf.phi_prime[1],sf.phi_prime[2]);
         }
      }

      random_ud();
      random_mom();
      dsdt=dSdt(c);

      if (ia<2)
         ie=check_bcd();
      else
         ie=check_sfbcd();

      error_root(ie!=1,1,"main [check3.c]",
                 "Boundary values are not preserved");
      
      eps=1.0e-4;   
      rot_ud(eps);
      act0=2.0*action0(0)/3.0;
      rot_ud(-eps);
      
      rot_ud(-eps);
      act1=2.0*action0(0)/3.0;
      rot_ud(eps);
      
      rot_ud(2.0*eps);
      act0-=action0(0)/12.0;
      rot_ud(-2.0*eps);
      
      rot_ud(-2.0*eps);
      act1-=action0(0)/12.0;
      rot_ud(2.0*eps);

      act0*=c;
      act1*=c;

      dact=(act0-act1)/eps;
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
         printf("Calculation of the force:\n");
         printf("Relative deviation of dS/dt = %.2e ",fabs(dev_frc/dsdt));
         printf("[significance loss = %d digits]\n\n",(int)(sig_loss));
      }
   }

   if (my_rank==0)
      fclose(flog);
   
   MPI_Finalize();    
   exit(0);
}
