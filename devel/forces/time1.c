
/*******************************************************************************
*
* File time1.c
*
* Copyright (C) 2005, 2008, 2009, 2010, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of plaq_frc(), sw_frc() and hop_frc()
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
#include "sflds.h"
#include "mdflds.h"
#include "forces.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,n,count;
   double wt1,wt2,wdt;
   FILE *flog=NULL;
   spinor_dble **wsd;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("time1.log","w",stdout);
      
      printf("\n");
      printf("Timing of plaq_frc(), sw_frc() and hop_frc()\n");
      printf("--------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,12345);
   geometry();

   set_lat_parms(6.0,1.0,0.0,0.0,0.0,1.2300,0.23,1.34);
   set_sw_parms(-0.1235);

   alloc_wsd(2);
   wsd=reserve_wsd(2);
   
   random_ud();
   random_sd(VOLUME,wsd[0],1.0);
   random_sd(VOLUME,wsd[1],1.0);
   bnd_sd2zero(ALL_PTS,wsd[0]);
   bnd_sd2zero(ALL_PTS,wsd[1]);

   plaq_frc();
   set_frc2zero();
   set_xt2zero();
   add_prod2xt(-0.5,wsd[0],wsd[1]);   
   add_prod2xv(-0.5,wsd[0],wsd[1]);   
   sw_frc(1.0);
   hop_frc(1.0);
   
   n=(int)(3.0e6/(double)(4*VOLUME));
   if (n<2)
      n=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<n;count++)
         plaq_frc();
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      n*=2;
   }

   wdt=2.0e6*wdt/((double)(n)*(double)(4*VOLUME));
   error_chk();

   if (my_rank==0)
   {
      printf("Time per link:\n");
      printf("plaq_frc():      %4.3f usec\n",wdt);
   }

   n=(int)(3.0e6/(double)(4*VOLUME));
   if (n<2)
      n=2;
   wdt=0.0;
   set_xt2zero();
         
   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<n;count++)
         add_prod2xt(0.0,wsd[0],wsd[1]);         
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      
      wdt=wt2-wt1;
      n*=2;
   }

   wdt=2.0e6*wdt/((double)(n)*(double)(4*VOLUME));
   error_chk();

   if (my_rank==0)
      printf("add_prod2xt():   %4.3f usec\n",wdt);

   n=(int)(3.0e6/(double)(4*VOLUME));
   if (n<2)
      n=2;
   wdt=0.0;
   set_xv2zero();
         
   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<n;count++)
         add_prod2xv(0.0,wsd[0],wsd[1]);         
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      
      wdt=wt2-wt1;
      n*=2;
   }

   wdt=2.0e6*wdt/((double)(n)*(double)(4*VOLUME));
   error_chk();

   if (my_rank==0)
      printf("add_prod2xv():   %4.3f usec\n",wdt);

   n=(int)(3.0e6/(double)(4*VOLUME));
   if (n<2)
      n=2;   
   wdt=0.0;
   set_frc2zero();
         
   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<n;count++)
         sw_frc(0.0);
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      n*=2;
   }

   wdt=2.0e6*wdt/((double)(n)*(double)(4*VOLUME));
   error_chk();

   if (my_rank==0)
      printf("sw_frc():        %4.3f usec\n",wdt);

   n=(int)(3.0e6/(double)(4*VOLUME));
   if (n<2)
      n=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<n;count++)
         hop_frc(0.0);
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      n*=2;
   }

   wdt=2.0e6*wdt/((double)(n)*(double)(4*VOLUME));
   error_chk();

   if (my_rank==0)
   {
      printf("hop_frc():       %4.3f usec\n\n",wdt);
      fclose(flog);
   }
   
   MPI_Finalize();   
   exit(0);
}
