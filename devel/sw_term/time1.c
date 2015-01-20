
/*******************************************************************************
*
* File time1.c
*
* Copyright (C) 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of the program sw_term()
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
#include "uflds.h"
#include "sw_term.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,count,nt;
   double wt1,wt2,wdt;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("time1.log","w",stdout);
      printf("\n");
      printf("Timing of the program sw_term()\n");
      printf("-------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

#if (defined AVX)
   printf("Using AVX instructions\n\n");
#elif (defined x64)
   printf("Using SSE3 instructions and up to 16 xmm registers\n\n");
#endif
   }

   start_ranlux(0,12345);
   geometry();
   
   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.456,1.0,1.0);
   set_sw_parms(-0.0123);
   random_ud();

   nt=(int)(5.0e5/(double)(VOLUME));
   if (nt<2)
      nt=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<nt;count++)
      {
         set_flags(UPDATED_UD);
         (void)sw_term(NO_PTS);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   error_chk();
   wdt=2.0e6*wdt/((double)(nt)*(double)(VOLUME));

   if (my_rank==0)
   {
      printf("Time per lattice point: %4.3f micro sec",wdt);
      printf(" (%d Mflops [%d bit arithmetic])\n",
             (int)(9936.0/wdt),(int)(sizeof(spinor_dble))/3);
   }

   nt=(int)(2.0e6/(double)(VOLUME));
   if (nt<2)
      nt=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<nt;count++)
      {
         set_flags(ERASED_SWD);
         (void)sw_term(NO_PTS);
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   error_chk();
   wdt=2.0e6*wdt/((double)(nt)*(double)(VOLUME));

   if (my_rank==0)
   {
      printf("                        %4.3f micro sec",wdt);
      printf(" (field tensor is up-to-date)\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
