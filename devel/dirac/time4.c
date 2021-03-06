
/*******************************************************************************
*
* File time4.c
*
* Copyright (C) 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of Dw_blk_dble() and Dwhat_blk_dble()
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
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,count,nt;
   int n,nb,isw,bs[4];
   double mu,wt1,wt2,wdt;
   block_t *b;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("time4.log","w",stdout);
      fin=freopen("check7.in","r",stdin);

      printf("\n");
      printf("Timing of Dw_blk_dble() and Dwhat_blk_dble()\n");
      printf("--------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      if (NPROC>1)
         printf("There are %d MPI processes\n",NPROC);
      else
         printf("There is 1 MPI process\n");
      
      if ((VOLUME*sizeof(double))<(64*1024))
      {      
         printf("The local size of the gauge field is %d KB\n",
                (int)((72*VOLUME*sizeof(double))/(1024)));
         printf("The local size of a quark field is %d KB\n",
                (int)((24*VOLUME*sizeof(double))/(1024)));
      }
      else
      {
         printf("The local size of the gauge field is %d MB\n",
                (int)((72*VOLUME*sizeof(double))/(1024*1024)));
         printf("The local size of a quark field is %d MB\n",
                (int)((24*VOLUME*sizeof(double))/(1024*1024)));
      }

#if (defined x64)
#if (defined AVX)
      printf("Using AVX instructions\n");
#else      
      printf("Using SSE3 instructions and 16 xmm registers\n");
#endif
#if (defined P3)
      printf("Assuming SSE prefetch instructions fetch 32 bytes\n");
#elif (defined PM)
      printf("Assuming SSE prefetch instructions fetch 64 bytes\n");
#elif (defined P4)
      printf("Assuming SSE prefetch instructions fetch 128 bytes\n");
#else
      printf("SSE prefetch instructions are not used\n");
#endif
#endif
      printf("\n");

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      fclose(fin);

      printf("bs = %d %d %d %d\n\n",bs[0],bs[1],bs[2],bs[3]);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);      

   start_ranlux(0,12345);
   geometry();
   set_dfl_parms(bs,4);   
   alloc_bgr(DFL_BLOCKS);   

   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.456,1.0,1.234);
   set_sw_parms(-0.0123);
   mu=0.0785;

   random_ud();
   sw_term(NO_PTS);
   assign_ud2udblk(DFL_BLOCKS,0);
   assign_swd2swdblk(DFL_BLOCKS,0,NO_PTS);

   b=blk_list(DFL_BLOCKS,&nb,&isw);
   random_sd((*b).vol,(*b).sd[0],1.0);
   
   nt=(int)(2.0e6f/(double)(VOLUME));
   if (nt<2)
      nt=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<nt;count++)
      {
         for (n=0;n<nb;n++)
         {
            Dw_blk_dble(DFL_BLOCKS,n,mu,0,1);
            Dw_blk_dble(DFL_BLOCKS,n,mu,1,2);
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   error_chk();
   wdt=1.0e6*wdt/((double)(nt)*(double)(VOLUME));

   if (my_rank==0)
   {
      printf("Time per lattice point for Dw_blk_dble():\n");
      printf("%4.3f micro sec (%d Mflops)\n\n",wdt,(int)(1920.0/wdt));
   }

   nt=(int)(2.0e6f/(double)(VOLUME));
   if (nt<2)
      nt=2;
   wdt=0.0;

   while (wdt<5.0)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();     
      for (count=0;count<nt;count++)
      {
         for (n=0;n<nb;n++)
         {
            Dwhat_blk_dble(DFL_BLOCKS,n,mu,0,1);
            Dwhat_blk_dble(DFL_BLOCKS,n,mu,1,2);
         }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();

      wdt=wt2-wt1;
      nt*=2;
   }

   error_chk();
   wdt=1.0e6*wdt/((double)(nt)*(double)(VOLUME));

   if (my_rank==0)
   {
      printf("Time per lattice point for Dwhat_blk_dble():\n");
      printf("%4.3f micro sec (%d Mflops)\n\n",wdt,(int)(1908.0/wdt));
      fclose(flog);
   }
   
   MPI_Finalize();
   exit(0);
}
