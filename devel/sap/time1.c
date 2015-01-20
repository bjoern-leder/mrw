
/*******************************************************************************
*
* File time1.c
*
* Copyright (C) 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Timing of blk_mres() and blk_eo_mres()
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
#include "sap.h"
#include "global.h"


int main(int argc,char *argv[])
{
   int my_rank,count,nt;
   int nb,isw,nmr,bs[4];
   int n,ie;   
   float mu;
   double wt1,wt2,wdt;
   spinor **ps;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("time1.log","w",stdout);
      fin=freopen("time1.in","r",stdin);

      printf("\n");
      printf("Timing of blk_mres() and blk_eo_mres()\n");
      printf("--------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

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
      read_line("nmr","%d",&nmr);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("nmr = %d\n\n",nmr);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);      
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   
   start_ranlux(0,12345);
   geometry();
   alloc_ws(1);
   set_sap_parms(bs,0,1,1);
   alloc_bgr(SAP_BLOCKS);   

   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.156,1.0,1.234);
   set_sw_parms(0.0123);
   mu=0.0785f;

   random_ud();
   sw_term(NO_PTS);
   assign_ud2ubgr(SAP_BLOCKS);
   assign_swd2swbgr(SAP_BLOCKS,NO_PTS);

   ps=reserve_ws(1);
   random_s(VOLUME,ps[0],1.0f);   
   blk_list(SAP_BLOCKS,&nb,&isw);
   
   nt=(int)(1.0e7/(double)(nmr*VOLUME));
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
            assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],1);
            blk_mres(n,mu,nmr);
         }
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
      printf("Field copying + block inversion using blk_mres():\n");
      printf("Time per lattice point: %.3f micro sec",wdt);
      printf(" (about %d Mflops)\n",(int)((double)(nmr*2256)/wdt));
      printf("Time per lattice point and MR iteration: %.3f micro sec\n\n",
             wdt/(double)(nmr));
   }

   ie=assign_swd2swbgr(SAP_BLOCKS,ODD_PTS);
   error_root(ie,1,"main [time1.c]",
              "The inversion of the SW term was not safe");
   
   nt=(int)(1.0e7/(double)(nmr*VOLUME));
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
            assign_s2sblk(SAP_BLOCKS,n,EVEN_PTS,ps[0],1);
            blk_eo_mres(n,mu,nmr);
         }
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
      printf("Field copying + block inversion using blk_eo_mres():\n");
      printf("Time per lattice point: %.3f micro sec",wdt);
      printf(" (about %d Mflops)\n",(int)((double)(nmr*2076)/wdt));
      printf("Time per lattice point and MR iteration: %.3f micro sec\n\n",
             wdt/(double)(nmr));
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
