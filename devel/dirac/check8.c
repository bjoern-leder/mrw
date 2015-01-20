
/*******************************************************************************
*
* File check8.c
*
* Copyright (C) 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Comparison of Dw_blk_dble(),... with Dw_dble()
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "block.h"
#include "dirac.h"
#include "global.h"


static void blk_sd2zero(int ic,spinor_dble *sd)
{
   int nb,isw;
   int nbh,n,nm,vol;
   block_t *b;

   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

   if (ic^isw)
      n=nbh;
   else
      n=0;

   nm=n+nbh;

   for (;n<nm;n++)   
   {
      set_sd2zero(vol,b[n].sd[0]);
      assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,0,sd);
   }
}


int main(int argc,char *argv[])
{
   int my_rank,nb,isw,nbh,ic,itm;
   int bs[4],n,nm,vol,volh,ie;
   double mu,d,dmax;
   spinor_dble **psd;
   block_t *b;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check8.log","w",stdout);
      fin=freopen("check7.in","r",stdin);

      printf("\n");
      printf("Comparison of Dw_blk_dble(),... with Dw_dble()\n");
      printf("----------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      fclose(fin);

      printf("bs = %d %d %d %d\n\n",bs[0],bs[1],bs[2],bs[3]);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);

   start_ranlux(0,1234);
   geometry();
   set_dfl_parms(bs,2);
   alloc_bgr(DFL_BLOCKS);
   alloc_wsd(4);
   
   set_lat_parms(5.6,1.0,0.0,0.0,0.0,0.1,1.3,1.15);
   set_sw_parms(0.05);
   mu=0.123;
   
   random_ud();
   sw_term(NO_PTS);
   
   psd=reserve_wsd(4);
   b=blk_list(DFL_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;
   volh=vol/2;

   for (itm=0;itm<2;itm++)
   {
      ie=0;
      dmax=0.0;
      set_tm_parms(itm);

      if (my_rank==0)
         printf("Twisted-mass flag = %d\n",itm);  
      
      for (ic=0;ic<2;ic++)
      {
         random_sd(VOLUME,psd[0],1.0);
         random_sd(VOLUME,psd[2],1.0);
         blk_sd2zero(ic^0x1,psd[0]);
         blk_sd2zero(ic^0x1,psd[2]);

         if (ic^isw)
            n=nbh;
         else
            n=0;

         nm=n+nbh;
      
         for (;n<nm;n++)
         {
            assign_ud2udblk(DFL_BLOCKS,n);
            assign_swd2swdblk(DFL_BLOCKS,n,NO_PTS);
         
            random_sd(vol,b[n].sd[1],1.0);
            assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[0],0);
            Dw_blk_dble(DFL_BLOCKS,n,mu,0,1);
            assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,0,psd[2]);
            assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,1,psd[3]);
         }      

         error_chk();
         Dw_dble(mu,psd[0],psd[1]);
         blk_sd2zero(ic^0x1,psd[1]);      
         blk_sd2zero(ic^0x1,psd[3]);
      
         mulr_spinor_add_dble(VOLUME,psd[0],psd[2],-1.0);
         mulr_spinor_add_dble(VOLUME,psd[1],psd[3],-1.0);   

         if (norm_square_dble(VOLUME,0,psd[0])!=0.0)
            ie=1;
   
         d=norm_square_dble(VOLUME,1,psd[1])/
            norm_square_dble(VOLUME,1,psd[3]);

         if (d>dmax)
            dmax=d;
      }
   
      error_chk();
      error(ie,1,"main [check8.c]",
            "Dw_blk_dble() changes the fields where it should not"); 

      dmax=sqrt(dmax);

      if (my_rank==0)
      {
         printf("The maximal relative deviations are:\n\n");
         printf("Dw_blk_dble():                    %.1e\n",dmax);
      }

      dmax=0.0;
      random_sd(VOLUME,psd[0],1.0);
      random_sd(VOLUME,psd[1],1.0);
   
      for (n=0;n<nb;n++)
      {
         assign_ud2udblk(DFL_BLOCKS,n);
         assign_swd2swdblk(DFL_BLOCKS,n,NO_PTS);
      
         assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[0],0);
         assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[1],1);
         Dwee_blk_dble(DFL_BLOCKS,n,mu,0,1);     
         assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,0,psd[2]);      
         assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,1,psd[3]);

         assign_sd2sdblk(DFL_BLOCKS,n,EVEN_PTS,psd[0],0);
         assign_sd2sdblk(DFL_BLOCKS,n,ODD_PTS,psd[1],0);      
         Dwee_blk_dble(DFL_BLOCKS,n,mu,0,0);
         mulr_spinor_add_dble(vol,b[n].sd[0],b[n].sd[1],-1.0);
         if (norm_square_dble(vol,0,b[n].sd[0])!=0.0)
            ie=1;
      }

      Dwee_dble(mu,psd[0],psd[1]);
      mulr_spinor_add_dble(VOLUME,psd[0],psd[2],-1.0);
      mulr_spinor_add_dble(VOLUME,psd[1],psd[3],-1.0);   

      if (norm_square_dble(VOLUME,0,psd[0])!=0.0)
         ie=1;
   
      d=norm_square_dble(VOLUME,1,psd[1])/
         norm_square_dble(VOLUME,1,psd[3]);

      if (d>dmax)
         dmax=d;

      random_sd(VOLUME,psd[0],1.0);
      random_sd(VOLUME,psd[1],1.0);
   
      for (n=0;n<nb;n++)
      {
         assign_ud2udblk(DFL_BLOCKS,n);
         assign_swd2swdblk(DFL_BLOCKS,n,NO_PTS);
         
         assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[0],0);
         assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[1],1);
         Dwoo_blk_dble(DFL_BLOCKS,n,mu,0,1);     
         assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,0,psd[2]);      
         assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,1,psd[3]);

         assign_sd2sdblk(DFL_BLOCKS,n,ODD_PTS,psd[0],0);
         assign_sd2sdblk(DFL_BLOCKS,n,EVEN_PTS,psd[1],0);      
         Dwoo_blk_dble(DFL_BLOCKS,n,mu,0,0);
         mulr_spinor_add_dble(vol,b[n].sd[0],b[n].sd[1],-1.0);
         if (norm_square_dble(vol,0,b[n].sd[0])!=0.0)
            ie=1;
      }

      Dwoo_dble(mu,psd[0],psd[1]);
      mulr_spinor_add_dble(VOLUME,psd[0],psd[2],-1.0);
      mulr_spinor_add_dble(VOLUME,psd[1],psd[3],-1.0);   

      if (norm_square_dble(VOLUME,0,psd[0])!=0.0)
         ie=1;
   
      d=norm_square_dble(VOLUME,1,psd[1])/
         norm_square_dble(VOLUME,1,psd[3]);

      if (d>dmax)
         dmax=d;
   
      error_chk();
      error(ie,1,"main [check8.c]","Dwee_blk_dble() or Dwoo_blk_dble() "
            "changes the fields where it should not"); 

      dmax=sqrt(dmax);
   
      if (my_rank==0)
         printf("Dwee_blk_dble(), Dwoo_blk_dble(): %.1e\n",dmax);

      dmax=0.0;

      for (ic=0;ic<2;ic++)
      {
         random_sd(VOLUME,psd[0],1.0);
         random_sd(VOLUME,psd[1],1.0);
         random_sd(VOLUME,psd[2],1.0);
         blk_sd2zero(ic^0x1,psd[0]);
         blk_sd2zero(ic^0x1,psd[2]);

         if (ic^isw)
            n=nbh;
         else
            n=0;

         nm=n+nbh;
      
         for (;n<nm;n++)
         {
            assign_ud2udblk(DFL_BLOCKS,n);
            assign_swd2swdblk(DFL_BLOCKS,n,NO_PTS);
         
            assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[0],0);
            assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[1],1);
            Dweo_blk_dble(DFL_BLOCKS,n,0,1);
            assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,0,psd[2]);
            assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,1,psd[3]);
         }      

         error_chk();
         Dweo_dble(psd[0],psd[1]);
         blk_sd2zero(ic^0x1,psd[1]);      
         blk_sd2zero(ic^0x1,psd[3]);
      
         mulr_spinor_add_dble(VOLUME,psd[0],psd[2],-1.0);
         mulr_spinor_add_dble(VOLUME,psd[1],psd[3],-1.0);   

         if (norm_square_dble(VOLUME,0,psd[0])!=0.0)
            ie=1;
   
         d=norm_square_dble(VOLUME,1,psd[1])/
            norm_square_dble(VOLUME,1,psd[3]);

         if (d>dmax)
            dmax=d;
      }
   
      error_chk();
      error(ie,1,"main [check8.c]",
            "Dweo_blk_dble() changes the fields where it should not"); 

      dmax=sqrt(dmax);
   
      if (my_rank==0)
         printf("Dweo_blk_dble():                  %.1e\n",dmax);

      dmax=0.0;

      for (ic=0;ic<2;ic++)
      {
         random_sd(VOLUME,psd[0],1.0);
         random_sd(VOLUME,psd[1],1.0);      
         random_sd(VOLUME,psd[2],1.0);
         blk_sd2zero(ic^0x1,psd[0]);
         blk_sd2zero(ic^0x1,psd[2]);

         if (ic^isw)
            n=nbh;
         else
            n=0;

         nm=n+nbh;
      
         for (;n<nm;n++)      
         {
            assign_ud2udblk(DFL_BLOCKS,n);
            assign_swd2swdblk(DFL_BLOCKS,n,NO_PTS);
         
            assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[0],0);
            assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[1],1);
            Dwoe_blk_dble(DFL_BLOCKS,n,0,1);
            assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,0,psd[2]);
            assign_sdblk2sd(DFL_BLOCKS,n,ALL_PTS,1,psd[3]);
         }      

         error_chk();
         Dwoe_dble(psd[0],psd[1]);
         blk_sd2zero(ic^0x1,psd[1]);      
         blk_sd2zero(ic^0x1,psd[3]);
      
         mulr_spinor_add_dble(VOLUME,psd[0],psd[2],-1.0);
         mulr_spinor_add_dble(VOLUME,psd[1],psd[3],-1.0);   

         if (norm_square_dble(VOLUME,0,psd[0])!=0.0)
            ie=1;
   
         d=norm_square_dble(VOLUME,1,psd[1])/
            norm_square_dble(VOLUME,1,psd[3]);

         if (d>dmax)
            dmax=d;
      }
   
      error_chk();
      error(ie,1,"main [check8.c]",
            "Dwoe_blk_dble() changes the fields where it should not"); 

      dmax=sqrt(dmax);
   
      if (my_rank==0)
         printf("Dwoe_blk_dble():                  %.1e\n",dmax);
   
      dmax=0.0;
      random_sd(VOLUME,psd[0],1.0);
      random_sd(VOLUME,psd[1],1.0);
   
      for (n=0;n<nb;n++)
      {
         assign_ud2udblk(DFL_BLOCKS,n);
         assign_swd2swdblk(DFL_BLOCKS,n,NO_PTS);

         assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[0],0);
         Dwoe_blk_dble(DFL_BLOCKS,n,0,1);
         Dwee_blk_dble(DFL_BLOCKS,n,mu,0,0);      
         Dwoo_blk_dble(DFL_BLOCKS,n,0.0,1,1);
         Dweo_blk_dble(DFL_BLOCKS,n,1,0);
      
         assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[0],1);
         Dwhat_blk_dble(DFL_BLOCKS,n,mu,1,2);
         mulr_spinor_add_dble(volh,b[n].sd[0],b[n].sd[2],-1.0);
         d=norm_square_dble(volh,0,b[n].sd[0]);
         if (d>dmax)
            dmax=d;

         assign_sd2sdblk(DFL_BLOCKS,n,ALL_PTS,psd[0],0);      
         mulr_spinor_add_dble(volh,b[n].sd[0]+volh,b[n].sd[1]+volh,-1.0);
         if (norm_square_dble(volh,0,b[n].sd[0]+volh)!=0.0)
            ie=1;
      }

      error_chk();
      error(ie,1,"main [check8.c]",
            "Dwhat_blk_dble() changes the fields where it should not"); 

      dmax=sqrt(dmax);
   
      if (NPROC>1)
      {
         d=dmax;
         MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
         MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);      
      }

      if (my_rank==0)
         printf("Dwhat_blk_dble():                 %.1e\n\n",dmax);
   }

   if (my_rank==0)
      fclose(flog);
   
   MPI_Finalize();
   exit(0);
}
