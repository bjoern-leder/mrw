
/*******************************************************************************
*
* File check7.c
*
* Copyright (C) 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Comparison of Dw_blk(),... with Dw()
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


static void blk_s2zero(int ic,spinor *s)
{
   int nb,isw;
   int nbh,n,nm,vol;
   block_t *b;

   b=blk_list(SAP_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;

   if (ic^isw)
      n=nbh;
   else
      n=0;

   nm=n+nbh;

   for (;n<nm;n++)
   {
      set_s2zero(vol,b[n].s[0]);
      assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,0,s);
   }
}


int main(int argc,char *argv[])
{
   int my_rank,nb,isw,nbh,ic,itm;
   int bs[4],n,nm,vol,volh,ie;
   float mu,d,dmax;
   spinor **ps;
   block_t *b;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check7.log","w",stdout);
      fin=freopen("check7.in","r",stdin);

      printf("\n");
      printf("Comparison of Dw_blk(),... with Dw()\n");
      printf("------------------------------------\n\n");

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
   set_sap_parms(bs,0,1,1);
   alloc_bgr(SAP_BLOCKS);
   alloc_ws(4);
   
   set_lat_parms(5.6,1.0,0.0,0.0,0.0,0.1,1.3,1.15);
   set_sw_parms(0.05);
   mu=0.123f;
   
   random_ud();
   sw_term(NO_PTS);

   assign_ud2u();
   assign_swd2sw();
   assign_ud2ubgr(SAP_BLOCKS);
   assign_swd2swbgr(SAP_BLOCKS,NO_PTS);
   
   ps=reserve_ws(4);
   b=blk_list(SAP_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;
   volh=vol/2;

   for (itm=0;itm<2;itm++)
   {
      ie=0;
      dmax=0.0f;
      set_tm_parms(itm);
      
      if (my_rank==0)
         printf("Twisted-mass flag = %d\n",itm);

      for (ic=0;ic<2;ic++)
      {
         random_s(VOLUME,ps[0],1.0f);
         random_s(VOLUME,ps[2],1.0f);
         blk_s2zero(ic^0x1,ps[0]);
         blk_s2zero(ic^0x1,ps[2]);

         if (ic^isw)
            n=nbh;
         else
            n=0;

         nm=n+nbh;
      
         for (;n<nm;n++)
         {
            random_s(vol,b[n].s[1],1.0f);
            assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],0);
            Dw_blk(SAP_BLOCKS,n,mu,0,1);
            assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,0,ps[2]);
            assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,1,ps[3]);
         }      

         error_chk();
         Dw(mu,ps[0],ps[1]);
         blk_s2zero(ic^0x1,ps[1]);      
         blk_s2zero(ic^0x1,ps[3]);
      
         mulr_spinor_add(VOLUME,ps[0],ps[2],-1.0f);
         mulr_spinor_add(VOLUME,ps[1],ps[3],-1.0f);   

         if (norm_square(VOLUME,0,ps[0])!=0.0f)
            ie=1;
   
         d=norm_square(VOLUME,1,ps[1])/
            norm_square(VOLUME,1,ps[3]);

         if (d>dmax)
            dmax=d;
      }
   
      error_chk();
      error(ie,1,"main [check7.c]",
            "Dw_blk() changes the fields where it should not"); 

      dmax=(float)(sqrt((double)(dmax)));

      if (my_rank==0)
      {
         printf("The maximal relative deviations are:\n\n");
         printf("Dw_blk():               %.1e\n",dmax);
      }

      dmax=0.0f;
      random_s(VOLUME,ps[0],1.0f);
      random_s(VOLUME,ps[1],1.0f);
   
      for (n=0;n<nb;n++)
      {
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],0);
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[1],1);
         Dwee_blk(SAP_BLOCKS,n,mu,0,1);     
         assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,0,ps[2]);      
         assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,1,ps[3]);

         assign_s2sblk(SAP_BLOCKS,n,EVEN_PTS,ps[0],0);
         assign_s2sblk(SAP_BLOCKS,n,ODD_PTS,ps[1],0);      
         Dwee_blk(SAP_BLOCKS,n,mu,0,0);
         mulr_spinor_add(vol,b[n].s[0],b[n].s[1],-1.0f);
         if (norm_square(vol,0,b[n].s[0])!=0.0f)
            ie=1;
      }

      Dwee(mu,ps[0],ps[1]);
      mulr_spinor_add(VOLUME,ps[0],ps[2],-1.0f);
      mulr_spinor_add(VOLUME,ps[1],ps[3],-1.0f);   

      if (norm_square(VOLUME,0,ps[0])!=0.0f)
         ie=1;
   
      d=norm_square(VOLUME,1,ps[1])/
         norm_square(VOLUME,1,ps[3]);

      if (d>dmax)
         dmax=d;

      random_s(VOLUME,ps[0],1.0f);
      random_s(VOLUME,ps[1],1.0f);
   
      for (n=0;n<nb;n++)
      {
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],0);
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[1],1);
         Dwoo_blk(SAP_BLOCKS,n,mu,0,1);     
         assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,0,ps[2]);      
         assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,1,ps[3]);

         assign_s2sblk(SAP_BLOCKS,n,ODD_PTS,ps[0],0);
         assign_s2sblk(SAP_BLOCKS,n,EVEN_PTS,ps[1],0);      
         Dwoo_blk(SAP_BLOCKS,n,mu,0,0);
         mulr_spinor_add(vol,b[n].s[0],b[n].s[1],-1.0f);
         if (norm_square(vol,0,b[n].s[0])!=0.0f)
            ie=1;
      }

      Dwoo(mu,ps[0],ps[1]);
      mulr_spinor_add(VOLUME,ps[0],ps[2],-1.0f);
      mulr_spinor_add(VOLUME,ps[1],ps[3],-1.0f);   

      if (norm_square(VOLUME,0,ps[0])!=0.0f)
         ie=1;
   
      d=norm_square(VOLUME,1,ps[1])/
         norm_square(VOLUME,1,ps[3]);

      if (d>dmax)
         dmax=d;
   
      error_chk();
      error(ie,1,"main [check7.c]",
            "Dwee_blk() or Dwoo_blk() changes the fields where it should not"); 

      dmax=(float)(sqrt((double)(dmax)));
   
      if (my_rank==0)
         printf("Dwee_blk(), Dwoo_blk(): %.1e\n",dmax);

      dmax=0.0f;

      for (ic=0;ic<2;ic++)
      {
         random_s(VOLUME,ps[0],1.0f);
         random_s(VOLUME,ps[1],1.0f);
         random_s(VOLUME,ps[2],1.0f);
         blk_s2zero(ic^0x1,ps[0]);
         blk_s2zero(ic^0x1,ps[2]);

         if (ic^isw)
            n=nbh;
         else
            n=0;

         nm=n+nbh;
      
         for (;n<nm;n++)
         {
            assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],0);
            assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[1],1);
            Dweo_blk(SAP_BLOCKS,n,0,1);
            assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,0,ps[2]);
            assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,1,ps[3]);
         }      

         error_chk();
         Dweo(ps[0],ps[1]);
         blk_s2zero(ic^0x1,ps[1]);      
         blk_s2zero(ic^0x1,ps[3]);
      
         mulr_spinor_add(VOLUME,ps[0],ps[2],-1.0f);
         mulr_spinor_add(VOLUME,ps[1],ps[3],-1.0f);   

         if (norm_square(VOLUME,0,ps[0])!=0.0f)
            ie=1;
   
         d=norm_square(VOLUME,1,ps[1])/
            norm_square(VOLUME,1,ps[3]);

         if (d>dmax)
            dmax=d;
      }
   
      error_chk();
      error(ie,1,"main [check7.c]",
            "Dweo_blk() changes the fields where it should not"); 

      dmax=(float)(sqrt((double)(dmax)));
   
      if (my_rank==0)
         printf("Dweo_blk():             %.1e\n",dmax);

      dmax=0.0f;

      for (ic=0;ic<2;ic++)
      {
         random_s(VOLUME,ps[0],1.0f);
         random_s(VOLUME,ps[1],1.0f);      
         random_s(VOLUME,ps[2],1.0f);
         blk_s2zero(ic^0x1,ps[0]);
         blk_s2zero(ic^0x1,ps[2]);

         if (ic^isw)
            n=nbh;
         else
            n=0;

         nm=n+nbh;
      
         for (;n<nm;n++)      
         {
            assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],0);
            assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[1],1);
            Dwoe_blk(SAP_BLOCKS,n,0,1);
            assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,0,ps[2]);
            assign_sblk2s(SAP_BLOCKS,n,ALL_PTS,1,ps[3]);
         }      

         error_chk();
         Dwoe(ps[0],ps[1]);
         blk_s2zero(ic^0x1,ps[1]);      
         blk_s2zero(ic^0x1,ps[3]);
      
         mulr_spinor_add(VOLUME,ps[0],ps[2],-1.0f);
         mulr_spinor_add(VOLUME,ps[1],ps[3],-1.0f);   

         if (norm_square(VOLUME,0,ps[0])!=0.0f)
            ie=1;
   
         d=norm_square(VOLUME,1,ps[1])/
            norm_square(VOLUME,1,ps[3]);

         if (d>dmax)
            dmax=d;
      }
   
      error_chk();
      error(ie,1,"main [check7.c]",
            "Dwoe_blk() changes the fields where it should not"); 

      dmax=(float)(sqrt((double)(dmax)));
   
      if (my_rank==0)
         printf("Dwoe_blk():             %.1e\n",dmax);
   
      dmax=0.0f;
      random_s(VOLUME,ps[0],1.0f);
      random_s(VOLUME,ps[1],1.0f);
   
      for (n=0;n<nb;n++)
      {
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],0);
         Dwoe_blk(SAP_BLOCKS,n,0,1);
         Dwee_blk(SAP_BLOCKS,n,mu,0,0);      
         Dwoo_blk(SAP_BLOCKS,n,0.0f,1,1);
         Dweo_blk(SAP_BLOCKS,n,1,0);
      
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],1);
         Dwhat_blk(SAP_BLOCKS,n,mu,1,2);
         mulr_spinor_add(volh,b[n].s[0],b[n].s[2],-1.0f);
         d=norm_square(volh,0,b[n].s[0]);
         if (d>dmax)
            dmax=d;

         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],0);      
         mulr_spinor_add(volh,b[n].s[0]+volh,b[n].s[1]+volh,-1.0f);
         if (norm_square(volh,0,b[n].s[0]+volh)!=0.0f)
            ie=1;
      }

      error_chk();
      error(ie,1,"main [check7.c]",
            "Dwhat_blk() changes the fields where it should not"); 

      dmax=(float)(sqrt((double)(dmax)));
   
      if (NPROC>1)
      {
         d=dmax;
         MPI_Reduce(&d,&dmax,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
         MPI_Bcast(&dmax,1,MPI_FLOAT,0,MPI_COMM_WORLD);      
      }

      if (my_rank==0)
         printf("Dwhat_blk():            %.1e\n\n",dmax);
   }

   if (my_rank==0)
      fclose(flog);   
   
   MPI_Finalize();
   exit(0);
}
