
/*******************************************************************************
*
* File check9.c
*
* Copyright (C) 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Comparison of Dw_bnd() with Dw()
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
#include "sap.h"
#include "dirac.h"
#include "global.h"

typedef union
{
   weyl w;
   float r[12];
} spin_t;


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


static void random_weyl(int vol,weyl *w)
{
   spin_t *ws,*wm;

   ws=(spin_t*)(w);
   wm=ws+vol;

   for (;ws<wm;ws++)
      gauss((*ws).r,12);
}


static void random_bnd(int ic)
{
   int nb,isw;
   int nbh,n,nm;
   block_t *b;
   bndry_t *bb;

   b=blk_list(SAP_BLOCKS,&nb,&isw);
   nbh=nb/2;

   if (ic^isw)
      n=nbh;
   else
      n=0;

   nm=n+nbh;   

   for (;n<nm;n++)
   {
      bb=b[n].bb;

      if ((cpr[0]==0)&&(b[n].bo[0]==0))
         random_weyl(bb[0].vol,bb[0].w[0]);

      if ((cpr[0]==(NPROC0-1))&&((b[n].bo[0]+b[n].bs[0])==L0))
         random_weyl(bb[1].vol,bb[1].w[0]);
   }   
}


int main(int argc,char *argv[])
{
   int my_rank,nb,isw,nbh,ic;
   int bs[4],n,nm,vol,ie;
   float mu,d,dmax;
   complex z;
   spinor **ps;
   block_t *b;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check9.log","w",stdout);
      fin=freopen("check7.in","r",stdin);

      printf("\n");
      printf("Comparison of Dw_bnd() with Dw()\n");
      printf("--------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
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
   
   ps=reserve_ws(4);
   b=blk_list(SAP_BLOCKS,&nb,&isw);
   nbh=nb/2;
   vol=(*b).vol;
   z.re=-1.0f;
   z.im=0.0f;

   ie=0;
   dmax=0.0f;

   for (ic=0;ic<2;ic++)
   {
      random_s(VOLUME,ps[0],1.0f);
      assign_s2s(VOLUME,ps[0],ps[3]);

      if (ic^isw)
         n=nbh;
      else
         n=0;
      nm=n+nbh;
      
      for (;n<nm;n++)
      {
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],0);
         Dw_bnd(SAP_BLOCKS,n,0,0);
      }      

      random_bnd(ic);
      random_s(VOLUME,ps[1],1.0f);
      assign_s2s(VOLUME,ps[1],ps[2]);
      sap_com(ic,ps[1]);
      mulc_spinor_add(VOLUME,ps[2],ps[1],z);

      blk_s2zero(ic^0x1,ps[0]);
      Dw(mu,ps[0],ps[1]);
      blk_s2zero(ic,ps[1]);
      
      mulc_spinor_add(VOLUME,ps[1],ps[2],z);
      d=norm_square(VOLUME,1,ps[1])/
         norm_square(VOLUME,1,ps[2]);

      if (d>dmax)
         dmax=d;

      if (ic^isw)
         n=nbh;
      else
         n=0;
      nm=n+nbh;
      
      for (;n<nm;n++)
      {
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[3],0);
         Dw_bnd(SAP_BLOCKS,n,0,0);
         assign_s2sblk(SAP_BLOCKS,n,ALL_PTS,ps[0],1);
         mulc_spinor_add(vol,b[n].s[0],b[n].s[1],z);
         d=norm_square(vol,0,b[n].s[0]);

         if (d!=0.0f)
            ie=1;
      }  
   }
   
   error_chk();

   error(ie,1,"main [check9.c]",
     "Dw_bnd() changes the input field where it should not");

   dmax=(float)(sqrt((double)(dmax)));
   
   if (my_rank==0)
   {
      printf("\n");
      printf("The maximal relative deviation is %.1e\n",dmax);
      fclose(flog);
   }
   
   MPI_Finalize();
   exit(0);
}
