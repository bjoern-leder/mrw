
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2005, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the gauge covariance of the SW term
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
#include "linalg.h"
#include "sw_term.h"
#include "global.h"

static int nfc[8],ofs[8];
static su3_dble *g,*gbuf;


static void pack_gbuf(void)
{
   int n,ix,iy,io;

   nfc[0]=FACE0/2;
   nfc[1]=FACE0/2;
   nfc[2]=FACE1/2;
   nfc[3]=FACE1/2;
   nfc[4]=FACE2/2;
   nfc[5]=FACE2/2;
   nfc[6]=FACE3/2;
   nfc[7]=FACE3/2;

   ofs[0]=0;
   ofs[1]=ofs[0]+nfc[0];
   ofs[2]=ofs[1]+nfc[1];
   ofs[3]=ofs[2]+nfc[2];
   ofs[4]=ofs[3]+nfc[3];
   ofs[5]=ofs[4]+nfc[4];
   ofs[6]=ofs[5]+nfc[5];
   ofs[7]=ofs[6]+nfc[6];

   for (n=0;n<8;n++)
   {
      io=ofs[n];

      for (ix=0;ix<nfc[n];ix++)
      {
         iy=map[io+ix];
         gbuf[io+ix]=g[iy];
      }
   }
}


static void send_gbuf(void)
{
   int n,mu,np,saddr,raddr;
   int nbf,tag;
   su3_dble *sbuf,*rbuf;
   MPI_Status stat;

   for (n=0;n<8;n++)
   {
      nbf=18*nfc[n];

      if (nbf>0)
      {
         tag=mpi_tag();
         mu=n/2;
         np=cpr[mu];

         if (n==(2*mu))
         {
            saddr=npr[n+1];
            raddr=npr[n];
         }
         else
         {
            saddr=npr[n-1];
            raddr=npr[n];
         }

         sbuf=gbuf+ofs[n];
         rbuf=g+ofs[n]+VOLUME;

         if ((np|0x1)!=np)
         {
            MPI_Send((double*)(sbuf),nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv((double*)(rbuf),nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,
                     &stat);
         }
         else
         {
            MPI_Recv((double*)(rbuf),nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,
                     &stat);
            MPI_Send((double*)(sbuf),nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}


static void random_g(void)
{
   su3_dble *gx,*gm;

   gm=g+VOLUME;

   for (gx=g;gx<gm;gx++)
      random_su3_dble(gx);

   if (BNDRY>0)
   {
      pack_gbuf();
      send_gbuf();
   }
}


static void transform_ud(void)
{
   int ix,iy,mu;
   su3_dble *ub,u,v,w;
   su3_dble gx,gxi,gy,gyi;

   ub=udfld();
   
   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      gx=g[ix];

      for (mu=0;mu<4;mu++)
      {
         iy=iup[ix][mu];
         gy=g[iy];
         u=ub[2*mu];
         _su3_dagger(gyi,gy);
         _su3_times_su3(v,u,gyi);
         _su3_times_su3(w,gx,v);
         ub[2*mu]=w;

         iy=idn[ix][mu];
         gy=g[iy];
         u=ub[2*mu+1];
         _su3_dagger(gxi,gx);
         _su3_times_su3(v,u,gxi);
         _su3_times_su3(w,gy,v);
         ub[2*mu+1]=w;
      }

      ub+=8;
   }

   set_flags(UPDATED_UD);
}


static void transform_sd(spinor_dble *pk,spinor_dble *pl)
{
   int ix;
   su3_dble gx;
   spinor_dble r,s;

   for (ix=0;ix<VOLUME;ix++)
   {
      s=pk[ix];
      gx=g[ix];

      _su3_multiply(r.c1,gx,s.c1);
      _su3_multiply(r.c2,gx,s.c2);
      _su3_multiply(r.c3,gx,s.c3);
      _su3_multiply(r.c4,gx,s.c4);

      pl[ix]=r;
   }
}


int main(int argc,char *argv[])
{
   int my_rank,i;
   double d;
   complex_dble z;
   spinor_dble **psd;
   pauli_dble *sw;
   sw_parms_t swp;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      printf("\n");
      printf("Gauge covariance of the SW term (random fields)\n");
      printf("-----------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,123456);
   geometry();
   alloc_wsd(4);
   psd=reserve_wsd(4);
   g=amalloc(NSPIN*sizeof(su3_dble),4);

   if (BNDRY>0)
      gbuf=amalloc((BNDRY/2)*sizeof(su3_dble),4);

   error((g==NULL)||((BNDRY>0)&&(gbuf==NULL)),1,"main [check2.c]",
         "Unable to allocate auxiliary arrays");

   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.456,1.0,1.234);
   swp=set_sw_parms(-0.0123);

   if (my_rank==0)
      printf("m0 = %.4e, csw = %.4e, cF = %.4e\n\n",swp.m0,swp.csw,swp.cF);
   
   random_g();
   random_ud();   
   z.re=-1.0;
   z.im=0.0;

   for (i=0;i<4;i++)
      random_sd(VOLUME,psd[i],1.0);

   (void)sw_term(NO_PTS);
   sw=swdfld();
   apply_sw_dble(VOLUME,0.789,sw,psd[0],psd[1]);

   transform_sd(psd[0],psd[2]);
   transform_ud();
   (void)sw_term(NO_PTS);
   sw=swdfld();
   apply_sw_dble(VOLUME,0.789,sw,psd[2],psd[3]);
   transform_sd(psd[1],psd[2]);

   mulc_spinor_add_dble(VOLUME,psd[3],psd[2],z);
   d=norm_square_dble(VOLUME,1,psd[3])/norm_square_dble(VOLUME,1,psd[0]);
   error_chk();

   if (my_rank==0)
   {
      printf("Maximal normalized difference = %.2e\n",sqrt(d));
      printf("(should be around 1*10^(-15) or so)\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
