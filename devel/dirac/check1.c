
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2005, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Gauge covariance of Dw()
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
#include "dirac.h"
#include "global.h"

static int nfc[8],ofs[8];
static su3 *g,*gbuf;


static void assign_u2ud(void)
{
   su3 *u,*um;
   su3_dble *ud;

   u=ufld();
   um=u+4*VOLUME;
   ud=udfld();

   for (;u<um;u++)
   {
      (*ud).c11.re=(double)((*u).c11.re);
      (*ud).c11.im=(double)((*u).c11.im);
      (*ud).c12.re=(double)((*u).c12.re);
      (*ud).c12.im=(double)((*u).c12.im);
      (*ud).c13.re=(double)((*u).c13.re);
      (*ud).c13.im=(double)((*u).c13.im);

      (*ud).c21.re=(double)((*u).c21.re);
      (*ud).c21.im=(double)((*u).c21.im);
      (*ud).c22.re=(double)((*u).c22.re);
      (*ud).c22.im=(double)((*u).c22.im);
      (*ud).c23.re=(double)((*u).c23.re);
      (*ud).c23.im=(double)((*u).c23.im);
   
      (*ud).c21.re=(double)((*u).c21.re);
      (*ud).c21.im=(double)((*u).c21.im);
      (*ud).c22.re=(double)((*u).c22.re);
      (*ud).c22.im=(double)((*u).c22.im);
      (*ud).c23.re=(double)((*u).c23.re);
      (*ud).c23.im=(double)((*u).c23.im);

      (*ud).c31.re=(double)((*u).c31.re);
      (*ud).c31.im=(double)((*u).c31.im);
      (*ud).c32.re=(double)((*u).c32.re);
      (*ud).c32.im=(double)((*u).c32.im);
      (*ud).c33.re=(double)((*u).c33.re);
      (*ud).c33.im=(double)((*u).c33.im);

      project_to_su3_dble(ud);
      ud+=1;
   }

   set_flags(UPDATED_UD);
}


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
   su3 *sbuf,*rbuf;
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
            MPI_Send((float*)(sbuf),nbf,MPI_FLOAT,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv((float*)(rbuf),nbf,MPI_FLOAT,raddr,tag,MPI_COMM_WORLD,
                     &stat);
         }
         else
         {
            MPI_Recv((float*)(rbuf),nbf,MPI_FLOAT,raddr,tag,MPI_COMM_WORLD,
                     &stat);
            MPI_Send((float*)(sbuf),nbf,MPI_FLOAT,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}


static void random_g(void)
{
   su3 *gx,*gm;

   gm=g+VOLUME;

   for (gx=g;gx<gm;gx++)
      random_su3(gx);

   if (BNDRY>0)
   {
      pack_gbuf();
      send_gbuf();
   }
}


static void transform_u(void)
{
   int ix,iy,mu;
   su3 *ub,u,v,w;
   su3 gx,gxi,gy,gyi;

   ub=ufld();
   
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

   set_flags(UPDATED_U);
}


static void transform_s(spinor *pk,spinor *pl)
{
   int ix;
   su3 gx;
   spinor r,s;

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
   float mu,d;
   complex z;
   spinor **ps;
   sw_parms_t swp;   
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);
      printf("\n");
      printf("Gauge covariance of Dw() (random fields)\n");
      printf("----------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,12345);
   geometry();
   alloc_ws(5);
   ps=reserve_ws(5);
   g=amalloc(NSPIN*sizeof(su3),4);

   if (BNDRY>0)
      gbuf=amalloc((BNDRY/2)*sizeof(su3),4);

   error((g==NULL)||((BNDRY>0)&&(gbuf==NULL)),1,"main [check1.c]",
         "Unable to allocate auxiliary arrays");

   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.456,1.0,1.234);
   swp=set_sw_parms(-0.0123);
   mu=0.0376;

   if (my_rank==0)
      printf("m0 = %.4e, mu= %.4e, csw = %.4e, cF = %.4e\n\n",
             swp.m0,mu,swp.csw,swp.cF);

   random_g();
   random_u();
   assign_u2ud();
   sw_term(NO_PTS);
   assign_swd2sw();
   z.re=-1.0f;
   z.im=0.0f;
   
   for (i=0;i<4;i++)
      random_s(NSPIN,ps[i],1.0f);

   assign_s2s(VOLUME,ps[0],ps[4]);
   bnd_s2zero(ALL_PTS,ps[4]);
   Dw(mu,ps[0],ps[1]);
   mulc_spinor_add(VOLUME,ps[4],ps[0],z);
   d=norm_square(VOLUME,1,ps[4]);
   error(d!=0.0f,1,"main [check1.c]","Dw() changes the input field");

   Dw(mu,ps[0],ps[4]);
   mulc_spinor_add(VOLUME,ps[4],ps[1],z);
   d=norm_square(VOLUME,1,ps[4]);
   error(d!=0.0f,1,"main [check1.c]","Action of Dw() depends "
         "on the boundary values of the input field");   
   
   assign_s2s(VOLUME,ps[1],ps[4]);
   bnd_s2zero(ALL_PTS,ps[4]);
   mulc_spinor_add(VOLUME,ps[4],ps[1],z);
   d=norm_square(VOLUME,1,ps[4]);
   error(d!=0.0f,1,"main [check1.c]",
         "Dw() does not vanish at global time 0 and NPROC0*L0-1 ");  
   
   transform_s(ps[0],ps[2]);   
   transform_u();
   assign_u2ud();
   sw_term(NO_PTS);
   assign_swd2sw();
   Dw(mu,ps[2],ps[3]);
   transform_s(ps[1],ps[2]);

   mulc_spinor_add(VOLUME,ps[3],ps[2],z);
   d=norm_square(VOLUME,1,ps[3])/norm_square(VOLUME,1,ps[0]);
   error_chk();

   if (my_rank==0)
   {
      printf("Normalized difference = %.2e\n",sqrt((double)(d)));
      printf("(should be around 1*10^(-6) or so)\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
