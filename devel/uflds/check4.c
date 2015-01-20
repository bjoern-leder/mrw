
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2005, 2007, 2008, 2009, 2010, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the program for the plaquette action of the double-precision
* gauge field
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

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


static void random_vec(int *svec)
{
   int mu,bs[4];
   double r[4];

   bs[0]=NPROC0*L0;
   bs[1]=NPROC1*L1;
   bs[2]=NPROC2*L2;
   bs[3]=NPROC3*L3;

   ranlxd(r,4);

   for (mu=0;mu<4;mu++)
   {
      svec[mu]=(int)((double)(bs[mu])*r[mu]);
      if (svec[mu]>(bs[mu]/2))
         svec[mu]-=bs[mu];
   }

   MPI_Bcast(svec,4,MPI_INT,0,MPI_COMM_WORLD);
}


int main(int argc,char *argv[])
{
   int my_rank,n,s[4];
   double cG,npl,p1,p2,eps;
   double phi[4];
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);

      printf("\n");
      printf("Plaquette sums of the double-precision gauge field\n");
      printf("--------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
      fflush(flog);
   }

   start_ranlux(0,12345);
   geometry();

   g=amalloc(NSPIN*sizeof(*g),4);

   if (BNDRY>0)
      gbuf=amalloc((BNDRY/2)*sizeof(*gbuf),4);

   error((g==NULL)||((BNDRY>0)&&(gbuf==NULL)),1,"main [check4.c]",
         "Unable to allocate auxiliary arrays");

   cG=1.3456;   
   npl=(double)(6*(N0-1))*(double)(N1*N2*N3);
   set_lat_parms(5.3,1.6667,0.0,0.0,0.0,1.234,cG,1.876);
   p1=plaq_wsum_dble(1);

   if (my_rank==0)
      printf("Average after initialization = %.8e (cG!=1), ",p1/npl);

   npl=(double)(6*N0-3)*(double)(N1*N2*N3);   
   p1=plaq_sum_dble(1);

   if (my_rank==0)
   {
      printf("%.8e (cG=1)\n",p1/npl);
      printf("(should be equal to 3.0)\n\n");
   }   
   
   print_flags();
   random_ud();

   p1=plaq_wsum_dble(1);
   random_g();
   transform_ud();
   p2=plaq_wsum_dble(1);

   eps=sqrt(npl)*(double)(DBL_EPSILON);   
   
   if (my_rank==0)
   {
      printf("Gauge invariance: absolute difference = %.1e\n",
             fabs(p1-p2));
      printf("(expected to be %.1e or so)\n\n",eps);
   }

   if (my_rank==0)
      printf("Translation invariance:\n");

   p1=plaq_wsum_dble(1);
      
   for (n=0;n<8;n++)
   {
      random_vec(s);
      s[0]=0;
      shift_ud(s);
      p2=plaq_wsum_dble(1);

      if (my_rank==0)
      {
         printf("s=(% 3d,% 3d,% 3d), ",s[1],s[2],s[3]);
         printf("absolute deviation = %.1e\n",fabs(p1-p2));
      }
   }

   set_lat_parms(5.3,1.6667,0.0,0.0,0.0,1.234,2.0,1.876);
   phi[0]=0.1;
   phi[1]=0.2;
   phi[2]=0.3;
   phi[3]=0.4;
   set_sf_parms(phi,phi+2);
   random_ud();
   
   p1=plaq_sum_dble(1);
   p2=plaq_wsum_dble(1);
   
   if (my_rank==0)
   {
      printf("\n");
      printf("Comparison of plaq_sum_dble() and plaq_wsum_dble() with\n");
      printf("SF boundary conditions: absolute deviation = %.1e\n\n",
             fabs(p1-p2-9.0*(double)(N1*N2*N3)));
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
