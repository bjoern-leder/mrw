
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of the SW term for abelian background fields
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

static int np[4];
static double t[3],a[4],p[4];
static double (*Fhat)[3];
static const su3_dble ud0={{0.0}};
static spinor_dble ws;


static void alloc_Fhat(void)
{
   Fhat=amalloc(VOLUME*sizeof(*Fhat),3);

   error(Fhat==NULL,1,"alloc_Fhat [check3.c]",
         "Unable to allocate auxiliary array");
}


static void set_parms(void)
{
   int my_rank;
   double pi;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      gauss_dble(t,2);
      t[2]=-t[0]-t[1];

      ranlxd(a,4);

      np[0]=(int)(a[0]*(double)(NPROC0*L0));
      np[1]=(int)(a[1]*(double)(NPROC1*L1));
      np[2]=(int)(a[2]*(double)(NPROC2*L2));
      np[3]=(int)(a[3]*(double)(NPROC3*L3));

      pi=4.0*atan(1.0);

      p[0]=(double)(np[0])*2.0*pi/(double)(NPROC0*L0);
      p[1]=(double)(np[1])*2.0*pi/(double)(NPROC1*L1);
      p[2]=(double)(np[2])*2.0*pi/(double)(NPROC2*L2);
      p[3]=(double)(np[3])*2.0*pi/(double)(NPROC3*L3);

      gauss_dble(a,4);
   }

   MPI_Bcast(t,3,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(a,4,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(p,4,MPI_DOUBLE,0,MPI_COMM_WORLD);
}


static void set_ud(void)
{
   int bo[4],x0,x1,x2,x3,ix,mu;
   double px,r1,r2;
   su3_dble *udb,*u;

   udb=udfld();
   
   bo[0]=cpr[0]*L0;
   bo[1]=cpr[1]*L1;
   bo[2]=cpr[2]*L2;
   bo[3]=cpr[3]*L3;

   for (x0=0;x0<L0;x0++)
   {
      for (x1=0;x1<L1;x1++)
      {
         for (x2=0;x2<L2;x2++)
         {
            for (x3=0;x3<L3;x3++)
            {
               ix=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*x0];

               if (ix>=(VOLUME/2))
               {
                  px=p[0]*(double)(x0+bo[0])+p[1]*(double)(x1+bo[1])+
                     p[2]*(double)(x2+bo[2])+p[3]*(double)(x3+bo[3]);
                  u=udb+8*(ix-(VOLUME/2));

                  for (mu=0;mu<4;mu++)
                  {
                     (*u)=ud0;
                     r1=a[mu]*sin(px);
                     r2=t[0]*r1;
                     (*u).c11.re=cos(r2);
                     (*u).c11.im=sin(r2);
                     r2=t[1]*r1;
                     (*u).c22.re=cos(r2);
                     (*u).c22.im=sin(r2);
                     r2=t[2]*r1;
                     (*u).c33.re=cos(r2);
                     (*u).c33.im=sin(r2);
                     u+=1;

                     (*u)=ud0;
                     r1=a[mu]*sin(px-p[mu]);
                     r2=t[0]*r1;
                     (*u).c11.re=cos(r2);
                     (*u).c11.im=sin(r2);
                     r2=t[1]*r1;
                     (*u).c22.re=cos(r2);
                     (*u).c22.im=sin(r2);
                     r2=t[2]*r1;
                     (*u).c33.re=cos(r2);
                     (*u).c33.im=sin(r2);
                     u+=1;
                  }
               }
            }
         }
      }
   }

   openbcd();
   set_flags(UPDATED_UD);
}


static void compute_Fhat(int mu,int nu)
{
   int bo[4],x0,x1,x2,x3,ix;
   int n,*bp,*bm;
   double px,fx,fy,fw,fz;

   bo[0]=cpr[0]*L0;
   bo[1]=cpr[1]*L1;
   bo[2]=cpr[2]*L2;
   bo[3]=cpr[3]*L3;

   for (x0=0;x0<L0;x0++)
   {
      for (x1=0;x1<L1;x1++)
      {
         for (x2=0;x2<L2;x2++)
         {
            for (x3=0;x3<L3;x3++)
            {
               ix=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*x0];

               px=p[0]*(double)(x0+bo[0])+p[1]*(double)(x1+bo[1])+
                  p[2]*(double)(x2+bo[2])+p[3]*(double)(x3+bo[3]);

               fx=a[nu]*(sin(px+p[mu])-sin(px))-a[mu]*(sin(px+p[nu])-sin(px));
               px-=p[mu];
               fy=a[nu]*(sin(px+p[mu])-sin(px))-a[mu]*(sin(px+p[nu])-sin(px));
               px-=p[nu];
               fz=a[nu]*(sin(px+p[mu])-sin(px))-a[mu]*(sin(px+p[nu])-sin(px));
               px+=p[mu];
               fw=a[nu]*(sin(px+p[mu])-sin(px))-a[mu]*(sin(px+p[nu])-sin(px));

               Fhat[ix][0]=0.25*(sin(t[0]*fx)+sin(t[0]*fy)+
                                 sin(t[0]*fw)+sin(t[0]*fz));
               Fhat[ix][1]=0.25*(sin(t[1]*fx)+sin(t[1]*fy)+
                                 sin(t[1]*fw)+sin(t[1]*fz));
               Fhat[ix][2]=0.25*(sin(t[2]*fx)+sin(t[2]*fy)+
                                 sin(t[2]*fw)+sin(t[2]*fz));
            }
         }
      }
   }

   if ((mu==0)||(nu==0))
   {
      bp=bnd_pts(&n);
      bm=bp+n;

      for (;bp<bm;bp++)
      {
         ix=(*bp);
         Fhat[ix][0]=0.0;
         Fhat[ix][1]=0.0;
         Fhat[ix][2]=0.0;         
      }
   }
}


static su3_vector_dble mul_cplx(complex_dble z,su3_vector_dble s)
{
   su3_vector_dble r;

   r.c1.re=z.re*s.c1.re-z.im*s.c1.im;
   r.c1.im=z.im*s.c1.re+z.re*s.c1.im;
   r.c2.re=z.re*s.c2.re-z.im*s.c2.im;
   r.c2.im=z.im*s.c2.re+z.re*s.c2.im;
   r.c3.re=z.re*s.c3.re-z.im*s.c3.im;
   r.c3.im=z.im*s.c3.re+z.re*s.c3.im;

   return r;
}


static spinor_dble mul_gamma(int mu,spinor_dble s)
{
   spinor_dble r;
   complex_dble i,m_i,m_1;

   i.re=0.0;
   i.im=1.0;

   m_i.re=0.0;
   m_i.im=-1.0;

   m_1.re=-1.0;
   m_1.im=0.0;

   if (mu==0)
   {
      r.c1=mul_cplx(m_1,s.c3);
      r.c2=mul_cplx(m_1,s.c4);
      r.c3=mul_cplx(m_1,s.c1);
      r.c4=mul_cplx(m_1,s.c2);
   }
   else if (mu==1)
   {
      r.c1=mul_cplx(m_i,s.c4);
      r.c2=mul_cplx(m_i,s.c3);
      r.c3=mul_cplx(i,s.c2);
      r.c4=mul_cplx(i,s.c1);
   }
   else if (mu==2)
   {
      r.c1=mul_cplx(m_1,s.c4);
      r.c2=s.c3;
      r.c3=s.c2;
      r.c4=mul_cplx(m_1,s.c1);
   }
   else if (mu==3)
   {
      r.c1=mul_cplx(m_i,s.c3);
      r.c2=mul_cplx(i,s.c4);
      r.c3=mul_cplx(i,s.c1);
      r.c4=mul_cplx(m_i,s.c2);
   }
   else
   {
      r.c1=s.c1;
      r.c2=s.c2;
      r.c3=mul_cplx(m_1,s.c3);
      r.c4=mul_cplx(m_1,s.c4);
   }

   return r;
}


static spinor_dble mul_sigma(int mu,int nu,spinor_dble s)
{
   complex_dble z;
   spinor_dble r1,r2;

   r1=mul_gamma(nu,s);
   r1=mul_gamma(mu,r1);

   r2=mul_gamma(mu,s);
   r2=mul_gamma(nu,r2);

   _vector_sub_assign(r1.c1,r2.c1);
   _vector_sub_assign(r1.c2,r2.c2);
   _vector_sub_assign(r1.c3,r2.c3);
   _vector_sub_assign(r1.c4,r2.c4);

   z.re=0.0;
   z.im=0.5;
   _vector_mulc(r2.c1,z,r1.c1);
   _vector_mulc(r2.c2,z,r1.c2);
   _vector_mulc(r2.c3,z,r1.c3);
   _vector_mulc(r2.c4,z,r1.c4);

   return r2;
}


static void muladd_pauli(double csw,int mu,int nu,
                         spinor_dble *pk,spinor_dble *pl)
{
   int ix;
   double r;

   compute_Fhat(mu,nu);

   csw=(-0.25)*csw;

   for (ix=0;ix<VOLUME;ix++)
   {
      ws=mul_sigma(mu,nu,*pk);

      r=csw*Fhat[ix][0];
      ws.c1.c1.re*=r;
      ws.c1.c1.im*=r;
      ws.c2.c1.re*=r;
      ws.c2.c1.im*=r;
      ws.c3.c1.re*=r;
      ws.c3.c1.im*=r;
      ws.c4.c1.re*=r;
      ws.c4.c1.im*=r;

      r=csw*Fhat[ix][1];
      ws.c1.c2.re*=r;
      ws.c1.c2.im*=r;
      ws.c2.c2.re*=r;
      ws.c2.c2.im*=r;
      ws.c3.c2.re*=r;
      ws.c3.c2.im*=r;
      ws.c4.c2.re*=r;
      ws.c4.c2.im*=r;

      r=csw*Fhat[ix][2];
      ws.c1.c3.re*=r;
      ws.c1.c3.im*=r;
      ws.c2.c3.re*=r;
      ws.c2.c3.im*=r;
      ws.c3.c3.re*=r;
      ws.c3.c3.im*=r;
      ws.c4.c3.re*=r;
      ws.c4.c3.im*=r;

      _vector_add_assign((*pl).c1,ws.c1);
      _vector_add_assign((*pl).c2,ws.c2);
      _vector_add_assign((*pl).c3,ws.c3);
      _vector_add_assign((*pl).c4,ws.c4);

      pk+=1;
      pl+=1;
   }
}


static void mul_swd(double m0,double csw,spinor_dble *pk,spinor_dble *pl)
{
   int mu,nu;
   double c;
   spinor_dble *pm;

   set_sd2zero(VOLUME,pl);

   for (mu=0;mu<3;mu++)
   {
      for (nu=(mu+1);nu<4;nu++)
         muladd_pauli(2.0*csw,mu,nu,pk,pl);
   }

   pm=pk+VOLUME;
   c=4.0+m0;

   for (;pk<pm;pk++)
   {
      _vector_mulr_assign((*pl).c1,c,(*pk).c1);
      _vector_mulr_assign((*pl).c2,c,(*pk).c2);
      _vector_mulr_assign((*pl).c3,c,(*pk).c3);
      _vector_mulr_assign((*pl).c4,c,(*pk).c4);

      pl+=1;
   }
}


static void bnd_corr(double cF,spinor_dble *pk,spinor_dble *pl)
{
   int *pt,*pm,n,iy;
   double c;

   c=cF-1.0;
   pt=bnd_pts(&n);
   pm=pt+n;

   for (;pt<pm;pt++)
   {
      pl[*pt]=pk[*pt];

      if (global_time(*pt)==0)
         iy=iup[*pt][0];
      else
         iy=idn[*pt][0];

      _vector_mulr_assign(pl[iy].c1,c,pk[iy].c1);
      _vector_mulr_assign(pl[iy].c2,c,pk[iy].c2);
      _vector_mulr_assign(pl[iy].c3,c,pk[iy].c3);
      _vector_mulr_assign(pl[iy].c4,c,pk[iy].c4);                          
   }
}


int main(int argc,char *argv[])
{
   int my_rank,n;
   double d;
   complex_dble z;
   pauli_dble *sw;
   spinor_dble **psd;
   sw_parms_t swp;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      printf("\n");
      printf("Check of the SW term for abelian background fields\n");
      printf("--------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      printf("For this test to pass, the calculated differences delta\n");
      printf("should be at most 1*10^(-14) or so\n\n");
   }

   start_ranlux(0,12345);
   geometry();
   alloc_Fhat();
   alloc_wsd(3);
   psd=reserve_wsd(3);

   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.456,1.0,1.234);
   set_sw_parms(-0.0123);
   swp=sw_parms();

   if (my_rank==0)
      printf("m0=%.4e, csw=%.4e, cF=%.4e\n\n",swp.m0,swp.csw,swp.cF);

   for (n=0;n<4;n++)
   {
      set_parms();
      set_ud();
      (void)sw_term(NO_PTS);
      sw=swdfld();

      random_sd(VOLUME,psd[0],1.0);
      apply_sw_dble(VOLUME,0.0,sw,psd[0],psd[1]);
      mul_swd(swp.m0,swp.csw,psd[0],psd[2]);
      bnd_corr(swp.cF,psd[0],psd[2]);
      
      z.re=-1.0;
      z.im=0.0;
      mulc_spinor_add_dble(VOLUME,psd[2],psd[1],z);
      d=norm_square_dble(VOLUME,1,psd[2])/norm_square_dble(VOLUME,1,psd[0]);

      if (my_rank==0)
      {
         printf("Field number %d:\n",n+1);
         printf("The parameters are:\n");
         printf("t=%.2f,%.2f,%.2f, a=%.2f,%.2f,%.2f,%.2f, ",
                t[0],t[1],t[2],a[0],a[1],a[2],a[3]);
         printf("np=%d,%d,%d,%d\n",np[0],np[1],np[2],np[3]);
         printf("delta = %.2e\n\n",sqrt(d));
      }
   }
   
   error_chk();

   if (my_rank==0)
   {
      printf("\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
