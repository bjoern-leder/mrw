
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2010, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Forward integration of the Wilson flow
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "random.h"
#include "su3fcts.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "forces.h"
#include "wflow.h"
#include "global.h"

static const su3_alg_dble fr0={0.0};
static su3_alg_dble XX ALIGNED16;
static su3_dble mm,uu,vv ALIGNED16;


static double cmp_ud(su3_dble *u,su3_dble *v)
{
   int i;
   double r[18],dev,dmax;

   r[ 0]=(*u).c11.re-(*v).c11.re;
   r[ 1]=(*u).c11.im-(*v).c11.im;
   r[ 2]=(*u).c12.re-(*v).c12.re;
   r[ 3]=(*u).c12.im-(*v).c12.im;
   r[ 4]=(*u).c13.re-(*v).c13.re;
   r[ 5]=(*u).c13.im-(*v).c13.im;

   r[ 6]=(*u).c21.re-(*v).c21.re;
   r[ 7]=(*u).c21.im-(*v).c21.im;
   r[ 8]=(*u).c22.re-(*v).c22.re;
   r[ 9]=(*u).c22.im-(*v).c22.im;
   r[10]=(*u).c23.re-(*v).c23.re;
   r[11]=(*u).c23.im-(*v).c23.im;

   r[12]=(*u).c31.re-(*v).c31.re;
   r[13]=(*u).c31.im-(*v).c31.im;
   r[14]=(*u).c32.re-(*v).c32.re;
   r[15]=(*u).c32.im-(*v).c32.im;
   r[16]=(*u).c33.re-(*v).c33.re;
   r[17]=(*u).c33.im-(*v).c33.im;   

   dmax=0.0;
   
   for (i=0;i<18;i+=2)
   {
      dev=r[i]*r[i]+r[i+1]*r[i+1];
      if (dev>dmax)
         dmax=dev;
   }

   return dmax;
}


static double max_dev_ud(su3_dble *v)
{
   double d,dmax;
   su3_dble *u,*um;

   u=udfld();
   um=u+4*VOLUME;
   dmax=0.0;
   
   for (;u<um;u++)
   {
      d=cmp_ud(u,v);

      if (d>dmax)
         dmax=d;

      v+=1;
   }

   if (NPROC>1)
   {
      d=dmax;
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   }
   
   return sqrt(dmax);
}


static double cmp_fd(su3_alg_dble *f,su3_alg_dble *g)
{
   int i;
   double r[8],dev,dmax;

   r[0]=(*f).c1-(*g).c1;
   r[1]=(*f).c2-(*g).c2;
   r[2]=(*f).c3-(*g).c3;
   r[3]=(*f).c4-(*g).c4;
   r[4]=(*f).c5-(*g).c5;
   r[5]=(*f).c6-(*g).c6;   
   r[6]=(*f).c7-(*g).c7;   
   r[7]=(*f).c8-(*g).c8;
   
   dmax=0.0;
   
   for (i=0;i<8;i++)
   {
      dev=fabs(r[i]);
      if (dev>dmax)
         dmax=dev;
   }

   return dmax;
}


static int ofs(int ix,int mu)
{
   int iy;
   
   if (ix<(VOLUME/2))
   {
      iy=iup[ix][mu];

      return 8*(iy-(VOLUME/2))+2*mu+1;
   }
   else
      return 8*(ix-(VOLUME/2))+2*mu;
}


static double chkfrc(void)
{
   int x0,x1,x2,x3,it;
   int ix,iy,iz,iw,mu,nu;
   double d,dmax;
   su3_alg_dble *frc;
   su3_dble *udb;
   mdflds_t *mdfs;

   udb=udfld();
   mdfs=mdflds();
   dmax=0.0;

   for (x0=1;x0<(L0-2);x0++)
   {
      for (x1=1;x1<(L1-2);x1++)
      {
         for (x2=1;x2<(L2-2);x2++)
         {
            for (x3=1;x3<(L3-2);x3++)
            {
               ix=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*x0];
               
               for (mu=0;mu<4;mu++)
               {
                  cm3x3_zero(1,&mm);
                  iy=iup[ix][mu];
                  
                  for (nu=0;nu<4;nu++)
                  {
                     if (nu!=mu)
                     {
                        iz=iup[ix][nu];

                        su3xsu3dag(udb+ofs(iy,nu),udb+ofs(iz,mu),&uu);
                        su3xsu3dag(&uu,udb+ofs(ix,nu),&vv);
                        cm3x3_add(&vv,&mm);

                        iz=idn[ix][nu];
                        iw=idn[iy][nu];

                        su3dagxsu3(udb+ofs(iz,mu),udb+ofs(iz,nu),&uu);
                        su3dagxsu3(udb+ofs(iw,nu),&uu,&vv);
                        cm3x3_add(&vv,&mm);
                     }
                  }

                  prod2su3alg(udb+ofs(ix,mu),&mm,&XX);                  

                  if (ix<(VOLUME/2))
                     frc=(*mdfs).frc+8*(iy-(VOLUME/2))+2*mu+1;
                  else
                     frc=(*mdfs).frc+8*(ix-(VOLUME/2))+2*mu;
                     
                  d=cmp_fd(&XX,frc);

                  if (d>dmax)
                     dmax=d;
               }
            }
         }
      }
   }

   it=0;
      
   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      x0=global_time(ix);

      if ((x0==0)||(x0==(NPROC0*L0-1)))
      {
         if (x0==0)
            frc=(*mdfs).frc+8*(ix-(VOLUME/2))+1;
         else
            frc=(*mdfs).frc+8*(ix-(VOLUME/2));

         it|=((*frc).c1!=0.0);
         it|=((*frc).c2!=0.0);
         it|=((*frc).c3!=0.0);
         it|=((*frc).c4!=0.0);
         it|=((*frc).c5!=0.0);
         it|=((*frc).c6!=0.0);
         it|=((*frc).c7!=0.0);
         it|=((*frc).c8!=0.0);            
      }
   }

   error(it!=0,1,"chkfrc [check1.c]",
         "Open boundary conditions are not respected by the force");
   
   if (NPROC>1)
   {
      d=dmax;
      MPI_Reduce(&d,&dmax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Bcast(&dmax,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   }
   
   return dmax;
}


static void bnd_frc(int ia,su3_alg_dble *frc)
{
   int k,npt,*pt,*ptm;
   su3_alg_dble *fr;

   pt=bnd_pts(&npt);
   ptm=pt+npt;
   pt+=(npt/2);

   for (;pt<ptm;pt++)
   {
      fr=frc+8*(pt[0]-(VOLUME/2));

      for (k=2;k<8;k++)
      {
         if (ia==0)
         {
            _su3_alg_mul_assign(fr[k],2.0);
         }
         else
            fr[k]=fr0;
      }
   }
}

   
int main(int argc,char *argv[])
{
   int my_rank,n,k,ia,ie;
   double eps,npl,cG,phi[2],phi_prime[2];
   double act0,act1,dev0,dev1;
   su3_dble *udb,*u,*um,**usv;
   su3_alg_dble *frc,**fsv;
   mdflds_t *mdfs;
   lat_parms_t lat;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);
      fin=freopen("check1.in","r",stdin);
      
      printf("\n");
      printf("Forward integration of the Wilson flow\n");
      printf("--------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("n","%d\n",&n);
      read_line("eps","%lf",&eps);      
      fclose(fin);

      printf("n = %d\n",n);
      printf("eps = %.3e\n\n",eps);
      fflush(flog);
   }

   MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&eps,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   start_ranlux(0,1234);
   geometry();
   alloc_wud(1);
   alloc_wfd(2);
   mdfs=mdflds();
   usv=reserve_wud(1);
   fsv=reserve_wfd(1);
   udb=udfld();

   npl=(double)(6*(NPROC0*L0-1));
   npl*=(double)(NPROC1*NPROC2*NPROC3*L1*L2*L3);

   for (ia=0;ia<2;ia++)
   {
      if (ia==0)
      {
         if (my_rank==0)
            printf("Open boundary conditions\n\n");
      }
      else
      {
         phi[0]=0.1;
         phi[1]=0.2;
         phi_prime[0]=-0.3;
         phi_prime[1]=0.4;
         set_sf_parms(phi,phi_prime);

         if (my_rank==0)
            printf("Schroedinger functional boundary conditions\n\n");
      }

      random_ud();
      plaq_frc();
      dev0=chkfrc();
      cm3x3_assign(4*VOLUME,udb,usv[0]);
      assign_alg2alg(4*VOLUME,(*mdfs).frc,fsv[0]);
      fwd_euler(1,eps);

      u=udb;
      um=u+4*VOLUME;
      frc=fsv[0];
      bnd_frc(ia,frc);

      for (;u<um;u++)
      {
         expXsu3(eps,frc,u);
         frc+=1;
      }

      set_flags(UPDATED_UD);   
      dev1=max_dev_ud(usv[0]);

      if (my_rank==0)
      {
         printf("Direct check of the generator: %.1e\n",dev0);
         printf("Check of the 1-step integration: %.1e\n\n",dev1);

         printf("Evolution of the Wilson action:\n\n");
         fflush(stdout);
      }

      random_ud();
      act0=3.0*npl-plaq_wsum_dble(1);

      if (my_rank==0)
         printf("k =  0: %.8e\n",act0/npl);

      for (k=1;k<=n;k++)
      {
         fwd_euler(1,eps);
         act1=3.0*npl-plaq_wsum_dble(1);

         error(((act1>act0)&&(eps>=0.0))||((act1<act0)&&(eps<=0.0)),1,
               "main [check1.c]","The Wilson action is not monotonic");

         act0=act1;
      
         if (my_rank==0)
            printf("k = %2d: %.8e\n",k,act0/npl);
      }         
      
      error_chk();

      if (ia==0)
         ie=check_bcd();
      else
         ie=check_sfbcd();

      error_root(ie!=1,1,"main [check3.c]","Boundary values changed");
   
      if (my_rank==0)
      {
         printf("\n");
         printf("Monotonicity check passed\n");
         fflush(stdout);
      }

      start_ranlux(0,1234);   
      random_ud();
      
      fwd_euler(n,eps);
      cm3x3_assign(4*VOLUME,udb,usv[0]);   
      
      start_ranlux(0,1234);   
      random_ud();
      
      fwd_rk2(n,eps);
      dev0=max_dev_ud(usv[0]);
      cm3x3_assign(4*VOLUME,udb,usv[0]);

      if (my_rank==0)
         printf("Comparison of fwd_euler() and fwd_rk2(): |dU| = %.1e\n",
                dev0);

      start_ranlux(0,1234);   
      random_ud();

      fwd_rk3(n,eps);
      dev0=max_dev_ud(usv[0]);

      if (my_rank==0)
         printf("Comparison of fwd_rk2() and fwd_rk3():   |dU| = %.1e\n",
                dev0);

      start_ranlux(0,1234);   
      random_ud();
      fwd_rk3(n,eps);
      cm3x3_assign(4*VOLUME,udb,usv[0]);

      cG=1.3456;
      lat=set_lat_parms(0.0,1.0,0.0,0.0,0.0,1.0,cG,1.0);
      error(lat.cG!=cG,1,"main [check1.c]","Parameter cG is not correctly set");
   
      start_ranlux(0,1234);   
      random_ud();
      fwd_rk3(n,eps);
      dev0=max_dev_ud(usv[0]);

      lat=set_lat_parms(0.0,1.0,0.0,0.0,0.0,1.0,1.0,1.0);      
      error(lat.cG!=1.0,1,"main [check1.c]",
            "Parameter cG is not correctly reset");
   
      if (my_rank==0)
      {
         printf("Integration with different values of cG: |dU| = %.1e\n\n",
                dev0);
         fflush(stdout);
      }
   }

   if (my_rank==0)
      fclose(flog);      
   
   MPI_Finalize();    
   exit(0);
}
