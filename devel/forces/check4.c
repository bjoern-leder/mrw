
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2005, 2008, 2009, 2010,    Martin Luescher, Filippo Palombi
*               2011, 2012, 2012           Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of sw_frc() and hop_frc()
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
#include "mdflds.h"
#include "sflds.h"
#include "linalg.h"
#include "sw_term.h"
#include "dirac.h"
#include "forces.h"
#include "global.h"

#define MAX_LEVELS 8
#define BLK_LENGTH 8

static int cnt[MAX_LEVELS];
static double smx[MAX_LEVELS];


static int is_Xt_zero(u3_alg_dble *X)
{
   int ie;

   ie=0;
   ie|=((*X).c1!=0.0);
   ie|=((*X).c2!=0.0);
   ie|=((*X).c3!=0.0);
   ie|=((*X).c4!=0.0);
   ie|=((*X).c5!=0.0);
   ie|=((*X).c6!=0.0);
   ie|=((*X).c7!=0.0);   
   ie|=((*X).c8!=0.0);
   ie|=((*X).c9!=0.0);   

   return ie;
}


static int is_Xv_zero(su3_dble *X)
{
   int ie;

   ie=0;
   ie|=((*X).c11.re!=0.0);
   ie|=((*X).c11.im!=0.0);
   ie|=((*X).c12.re!=0.0);
   ie|=((*X).c12.im!=0.0);   
   ie|=((*X).c13.re!=0.0);
   ie|=((*X).c13.im!=0.0);

   ie|=((*X).c21.re!=0.0);
   ie|=((*X).c21.im!=0.0);
   ie|=((*X).c22.re!=0.0);
   ie|=((*X).c22.im!=0.0);   
   ie|=((*X).c23.re!=0.0);
   ie|=((*X).c23.im!=0.0);

   ie|=((*X).c31.re!=0.0);
   ie|=((*X).c31.im!=0.0);
   ie|=((*X).c32.re!=0.0);
   ie|=((*X).c32.im!=0.0);   
   ie|=((*X).c33.re!=0.0);
   ie|=((*X).c33.im!=0.0);   

   return ie;
}


static void check_Xtbnd(void)
{
   int n,ie;
   int npt,*pt,*ptm;
   u3_alg_dble **xt;

   xt=xtensor();   
   pt=bnd_pts(&npt);
   ptm=pt+npt;
   ie=0;

   for (;pt<ptm;pt+=1)
   {
      for (n=0;n<6;n++)
         ie|=is_Xt_zero(xt[n]+pt[0]);
   }

   error(ie!=0,1,"check_Xtbnd [check4.c]",
         "X tensor field does not vanish at time 0 and N0-1");
}


static void check_Xvbnd(void)
{
   int ie;
   int npt,*pt,*ptm;
   su3_dble *xv,*u,*um;

   xv=xvector();
   pt=bnd_pts(&npt);
   ptm=pt+npt;
   pt+=npt/2;
   ie=0;

   for (;pt<ptm;pt++)
   {
      u=xv+8*(pt[0]-(VOLUME/2));
      um=u+8;

      for (;u<um;u++)
         ie|=is_Xv_zero(u);
   }

   error(ie!=0,1,"check_Xvbnd [check4.c]",
         "X vector field does not vanish at time 0 and N0-1");
}


static void rot_ud(double t)
{
   su3_dble *u,*um;
   su3_alg_dble *mom;
   mdflds_t *mdfs;

   u=udfld();
   um=u+4*VOLUME;   

   mdfs=mdflds();
   mom=(*mdfs).mom;
   
   for (;u<um;u++)
   {
      expXsu3(t,mom,u);
      mom+=1;
   }

   set_flags(UPDATED_UD);
}


static double action(int k,spinor_dble **phi)
{
   int l;
   spinor_dble **wsd;
   double act;
   
   wsd=reserve_wsd(2);
   sw_term(NO_PTS);
   assign_sd2sd(VOLUME,phi[0],wsd[0]);

   for (l=0;l<k;l++)
   {
     Dw_dble(0.0,wsd[0],wsd[1]);
     mulg5_dble(VOLUME,wsd[1]);
     scale_dble(VOLUME,0.125,wsd[1]);
     assign_sd2sd(VOLUME,wsd[1],wsd[0]);
   }
   
   act=spinor_prod_re_dble(VOLUME,0,phi[0],wsd[0]);
   release_wsd();
   
   return act;
}


static double dSdt(int k,spinor_dble **phi)
{
   int l;
   spinor_dble **wsd;
   mdflds_t *mdfs;

   wsd=reserve_wsd(k);
   sw_term(NO_PTS);   
   assign_sd2sd(VOLUME,phi[0],wsd[0]);

   for (l=1;l<k;l++)
   {
      Dw_dble(0.0,wsd[l-1],wsd[l]);
      mulg5_dble(VOLUME,wsd[l]);
      scale_dble(VOLUME,0.125,wsd[l]);
   }
   
   set_frc2zero();
   set_xt2zero();
   set_xv2zero();

   for (l=0;l<k;l++)
   {
      add_prod2xt(-0.0625,wsd[l],wsd[k-l-1]);
      add_prod2xv(-0.0625,wsd[l],wsd[k-l-1]);
   }

   check_Xtbnd();
   check_Xvbnd();
   
   sw_frc(1.0);
   hop_frc(1.0);
   release_wsd();

   mdfs=mdflds();

   return scalar_prod_alg(4*VOLUME,0,(*mdfs).mom,(*mdfs).frc);
}   


static double action_det(ptset_t set)
{
   int ie,io;
   int vol,ofs,ix,im,t,n;
   double c,p;
   complex_dble z;
   pauli_dble *m;
   sw_parms_t swp;

   if (set==NO_PTS)
      return 0.0;
      
   swp=sw_parms();

   if ((4.0+swp.m0)>1.0)
      c=pow(4.0+swp.m0,-6.0);
   else
      c=1.0;

   for (n=0;n<MAX_LEVELS;n++)
   {
      cnt[n]=0;
      smx[n]=0.0;
   }

   if (query_flags(SWD_UP2DATE)!=1)
      sw_term(NO_PTS);
   else
   {
      ie=query_flags(SWD_E_INVERTED);
      io=query_flags(SWD_O_INVERTED);

      if (((ie==1)&&((set==ALL_PTS)||(set==EVEN_PTS)))||
          ((io==1)&&((set==ALL_PTS)||(set==ODD_PTS))))
         sw_term(NO_PTS);
   }
   
   if (set==ODD_PTS)
      ofs=(VOLUME/2);
   else
      ofs=0;

   if (set==EVEN_PTS)
      vol=(VOLUME/2);
   else
      vol=VOLUME;

   ix=ofs;
   m=swdfld()+2*ofs;
   
   while (ix<vol)
   {
      im=ix+BLK_LENGTH;
      if (im>vol)
         im=vol;
      p=1.0;

      for (;ix<im;ix++)
      {
         t=global_time(ix);

         if ((t>0)&&(t<(NPROC0*L0-1)))
         {
            z=det_pauli_dble(0.0,m);
            p*=(c*z.re);
            z=det_pauli_dble(0.0,m+1);
            p*=(c*z.re);            
         }

         m+=2;
      }

      cnt[0]+=1;
      smx[0]-=log(fabs(p));

      for (n=1;(cnt[n-1]>=BLK_LENGTH)&&(n<MAX_LEVELS);n++)
      {
         cnt[n]+=1;
         smx[n]+=smx[n-1];

         cnt[n-1]=0;
         smx[n-1]=0.0;
      }
   }

   for (n=1;n<MAX_LEVELS;n++)
      smx[0]+=smx[n];

   return 2.0*smx[0];
}


static double dSdt_det(ptset_t set)
{
   int ifail;
   mdflds_t *mdfs;

   set_xt2zero();
   ifail=add_det2xt(2.0,set);   
   error_root(ifail!=0,1,"dSdt_det [check4.c]",
              "Inversion of the SW term was not safe");
   check_Xtbnd();

   set_frc2zero();   
   sw_frc(1.0);
   mdfs=mdflds();

   return scalar_prod_alg(4*VOLUME,0,(*mdfs).mom,(*mdfs).frc);
}   


int main(int argc,char *argv[])
{
   int my_rank,k;
   double eps,act0,act1,dsdt;
   double dev_frc,sig_loss,s[2],r[2];
   spinor_dble **phi;
   ptset_t set;
   sw_parms_t swp;
   FILE *flog=NULL;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);
      
      printf("\n");
      printf("Check of sw_frc() and hop_frc()\n");
      printf("-------------------------------\n\n");
      
      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }
   
   start_ranlux(0,1245);
   geometry();

   set_lat_parms(6.0,1.0,0.0,0.0,0.0,1.234,1.0,1.34);
   swp=set_sw_parms(-0.0123);
   
   if (my_rank==0)
      printf("m0 = %.4e, mu = 0.0, csw = %.4e, cF = %.4e\n\n",
             swp.m0,swp.csw,swp.cF);

   alloc_wsd(6);
   phi=reserve_wsd(1);

   for (k=1;k<=4;k++)
   {
      random_ud();
      random_mom();
      random_sd(VOLUME,phi[0],1.0);
      bnd_sd2zero(ALL_PTS,phi[0]);
      dsdt=dSdt(k,phi);
      
      eps=5.0e-5;   
      rot_ud(eps);
      act0=2.0*action(k,phi)/3.0;
      rot_ud(-eps);
      
      rot_ud(-eps);
      act1=2.0*action(k,phi)/3.0;
      rot_ud(eps);
      
      rot_ud(2.0*eps);
      act0-=action(k,phi)/12.0;
      rot_ud(-2.0*eps);
      
      rot_ud(-2.0*eps);
      act1-=action(k,phi)/12.0;
      rot_ud(2.0*eps);
      
      s[0]=dsdt-(act0-act1)/eps;
      s[1]=dsdt;

      if (NPROC>1)
      {
         MPI_Reduce(s,r,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
         MPI_Bcast(r,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      }      
      else
      {
         r[0]=s[0];
         r[1]=s[1];
      }

      dev_frc=fabs(r[0]/r[1]);
      sig_loss=-log10(fabs(1.0-act0/act1));      

      error_chk();
      
      if (my_rank==0)
      {
         printf("Calculation of the force for S=(phi,Q^%d*phi):\n",k);
         printf("Relative deviation of dS/dt = %.2e ",dev_frc);
         printf("[significance loss = %d digits]\n\n",(int)(sig_loss));
      }
   }

   if (my_rank==0)
      printf("Calculation of the force for S=-2*Tr{ln(SW term)}:\n");
   
   for (k=0;k<4;k++)
   {
      if (k==0)
         set=NO_PTS;
      else if (k==1)
         set=EVEN_PTS;
      else if (k==2)
         set=ODD_PTS;
      else
         set=ALL_PTS;
      
      random_ud();
      random_mom();
      dsdt=dSdt_det(set);
      
      eps=5.0e-4;   
      rot_ud(eps);
      act0=2.0*action_det(set)/3.0;
      rot_ud(-eps);
      
      rot_ud(-eps);
      act1=2.0*action_det(set)/3.0;
      rot_ud(eps);
      
      rot_ud(2.0*eps);
      act0-=action_det(set)/12.0;
      rot_ud(-2.0*eps);
      
      rot_ud(-2.0*eps);
      act1-=action_det(set)/12.0;
      rot_ud(2.0*eps);

      s[0]=dsdt-(act0-act1)/eps;
      s[1]=dsdt;

      if (NPROC>1)
      {
         MPI_Reduce(s,r,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
         MPI_Bcast(r,2,MPI_DOUBLE,0,MPI_COMM_WORLD);
      }      
      else
      {
         r[0]=s[0];
         r[1]=s[1];
      }
   
      if (k>0)
      {
         dev_frc=fabs(r[0]/r[1]);
         sig_loss=-log10(fabs(1.0-act0/act1));
      }
      else
         dev_frc=fabs(r[0]);

      error_chk();
      
      if (my_rank==0)
      {
         if (k==0)
            printf("set=NO_PTS:   ");
         else if (k==1)
            printf("set=EVEN_PTS: ");
         else if (k==2)
            printf("set=ODD_PTS:  ");
         else
            printf("set=ALL_PTS:  ");

         if (k>0)
         {
            printf("relative deviation of dS/dt = %.2e ",dev_frc);
            printf("[significance loss = %d digits]\n",(int)(sig_loss));
         }
         else
            printf("absolute deviation of dS/dt = %.2e\n",dev_frc);
      }
   }
   
   if (my_rank==0)
   {
      printf("\n");
      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
