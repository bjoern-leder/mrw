
/*******************************************************************************
*
* File bcnds.c
*
* Copyright (C) 2005, 2010, 2011, 2012 Martin Luescher, John Bulava
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs related to the boundary conditions in the time direction
*
*   void openbc(void)
*     Sets the time-like single-precision link variables at time
*     NPROC0*L0-1 to zero.
*     
*   void openbcd(void)
*     Sets the time-like double-precision link variables at time
*     NPROC0*L0-1 to zero.
*
*   void sfbc(void)
*     Sets the time-like single-precision link variables at time
*     NPROC0*L0-1 to zero and the spatial link variables there and
*     at time 0 to the values specified in the parameter data base
*     (see flags/sw_parms.c).
*
*   void sfbcd(void)
*     Sets the time-like double-precision link variables at time
*     NPROC0*L0-1 to zero and the spatial link variables there and
*     at time 0 to the values specified in the parameter data base
*     (see flags/sw_parms.c).
*
*   int check_bc(void)
*     Returns 1 if the time-like single-precision link variables at time
*     NPROC0*L0-1 are all equal to zero and 0 otherwise. An error occurs
*     if only some of these link variables vanish.
*
*   int check_bcd(void)
*     Returns 1 if the time-like double-precision link variables at time
*     NPROC0*L0-1 are all equal to zero and 0 otherwise. An error occurs
*     if only some of these link variables vanish.
*
*   int check_sfbc(void)
*     Returns 1 if checkbc() returns 1 and if the single-precision gauge
*     field has the Schroedinger functional boundary values specified in
*     the parameter data base. In all other cases the program returns 0.
*     An error occurs if the boundary values are not set.
*
*   int check_sfbcd(void)
*     Returns 1 if checkbcd() returns 1 and if the double-precision gauge
*     field has the Schroedinger functional boundary values specified in
*     the parameter data base. In all other cases the program returns 0.
*     An error occurs if the boundary values are not set.
*
*   int *bnd_lks(int *n)
*     Returns the starting address of an array of length n whose elements
*     are the integer offsets of the time-like link variables on the local
*     lattice at global time NPROC0*L0-1.
*
*   int *bnd_pts(int *n)
*     Returns the starting address of an array of length n whose elements
*     are the indices of the points on the local lattice at global time 0
*     and NPROC0*L0-1. The ordering of the indices is such that the first
*     and the second n/2 indices are, respectively, those of the even and
*     the odd points at these times.
*
*   void bnd_s2zero(ptset_t set,spinor *s)
*     Sets the components of the single-precision spinor field s on the
*     specified set of points at global time 0 and NPROC0*L0-1 to zero.
*
*   void bnd_sd2zero(ptset_t set,spinor_dble *sd)
*     Sets the components of the double-precision spinor field sd on the
*     specified set of points at global time 0 and NPROC0*L0-1 to zero.
*
* Notes:
*
* These programs act globally and should therefore be called simultaneously
* on all processes. 
*
* Only the basic field variables on the local lattices (those with address
* offsets less than 4*VOLUME) are modified by openbc(), openbcd(), sfbc()
* and sfbcd(). 
*
*******************************************************************************/

#define BCNDS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "uflds.h"
#include "lattice.h"
#include "global.h"

#define N0 (NPROC0*L0)

static int init0=0,nlks=0,*lks=NULL;
static int init1=0,npts=0,*pts=NULL;
static int init2=0;
static const su3 u0={{0.0f}};
static const su3_dble ud0={{0.0}};
static const spinor s0={{{0.0f}}};
static const spinor_dble sd0={{{0.0}}};
static su3 ubnd[6];
static su3_dble udbnd[6];


static void alloc_lks(void)
{
   int ix,t,*lk;

   if (iup[0][0]==0)
      geometry();

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      if (NPROC0>1)
         nlks=(L1*L2*L3)/2;
      else
         nlks=L1*L2*L3;

      lks=amalloc(nlks*sizeof(*lks),3);

      if (lks!=NULL)
      {
         lk=lks;
         
         for (ix=(VOLUME/2);ix<VOLUME;ix++)
         {
            t=global_time(ix);

            if (t==0)
            {
               (*lk)=8*(ix-(VOLUME/2))+1;
               lk+=1;
            }

            if (t==(N0-1))
            {
               (*lk)=8*(ix-(VOLUME/2));
               lk+=1;
            }
         }
      }
   }

   error((nlks>0)&&(lks==NULL),1,"alloc_lks [bcnds.c]",
         "Unable to allocate index array");
   init0=1;
}


static void alloc_pts(void)
{
   int ix,t,*pt;

   if (iup[0][0]==0)
      geometry();

   if ((cpr[0]==0)||(cpr[0]==(NPROC0-1)))
   {
      if (NPROC0>1)
         npts=L1*L2*L3;
      else
         npts=2*L1*L2*L3;

      pts=amalloc(npts*sizeof(*pts),3);

      if (pts!=NULL)
      {
         pt=pts;
         
         for (ix=0;ix<VOLUME;ix++)
         {
            t=global_time(ix);

            if ((t==0)||(t==(N0-1)))
            {
               (*pt)=ix;
               pt+=1;
            }
         }
      }
   }

   error((npts>0)&&(pts==NULL),1,"alloc_pts [bcnds.c]",
         "Unable to allocate index array");
   init1=1;
}


static void set_ubnd(void)
{
   int k;
   double s[3];
   sf_parms_t sf;

   if (init0==0)
      alloc_lks();
   if (init1==0)
      alloc_pts();
   
   sf=sf_parms();
   error_root(sf.flg!=1,1,"set_ubnd [bcnds.c]",
              "SF boundary values are not set");

   s[0]=(double)(NPROC1*L1);
   s[1]=(double)(NPROC2*L2);
   s[2]=(double)(NPROC3*L3);   
   
   for (k=0;k<6;k++)
   {
      ubnd[k]=u0;
      udbnd[k]=ud0;
   }

   for (k=0;k<3;k++)
   {
      udbnd[k].c11.re=cos(sf.phi[0]/s[k]);
      udbnd[k].c11.im=sin(sf.phi[0]/s[k]);
      udbnd[k].c22.re=cos(sf.phi[1]/s[k]);
      udbnd[k].c22.im=sin(sf.phi[1]/s[k]);
      udbnd[k].c33.re=cos(sf.phi[2]/s[k]);
      udbnd[k].c33.im=sin(sf.phi[2]/s[k]);

      udbnd[3+k].c11.re=cos(sf.phi_prime[0]/s[k]);
      udbnd[3+k].c11.im=sin(sf.phi_prime[0]/s[k]);
      udbnd[3+k].c22.re=cos(sf.phi_prime[1]/s[k]);
      udbnd[3+k].c22.im=sin(sf.phi_prime[1]/s[k]);
      udbnd[3+k].c33.re=cos(sf.phi_prime[2]/s[k]);
      udbnd[3+k].c33.im=sin(sf.phi_prime[2]/s[k]);      
   }
   
   for (k=0;k<6;k++)
   {
      ubnd[k].c11.re=(float)(udbnd[k].c11.re);
      ubnd[k].c11.im=(float)(udbnd[k].c11.im);
      ubnd[k].c22.re=(float)(udbnd[k].c22.re);
      ubnd[k].c22.im=(float)(udbnd[k].c22.im);
      ubnd[k].c33.re=(float)(udbnd[k].c33.re);
      ubnd[k].c33.im=(float)(udbnd[k].c33.im);
   }

   init2=1;
}


void openbc(void)
{
   int *lk,*lkm;
   su3 *ub;

   if (init0==0)
      alloc_lks();

   ub=ufld();
   lk=lks;
   lkm=lk+nlks;

   for (;lk<lkm;lk++)
      ub[*lk]=u0;

   set_flags(UPDATED_U);
}


void openbcd(void)
{
   int *lk,*lkm;
   su3_dble *ub;

   if (init0==0)
      alloc_lks();

   ub=udfld();
   lk=lks;
   lkm=lk+nlks;

   for (;lk<lkm;lk++)
      ub[*lk]=ud0;

   set_flags(UPDATED_UD);
}


void sfbc(void)
{
   int *lk,*lkm;
   int t,k,*pt,*ptm;
   su3 *ub,*u;

   if (init2==0)
      set_ubnd();
   
   ub=ufld();
   lk=lks;
   lkm=lk+nlks;

   for (;lk<lkm;lk++)
      ub[*lk]=u0;

   pt=pts+(npts/2);
   ptm=pts+npts;
   
   for (;pt<ptm;pt++)
   {
      t=global_time(pt[0]);
      u=ub+8*(pt[0]-(VOLUME/2));

      if (t==0)
      {
         for (k=0;k<3;k++)
         {
            u[2+2*k]=ubnd[k];
            u[3+2*k]=ubnd[k];
         }
      }
      else if (t==(N0-1))
      {
         for (k=0;k<3;k++)
         {
            u[2+2*k]=ubnd[3+k];
            u[3+2*k]=ubnd[3+k];
         }
      }
   }
   
   set_flags(UPDATED_U);
}


void sfbcd(void)
{
   int *lk,*lkm;
   int t,k,*pt,*ptm;
   su3_dble *ub,*u;

   if (init2==0)
      set_ubnd();

   ub=udfld();
   lk=lks;
   lkm=lk+nlks;

   for (;lk<lkm;lk++)
      ub[*lk]=ud0;

   pt=pts+(npts/2);
   ptm=pts+npts;
   
   for (;pt<ptm;pt++)
   {
      t=global_time(pt[0]);
      u=ub+8*(pt[0]-(VOLUME/2));

      if (t==0)
      {
         for (k=0;k<3;k++)
         {
            u[2+2*k]=udbnd[k];
            u[3+2*k]=udbnd[k];
         }
      }
      else if (t==(N0-1))
      {
         for (k=0;k<3;k++)
         {
            u[2+2*k]=udbnd[3+k];
            u[3+2*k]=udbnd[3+k];
         }
      }
   }

   set_flags(UPDATED_UD);
}


static int is_zero(su3 *u)
{
   int it;

   it =((*u).c11.re==0.0f);
   it&=((*u).c11.im==0.0f);
   it&=((*u).c12.re==0.0f);
   it&=((*u).c12.im==0.0f);
   it&=((*u).c13.re==0.0f);
   it&=((*u).c13.im==0.0f);

   it&=((*u).c21.re==0.0f);
   it&=((*u).c21.im==0.0f);
   it&=((*u).c22.re==0.0f);
   it&=((*u).c22.im==0.0f);
   it&=((*u).c23.re==0.0f);
   it&=((*u).c23.im==0.0f);

   it&=((*u).c31.re==0.0f);
   it&=((*u).c31.im==0.0f);
   it&=((*u).c32.re==0.0f);
   it&=((*u).c32.im==0.0f);
   it&=((*u).c33.re==0.0f);
   it&=((*u).c33.im==0.0f);

   return it;
}


static int is_zero_dble(su3_dble *u)
{
   int it;

   it =((*u).c11.re==0.0);
   it&=((*u).c11.im==0.0);
   it&=((*u).c12.re==0.0);
   it&=((*u).c12.im==0.0);
   it&=((*u).c13.re==0.0);
   it&=((*u).c13.im==0.0);

   it&=((*u).c21.re==0.0);
   it&=((*u).c21.im==0.0);
   it&=((*u).c22.re==0.0);
   it&=((*u).c22.im==0.0);
   it&=((*u).c23.re==0.0);
   it&=((*u).c23.im==0.0);

   it&=((*u).c31.re==0.0);
   it&=((*u).c31.im==0.0);
   it&=((*u).c32.re==0.0);
   it&=((*u).c32.im==0.0);
   it&=((*u).c33.re==0.0);
   it&=((*u).c33.im==0.0);

   return it;
}


int check_bc(void)
{
   int ibc,iba,*lk,*lkm;
   su3 *ub;

   if (init0==0)
      alloc_lks();

   if (nlks>0)
   {
      ub=ufld();         
      lk=lks;
      lkm=lk+nlks;

      ibc=is_zero(ub+(*lk));
      lk+=1;
      
      for (;lk<lkm;lk++)
      {
         if (is_zero(ub+(*lk))!=ibc)
         {
            ibc=-1;
            break;
         }
      }
   }
   else
      ibc=1;

   MPI_Reduce(&ibc,&iba,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Bcast(&iba,1,MPI_INT,0,MPI_COMM_WORLD);

   error_root(iba==-1,1,"check_bc [bcnds.c]",
              "Mixed zero and non-zero boundary link variables");
   
   return iba;
}


int check_bcd(void)
{
   int ibc,iba,*lk,*lkm;
   su3_dble *ub;

   if (init0==0)
      alloc_lks();

   if (nlks>0)
   {
      ub=udfld();      
      lk=lks;
      lkm=lk+nlks;
   
      ibc=is_zero_dble(ub+(*lk));
      lk+=1;

      for (;lk<lkm;lk++)
      {
         if (is_zero_dble(ub+(*lk))!=ibc)
         {
            ibc=-1;
            break;
         }
      }
   }
   else
      ibc=1;

   MPI_Reduce(&ibc,&iba,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Bcast(&iba,1,MPI_INT,0,MPI_COMM_WORLD);

   error_root(iba==-1,1,"check_bcd [bcnds.c]",
              "Mixed zero and non-zero boundary link variables");
   
   return iba;
}


static int cmp_su3(su3 *u,su3 *v)
{
   int it;

   it =((*u).c11.re==(*v).c11.re);
   it&=((*u).c11.im==(*v).c11.im);   
   it&=((*u).c12.re==(*v).c12.re);
   it&=((*u).c12.im==(*v).c12.im);    
   it&=((*u).c13.re==(*v).c13.re);
   it&=((*u).c13.im==(*v).c13.im);   

   it&=((*u).c21.re==(*v).c21.re);
   it&=((*u).c21.im==(*v).c21.im);   
   it&=((*u).c22.re==(*v).c22.re);
   it&=((*u).c22.im==(*v).c22.im);    
   it&=((*u).c23.re==(*v).c23.re);
   it&=((*u).c23.im==(*v).c23.im);

   it&=((*u).c31.re==(*v).c31.re);
   it&=((*u).c31.im==(*v).c31.im);   
   it&=((*u).c32.re==(*v).c32.re);
   it&=((*u).c32.im==(*v).c32.im);    
   it&=((*u).c33.re==(*v).c33.re);
   it&=((*u).c33.im==(*v).c33.im);    

   return it;
}


static int cmp_su3_dble(su3_dble *u,su3_dble *v)
{
   int it;

   it =((*u).c11.re==(*v).c11.re);
   it&=((*u).c11.im==(*v).c11.im);   
   it&=((*u).c12.re==(*v).c12.re);
   it&=((*u).c12.im==(*v).c12.im);    
   it&=((*u).c13.re==(*v).c13.re);
   it&=((*u).c13.im==(*v).c13.im);   

   it&=((*u).c21.re==(*v).c21.re);
   it&=((*u).c21.im==(*v).c21.im);   
   it&=((*u).c22.re==(*v).c22.re);
   it&=((*u).c22.im==(*v).c22.im);    
   it&=((*u).c23.re==(*v).c23.re);
   it&=((*u).c23.im==(*v).c23.im);

   it&=((*u).c31.re==(*v).c31.re);
   it&=((*u).c31.im==(*v).c31.im);   
   it&=((*u).c32.re==(*v).c32.re);
   it&=((*u).c32.im==(*v).c32.im);    
   it&=((*u).c33.re==(*v).c33.re);
   it&=((*u).c33.im==(*v).c33.im);    

   return it;
}


int check_sfbc(void)
{
   int ibc,iba,*pt,*ptm;
   int t,k;
   su3 *ub,*u;

   if (check_bc()==0)
      return 0;
   
   if (init2==0)
      set_ubnd();
   
   if (npts>0)
   {
      ub=ufld();      
      pt=pts+(npts/2);
      ptm=pts+npts;
      ibc=1;
      
      for (;pt<ptm;pt++)
      {
         t=global_time(pt[0]);
         u=ub+8*(pt[0]-(VOLUME/2));

         if (t==0)
         {
            for (k=0;k<3;k++)
            {
               ibc&=cmp_su3(u+2+2*k,ubnd+k);
               ibc&=cmp_su3(u+3+2*k,ubnd+k);
            }
         }
         else if (t==(N0-1))
         {
            for (k=0;k<3;k++)
            {
               ibc&=cmp_su3(u+2+2*k,ubnd+3+k);
               ibc&=cmp_su3(u+3+2*k,ubnd+3+k);
            }
         }
      }
   }
   else
      ibc=1;

   MPI_Reduce(&ibc,&iba,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Bcast(&iba,1,MPI_INT,0,MPI_COMM_WORLD);

   return iba;
}


int check_sfbcd(void)
{
   int ibc,iba,*pt,*ptm;
   int t,k;
   su3_dble *ub,*u;

   if (check_bcd()==0)
      return 0;
   
   if (init2==0)
      set_ubnd();
   
   if (npts>0)
   {
      ub=udfld();      
      pt=pts+(npts/2);
      ptm=pts+npts;
      ibc=1;
      
      for (;pt<ptm;pt++)
      {
         t=global_time(pt[0]);
         u=ub+8*(pt[0]-(VOLUME/2));

         if (t==0)
         {
            for (k=0;k<3;k++)
            {
               ibc&=cmp_su3_dble(u+2+2*k,udbnd+k);
               ibc&=cmp_su3_dble(u+3+2*k,udbnd+k);
            }
         }
         else if (t==(N0-1))
         {
            for (k=0;k<3;k++)
            {
               ibc&=cmp_su3_dble(u+2+2*k,udbnd+3+k);
               ibc&=cmp_su3_dble(u+3+2*k,udbnd+3+k);
            }
         }
      }
   }
   else
      ibc=1;

   MPI_Reduce(&ibc,&iba,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
   MPI_Bcast(&iba,1,MPI_INT,0,MPI_COMM_WORLD);

   return iba;
}


int *bnd_lks(int *n)
{
   if (init0==0)
      alloc_lks();

   (*n)=nlks;
   
   return lks;
}


int *bnd_pts(int *n)
{
   if (init1==0)
      alloc_pts();

   (*n)=npts;
   
   return pts;
}


void bnd_s2zero(ptset_t set,spinor *s)
{
   int *pt,*pm;

   if (init1==0)
      alloc_pts();

   if (npts>0)
   {
      if (set==ALL_PTS)
      {
         pt=pts;
         pm=pts+npts;
      }
      else if (set==EVEN_PTS)
      {
         pt=pts;
         pm=pts+npts/2;
      }
      else if (set==ODD_PTS)
      {
         pt=pts+npts/2;
         pm=pts+npts;
      }
      else
         return;

      for (;pt<pm;pt++)
         s[*pt]=s0;
   }
}


void bnd_sd2zero(ptset_t set,spinor_dble *sd)
{
   int *pt,*pm;

   if (init1==0)
      alloc_pts();

   if (npts>0)
   {
      if (set==ALL_PTS)
      {
         pt=pts;
         pm=pts+npts;
      }
      else if (set==EVEN_PTS)
      {
         pt=pts;
         pm=pts+npts/2;
      }
      else if (set==ODD_PTS)
      {
         pt=pts+npts/2;
         pm=pts+npts;
      }
      else
         return;

      for (;pt<pm;pt++)
         sd[*pt]=sd0;
   }
}
