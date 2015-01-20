
/*******************************************************************************
*
* File uflds.c
*
* Copyright (C) 2006, 2010, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation and initialization of the global gauge fields
*
* The externally accessible functions are
*
*   su3 *ufld(void)
*     Returns the base address of the single-precision gauge field. If it
*     is not already allocated, the field is allocated and initialized to
*     unity except for the time-like link variables at time NPROC0*L0-1,
*     which are set to zero.
*
*   su3_dble *udfld(void)
*     Returns the base address of the double-precision gauge field. If it
*     is not already allocated, the field is allocated and initialized to
*     unity except for the time-like link variables at time NPROC0*L0-1,
*     which are set to zero.
*
*   void random_u(void)
*     Initializes the single-precision gauge field to uniformly distributed
*     random SU(3) matrices. Open or Schroedinger functional boundary
*     conditions are then imposed depending on what is specified in the
*     parameter data base.
*
*   void random_ud(void)
*     Initializes the double-precision gauge field to uniformly distributed
*     random SU(3) matrices. Open or Schroedinger functional boundary
*     conditions are then imposed depending on what is specified in the
*     parameter data base.
*
*   void renormalize_ud(void)
*     Projects the double-precision gauge field back to SU(3). Only the
*     active link variables are projected.
*
*   void assign_ud2u(void)
*     Assigns the double-precision gauge field to the single-precision
*     gauge field.
*
* Notes:
*
* All these programs act globally and must be called from all processes
* simultaneously. 
*
*******************************************************************************/

#define UFLDS_C

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
#include "global.h"

#define N0 (NPROC0*L0)

static const su3 u0={{0.0f}};
static const su3_dble ud0={{0.0}};

static su3 *ub=NULL;
static su3_dble *udb=NULL;


static void alloc_u(void)
{
   size_t n;
   su3 unity,*u,*um;

   error_root(sizeof(su3)!=(18*sizeof(float)),1,"alloc_u [uflds.c]",
              "The su3 structures are not properly packed");

   n=4*VOLUME;
   ub=amalloc(n*sizeof(*ub),ALIGN);
   error(ub==NULL,1,"alloc_u [uflds.c]",
         "Could not allocate memory space for the gauge field");

   unity=u0;
   unity.c11.re=1.0f;
   unity.c22.re=1.0f;
   unity.c33.re=1.0f;
   u=ub;
   um=ub+n;

   for (;u<um;u++)
      (*u)=unity;

   openbc();
}


su3 *ufld(void)
{
   if (ub==NULL)
      alloc_u();

   return ub;
}


static void alloc_ud(void)
{
   size_t n;
   su3_dble unity,*u,*um;

   error_root(sizeof(su3_dble)!=(18*sizeof(double)),1,"alloc_ud [uflds.c]",
              "The su3_dble structures are not properly packed");

   n=4*VOLUME+7*(BNDRY/4);
   udb=amalloc(n*sizeof(*udb),ALIGN);
   error(udb==NULL,1,"alloc_ud [uflds.c]",
         "Could not allocate memory space for the gauge field");

   unity=ud0;
   unity.c11.re=1.0;
   unity.c22.re=1.0;
   unity.c33.re=1.0;
   u=udb;
   um=udb+n;

   for (;u<um;u++)
      (*u)=unity;

   openbcd();
}


su3_dble *udfld(void)
{
   if (udb==NULL)
      alloc_ud();

   return udb;
}


void random_u(void)
{
   su3 *u,*um;

   u=ufld();
   um=u+4*VOLUME;

   for (;u<um;u++)
      random_su3(u);

   if (sf_flg()==0)
      openbc();
   else
      sfbc();
}


void random_ud(void)
{
   su3_dble *u,*um;
   
   u=udfld();
   um=u+4*VOLUME;

   for (;u<um;u++)
      random_su3_dble(u);

   if (sf_flg()==0)
      openbcd();
   else
      sfbcd();   
}


void renormalize_ud(void)
{
   int sf,ix,t,k;
   su3_dble *u;

   error(udb==NULL,1,"renormalize_ud [uflds.c]",
         "Attempt to access unallocated memory space");

   sf=sf_flg();
   u=udfld();

   for (ix=(VOLUME/2);ix<VOLUME;ix++)
   {
      t=global_time(ix);

      if (t==0)
      {
         project_to_su3_dble(u);

         if (sf==0)
         {
            for (k=2;k<8;k++)
               project_to_su3_dble(u+k);
         }
      }
      else if (t==(N0-1))
      {
         project_to_su3_dble(u+1);

         if (sf==0)
         {
            for (k=2;k<8;k++)
               project_to_su3_dble(u+k);
         }
      }
      else
      {
         for (k=0;k<8;k++)
            project_to_su3_dble(u+k);
      }

      u+=8;
   }

   set_flags(UPDATED_UD);
}


void assign_ud2u(void)
{
   su3 *u,*um;
   su3_dble *ud;
   
   error(udb==NULL,1,"assign_ud2u [uflds.c]",
         "Attempt to access unallocated memory space");

   u=ufld();
   um=u+4*VOLUME;
   ud=udfld();

   for (;u<um;u++)
   {   
      (*u).c11.re=(float)((*ud).c11.re);
      (*u).c11.im=(float)((*ud).c11.im);
      (*u).c12.re=(float)((*ud).c12.re);
      (*u).c12.im=(float)((*ud).c12.im);
      (*u).c13.re=(float)((*ud).c13.re);
      (*u).c13.im=(float)((*ud).c13.im);

      (*u).c21.re=(float)((*ud).c21.re);
      (*u).c21.im=(float)((*ud).c21.im);
      (*u).c22.re=(float)((*ud).c22.re);
      (*u).c22.im=(float)((*ud).c22.im);
      (*u).c23.re=(float)((*ud).c23.re);
      (*u).c23.im=(float)((*ud).c23.im);

      (*u).c31.re=(float)((*ud).c31.re);
      (*u).c31.im=(float)((*ud).c31.im);
      (*u).c32.re=(float)((*ud).c32.re);
      (*u).c32.im=(float)((*ud).c32.im);
      (*u).c33.re=(float)((*ud).c33.re);
      (*u).c33.im=(float)((*ud).c33.im);
      
      ud+=1;
   }

   set_flags(ASSIGNED_UD2U);
}
