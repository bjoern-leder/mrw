
/*******************************************************************************
*
* File mrw.c
*
* Copyright (C) 2012, 2013 Martin Luescher, 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Mass and twisted-mass reweighting factors
*
* The externally accessible functions are
*
*   complex_dble mrw1(mrw_masses_t ms,int tm,int isp,double *sqnp,double *sqne,
*                    int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqne, solves the twisted-mass Dirac equation
*     using the mass/twisted-mass parameters in ms, sets sqnp to the square norm
*     of the solution and returns -ln(w^(1)) (see the documentation).
*     The solver is specified by the parameter set number isp.
*     The argument status must be pointing to an array of at least 1,1
*     and 3 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program.
*
*  complex_dble mrw2(mrw_masses_t ms,int tm,int *isp,complex_dble *lnw1,
*                   double *sqnp,double *sqne,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqne, solves the twisted-mass Dirac equation
*     twice using the mass/twisted-mass parameters in ms, sets sqnp[0,1] to the
*     square norm of the solutions, sets lnw1[0,1] to -ln(w^(1)) and returns
*     -ln(w^(2)) (see the documentation).
*     The solvers for the two solves are specified by the parameter set numbers
*     isp[0,1].
*     The argument status must be pointing to an array of at least 2,2
*     and 6 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program for first (first
*     half of the array) and second solve (second half).
*
* double mrw3(mrw_masses_t ms,int *isp,complex_dble *lnw1,double *sqnp,
*             double *sqne,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqne, solves the twisted-mass Dirac equation
*     tree times using the mass/twisted-mass parameters in ms, sets sqnp[0,1] to the
*     square norm of the first two solutions, sets lnw1[0,1] to -ln(w^(1)) and returns
*     -ln(w^(4,tm)) (see the documentation).
*     The solvers for the two solves are specified by the parameter set numbers
*     isp[0,1].
*     The argument status must be pointing to an array of at least 4,4
*     and 9 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program for first (first
*     third of the array), second solve (second third) and third solve
*     (last third).
*     If ms.d1==-ms.d2 and ms.m1==ms.m2 the second solve is skipped and
*     sqnp[1], lnw1[1] and status[3-5] are set to zero.
* 
* 
* Notes:
* 
* See doc/mrw.pdf for more details.
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define MRW_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "sflds.h"
#include "linalg.h"
#include "sap.h"
#include "dfl.h"
#include "forces.h"
#include "mrw.h"
#include "global.h"
#include "dirac.h"


static double set_eta(spinor_dble *eta)
{
   random_sd(VOLUME,eta,1.0);
   bnd_sd2zero(ALL_PTS,eta);
   
   return norm_square_dble(VOLUME,1,eta);
}


static void get_psi(double m,double mu,int ihc,spinor_dble *eta,
                    spinor_dble *psi,int isp, int* stat)
{
   spinor_dble **wsd;
   solver_parms_t sp;
   sap_parms_t sap;

   stat[0]=0;
   stat[1]=0;
   stat[2]=0;
      
   sp=solver_parms(isp);

   set_sw_parms(m);

   if (sp.solver==CGNE)
   {
      wsd=reserve_wsd(1);
      if (ihc==0)
      {
         mu*=-1.0;
         mulg5_dble(VOLUME,eta);
      }
   }
   else
   {   
      if (ihc==1)
      {
         mu*=-1.0;
         mulg5_dble(VOLUME,eta);
      }
   }
      
   if (sp.solver==CGNE)
   {
      tmcg(sp.nmx,sp.res,mu,eta,wsd[0],stat);
      error_root(stat[0]<0,1,"get_psi [mrw.c]",
               "CGNE solver failed (mu = %.4e, parameter set no %d, "
               "status = %d)",mu,isp,stat[0]);     
      Dw_dble(mu,wsd[0],psi);      
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      sap_gcr(sp.nkv,sp.nmx,sp.res,mu,eta,psi,stat);
      error_root(stat[0]<0,1,"get_psi [mrw.c]",
               "SAP_GCR solver failed (mu = %.4e, parameter set no %d, "
               "status = %d)",mu,isp,stat[0]);      
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,mu,eta,psi,stat);      
      error_root((stat[0]<0)||(stat[1]<0),1,
               "get_psi [mrw.c]","DFL_SAP_GCR solver failed "
               "(mu = %.4e, parameter set no %d, status = (%d,%d,%d))",
               mu,isp,stat[0],stat[1],stat[2]);
   }
   else
      error_root(1,1,"get_psi [mrw.c]","Unknown solver");

   if (sp.solver==CGNE)
   {
      release_wsd();
      if (ihc==0)
      {
         mulg5_dble(VOLUME,psi);
         mulg5_dble(VOLUME,eta);
      }
   }
   else
   {   
      if (ihc==1)
      {
         mulg5_dble(VOLUME,psi);
         mulg5_dble(VOLUME,eta);
      }
   }
}


complex_dble mrw1(mrw_masses_t ms,int tm,int isp,double *sqnp,double *sqne,int *status)
{
   complex_dble lnw,z;
   spinor_dble *eta,*psi1,**wsd;

   lnw.re=0.0;
   lnw.im=0.0;
   (*sqne)=0.0;
   (*sqnp)=0.0;
   status[0]=0;
   status[1]=0;
   status[2]=0;
   
   if (ms.d1==0.0)
      return lnw;
   
   wsd=reserve_wsd(2);
   
   eta=wsd[0];
   psi1=wsd[1];
   (*sqne)=set_eta(eta);

   get_psi(ms.m1,ms.mu1,1,eta,psi1,isp,status);
   
   if (tm)
   {
      z=spinor_prod5_dble(VOLUME,1,psi1,eta);
      lnw.re=-ms.d1*z.im;
      lnw.im=ms.d1*z.re;      
   }
   else
   {
      z=spinor_prod_dble(VOLUME,1,psi1,eta);
      lnw.re=ms.d1*z.re;
      lnw.im=ms.d1*z.im;
   }
      
   (*sqnp)=norm_square_dble(VOLUME,1,psi1);
   
   release_wsd();

   return lnw;
}


complex_dble mrw2(mrw_masses_t ms,int tm,int *isp,complex_dble *lnw1,
                  double *sqnp,double *sqne,int *status)
{
   complex_dble lnw,z;
   spinor_dble *eta,*psi1,*psi2,**wsd;

   lnw.re=0.0;
   lnw.im=0.0;
   (*sqne)=0.0;
   
   if ((ms.d1==0.0) && (ms.d2==0.0))
      return lnw;
   
   wsd=reserve_wsd(3);
   
   psi1=wsd[0];
   psi2=wsd[1];
   eta=wsd[2];
   (*sqne)=set_eta(eta);

   get_psi(ms.m1,ms.mu1,1,eta,psi1,isp[0],status);
   get_psi(ms.m2,ms.mu2,tm,eta,psi2,isp[1],status+3);

   if (tm)
   {
      z=spinor_prod5_dble(VOLUME,1,psi1,eta);
      lnw1[0].re=-ms.d1*z.im;
      lnw1[0].im=ms.d1*z.re;
      z=spinor_prod5_dble(VOLUME,1,psi2,eta);
      lnw1[1].re=-ms.d2*z.im;
      lnw1[1].im=ms.d2*z.re;
   }
   else
   {
      z=spinor_prod_dble(VOLUME,1,psi1,eta);
      lnw1[0].re=ms.d1*z.re;
      lnw1[0].im=ms.d1*z.im;
      z=spinor_prod_dble(VOLUME,1,psi2,eta);
      lnw1[1].re=ms.d2*z.re;
      lnw1[1].im=ms.d2*z.im;
   }

   z=spinor_prod_dble(VOLUME,1,psi1,psi2);
   lnw.re=lnw1[0].re+lnw1[1].re+ms.d1*ms.d2*z.re;
   lnw.im=lnw1[0].im-lnw1[1].im+ms.d1*ms.d2*z.im;

   if (tm==0)
      lnw1[1].im*=-1.0;
   
   sqnp[0]=norm_square_dble(VOLUME,1,psi1);
   sqnp[1]=norm_square_dble(VOLUME,1,psi2);

   release_wsd();

   return lnw;
}


double mrw3(mrw_masses_t ms,int *isp,complex_dble *lnw1,
                  double *sqnp,double *sqne,int *status)
{
   double d1,d2,lnw;
   complex_dble z;
   spinor_dble *eta,*psi1,*psi2,**wsd;

   lnw=0.0;
   (*sqne)=0.0;
   
   if ((ms.d1==0.0) && (ms.d2==0.0))
      return lnw;
   
   d1=-ms.mu1+sqrt(ms.mu1*ms.mu1+ms.d1);
   d2=-ms.mu2+sqrt(ms.mu2*ms.mu2+ms.d2);;

   wsd=reserve_wsd(3);
   
   psi1=wsd[0];
   psi2=wsd[1];
   eta=wsd[2];
   (*sqne)=set_eta(eta);

   get_psi(ms.m1,ms.mu1,1,eta,psi1,isp[0],status);

   z=spinor_prod5_dble(VOLUME,1,psi1,eta);
   lnw1[0].re=-d1*z.im;
   lnw1[0].im=d1*z.re;
   sqnp[0]=norm_square_dble(VOLUME,1,psi1);
   
   if ((ms.d2==-ms.d1)&&(ms.m1==ms.m2))
   {
      lnw1[1].re=0.0;
      lnw1[1].im=0.0;
      sqnp[1]=0.0;
      status[3]=0;
      status[4]=0;
      status[5]=0;
   }
   else
   {
      get_psi(ms.m2,ms.mu2,1,eta,psi2,isp[1],status+3);

      z=spinor_prod5_dble(VOLUME,1,psi2,eta);
      lnw1[1].re=-d2*z.im;
      lnw1[1].im=d2*z.re;
      sqnp[1]=norm_square_dble(VOLUME,1,psi2);
   
      if (ms.d2==-ms.d1)
      {
         assign_sd2sd(VOLUME,psi1,eta);
         mulr_spinor_add_dble(VOLUME,eta,psi2,-1.0);
         mulr_spinor_add_dble(VOLUME,psi2,psi1,1.0);
         z=spinor_prod_dble(VOLUME,1,eta,psi2);
      }
   }

   get_psi(ms.m2,ms.mu2,0,psi1,psi2,isp[1],status+6);

   lnw=norm_square_dble(VOLUME,1,psi2);
   
   if ((ms.d2==-ms.d1)&&(ms.m1==ms.m2))
      lnw*=(ms.d1*(ms.mu2*ms.mu2-ms.mu1*ms.mu1)-ms.d1*ms.d1);
   else
   {      
      if (ms.d2==-ms.d1)
         lnw=ms.d1*(z.re-ms.d1*lnw);
      else
         lnw=ms.d1*sqnp[0]+ms.d2*sqnp[1]+ms.d1*ms.d2*lnw;
   }

   release_wsd();

   return lnw;
}
