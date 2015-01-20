
/*******************************************************************************
*
* File rwtmeo.c
*
* Copyright (C) 2012, 2013 Martin Luescher, Stefan Schaefer
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Twisted-mass reweighting factors (even-odd preconditioned version)
*
* The externally accessible functions are
*
*   double rwtm1eo(double mu,int isp,double *sqn,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqn and returns -ln(r1) (see the notes).
*     The twisted-mass Dirac equation is solved using the solver specified
*     by the parameter set number isp.
*      The argument status must be pointing to an array of at least 1,1
*     and 3 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program.
*
*   double rwtm2eo(double mu,int isp,double *sqn,int *status)
*     Generates a random pseudo-fermion field with normal distribution,
*     assigns its square norm to sqn and returns -ln(r2) (see the notes).
*     The twisted-mass Dirac equation is solved using the solver specified
*     by the parameter set number isp.
*      The argument status must be pointing to an array of at least 2,2
*     and 6 elements, respectively, in the case of the CGNE, SAP_GCR and
*     DFL_SAP_GCR solver. On exit the array elements return the status
*     values reported by the solver program for twisted mass 0 (first
*     half of the array) and twisted mass sqrt(2)*mu (second half).
*
* Notes:
*
* Twisted-mass reweighting of the quark determinant was introduced by
*
*  M. Luescher, F. Palombi: "Fluctuations and reweighting of the quark
*  determinant on large lattices", PoS LATTICE2008 (2008) 049
*
* The stochastic reweighting factors computed here coincide with the ones
* defined in this paper, except for the fact that the Wilson-Dirac operator
* is replaced by the even-odd preconditioned operator.
*
* For a given random pseudo-fermion field eta with distribution proportional
* to exp{-(eta,eta)}, the factors r1 and r2 are given by
*
*  r1=exp{-mu^2*(eta,(Dwhat^dag*Dwhat)^(-1)*eta)},
*
*  r2=exp{-mu^4*(eta,(Dwhat^dag*Dwhat*(Dwhat^dag*Dwhat+2*mu^2))^(-1)*eta)},
*
* where Dwhat denotes the even-odd preconditioned, massive O(a)-improved
* Wilson-Dirac operator. Note that the pseudo-fermion field vanishes on
* the odd sites of the lattice. The bare quark mass is taken to be the one
* last set by sw_parms() [flags/parms.c] and it is assumed that the chosen 
* solver parameters have been set by set_solver_parms() [flags/sparms.c].
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define RWTMEO_C

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
#include "update.h"
#include "global.h"


static double set_eta(spinor_dble *eta)
{
   random_sd(VOLUME/2,eta,1.0);
   set_sd2zero(VOLUME/2,eta+(VOLUME/2));
   bnd_sd2zero(EVEN_PTS,eta);
   
   return norm_square_dble(VOLUME/2,1,eta);
}


double rwtm1eo(double mu,int isp,double *sqn,int *status)
{
   double lnr;
   spinor_dble *eta,*phi,**wsd;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);

   wsd=reserve_wsd(2);
   eta=wsd[0];
   phi=wsd[1];
   (*sqn)=set_eta(eta);
   sp=solver_parms(isp);   

   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.res,0.0,eta,phi,status);

      error_root(status[0]<0,1,"rwtm1eo [rwtmeo.c]",
                 "CGNE solver failed (mu = 0.0, parameter set no %d, "
                 "status = %d)",isp,status[0]);

      lnr=spinor_prod_re_dble(VOLUME/2,1,eta,phi);
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
      mulg5_dble(VOLUME/2,eta);
      sap_gcr(sp.nkv,sp.nmx,sp.res,0.0,eta,phi,status);
      
      error_root(status[0]<0,1,"rwtm1eo [rwtmeo.c]",
                 "SAP_GCR solver failed (mu = 0.0, parameter set no %d, "
                 "status = %d)",isp,status[0]);      

      lnr=norm_square_dble(VOLUME/2,1,phi);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
      mulg5_dble(VOLUME/2,eta);
      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,0.0,eta,phi,status);
      
      error_root((status[0]<0)||(status[1]<0),1,
                 "rwtm1eo [rwtmeo.c]","DFL_SAP_GCR solver failed "
                 "(mu = 0.0, parameter set no %d, status = (%d,%d,%d))",
                 isp,status[0],status[1],status[2]);
      
      lnr=norm_square_dble(VOLUME/2,1,phi);
   }
   else
   {
      lnr=0.0;
      error_root(1,1,"rwtm1eo [rwtmeo.c]","Unknown solver");
   }
   
   release_wsd();

   return mu*mu*lnr;
}


double rwtm2eo(double mu,int isp,double *sqn,int *status)
{
   double lnr;
   spinor_dble *eta,*phi,**wsd;
   solver_parms_t sp;
   sap_parms_t sap;
   tm_parms_t tm;

   tm=tm_parms();
   if (tm.eoflg!=1)
      set_tm_parms(1);
   
   wsd=reserve_wsd(2);
   eta=wsd[0];
   phi=wsd[1];
   (*sqn)=set_eta(eta);
   sp=solver_parms(isp);   

   if (sp.solver==CGNE)
   {
      tmcgeo(sp.nmx,sp.res,0.0,eta,phi,status);

      error_root(status[0]<0,1,"rwtm2eo [rwtmeo.c]",
                 "CGNE solver failed (mu =0.0, parameter set no %d, "
                 "status = %d)",isp,status[0]);

      tmcgeo(sp.nmx,sp.res,sqrt(2.0)*mu,eta,eta,status+1);

      error_root(status[1]<0,1,"rwtm2eo [rwtmeo.c]",
                 "CGNE solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",sqrt(2.0)*mu,isp,status[1]);      

      lnr=spinor_prod_re_dble(VOLUME/2,1,eta,phi);
   }
   else if (sp.solver==SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);
      mulg5_dble(VOLUME/2,eta);      
      sap_gcr(sp.nkv,sp.nmx,sp.res,0.0,eta,phi,status);
      
      error_root(status[0]<0,1,"rwtm2eo [rwtmeo.c]",
                 "SAP_GCR solver failed (mu = 0.0, parameter set no %d, "
                 "status = %d)",isp,status[0]);      

      mulg5_dble(VOLUME/2,phi);
      set_sd2zero(VOLUME/2,phi+(VOLUME/2));
      sap_gcr(sp.nkv,sp.nmx,sp.res,sqrt(2.0)*mu,phi,eta,status+1);
      
      error_root(status[1]<0,2,"rwtm2eo [rwtmeo.c]",
                 "SAP_GCR solver failed (mu = %.4e, parameter set no %d, "
                 "status = %d)",sqrt(2.0)*mu,isp,status[1]);
      
      lnr=norm_square_dble(VOLUME/2,1,eta);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      sap=sap_parms();
      set_sap_parms(sap.bs,sp.isolv,sp.nmr,sp.ncy);

      mulg5_dble(VOLUME/2,eta);
      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,0.0,eta,phi,status);
      
      error_root((status[0]<0)||(status[1]<0),1,
                 "rwtm2eo [rwtmeo.c]","DFL_SAP_GCR solver failed "
                 "(mu = 0.0, parameter set no %d, status = (%d,%d,%d))",
                 isp,status[0],status[1],status[2]);

      mulg5_dble(VOLUME/2,phi);
      set_sd2zero(VOLUME/2,phi+(VOLUME/2));

      dfl_sap_gcr2(sp.nkv,sp.nmx,sp.res,sqrt(2.0)*mu,phi,eta,status+3);
      
      error_root((status[3]<0)||(status[4]<0),2,
                 "rwtm2eo [rwtmeo.c]","DFL_SAP_GCR solver failed "
                 "(mu = %.4e, parameter set no %d, status = (%d,%d,%d)",
                 sqrt(2.0)*mu,isp,status[3],status[4],status[5]);
      
      lnr=norm_square_dble(VOLUME/2,1,eta);
   }
   else
   {
      lnr=0.0;
      error_root(1,1,"rwtm2eo [rwtmeo.c]","Unknown solver");
   }

   release_wsd();
   mu=mu*mu;

   return mu*mu*lnr;
}
