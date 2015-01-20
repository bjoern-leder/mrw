
/*******************************************************************************
*
* File lat_parms.c
*
* Copyright (C) 2009, 2010, 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Lattice parameters
*
* The externally accessible functions are
*
*   lat_parms_t set_lat_parms(double beta,double c0,
*                             double kappa_u,double kappa_s,double kappa_c,
*                             double csw,double cG,double cF)
*     Sets the basic lattice parameters. The parameters are
*
*       beta           Inverse bare coupling (beta=6/g_0^2).
*
*       c0             Coefficient of the plaquette loops in the gauge
*                      action (see doc/gauge_action.pdf).
*
*       kappa_{u,s,c}  Hopping parameters of the u, s and c sea quarks. The
*                      u and the d quark have the same hopping parameter and
*                      quarks with vanishing hopping parameter are ignored.
*
*       csw            Coefficient of the Sheikholeslami-Wohlert term.
*
*       cG,cF          Coefficients of the gauge and fermion O(a) boundary
*                      counterterms.
*
*     The return value is a structure that contains the lattice parameters
*     and the associated bare quark masses m0u, m0s and m0c.
*
*   lat_parms_t lat_parms(void)
*     Returns the current lattice parameters in a structure that contains
*     the above parameters plus the bare quark masses.
*
*   void print_lat_parms(void)
*     Prints the lattice parameters to stdout on MPI process 0.
*
*   void write_lat_parms(FILE *fdat)
*     Writes the (global) lattices sizes and lattice parameters to the
*     file fdat on MPI process 0.
*
*   void check_lat_parms(FILE *fdat)
*     Compares the (global) lattice sizes and the lattice parameters with
*     the values stored on the file fdat on MPI process 0, assuming the
*     latter were written to the file by the program write_lat_parms().
*
*   double sea_quark_mass(int im0)
*     Returns the bare sea quark mass m0u if im0=0, m0s if im0=1 and m0c 
*     if im0=2. In all other cases DBL_MAX is returned.
*
*   sw_parms_t set_sw_parms(double m0)
*     Sets the parameters of the SW term. The parameter is
*
*       m0            Bare quark mass.
*
*     The return value is a structure that contains the mass m0 and the
*     improvement coefficients csw and cF, the latter being copied from
*     the list of the lattice parameters.
*
*   sw_parms_t sw_parms(void)
*     Returns the parameters currently set for the SW term. The values
*     of the coefficients csw and cF are copied from the lattice parameter
*     list.
*
*   tm_parms_t set_tm_parms(int eoflg)
*     Sets the twisted-mass flag. The parameter is
*
*       eoflg         Twisted-mass flag. If the flag is set (eoflg=1),
*                     the twisted-mass term in the Dirac operator, the
*                     SAP preconditioner and the little Dirac operator
*                     is turned off on all odd lattice sites. 
*
*     The return value is structure that contains the twisted-mass flag.
*
*   tm_parms_t tm_parms(void)
*     Returns a structure containing the twisted-mass flag.
*
* Notes:
*
* To ensure the consistency of the data base, the parameters must be set
* simultaneously on all processes. The types lat_parms_t, ... are defined
* in the file flags.h.
*
*******************************************************************************/

#define LAT_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static lat_parms_t lat={0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,
                        DBL_MAX,DBL_MAX,DBL_MAX};
static sw_parms_t sw={DBL_MAX,1.0,1.0};
static tm_parms_t tm={0};


lat_parms_t set_lat_parms(double beta,double c0,
                          double kappa_u,double kappa_s,double kappa_c,
                          double csw,double cG,double cF)
{
   double dprms[8];

   if (NPROC>1)
   {
      dprms[0]=beta;
      dprms[1]=c0;
      dprms[2]=kappa_u;
      dprms[3]=kappa_s;
      dprms[4]=kappa_c;
      dprms[5]=csw;      
      dprms[6]=cG;
      dprms[7]=cF;

      MPI_Bcast(dprms,8,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error((dprms[0]!=beta)||(dprms[1]!=c0)||
            (dprms[2]!=kappa_u)||(dprms[3]!=kappa_s)||(dprms[4]!=kappa_c)||
            (dprms[5]!=csw)||(dprms[6]!=cG)||(dprms[7]!=cF),1,
            "set_lat_parms [lat_parms.c]","Parameters are not global");
   }

   error_root(c0<=0,1,"set_lat_parms [lat_parms.c]",
              "Parameter c0 must be positive");
   
   if ((csw!=lat.csw)||(cF!=lat.cF))
   {
      set_flags(ERASED_SW);
      set_flags(ERASED_SWD);
      set_grid_flags(SAP_BLOCKS,ERASED_SW);
      set_flags(ERASED_AW);
      set_flags(ERASED_AWHAT);
   }
   
   lat.beta=beta;
   lat.c0=c0;
   lat.c1=0.125*(1.0-c0);
   lat.kappa_u=kappa_u;
   lat.kappa_s=kappa_s;
   lat.kappa_c=kappa_c;   
   lat.csw=csw;
   lat.cG=cG;
   lat.cF=cF;

   if (kappa_u!=0.0)
      lat.m0u=1.0/(2.0*kappa_u)-4.0;
   else
      lat.m0u=DBL_MAX;

   if (kappa_s!=0.0)
      lat.m0s=1.0/(2.0*kappa_s)-4.0;
   else
      lat.m0s=DBL_MAX;   

   if (kappa_c!=0.0)
      lat.m0c=1.0/(2.0*kappa_c)-4.0;
   else
      lat.m0c=DBL_MAX;
   
   return lat;
}


lat_parms_t lat_parms(void)
{
   return lat;
}


void print_lat_parms(void)
{
   int my_rank,n;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      printf("Lattice parameters:\n");
      n=fdigits(lat.beta);
      printf("beta = %.*f\n",IMAX(n,1),lat.beta);
      n=fdigits(lat.c0);
      printf("c0 = %.*f, ",IMAX(n,1),lat.c0);
      n=fdigits(lat.c1);
      printf("c1 = %.*f\n",IMAX(n,1),lat.c1);
      n=fdigits(lat.kappa_u);
      printf("kappa_u = %.*f\n",IMAX(n,6),lat.kappa_u);      
      n=fdigits(lat.kappa_s);
      printf("kappa_s = %.*f\n",IMAX(n,6),lat.kappa_s);
      n=fdigits(lat.kappa_c);
      printf("kappa_c = %.*f\n",IMAX(n,6),lat.kappa_c);
      n=fdigits(lat.csw);
      printf("csw = %.*f\n",IMAX(n,1),lat.csw);
      n=fdigits(lat.cG);
      printf("cG = %.*f\n",IMAX(n,1),lat.cG);      
      n=fdigits(lat.cF);
      printf("cF = %.*f\n\n",IMAX(n,1),lat.cF);      
   }
}


void write_lat_parms(FILE *fdat)
{
   int my_rank,endian;
   int iw;
   stdint_t istd[4];
   double dstd[9];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();
   
   if (my_rank==0)
   {
      istd[0]=(stdint_t)(N0);
      istd[1]=(stdint_t)(N1);
      istd[2]=(stdint_t)(N2);
      istd[3]=(stdint_t)(N3);
      
      dstd[0]=lat.beta;
      dstd[1]=lat.c0;
      dstd[2]=lat.c1;      
      dstd[3]=lat.kappa_u;
      dstd[4]=lat.kappa_s;
      dstd[5]=lat.kappa_c;
      dstd[6]=lat.csw;
      dstd[7]=lat.cG;
      dstd[8]=lat.cF;
            
      if (endian==BIG_ENDIAN)
      {
         bswap_int(4,istd);
         bswap_double(9,dstd);
      }

      iw=fwrite(istd,sizeof(stdint_t),4,fdat);      
      iw+=fwrite(dstd,sizeof(double),9,fdat);
      error_root(iw!=13,1,"write_lat_parms [lat_parms.c]",
                 "Incorrect write count");
   }
}


void check_lat_parms(FILE *fdat)
{
   int my_rank,endian;
   int ir,ie;
   stdint_t istd[4];
   double dstd[9];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();
   
   if (my_rank==0)
   {
      ir=fread(istd,sizeof(stdint_t),4,fdat);
      ir+=fread(dstd,sizeof(double),9,fdat);
      error_root(ir!=13,1,"check_lat_parms [lat_parms.c]",
                 "Incorrect read count");         

      if (endian==BIG_ENDIAN)
      {
         bswap_int(4,istd);
         bswap_double(9,dstd);
      }
      
      ie=0;
      ie|=(istd[0]!=(stdint_t)(N0));
      ie|=(istd[1]!=(stdint_t)(N1));
      ie|=(istd[2]!=(stdint_t)(N2));
      ie|=(istd[3]!=(stdint_t)(N3));

      ie|=(dstd[0]!=lat.beta);
      ie|=(dstd[1]!=lat.c0);
      ie|=(dstd[2]!=lat.c1);      
      ie|=(dstd[3]!=lat.kappa_u);
      ie|=(dstd[4]!=lat.kappa_s);
      ie|=(dstd[5]!=lat.kappa_c);
      ie|=(dstd[6]!=lat.csw);
      ie|=(dstd[7]!=lat.cG);
      ie|=(dstd[8]!=lat.cF);
         
      error_root(ie!=0,1,"check_lat_parms [lat_parms.c]",
                 "Parameters do not match");
   }
}


double sea_quark_mass(int im0)
{
   if (im0==0)
      return lat.m0u;
   else if (im0==1)
      return lat.m0s;
   else if (im0==2)
      return lat.m0c;
   else
      return DBL_MAX;
}


sw_parms_t set_sw_parms(double m0)
{
   double dprms[1];

   if (NPROC>1)
   {
      dprms[0]=m0;

      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      error(dprms[0]!=m0,1,
            "set_sw_parms [lat_parms.c]","Parameters are not global");
   }

   if (m0!=sw.m0)
   {
      set_flags(ERASED_SW);
      set_flags(ERASED_SWD);
      set_grid_flags(SAP_BLOCKS,ERASED_SW);
      set_flags(ERASED_AWHAT);
   }
   
   sw.m0=m0;
   sw.csw=lat.csw;
   sw.cF=lat.cF;
   
   return sw;
}


sw_parms_t sw_parms(void)
{
   sw.csw=lat.csw;
   sw.cF=lat.cF;
   
   return sw;
}


tm_parms_t set_tm_parms(int eoflg)
{
   int iprms[1];

   if (NPROC>1)
   {
      iprms[0]=eoflg;

      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);

      error(iprms[0]!=eoflg,1,
            "set_tm_parms [lat_parms.c]","Parameters are not global");
   }

   if (eoflg!=tm.eoflg)
      set_flags(ERASED_AWHAT);

   tm.eoflg=eoflg;

   return tm;
}


tm_parms_t tm_parms(void)
{
   return tm;
}
