
/*******************************************************************************
*
* File mrw_parms.c
*
* Copyright (C) 2012, 2013 Martin Luescher, 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Reweighting factor parameter data base
*
* The externally accessible functions are
*
*   void init_mrw(void)
*     Initialize the structure array containing the reweighting factor
*     parameter sets. This resets all previously added parameter sets.
*
*   mrw_parms_t set_mrw_parms(int irw,mrwfact_t mrwfact,double kappa0,double kappa,
*                             double mu0,double mu,double gamma,double kappa2,
*                             int isp1,int isp2,int nm,int pwr,int nsrc,int tmeo)
*     Sets the parameters in the reweighting factor parameter set number
*     irw and returns a structure containing them (see the notes).
*
*   mrw_parms_t mrw_parms(int irw)
*     Returns a structure containing the reweighting factor parameter set
*     number irw (see the notes).
*
*   void read_mrw_parms(int irw)
*     On process 0, this program scans stdin for a line starting with the
*     string "[Reweighting factor <int>]" (after any number of blanks), where
*     <int> is the integer value passed by the argument. An error occurs if
*     no such line or more than one is found. The lines 
*
*       mrwfact  <mrwfact_t>
*       kappa0   <double>
*       kappa    <double>
*       mu0      <double>
*       mu       <double>
*       isp      <int> [<int>]
*       nm       <int>
*       pwr      <int>
*       nsrc     <int>
*       tmeo     <int>
*
*     are then read using read_line() [utils/mutils.c] and the data are
*     added to the data base by calling set_mrw_parms(irw,...). Depending
*     on the value of "mrwfact", some lines are not read and can be omitted 
*     in the input file (see the notes).
*
*   void print_mrw_parms(void)
*     Prints the defined reweighting factor parameter sets to stdout on
*     MPI process 0.
*
*   void write_mrw_parms(FILE *fdat)
*     Writes the defined reweighting factor parameter sets to the file fdat 
*     on MPI process 0.
*
*   void check_mrw_parms(FILE *fdat)
*     Compares the defined reweighting factor parameter sets with those 
*     on the file fdat on MPI process 0, assuming the latter were written
*     to the file by the program write_mrw_parms().
*
*   mrw_masses_t get_mrw_masses(int irw,int k)
*     Returns the masses of mass interpolations step k for the reweighting
*     factor parameter set irw 
*
* Notes:
*
* The elements of a structure of type mrw_parms_t are:
*
*   mrwfact Reweighting factor program used. This parameter is an enum 
*           type with one of the following values:
*
*            TMRW,TMRW1,MRW             (program mrw1() [mrw/mrw.c]),
*
*            TMRW3,MRW_ISO,MRW_TF       (program mrw2() [mrw/mrw.c])
* 
*            TMRW[2,4]                  (program mrw3() [mrw/mrw.c])
*
*            TMRW_EO,TMRW1_EO,MRW_EO    (program mrw1eo() [mrw/mrweo.c]),
*
*            TMRW3_EO                   (program mrw2eo() [mrw/mrweo.c]),
*
*            TMRW[2,4]_EO               (program mrw3eo() [mrw/mrweo.c]),
*
*   kappa0  Ensemble kappa parameter.
*
*   kappa   Ensemble or target kappa parameter (depending on rew. factor).
*
*   mu0     Ensemble twisted-mass parameter.
*
*   mu      Ensemble or target twisted-mass parameter.
*
*   nm      Number of mass/twisted-mass interpolation steps.
*
*   pwr     Power of the interpolation (default: 0).
*
*   gamma   Tuning parameter for two-flavor reweighting (default: 1.0).
* 
*   kappa2  Target kappa parameter two-flavor reweighting.
*
*   isp     Solver parameter set index to be used.
*
*   nsrc    Number of random source fields used for the stochastic
*           estimation of the reweighting factor.
*
*   tmeo    Twisted-mass only on even sites [1] or on all sites [0].
*           Only has effect for mass reweighting factors.
*
* Valid examples of sections that can be read by read_mrw_parms() are
*
*   [Reweighting factor 1]
*    mrwfact  TMRW        # or TMRW_EO
*    kappa0   0.1351
*    mu0      0.0
*    mu       0.001
*    nm       2
*    pwr      0
*    isp      3
*    nsrc     12
*
* Up to 32 parameter sets, labeled by an index irw=0,1,..,31, can be
* specified. Once a set is defined, it cannot be changed by calling
* set_mrw_parms() again. All parameters must be globally the same.
*
* Except for mrw_parms(), the programs in this module perform global
* operations and must be called simultaneously on all MPI processes.
*
*******************************************************************************/

#define MRW_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "mrw.h"
#include "global.h"

#define IMRWMAX 32


static int init=0;
static char mrwtag[NAME_SIZE][MRWFACTS]={"TMRW","TMRW_EO","TMRW1","TMRW1_EO",
                          "TMRW2","TMRW2_EO","TMRW3","TMRW3_EO","TMRW4","TMRW4_EO",
                          "MRW","MRW_EO","MRW_ISO","MRW_TF"};
static mrwfact_t mrwfact[]={TMRW,TMRW_EO,TMRW1,TMRW1_EO,TMRW2,TMRW2_EO,
                            TMRW3,TMRW3_EO,TMRW4,TMRW4_EO,MRW,MRW_EO,MRW_ISO,MRW_TF};
static mrw_parms_t rw0={MRWFACTS,0,{0,0},0,0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
static mrw_parms_t rw[IMRWMAX];


void init_mrw(void)
{
   int irw;
   
   for (irw=0;irw<IMRWMAX;irw++)
      rw[irw]=rw0;

   init=1;
}


mrw_parms_t set_mrw_parms(int irw,mrwfact_t mrwfact,double kappa0,double kappa,
                          double mu0,double mu,double gamma,double kappa2,
                          int isp1,int isp2,int nm,int pwr,int nsrc,int tmeo)
{
   int iprms[8],ie;
   double dprms[6];
   double m,m0;
   
   if (init==0)
      init_mrw();
   
   m0=0.0;
   m=0.0;
   
   if ((mrwfact==TMRW)||(mrwfact==TMRW_EO)||(mrwfact==TMRW1)||(mrwfact==TMRW1_EO))
   {
      gamma=0.0;
      m0=0.5/kappa0-4.0;
      m=m0;
      kappa=0.5/(m+4.0);
   }
   else if ((mrwfact==TMRW2)||(mrwfact==TMRW2_EO)||(mrwfact==TMRW3)||
            (mrwfact==TMRW3_EO)||(mrwfact==TMRW4)||(mrwfact==TMRW4_EO))
   {
      gamma=0.0;
      m0=0.5/kappa0-4.0;;
      m=0.5/kappa-4.0;
   }
   else if ((mrwfact==MRW)||(mrwfact==MRW_EO))
   {
      gamma=0.0;
      m0=0.5/kappa0-4.0;;
      m=0.5/kappa-4.0;
      mu=mu0;
   }
   else if (mrwfact==MRW_ISO)
   {
      gamma=0.0;
      m0=0.5/kappa0-4.0;;
      m=0.5/kappa-4.0;
      mu=mu0;
   }
   else if (mrwfact==MRW_TF)
   {
      m0=0.5/kappa0-4.0;;
      m=0.5/kappa-4.0;
   }

   
   if (NPROC>1)
   {
      iprms[0]=irw;
      iprms[1]=(int)(mrwfact);
      iprms[2]=nm;
      iprms[3]=isp1;
      iprms[4]=isp2;
      iprms[5]=nsrc;
      iprms[6]=pwr;
      iprms[7]=tmeo;
      dprms[0]=kappa0;
      dprms[1]=kappa;
      dprms[2]=mu0;
      dprms[3]=mu;
      dprms[4]=gamma;
      dprms[5]=kappa2;
      
      MPI_Bcast(iprms,8,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,6,MPI_DOUBLE,0,MPI_COMM_WORLD);

      ie=0;
      ie|=(iprms[0]!=irw);
      ie|=(iprms[1]!=(int)(mrwfact));
      ie|=(iprms[2]!=nm);
      ie|=(iprms[3]!=isp1);      
      ie|=(iprms[4]!=isp2);      
      ie|=(iprms[5]!=nsrc);
      ie|=(iprms[6]!=pwr);
      ie|=(iprms[7]!=tmeo);
      ie|=(dprms[0]!=kappa0);
      ie|=(dprms[1]!=kappa);
      ie|=(dprms[2]!=mu0);
      ie|=(dprms[3]!=mu);
      ie|=(dprms[4]!=gamma);
      ie|=(dprms[5]!=kappa2);
      
      error(ie!=0,1,"set_mrw_parms [mrw_parms.c]",
            "Parameters are not global");
   }

   ie=0;
   ie|=((irw<0)||(irw>=IMRWMAX));
   ie|=(nm<1);
   ie|=((pwr<0)||(pwr>4));
   ie|=(nsrc<1);

   if ((mrwfact==TMRW1)||(mrwfact==TMRW2)||(mrwfact==TMRW4))
   {
      ie|=(mu0<0.0);
      ie|=(mu<0.0);
   }
   if (mrwfact==TMRW2)
      ie|=(mu0<mu);
   
   error_root(ie!=0,1,"set_mrw_parms [mrw_parms.c]",
              "Parameters are out of range");

   error_root(rw[irw].mrwfact!=MRWFACTS,1,"set_mrw_parms [mrw_parms.c]",
              "Attempt to reset an already specified parameter set");

   rw[irw].mrwfact=mrwfact;
   rw[irw].kappa0=kappa0;
   rw[irw].kappa=kappa;
   rw[irw].mu0=mu0;
   rw[irw].mu=mu;
   rw[irw].gamma=gamma;
   rw[irw].kappa2=kappa2;
   rw[irw].nm=nm;
   rw[irw].pwr=pwr;
   rw[irw].isp[0]=isp1;
   rw[irw].isp[1]=isp2;
   rw[irw].nsrc=nsrc;
   rw[irw].tmeo=tmeo;
   
   rw[irw].m0=m0;
   rw[irw].m=m;
   
   return rw[irw];
}


mrw_parms_t mrw_parms(int irw)
{
   if (init==0)
      init_mrw();

   if ((irw>=0)&&(irw<IMRWMAX))
      return rw[irw];
   else
   {
      error_loc(1,1,"mrw_parms [mrw_parms.c]",
                "Reweighting factor index is out of range");
      return rw[IMRWMAX];
   }
}


void read_mrw_parms(int irw)
{
   int my_rank;
   int idr,nm,nsrc,isp[2],pwr,isp2,tmeo;
   double kappa0,kappa,mu0,mu,gamma,kappa2;
   char line[NAME_SIZE];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   idr=0;
   isp2=0;
   nsrc=0;
   nm=0;
   pwr=0;
   tmeo=0;
   isp[0]=0;
   isp[1]=0;
   kappa0=0.0;
   kappa=0.0;
   mu=0.0;
   mu0=0.0;
   gamma=1.0;
   kappa2=0.0;
   
   if (my_rank==0)
   {
      sprintf(line,"Reweighting factor %d",irw);
      find_section(line);

      read_line("mrwfact","%s",line);

      if (strcmp(line,"TMRW")==0)
      {
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
      }
      else if (strcmp(line,"TMRW_EO")==0)
      {
         idr=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
      }
      else if (strcmp(line,"TMRW1")==0)
      {
         idr=2;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
      }
      else if (strcmp(line,"TMRW1_EO")==0)
      {
         idr=3;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
      }
      else if (strcmp(line,"TMRW2")==0)
      {
         idr=4;
         isp2=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
         read_line("kappa","%lf",&kappa);
      }
      else if (strcmp(line,"TMRW2_EO")==0)
      {
         idr=5;
         isp2=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
         read_line("kappa","%lf",&kappa);
      }
      else if (strcmp(line,"TMRW3")==0)
      {
         idr=6;
         isp2=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("kappa","%lf",&kappa);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
      }
      else if (strcmp(line,"TMRW3_EO")==0)
      {
         idr=7;
         isp2=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("kappa","%lf",&kappa);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
      }
      else if (strcmp(line,"TMRW4")==0)
      {
         idr=8;
         isp2=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("kappa","%lf",&kappa);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
      }
      else if (strcmp(line,"TMRW4_EO")==0)
      {
         idr=9;
         isp2=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("kappa","%lf",&kappa);
         read_line("mu0","%lf",&mu0);
         read_line("mu","%lf",&mu);
      }
      else if (strcmp(line,"MRW")==0)
      {
         idr=10;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("kappa","%lf",&kappa);
      }
      else if (strcmp(line,"MRW_EO")==0)
      {
         idr=11;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("kappa","%lf",&kappa);
      }
      else if (strcmp(line,"MRW_ISO")==0)
      {
         idr=12;
         isp2=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("kappa","%lf",&kappa);
      }
      else if (strcmp(line,"MRW_TF")==0)
      {
         idr=13;
         isp2=1;
         read_line("kappa0","%lf",&kappa0);
         read_line("mu0","%lf",&mu0);
         read_line("kappa","%lf",&kappa);
         read_line("mu","%lf",&mu);
         read_line("gamma","%lf",&gamma);
         read_line("kappa2","%lf",&kappa2);
      }
      else
         error_root(1,1,"read_mrw_parms [mrw_parms.c]",
                    "Unknown reweighting factor %s",line);
   }

   if (my_rank==0)
   {
      read_line("nm","%d",&nm);         
      read_line("pwr","%d",&pwr);         
      read_line("nsrc","%d",&nsrc); 
      
      if (idr>=10)
         read_line("tmeo","%d",&tmeo); 
   
      if (isp2)
         read_iprms("isp",2,isp);
      else
      {
         read_line("isp","%d",isp);
         isp[1]=isp[0];
      }
   }
   
   if (NPROC>1)
   {
      MPI_Bcast(&idr,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&kappa0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&mu0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&gamma,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&kappa2,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&nm,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&pwr,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(isp,2,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&nsrc,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&tmeo,1,MPI_INT,0,MPI_COMM_WORLD);
   }
   
   set_mrw_parms(irw,mrwfact[idr],kappa0,kappa,mu0,mu,gamma,kappa2,
                 isp[0],isp[1],nm,pwr,nsrc,tmeo);
}


void print_mrw_parms(void)
{
   int my_rank,irw,n;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if ((my_rank==0)&&(init==1))
   {
      for (irw=0;irw<IMRWMAX;irw++)
      {
         if (rw[irw].mrwfact!=MRWFACTS)
         {
            printf("Reweighting factor %d:\n",irw);
            printf("%s factor\n",mrwtag[rw[irw].mrwfact]);

            if ((rw[irw].mrwfact==TMRW)||(rw[irw].mrwfact==TMRW_EO))
            {
               n=fdigits(rw[irw].kappa0);
               printf("kappa0 = %.*f\n",IMAX(n,1),rw[irw].kappa0);
               n=fdigits(rw[irw].mu0);
               printf("mu0 = %.*f\n",IMAX(n,1),rw[irw].mu0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               printf("nm = %d\n",rw[irw].nm);
               printf("pwr = %d\n",rw[irw].pwr);
               printf("isp = %d\n",rw[irw].isp[0]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }
            else if ((rw[irw].mrwfact==TMRW1)||(rw[irw].mrwfact==TMRW1_EO))
            {
               n=fdigits(rw[irw].kappa0);
               printf("kappa0 = %.*f\n",IMAX(n,1),rw[irw].kappa0);
               n=fdigits(rw[irw].mu0);
               printf("mu0 = %.*f\n",IMAX(n,1),rw[irw].mu0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               printf("nm = %d\n",rw[irw].nm);
               printf("pwr = %d\n",rw[irw].pwr);
               printf("isp = %d\n",rw[irw].isp[0]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }
            else if ((rw[irw].mrwfact==TMRW2)||(rw[irw].mrwfact==TMRW2_EO))
            {
               n=fdigits(rw[irw].kappa0);
               printf("kappa0 = %.*f\n",IMAX(n,1),rw[irw].kappa0);
               n=fdigits(rw[irw].mu0);
               printf("mu0 = %.*f\n",IMAX(n,1),rw[irw].mu0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               n=fdigits(rw[irw].kappa);
               printf("kappa = %.*f\n",IMAX(n,1),rw[irw].kappa);
               printf("nm = %d\n",rw[irw].nm);
               printf("pwr = %d\n",rw[irw].pwr);
               printf("isp = %d\n",rw[irw].isp[0]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }
            else if ((rw[irw].mrwfact==TMRW3)||(rw[irw].mrwfact==TMRW3_EO))
            {
               n=fdigits(rw[irw].kappa0);
               printf("kappa0 = %.*f\n",IMAX(n,1),rw[irw].kappa0);
               n=fdigits(rw[irw].mu0);
               printf("mu0 = %.*f\n",IMAX(n,1),rw[irw].mu0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               n=fdigits(rw[irw].kappa);
               printf("kappa = %.*f\n",IMAX(n,1),rw[irw].kappa);
               printf("nm = %d\n",rw[irw].nm);
               printf("pwr = %d\n",rw[irw].pwr);
               printf("isp = %d %d\n",rw[irw].isp[0],rw[irw].isp[1]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }
            else if ((rw[irw].mrwfact==TMRW4)||(rw[irw].mrwfact==TMRW4_EO))
            {
               n=fdigits(rw[irw].kappa0);
               printf("kappa0 = %.*f\n",IMAX(n,1),rw[irw].kappa0);
               n=fdigits(rw[irw].mu0);
               printf("mu0 = %.*f\n",IMAX(n,1),rw[irw].mu0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               n=fdigits(rw[irw].kappa);
               printf("kappa = %.*f\n",IMAX(n,1),rw[irw].kappa);
               printf("nm = %d\n",rw[irw].nm);
               printf("pwr = %d\n",rw[irw].pwr);
               printf("isp = %d %d\n",rw[irw].isp[0],rw[irw].isp[1]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }
            else if ((rw[irw].mrwfact==MRW)||(rw[irw].mrwfact==MRW_EO))
            {
               n=fdigits(rw[irw].kappa0);
               printf("kappa0 = %.*f\n",IMAX(n,1),rw[irw].kappa0);
               n=fdigits(rw[irw].mu0);
               printf("mu0 = %.*f\n",IMAX(n,1),rw[irw].mu0);
               n=fdigits(rw[irw].kappa);
               printf("kappa = %.*f\n",IMAX(n,1),rw[irw].kappa);
               printf("nm = %d\n",rw[irw].nm);
               printf("pwr = %d\n",rw[irw].pwr);
               printf("isp = %d\n",rw[irw].isp[0]);
               printf("nsrc = %d\n",rw[irw].nsrc);
               printf("tmeo = %d\n\n",rw[irw].tmeo);
            }
            else if (rw[irw].mrwfact==MRW_ISO)
            {
               n=fdigits(rw[irw].kappa0);
               printf("kappa0 = %.*f\n",IMAX(n,1),rw[irw].kappa0);
               n=fdigits(rw[irw].mu0);
               printf("mu0 = %.*f\n",IMAX(n,1),rw[irw].mu0);
               n=fdigits(rw[irw].kappa);
               printf("kappa = %.*f\n",IMAX(n,1),rw[irw].kappa);
               printf("nm = %d\n",rw[irw].nm);
               printf("pwr = %d\n",rw[irw].pwr);
               printf("isp = %d %d\n",rw[irw].isp[0],rw[irw].isp[1]);
               printf("nsrc = %d\n",rw[irw].nsrc);
               printf("tmeo = %d\n\n",rw[irw].tmeo);
            }
            else if (rw[irw].mrwfact==MRW_TF)
            {
               n=fdigits(rw[irw].kappa0);
               printf("kappa0 = %.*f\n",IMAX(n,1),rw[irw].kappa0);
               n=fdigits(rw[irw].mu0);
               printf("mu0 = %.*f\n",IMAX(n,1),rw[irw].mu0);
               n=fdigits(rw[irw].kappa);
               printf("kappa = %.*f\n",IMAX(n,1),rw[irw].kappa);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               n=fdigits(rw[irw].gamma);
               printf("gamma = %.*f\n",IMAX(n,1),rw[irw].gamma);
               n=fdigits(rw[irw].kappa2);
               printf("kappa2 = %.*f\n",IMAX(n,1),rw[irw].kappa2);
               printf("nm = %d\n",rw[irw].nm);
               printf("pwr = %d\n",rw[irw].pwr);
               printf("isp = %d %d\n",rw[irw].isp[0],rw[irw].isp[1]);
               printf("nsrc = %d\n",rw[irw].nsrc);
               printf("tmeo = %d\n\n",rw[irw].tmeo);
            }
         }
      }
   }
}


void write_mrw_parms(FILE *fdat)
{
   int my_rank,endian;
   int iw,irw;
   stdint_t istd[8];
   double dstd[6];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();
      
   if ((my_rank==0)&&(init==1))
   {
      for (irw=0;irw<IMRWMAX;irw++)
      {
         if (rw[irw].mrwfact!=MRWFACTS)
         {
            istd[0]=(stdint_t)(irw);            
            istd[1]=(stdint_t)(rw[irw].mrwfact);
            istd[2]=(stdint_t)(rw[irw].nm);
            istd[3]=(stdint_t)(rw[irw].isp[0]);
            istd[4]=(stdint_t)(rw[irw].isp[1]);
            istd[5]=(stdint_t)(rw[irw].nsrc);
            istd[6]=(stdint_t)(rw[irw].pwr);
            istd[7]=(stdint_t)(rw[irw].tmeo);
            dstd[0]=rw[irw].kappa0;
            dstd[1]=rw[irw].kappa;
            dstd[2]=rw[irw].mu0;
            dstd[3]=rw[irw].mu;
            dstd[4]=rw[irw].gamma;
            dstd[5]=rw[irw].kappa2;
            
            if (endian==BIG_ENDIAN)
            {
               bswap_int(8,istd);
               bswap_double(6,dstd);
            }
            
            iw=fwrite(istd,sizeof(stdint_t),8,fdat);
            iw+=fwrite(dstd,sizeof(double),6,fdat);

            error_root(iw!=(8+6),1,"write_mrw_parms [mrw_parms.c]",
                       "Incorrect write count");
         }
      }
   }
}


void check_mrw_parms(FILE *fdat)
{
   int my_rank,endian;
   int ir,irw,ie;
   stdint_t istd[8];
   double dstd[6];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();
      
   if ((my_rank==0)&&(init==1))
   {
      ie=0;
      
      for (irw=0;irw<IMRWMAX;irw++)
      {
         if (rw[irw].mrwfact!=MRWFACTS)
         {
            ir=fread(istd,sizeof(stdint_t),8,fdat);
            ir+=fread(dstd,sizeof(double),6,fdat);

            if (endian==BIG_ENDIAN)
            {
               bswap_int(8,istd);
               bswap_double(6,dstd);
            }
            
            ie|=(istd[0]!=(stdint_t)(irw));            
            ie|=(istd[1]!=(stdint_t)(rw[irw].mrwfact));
            ie|=(istd[2]!=(stdint_t)(rw[irw].nm));
            ie|=(istd[3]!=(stdint_t)(rw[irw].isp[0]));
            ie|=(istd[4]!=(stdint_t)(rw[irw].isp[1]));
            ie|=(istd[5]!=(stdint_t)(rw[irw].nsrc));
            ie|=(istd[6]!=(stdint_t)(rw[irw].pwr));
            ie|=(istd[7]!=(stdint_t)(rw[irw].tmeo));
            ie|=(dstd[0]!=rw[irw].kappa0);
            ie|=(dstd[1]!=rw[irw].kappa);
            ie|=(dstd[2]!=rw[irw].mu0);
            ie|=(dstd[3]!=rw[irw].mu);
            ie|=(dstd[4]!=rw[irw].gamma);
            ie|=(dstd[5]!=rw[irw].kappa2);

            error_root(ir!=(8+6),1,"check_mrw_parms [mrw_parms.c]",
                       "Incorrect read count");
         }
      }
         
      error_root(ie!=0,1,"check_mrw_parms [mrw_parms.c]",
                 "Parameters do not match");         
   }
}


static int cnp(int n,int p)
{
   int c;
   
   c=0;
   if (p==0)
      c=n;
   if (p==1)
      c=(n*(n+1))/2;
   if (p==2)
      c=(n*(n+1)*(2*n+1))/6;
   if (p==3)
      c=(n*n*(n+1)*(n+1))/4;
   if (p==4)
      c=(n*(n+1)*(2*n+1)*(3*n*n+3*n-1))/30;
   
   return c;
}


static double interpolation(int l,double x0,double x,int n,int p,double *d)
{
   int cl,cn,i;
   double r;

   cl=cnp(l,p);
   cn=cnp(n,p);
   
   *d=(x0-x)/((double)cn);
   r=x+((double)cl)*(*d);
   
   for (i=1;i<=p;i++)
      *d*=((double)(l+1));
   
   return r;
}


mrw_masses_t get_mrw_masses(int irw,int k)
{
   mrw_parms_t rw;
   mrw_masses_t ms;
   mrwfact_t mrwfact;   
   
   ms.m1=0.0;
   ms.mu1=0.0;
   ms.d1=0.0;
   ms.m2=0.0;
   ms.mu2=0.0;
   ms.d2=0.0;
   
   rw=mrw_parms(irw);
   mrwfact=rw.mrwfact;
   
   error_root(((k<0)||(k>=rw.nm)),1,"get_mrw_masses [mrw_parms.c]",
               "Mass interpolation step out of bounds");
   
   if ((mrwfact==TMRW)||(mrwfact==TMRW_EO))
   {
      ms.m1=rw.m0;
      ms.mu1=interpolation(k,rw.mu0,rw.mu,rw.nm,rw.pwr,&(ms.d1));
   }
   else if ((mrwfact==TMRW1)||(mrwfact==TMRW1_EO))
   {
      ms.m1=rw.m0;
      ms.mu1=interpolation(k,rw.mu0*rw.mu0,rw.mu*rw.mu,rw.nm,rw.pwr,&(ms.d1));
      ms.mu1=sqrt(ms.mu1);
   }
   else if ((mrwfact==TMRW2)||(mrwfact==TMRW2_EO))
   {
      ms.m1=rw.m0;
      ms.mu1=interpolation(k,rw.mu0*rw.mu0,rw.mu*rw.mu,rw.nm,rw.pwr,&(ms.d1));

      ms.m2=rw.m;
      ms.d2=-ms.d1;
      ms.mu2=sqrt(ms.mu1+rw.mu0*rw.mu0-rw.mu*rw.mu+ms.d1);
      
      ms.mu1=sqrt(ms.mu1);
   }
   else if ((mrwfact==TMRW3)||(mrwfact==TMRW3_EO))
   {
      ms.m1=rw.m0;
      ms.mu1=interpolation(k,rw.mu0,rw.mu,rw.nm,rw.pwr,&(ms.d1));

      ms.m2=rw.m;
      ms.d2=ms.d1;
      ms.mu2=-(ms.mu1+ms.d1);
   }
   else if ((mrwfact==TMRW4)||(mrwfact==TMRW4_EO))
   {
      ms.m1=rw.m0;
      ms.mu1=interpolation(k,rw.mu0*rw.mu0,rw.mu*rw.mu,rw.nm,rw.pwr,&(ms.d1));

      ms.m2=rw.m;
      ms.d2=-ms.d1;
      if (k==rw.nm-1)
         ms.mu2=rw.mu0;
      else
         ms.mu2=sqrt(ms.mu1+ms.d1);
      
      ms.mu1=sqrt(ms.mu1);
   }
   else if ((mrwfact==MRW)||(mrwfact==MRW_EO))
   {
      ms.mu1=rw.mu0;
      ms.m1=interpolation(k,rw.m0,rw.m,rw.nm,rw.pwr,&(ms.d1));
   }
   else if (mrwfact==MRW_ISO)
   {
      ms.mu1=rw.mu0;
      ms.m1=interpolation(k,rw.m0,rw.m,rw.nm,rw.pwr,&(ms.d1));

      ms.mu2=rw.mu0;
      ms.d2=-ms.d1;
      ms.m2=rw.m0-rw.m+ms.m1+ms.d1;
   }
   else if (mrwfact==MRW_TF)
   {
      ms.mu1=rw.mu;
      ms.m1=interpolation(k,rw.m,0.5/rw.kappa2-4.0,rw.nm,rw.pwr,&(ms.d1));

      ms.mu2=rw.mu0;
      ms.d2=-rw.gamma*ms.d1;
      ms.m2=rw.m0+rw.gamma*(ms.m1+ms.d1-0.5/rw.kappa2+4.0);
   }
   else
      error_root(1,1,"get_mrw_masses [mrw_parms.c]",
                  "Unknown reweighting factor (irw=%d)",irw);
   
   return ms;
}
