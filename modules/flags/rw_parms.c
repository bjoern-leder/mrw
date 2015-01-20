
/*******************************************************************************
*
* File rw_parms.c
*
* Copyright (C) 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Reweighting factor parameter data base
*
* The externally accessible functions are
*
*   rw_parms_t set_rw_parms(int irw,rwfact_t rwfact,int im0,double mu,
*                           int irp,int n,int *np,int *isp,int nsrc)
*     Sets the parameters in the reweighting factor parameter set number
*     irw and returns a structure containing them (see the notes).
*
*   rw_parms_t rw_parms(int irw)
*     Returns a structure containing the reweighting factor parameter set
*     number irw (see the notes).
*
*   void read_rw_parms(int irw)
*     On process 0, this program scans stdin for a line starting with the
*     string "[Reweighting factor <int>]" (after any number of blanks), where
*     <int> is the integer value passed by the argument. An error occurs if
*     no such line or more than one is found. The lines 
*
*       rwfact   <rwfact_t>
*       im0      <int>
*       irp      <int>
*       mu       <double>
*       np       <int> [<int>]
*       isp      <int> [<int>]
*       nsrc     <int>
*
*     are then read using read_line() [utils/mutils.c] and the data are
*     added to the data base by calling set_rw_parms(irw,...). Depending
*     on the value of "rwfact", some lines are not read and can be omitted 
*     in the input file. The number n of integer items on the lines with 
*     tag "np" and "isp" depends on the reweighting factor too (see the
*     notes).
*
*   void print_rw_parms(void)
*     Prints the defined reweighting factor parameter sets to stdout on
*     MPI process 0.
*
*   void write_rw_parms(FILE *fdat)
*     Writes the defined reweighting factor parameter sets to the file fdat 
*     on MPI process 0.
*
*   void check_rw_parms(FILE *fdat)
*     Compares the defined reweighting factor parameter sets with those 
*     on the file fdat on MPI process 0, assuming the latter were written
*     to the file by the program write_rw_parms().
*
* Notes:
*
* The elements of a structure of type rw_parms_t are:
*
*   rwfact  Reweighting factor program used. This parameter is an enum 
*           type with one of the following values:
*
*            RWTM1       (program rwtm1() [update/rwtm.c]),
*
*            RWTM1_EO    (program rwtm1eo() [update/rwtmeo.c]),
*
*            RWTM2       (program rwtm2() [update/rwtm.c]),
*
*            RWTM2_EO    (program rwtm2eo() [update/rwtmeo.c]),
*
*            RWRAT       (program rwrat() [update/rwrat.c]).
*
*   im0     Bare mass index (0: m0u, 1: m0s, 2: m0c; see flags/lat_parms.c).
*
*   mu      Twisted-mass parameter.
*
*   irp     Rational function index.
*
*   n       Number of rational function parts.
*
*   np      Array of the numbers of poles of the rational function parts
*           (n elements).
*
*   isp     Array of the solver parameter set indices to be used. Only one
*           index needs to be specified if rwfact={RWTM1,RWTM2}, while if
*           rwfact=RWRAT there must be n indices on this line, one for each
*           part of the rational function.
*
*   nsrc    Number of random source fields used for the stochastic
*           estimation of the reweighting factor.
*
* In the case of RWTM1* and RWTM2*, irp and n are set to 0 and 1, respectively,
* by set_rw_parms(), while mu is set to 0.0 in the case of RWRAT.
*
* Valid examples of sections that can be read by read_rw_parms() are
*
*   [Reweighting factor 1]
*    rwfact   RWTM1        # or RWTM1_EO, RWTM2, RWTM2_EO
*    im0      0
*    mu       0.001
*    isp      3
*    nsrc     12
*
*   [Reweighting factor 4]
*    rwfact   RWRAT 
*    im0      1
*    irp      0
*    np       6 2 2
*    isp      3 5 6
*    nsrc     1
*
* Up to 32 parameter sets, labeled by an index irw=0,1,..,31, can be
* specified. Once a set is defined, it cannot be changed by calling
* set_rw_parms() again. All parameters must be globally the same.
*
* Except for rw_parms(), the programs in this module perform global
* operations and must be called simultaneously on all MPI processes.
*
*******************************************************************************/

#define RW_PARMS_C

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include "utils.h"
#include "flags.h"
#include "global.h"

#define IRWMAX 32

static int init=0;
static rwfact_t rwfact[]={RWTM1,RWTM1_EO,RWTM2,RWTM2_EO,RWRAT};
static rw_parms_t rw[IRWMAX+1]={{RWFACTS,0,0,0,0,NULL,NULL,0.0}};


static void init_rw(void)
{
   int irw;
   
   for (irw=1;irw<=IRWMAX;irw++)
      rw[irw]=rw[0];

   init=1;
}


rw_parms_t set_rw_parms(int irw,rwfact_t rwfact,int im0,double mu,
                        int irp,int n,int *np,int *isp,int nsrc)
{
   int iprms[6],np0[1],i,ie;
   double dprms[1];
   
   if (init==0)
      init_rw();

   if ((rwfact==RWTM1)||(rwfact==RWTM1_EO)||
       (rwfact==RWTM2)||(rwfact==RWTM2_EO))
   {
      irp=0;
      n=1;
      np=np0;

      if ((rwfact==RWTM1)||(rwfact==RWTM1_EO))
         np[0]=1;
      else
         np[0]=2;
   }
   else if (rwfact==RWRAT)
      mu=0.0;
   
   if (NPROC>1)
   {
      iprms[0]=irw;
      iprms[1]=(int)(rwfact);
      iprms[2]=im0;
      iprms[3]=irp;
      iprms[4]=n;
      iprms[5]=nsrc;
      dprms[0]=mu;
      
      MPI_Bcast(iprms,6,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(dprms,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

      ie=0;
      ie|=(iprms[0]!=irw);
      ie|=(iprms[1]!=(int)(rwfact));
      ie|=(iprms[2]!=im0);
      ie|=(iprms[3]!=irp);
      ie|=(iprms[4]!=n);      
      ie|=(iprms[5]!=nsrc);
      ie|=(dprms[0]!=mu);
      
      error(ie!=0,1,"set_rw_parms [rw_parms.c]",
            "Parameters are not global");
   }

   ie=0;
   ie|=((irw<0)||(irw>=IRWMAX));
   ie|=((im0<0)||(im0>2));
   ie|=(irp<0);
   ie|=(n<1);
   ie|=(nsrc<1);

   error_root(ie!=0,1,"set_rw_parms [rw_parms.c]",
              "Parameters are out of range");

   if (NPROC>1)
   {
      for (i=0;i<n;i++)
      {
         iprms[0]=np[i];
         iprms[1]=isp[i];
      
         MPI_Bcast(iprms,2,MPI_INT,0,MPI_COMM_WORLD);

         ie|=(iprms[0]!=np[i]);
         ie|=(iprms[1]!=isp[i]);
      }
      
      error(ie!=0,1,"set_rw_parms [rw_parms.c]",
            "Parameters np or isp are not global");
   }

   error_root(rw[irw].rwfact!=RWFACTS,1,"set_rw_parms [rw_parms.c]",
              "Attempt to reset an already specified parameter set");

   rw[irw].rwfact=rwfact;
   rw[irw].im0=im0;
   rw[irw].mu=mu;
   rw[irw].irp=irp;
   rw[irw].n=n;

   rw[irw].np=malloc(2*n*sizeof(*np));
   rw[irw].isp=rw[irw].np+n;

   error_root(rw[irw].np==NULL,1,"set_rw_parms [rw_parms.c]",
              "Unable to allocate parameter arrays");
   
   for (i=0;i<n;i++)
   {
      rw[irw].np[i]=np[i];
      rw[irw].isp[i]=isp[i];
   }

   rw[irw].nsrc=nsrc;
   
   return rw[irp];
}


rw_parms_t rw_parms(int irw)
{
   if (init==0)
      init_rw();

   if ((irw>=0)&&(irw<IRWMAX))
      return rw[irw];
   else
   {
      error_loc(1,1,"rw_parms [rw_parms.c]",
                "Reweighting factor index is out of range");
      return rw[IRWMAX];
   }
}


void read_rw_parms(int irw)
{
   int my_rank;
   int idr,im0,irp,nsrc;
   int n,*np,*isp;
   double mu;
   char line[NAME_SIZE];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   idr=0;
   im0=0;
   irp=0;
   nsrc=0;
   n=1;
   mu=0.0;
   
   if (my_rank==0)
   {
      sprintf(line,"Reweighting factor %d",irw);
      find_section(line);

      read_line("rwfact","%s",line);

      if (strcmp(line,"RWTM1")==0)
      {
         read_line("im0","%d",&im0);
         read_line("mu","%lf",&mu);
         read_line("nsrc","%d",&nsrc);         
      }
      else if (strcmp(line,"RWTM1_EO")==0)
      {
         idr=1;
         read_line("im0","%d",&im0);
         read_line("mu","%lf",&mu);
         read_line("nsrc","%d",&nsrc);
      }
      else if (strcmp(line,"RWTM2")==0)
      {
         idr=2;
         read_line("im0","%d",&im0);
         read_line("mu","%lf",&mu);  
         read_line("nsrc","%d",&nsrc);         
      }
      else if (strcmp(line,"RWTM2_EO")==0)
      {
         idr=3;
         read_line("im0","%d",&im0);
         read_line("mu","%lf",&mu);
         read_line("nsrc","%d",&nsrc);
      }
      else if (strcmp(line,"RWRAT")==0)
      {
         idr=4;
         read_line("im0","%d",&im0);
         read_line("irp","%d",&irp);
         n=count_tokens("np");
         error_root(n<1,1,"read_rw_parms [rw_parms.c]",
                    "No data on line with tag np");
         read_line("nsrc","%d",&nsrc);
      }
      else
         error_root(1,1,"read_rw_parms [rw_parms.c]",
                    "Unknown reweighting factor %s",line);
   }

   if (NPROC>1)
   {
      MPI_Bcast(&idr,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&im0,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
      MPI_Bcast(&irp,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&n,1,MPI_INT,0,MPI_COMM_WORLD);      
   }

   np=malloc(2*n*sizeof(*np));
   isp=np+n;
   error(np==NULL,1,"read_rw_parms [rw_parms.c]",
         "Unable to allocated data array");

   if (my_rank==0)
   {
      if ((idr==0)||(idr==1))
         np[0]=1;
      else if ((idr==2)||(idr==3))
         np[0]=2;
      else if (idr==4)
         read_iprms("np",n,np);

      read_iprms("isp",n,isp);
   }

   if (NPROC>1)
   {
      MPI_Bcast(np,n,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(isp,n,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&nsrc,1,MPI_INT,0,MPI_COMM_WORLD);
   }
   
   set_rw_parms(irw,rwfact[idr],im0,mu,irp,n,np,isp,nsrc);
   free(np);
}


void print_rw_parms(void)
{
   int my_rank,irw,n,l;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if ((my_rank==0)&&(init==1))
   {
      for (irw=0;irw<IRWMAX;irw++)
      {
         if (rw[irw].rwfact!=RWFACTS)
         {
            printf("Reweighting factor %d:\n",irw);

            if (rw[irw].rwfact==RWTM1)
            {
               printf("RWTM1 factor\n");
               printf("im0 = %d\n",rw[irw].im0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               printf("isp = %d\n",rw[irw].isp[0]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }
            else if (rw[irw].rwfact==RWTM1_EO)
            {
               printf("RWTM1_EO factor\n");
               printf("im0 = %d\n",rw[irw].im0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               printf("isp = %d\n",rw[irw].isp[0]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }
            else if (rw[irw].rwfact==RWTM2)
            {
               printf("RWTM2 factor\n");
               printf("im0 = %d\n",rw[irw].im0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               printf("isp = %d\n",rw[irw].isp[0]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);               
            }
            else if (rw[irw].rwfact==RWTM2_EO)
            {
               printf("RWTM2_EO factor\n");
               printf("im0 = %d\n",rw[irw].im0);
               n=fdigits(rw[irw].mu);
               printf("mu = %.*f\n",IMAX(n,1),rw[irw].mu);
               printf("isp = %d\n",rw[irw].isp[0]);
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }
            else if (rw[irw].rwfact==RWRAT)
            {
               printf("RWRAT factor\n");
               printf("im0 = %d\n",rw[irw].im0);
               printf("irp = %d\n",rw[irw].irp);
               n=rw[irw].n;
               printf("np = %d",rw[irw].np[0]);
               for (l=1;l<n;l++)
                  printf(" %d",rw[irw].np[l]);
               printf("\n");
               printf("isp = %d",rw[irw].isp[0]);
               for (l=1;l<n;l++)
                  printf(" %d",rw[irw].isp[l]);
               printf("\n");              
               printf("nsrc = %d\n\n",rw[irw].nsrc);
            }            
         }
      }
   }
}


void write_rw_parms(FILE *fdat)
{
   int my_rank,endian;
   int iw,irw,n,l;
   stdint_t istd[6];
   double dstd[1];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();
      
   if ((my_rank==0)&&(init==1))
   {
      for (irw=0;irw<IRWMAX;irw++)
      {
         if (rw[irw].rwfact!=RWFACTS)
         {
            istd[0]=(stdint_t)(irw);            
            istd[1]=(stdint_t)(rw[irw].rwfact);
            istd[2]=(stdint_t)(rw[irw].im0); 
            istd[3]=(stdint_t)(rw[irw].irp);
            istd[4]=(stdint_t)(rw[irw].n);
            istd[5]=(stdint_t)(rw[irw].nsrc);
            dstd[0]=rw[irw].mu;
            
            if (endian==BIG_ENDIAN)
            {
               bswap_int(6,istd);
               bswap_double(1,dstd);
            }
            
            iw=fwrite(istd,sizeof(stdint_t),6,fdat);
            iw+=fwrite(dstd,sizeof(double),1,fdat);

            n=rw[irw].n;

            for (l=0;l<n;l++)
            {
               istd[0]=(stdint_t)(rw[irw].np[l]); 
               istd[1]=(stdint_t)(rw[irw].isp[l]);

               if (endian==BIG_ENDIAN)
                  bswap_int(2,istd);
               
               iw+=fwrite(istd,sizeof(stdint_t),2,fdat);
            }
            
            error_root(iw!=(7+2*n),1,"write_rw_parms [rw_parms.c]",
                       "Incorrect write count");
         }
      }
   }
}


void check_rw_parms(FILE *fdat)
{
   int my_rank,endian;
   int ir,irw,n,l,ie;
   stdint_t istd[6];
   double dstd[1];

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   endian=endianness();
      
   if ((my_rank==0)&&(init==1))
   {
      ie=0;
      
      for (irw=0;irw<IRWMAX;irw++)
      {
         if (rw[irw].rwfact!=RWFACTS)
         {
            ir=fread(istd,sizeof(stdint_t),6,fdat);
            ir+=fread(dstd,sizeof(double),1,fdat);

            if (endian==BIG_ENDIAN)
            {
               bswap_int(6,istd);
               bswap_double(1,dstd);
            }
            
            ie|=(istd[0]!=(stdint_t)(irw));            
            ie|=(istd[1]!=(stdint_t)(rw[irw].rwfact));
            ie|=(istd[2]!=(stdint_t)(rw[irw].im0));
            ie|=(istd[3]!=(stdint_t)(rw[irw].irp));
            ie|=(istd[4]!=(stdint_t)(rw[irw].n));
            ie|=(istd[5]!=(stdint_t)(rw[irw].nsrc));
            ie|=(dstd[0]!=rw[irw].mu);

            n=rw[irw].n;
            
            for (l=0;l<n;l++)
            {
               ir+=fread(istd,sizeof(stdint_t),2,fdat);

               if (endian==BIG_ENDIAN)
                  bswap_int(2,istd);

               ie|=(istd[0]!=(stdint_t)(rw[irw].np[l])); 
               ie|=(istd[1]!=(stdint_t)(rw[irw].isp[l]));
            }

            error_root(ir!=(7+2*n),1,"check_rw_parms [rw_parms.c]",
                       "Incorrect read count");
         }
      }
         
      error_root(ie!=0,1,"check_rw_parms [rw_parms.c]",
                 "Parameters do not match");         
   }
}
