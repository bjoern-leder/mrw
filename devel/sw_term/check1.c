
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2005, 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Allocation, assignment and inversion of the global SW arrays
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "su3fcts.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "sw_term.h"
#include "global.h"

typedef union
{
   weyl_dble w;
   complex_dble c[6];
} spin_dble_t;

static pauli_dble *sswd=NULL;
static spin_dble_t vd ALIGNED32;
static const weyl_dble vd0={{{0.0}}};


static void save_swd(void)
{
   pauli_dble *pa,*pb,*pm;

   if (sswd==NULL)
   {
      sswd=amalloc(2*VOLUME*sizeof(*sswd),ALIGN);
      error(sswd==NULL,1,"save_swd [check1.c]",
            "Unable to allocate auxiliary array"); 
   }
   
   pa=swdfld();
   pb=sswd;
   pm=pa+2*VOLUME;

   for (;pa<pm;pa++)
   {
      (*pb)=(*pa);
      pb+=1;
   }
}


static double cmp_swd(ptset_t set)
{
   int k;
   double d,dmax;
   pauli_dble *pa,*pb,*pm;

   pa=swdfld();
   pb=sswd;
   pm=pa;

   if (set==EVEN_PTS)
      pm=pa+VOLUME;
   else if (set==ODD_PTS)
   {
      pa+=VOLUME;
      pb+=VOLUME;
      pm=pa+VOLUME;
   }
   else if (set==ALL_PTS)
      pm=pa+2*VOLUME;

   dmax=0.0;

   for (;pa<pm;pa++)
   {
      for (k=0;k<36;k++)
      {
         d=fabs((*pa).u[k]-(*pb).u[k]);

         if (d>dmax)
            dmax=d;
      }

      pb+=1;
   }

   return dmax;
}


static double cmp_iswd(ptset_t set)
{
   int k,l;
   double d,dmax;
   pauli_dble *pa,*pb,*pm;

   pa=swdfld();
   pb=sswd;
   pm=pa;

   if (set==EVEN_PTS)
      pm=pa+VOLUME;
   else if (set==ODD_PTS)
   {
      pa+=VOLUME;
      pb+=VOLUME;
      pm=pa+VOLUME;
   }
   else if (set==ALL_PTS)
      pm=pa+2*VOLUME;

   dmax=0.0;

   for (;pa<pm;pa++)
   {
      for (k=0;k<6;k++)
      {
         vd.w=vd0;
         vd.c[k].re=1.0;

         mul_pauli_dble(0.0,pa,&(vd.w),&(vd.w));
         mul_pauli_dble(0.0,pb,&(vd.w),&(vd.w));
         vd.c[k].re-=1.0;

         for (l=0;l<6;l++)
         {
            d=vd.c[l].re*vd.c[l].re+vd.c[l].im*vd.c[l].im;
            if (d>dmax)
               dmax=d;
         }
      }

      pb+=1;
   }

   return sqrt(dmax);
}


static double cmp_sw2swd(ptset_t set)
{
   int k;
   double d,dmax;
   pauli *pa,*pm;
   pauli_dble *pb;

   pa=swfld();
   pb=swdfld();
   pm=pa;

   if (set==EVEN_PTS)
      pm=pa+VOLUME;
   else if (set==ODD_PTS)
   {
      pa+=VOLUME;
      pb+=VOLUME;
      pm=pa+VOLUME;
   }
   else if (set==ALL_PTS)
      pm=pa+2*VOLUME;

   dmax=0.0;

   for (;pa<pm;pa++)
   {
      for (k=0;k<36;k++)
      {
         d=fabs((double)((*pa).u[k])-(*pb).u[k]);

         if (d>dmax)
            dmax=d;
      }

      pb+=1;
   }

   return dmax;
}


int main(int argc,char *argv[])
{
   int my_rank,ix,k,ifail;
   float *r;
   double *rd,d,dmax,dmax_all;
   pauli *sw;
   pauli_dble *swd;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);
      printf("\n");
      printf("Initialization and inversion of the global SW arrays\n");
      printf("----------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,123456);
   geometry();
   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.456,1.0,1.234);
   set_sw_parms(-0.0123);

   sw=swfld();
   swd=swdfld();
   dmax=0.0;

   for (ix=0;ix<(2*VOLUME);ix++)
   {
      r=sw[ix].u;

      for (k=0;k<36;k++)
      {
         if (k<6)
            d=fabs((double)(r[k]-1.0f));
         else
            d=fabs((double)(r[k]));

         if (d>dmax)
            dmax=d;
      }
   }

   MPI_Reduce(&dmax,&dmax_all,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Allocated global single-precision SW field\n");
      printf("max|p-1| = %.1e\n\n",dmax_all);
   }

   dmax=0.0;

   for (ix=0;ix<(2*VOLUME);ix++)
   {
      rd=swd[ix].u;

      for (k=0;k<36;k++)
      {
         if (k<6)
            d=fabs(rd[k]-1.0);
         else
            d=fabs(rd[k]);

         if (d>dmax)
            dmax=d;
      }
   }

   MPI_Reduce(&dmax,&dmax_all,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Allocated global double-precision SW field\n");
      printf("max|p-1| = %.1e\n\n",dmax_all);
   }

   print_flags();
   random_ud();
   sw_term(NO_PTS);
   save_swd();

   ifail=sw_term(EVEN_PTS);
   error(ifail!=0,1,"main [check1.c]","Unsafe inversion of swd_e");
   dmax=cmp_iswd(EVEN_PTS);
   MPI_Reduce(&dmax,&dmax_all,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Inverted swd_e\n");
      printf("Maximal deviation of swd_e = %.1e\n",dmax_all);
   }

   dmax=cmp_swd(ODD_PTS);
   MPI_Reduce(&dmax,&dmax_all,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
      printf("Maximal deviation of swd_o = %.1e\n\n",dmax_all);

   print_flags();
   random_ud();
   sw_term(NO_PTS);
   save_swd();

   ifail=sw_term(ODD_PTS);
   error(ifail!=0,1,"main [check1.c]","Unsafe inversion of swd_o");
   dmax=cmp_swd(EVEN_PTS);
   MPI_Reduce(&dmax,&dmax_all,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Inverted swd_o\n");
      printf("Maximal deviation of swd_e = %.1e\n",dmax_all);
   }

   dmax=cmp_iswd(ODD_PTS);
   MPI_Reduce(&dmax,&dmax_all,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
      printf("Maximal deviation of swd_o = %.1e\n\n",dmax_all);

   print_flags();
   assign_swd2sw();
   dmax=cmp_sw2swd(ALL_PTS);
   MPI_Reduce(&dmax,&dmax_all,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Assigned swd to sw\n");
      printf("Maximal deviation = %.1e\n\n",dmax_all);
   }

   print_flags();
   random_ud();
   sw_term(NO_PTS);
   save_swd();

   ifail=sw_term(ALL_PTS);
   error(ifail!=0,1,"main [check1.c]","Unsafe inversion of swd");
   dmax=cmp_iswd(ALL_PTS);
   MPI_Reduce(&dmax,&dmax_all,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Inverted swd\n");
      printf("Maximal deviation = %.1e\n\n",dmax_all);
   }

   print_flags();
   free_sw();
   free_swd();

   if (my_rank==0)
      printf("Freed sw and swd\n\n");

   print_flags();

   if (my_rank==0)
      fclose(flog);

   MPI_Finalize();
   exit(0);
}
