
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2009, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Initialization of the link variables
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "random.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "global.h"

#define N0 (NPROC0*L0)


static complex det(su3 *u)
{
   complex det1,det2,det3,detu;

   det1.re=
      ((*u).c22.re*(*u).c33.re-(*u).c22.im*(*u).c33.im)-
      ((*u).c23.re*(*u).c32.re-(*u).c23.im*(*u).c32.im);
   det1.im=
      ((*u).c22.re*(*u).c33.im+(*u).c22.im*(*u).c33.re)-
      ((*u).c23.re*(*u).c32.im+(*u).c23.im*(*u).c32.re);
   det2.re=
      ((*u).c21.re*(*u).c33.re-(*u).c21.im*(*u).c33.im)-
      ((*u).c23.re*(*u).c31.re-(*u).c23.im*(*u).c31.im);
   det2.im=
      ((*u).c21.re*(*u).c33.im+(*u).c21.im*(*u).c33.re)-
      ((*u).c23.re*(*u).c31.im+(*u).c23.im*(*u).c31.re);
   det3.re=
      ((*u).c21.re*(*u).c32.re-(*u).c21.im*(*u).c32.im)-
      ((*u).c22.re*(*u).c31.re-(*u).c22.im*(*u).c31.im);
   det3.im=
      ((*u).c21.re*(*u).c32.im+(*u).c21.im*(*u).c32.re)-
      ((*u).c22.re*(*u).c31.im+(*u).c22.im*(*u).c31.re);

   detu.re=
      ((*u).c11.re*det1.re-(*u).c11.im*det1.im)-
      ((*u).c12.re*det2.re-(*u).c12.im*det2.im)+
      ((*u).c13.re*det3.re-(*u).c13.im*det3.im);
   detu.im=
      ((*u).c11.re*det1.im+(*u).c11.im*det1.re)-
      ((*u).c12.re*det2.im+(*u).c12.im*det2.re)+
      ((*u).c13.re*det3.im+(*u).c13.im*det3.re);

   return detu;
}


static complex_dble det_dble(su3_dble *u)
{
   complex_dble det1,det2,det3,detu;

   det1.re=
      ((*u).c22.re*(*u).c33.re-(*u).c22.im*(*u).c33.im)-
      ((*u).c23.re*(*u).c32.re-(*u).c23.im*(*u).c32.im);
   det1.im=
      ((*u).c22.re*(*u).c33.im+(*u).c22.im*(*u).c33.re)-
      ((*u).c23.re*(*u).c32.im+(*u).c23.im*(*u).c32.re);
   det2.re=
      ((*u).c21.re*(*u).c33.re-(*u).c21.im*(*u).c33.im)-
      ((*u).c23.re*(*u).c31.re-(*u).c23.im*(*u).c31.im);
   det2.im=
      ((*u).c21.re*(*u).c33.im+(*u).c21.im*(*u).c33.re)-
      ((*u).c23.re*(*u).c31.im+(*u).c23.im*(*u).c31.re);
   det3.re=
      ((*u).c21.re*(*u).c32.re-(*u).c21.im*(*u).c32.im)-
      ((*u).c22.re*(*u).c31.re-(*u).c22.im*(*u).c31.im);
   det3.im=
      ((*u).c21.re*(*u).c32.im+(*u).c21.im*(*u).c32.re)-
      ((*u).c22.re*(*u).c31.im+(*u).c22.im*(*u).c31.re);

   detu.re=
      ((*u).c11.re*det1.re-(*u).c11.im*det1.im)-
      ((*u).c12.re*det2.re-(*u).c12.im*det2.im)+
      ((*u).c13.re*det3.re-(*u).c13.im*det3.im);
   detu.im=
      ((*u).c11.re*det1.im+(*u).c11.im*det1.re)-
      ((*u).c12.re*det2.im+(*u).c12.im*det2.re)+
      ((*u).c13.re*det3.im+(*u).c13.im*det3.re);

   return detu;
}


static float dev_unity(su3 *u)
{
   int i;
   float r[18],d,dmax;

   r[ 0]=(*u).c11.re-1.0f;
   r[ 1]=(*u).c11.im;
   r[ 2]=(*u).c12.re;
   r[ 3]=(*u).c12.im;
   r[ 4]=(*u).c13.re;
   r[ 5]=(*u).c13.im;

   r[ 6]=(*u).c21.re;
   r[ 7]=(*u).c21.im;
   r[ 8]=(*u).c22.re-1.0f;
   r[ 9]=(*u).c22.im;
   r[10]=(*u).c23.re;
   r[11]=(*u).c23.im;

   r[12]=(*u).c31.re;
   r[13]=(*u).c31.im;
   r[14]=(*u).c32.re;
   r[15]=(*u).c32.im;
   r[16]=(*u).c33.re-1.0f;
   r[17]=(*u).c33.im;

   dmax=0.0f;

   for (i=0;i<18;i++)
   {
      d=(float)fabs((double)(r[i]));
      if (d>dmax)
         dmax=d;
   }

   return dmax;
}


static double dev_unity_dble(su3_dble *u)
{
   int i;
   double r[18],d,dmax;

   r[ 0]=(*u).c11.re-1.0;
   r[ 1]=(*u).c11.im;
   r[ 2]=(*u).c12.re;
   r[ 3]=(*u).c12.im;
   r[ 4]=(*u).c13.re;
   r[ 5]=(*u).c13.im;

   r[ 6]=(*u).c21.re;
   r[ 7]=(*u).c21.im;
   r[ 8]=(*u).c22.re-1.0;
   r[ 9]=(*u).c22.im;
   r[10]=(*u).c23.re;
   r[11]=(*u).c23.im;

   r[12]=(*u).c31.re;
   r[13]=(*u).c31.im;
   r[14]=(*u).c32.re;
   r[15]=(*u).c32.im;
   r[16]=(*u).c33.re-1.0;
   r[17]=(*u).c33.im;

   dmax=0.0;

   for (i=0;i<18;i++)
   {
      d=fabs(r[i]);
      if (d>dmax)
         dmax=d;
   }

   return dmax;
}


static float dev_zero(su3 *u)
{
   int i;
   float r[18],d,dmax;

   r[ 0]=(*u).c11.re;
   r[ 1]=(*u).c11.im;
   r[ 2]=(*u).c12.re;
   r[ 3]=(*u).c12.im;
   r[ 4]=(*u).c13.re;
   r[ 5]=(*u).c13.im;

   r[ 6]=(*u).c21.re;
   r[ 7]=(*u).c21.im;
   r[ 8]=(*u).c22.re;
   r[ 9]=(*u).c22.im;
   r[10]=(*u).c23.re;
   r[11]=(*u).c23.im;

   r[12]=(*u).c31.re;
   r[13]=(*u).c31.im;
   r[14]=(*u).c32.re;
   r[15]=(*u).c32.im;
   r[16]=(*u).c33.re;
   r[17]=(*u).c33.im;

   dmax=0.0f;

   for (i=0;i<18;i++)
   {
      d=(float)fabs((double)(r[i]));
      if (d>dmax)
         dmax=d;
   }

   return dmax;
}


static double dev_zero_dble(su3_dble *u)
{
   int i;
   double r[18],d,dmax;

   r[ 0]=(*u).c11.re;
   r[ 1]=(*u).c11.im;
   r[ 2]=(*u).c12.re;
   r[ 3]=(*u).c12.im;
   r[ 4]=(*u).c13.re;
   r[ 5]=(*u).c13.im;

   r[ 6]=(*u).c21.re;
   r[ 7]=(*u).c21.im;
   r[ 8]=(*u).c22.re;
   r[ 9]=(*u).c22.im;
   r[10]=(*u).c23.re;
   r[11]=(*u).c23.im;

   r[12]=(*u).c31.re;
   r[13]=(*u).c31.im;
   r[14]=(*u).c32.re;
   r[15]=(*u).c32.im;
   r[16]=(*u).c33.re;
   r[17]=(*u).c33.im;

   dmax=0.0;

   for (i=0;i<18;i++)
   {
      d=fabs(r[i]);
      if (d>dmax)
         dmax=d;
   }

   return dmax;
}


static float dev_uudag(su3 *u)
{
   su3 udag,w;

   _su3_dagger(udag,(*u));
   _su3_times_su3(w,(*u),udag);

   return dev_unity(&w);
}


static double dev_uudag_dble(su3_dble *u)
{
   su3_dble udag,w;

   _su3_dagger(udag,(*u));
   _su3_times_su3(w,(*u),udag);

   return dev_unity_dble(&w);
}


static float dev_detu(su3 *u)
{
   float d,dmax;
   complex detu;

   detu=det(u);
   dmax=0.0f;

   d=(float)fabs((double)(1.0f-detu.re));
   if (d>dmax)
      dmax=d;
   d=(float)fabs((double)(detu.im));
   if (d>dmax)
      dmax=d;

   return dmax;
}


static double dev_detu_dble(su3_dble *u)
{
   double d,dmax;
   complex_dble detu;

   detu=det_dble(u);
   dmax=0.0;

   d=fabs(1.0-detu.re);
   if (d>dmax)
      dmax=d;
   d=fabs(detu.im);
   if (d>dmax)
      dmax=d;

   return dmax;
}


static double dev_udu_dble(su3_dble *ud,su3 *u)
{
   int i;
   double r[18],d,dmax;

   r[ 0]=(*ud).c11.re-(double)((*u).c11.re);
   r[ 1]=(*ud).c11.im-(double)((*u).c11.im);
   r[ 2]=(*ud).c12.re-(double)((*u).c12.re);
   r[ 3]=(*ud).c12.im-(double)((*u).c12.im);
   r[ 4]=(*ud).c13.re-(double)((*u).c13.re);
   r[ 5]=(*ud).c13.im-(double)((*u).c13.im);
   r[ 6]=(*ud).c21.re-(double)((*u).c21.re);
   r[ 7]=(*ud).c21.im-(double)((*u).c21.im);
   r[ 8]=(*ud).c22.re-(double)((*u).c22.re);
   r[ 9]=(*ud).c22.im-(double)((*u).c22.im);
   r[10]=(*ud).c23.re-(double)((*u).c23.re);
   r[11]=(*ud).c23.im-(double)((*u).c23.im);
   r[12]=(*ud).c31.re-(double)((*u).c31.re);
   r[13]=(*ud).c31.im-(double)((*u).c31.im);
   r[14]=(*ud).c32.re-(double)((*u).c32.re);
   r[15]=(*ud).c32.im-(double)((*u).c32.im);
   r[16]=(*ud).c33.re-(double)((*u).c33.re);
   r[17]=(*ud).c33.im-(double)((*u).c33.im);

   dmax=0.0;

   for (i=0;i<18;i++)
   {
      d=fabs(r[i]);
      if (d>dmax)
         dmax=d;
   }

   return dmax;
}


int main(int argc,char *argv[])
{
   int my_rank,iu,ix,ifc,x0;
   float d1,d2,dmax1,dmax2;
   float dmax1_all,dmax2_all;
   su3 *u,*ub,*um;
   su3_dble *ud,*udb,*udm;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);

      printf("\n");
      printf("Initialization of the link variables\n");
      printf("------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,123456);
   geometry();

   ub=ufld();
   um=ub+4*VOLUME;
   dmax1=0.0f;

   for (u=ub;u<um;u++)
   {
      iu=(u-ub);
      ix=iu/8+VOLUME/2;
      ifc=iu%8;
      x0=global_time(ix);
      
      if (((x0==0)&&(ifc==1))||((x0==(N0-1))&&(ifc==0)))
         d1=dev_zero(u);
      else
         d1=dev_unity(u);

      if (d1>dmax1)
         dmax1=d1;
   }

   MPI_Reduce(&dmax1,&dmax1_all,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Allocate single-precision gauge field\n");
      printf("|u-1| = %.2e\n\n",dmax1_all);
   }

   print_flags();

   random_u();
   dmax1=0.0f;
   dmax2=0.0f;

   for (u=ub;u<um;u++)
   {
      iu=(u-ub);
      ix=iu/8+VOLUME/2;
      ifc=iu%8;
      x0=global_time(ix);
      
      if (((x0==0)&&(ifc==1))||((x0==(N0-1))&&(ifc==0)))
      {
         d1=dev_zero(u);
         d2=d1;
      }
      else
      {
         d1=dev_uudag(u);
         d2=dev_detu(u);
      }

      if (d1>dmax1)
         dmax1=d1;
      if (d2>dmax2)
         dmax2=d2;
   }

   MPI_Reduce(&dmax1,&dmax1_all,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&dmax2,&dmax2_all,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Call random_u\n");
      printf("|u^dag*u-1| = %.2e\n",dmax1_all);
      printf("|det{u}-1| = %.2e\n\n",dmax2_all);
   }

   print_flags();

   udb=udfld();
   udm=udb+4*VOLUME;
   dmax1=0.0f;

   for (ud=udb;ud<udm;ud++)
   {
      iu=(ud-udb);
      ix=iu/8+VOLUME/2;
      ifc=iu%8;
      x0=global_time(ix);
      
      if (((x0==0)&&(ifc==1))||((x0==(N0-1))&&(ifc==0)))
         d1=(float)(dev_zero_dble(ud));
      else
         d1=(float)(dev_unity_dble(ud));

      if (d1>dmax1)
         dmax1=d1;
   }

   MPI_Reduce(&dmax1,&dmax1_all,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Allocate double-precision gauge field\n");
      printf("|ud-1| = %.2e\n\n",dmax1_all);
   }

   print_flags();

   random_ud();
   random_u();
   assign_ud2u();

   ud=udb;
   udm=udb+4*VOLUME;
   u=ub;
   dmax1=0.0f;

   for (ud=udb;ud<udm;ud++)
   {
      d1=(float)(dev_udu_dble(ud,u));      

      if (d1>dmax1)
         dmax1=d1;

      u+=1;
   }

   MPI_Reduce(&dmax1,&dmax1_all,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Random fields\n");
      printf("Assign double-precision to single-precision field\n");
      printf("Maximal deviation = %.2e\n\n",dmax1_all);
   }

   print_flags();

   random_ud();
   dmax1=0.0f;
   dmax2=0.0f;

   for (ud=udb;ud<udm;ud++)
   {
      iu=(ud-udb);
      ix=iu/8+VOLUME/2;
      ifc=iu%8;
      x0=global_time(ix);
      
      if (((x0==0)&&(ifc==1))||((x0==(N0-1))&&(ifc==0)))
      {
         d1=(float)(dev_zero_dble(ud));
         d2=d1;
      }
      else
      {
         d1=(float)(dev_uudag_dble(ud));
         d2=(float)(dev_detu_dble(ud));
      }

      if (d1>dmax1)
         dmax1=d1;
      if (d2>dmax2)
         dmax2=d2;
   }

   MPI_Reduce(&dmax1,&dmax1_all,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);
   MPI_Reduce(&dmax2,&dmax2_all,1,MPI_FLOAT,MPI_MAX,0,MPI_COMM_WORLD);

   if (my_rank==0)
   {
      printf("Call random_ud\n");
      printf("|u^dag*u-1| = %.2e\n",dmax1_all);
      printf("|det{u}-1| = %.2e\n\n",dmax2_all);
   }

   print_flags();
   
   if (my_rank==0)
      fclose(flog);
   MPI_Finalize();
   exit(0);
}
