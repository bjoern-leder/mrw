
/*******************************************************************************
*
* File check11.c
*
* Copyright (C) 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check interpolations
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "global.h"
#include "mrw.h"

static mrwfact_t mrwfact[]={TMRW,TMRW_EO,TMRW1,TMRW1_EO,TMRW2,TMRW2_EO,
                            TMRW3,TMRW3_EO,TMRW4,TMRW4_EO,MRW,MRW_EO,MRW_ISO,MRW_TF};
static char mrwfacts[NAME_SIZE][MRWFACTS]={"TMRW","TMRW_EO","TMRW1","TMRW1_EO",
                          "TMRW2","TMRW2_EO","TMRW3","TMRW3_EO","TMRW4","TMRW4_EO",
                          "MRW","MRW_EO","MRW_ISO","MRW_TF"};

  
static void start_end(int irw,double *x0,double* x)
{
   int i;
   mrw_parms_t rw;
   mrwfact_t mrwfact;
   
   rw=mrw_parms(irw);
   mrwfact=rw.mrwfact;
   
   for (i=0;i<2;i++)
   {
      x0[i]=0.0;
      x[i]=0.0;
   }
   
   if ((mrwfact==TMRW)||(mrwfact==TMRW_EO)||(mrwfact==TMRW1)||(mrwfact==TMRW1_EO))
   {
      x0[0]=rw.mu0;
      x[0]=rw.mu;
   }
   else if ((mrwfact==TMRW2)||(mrwfact==TMRW2_EO))
   {
      x0[0]=rw.mu0;
      x[0]=rw.mu;

      x0[1]=rw.mu0;
      x[1]=sqrt(2.0*rw.mu0*rw.mu0-rw.mu*rw.mu);
   }
   else if ((mrwfact==TMRW3)||(mrwfact==TMRW3_EO))
   {
      x0[0]=rw.mu0;
      x[0]=rw.mu;

      x0[1]=-rw.mu;
      x[1]=-rw.mu0;
   }
   else if ((mrwfact==TMRW4)||(mrwfact==TMRW4_EO))
   {
      x0[0]=rw.mu0;
      x[0]=rw.mu;

      x0[1]=rw.mu;
      x[1]=rw.mu0;
   }
   else if ((mrwfact==MRW)||(mrwfact==MRW_EO))
   {
      x0[0]=rw.m0;
      x[0]=rw.m;
   }
   else if (mrwfact==MRW_ISO)
   {
      x0[0]=rw.m0;
      x[0]=rw.m;
      
      x0[1]=rw.m0;
      x[1]=2.0*rw.m0-rw.m;
   }
   else if (mrwfact==MRW_TF)
   {
      x0[0]=rw.m;
      x[0]=0.5/rw.kappa2-4.0;
      
      x0[1]=rw.m0;
      x[1]=rw.m0+rw.gamma*(rw.m-0.5/rw.kappa2+4.0);
   }
}


static double check_mrw_inter(int irw,int *ie)
{
   int l,my_rank;
   double delta1,delta2,x[2],x0[2];
   double d0,d1,d2;
   mrw_masses_t ms,ms0;
   mrw_parms_t rw;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   rw=mrw_parms(irw);

   delta1=0.0;
   delta2=0.0;         
   d0=0.0;
   d1=0.0;
   d2=0.0;
   *ie=0;

   start_end(irw,x0,x);

   ms0=get_mrw_masses(irw,0);
   ms=ms0;
   
/*
   printf("\nx0_1 %.16f\n",x0[0]);
   printf("x_1  %.16f\n",x[0]);
   printf("x0_2 %.16f\n",x0[1]);
   printf("x_2  %.16f\n",x[1]);
   
   printf("\nm1  %.16f\n",ms0.m1);
   printf("mu1 %.16f\n",ms0.mu1);
   printf("d1  %.16f\n",ms0.d1);
   printf("m2  %.16f\n",ms0.m2);
   printf("mu2 %.16f\n",ms0.mu2);
   printf("d2  %.16f\n",ms0.d2);
*/         
   delta1+=ms0.d1;
   delta2+=ms0.d2;         
   
   if (rw.mrwfact<MRW)
   {
      d0+=fabs(x[0]-ms0.mu1);
      if ((rw.mrwfact==TMRW1)||(rw.mrwfact==TMRW2)||(rw.mrwfact==TMRW4)||
          (rw.mrwfact==TMRW1_EO)||(rw.mrwfact==TMRW2_EO)||(rw.mrwfact==TMRW4_EO))
         d0+=fabs(x0[1]*x0[1]-ms0.mu2*ms0.mu2-ms0.d2);
      else
         d0+=fabs(x0[1]-ms0.mu2-ms0.d2);
   }
   else
   {
      d0+=fabs(x[0]-ms0.m1);
      d0+=fabs(x0[1]-ms0.m2-ms0.d2);
   }
   
   for (l=1;l<rw.nm;l++)
   {
      ms=get_mrw_masses(irw,l);
/*            
      printf("\nm1  %.16f\n",ms.m1);
      printf("mu1 %.16f\n",ms.mu1);
      printf("d1  %.16f\n",ms.d1);
      printf("m2  %.16f\n",ms.m2);
      printf("mu2 %.16f\n",ms.mu2);
      printf("d2  %.16f\n",ms.d2);
*/
      if (rw.mrwfact<MRW)
      {
         if ((rw.mrwfact==TMRW1)||(rw.mrwfact==TMRW2)||(rw.mrwfact==TMRW4)||
            (rw.mrwfact==TMRW1_EO)||(rw.mrwfact==TMRW2_EO)||(rw.mrwfact==TMRW4_EO))
         {
            d1+=fabs(ms.mu1*ms.mu1-ms0.mu1*ms0.mu1-ms0.d1);
            d1+=fabs(ms0.mu2*ms0.mu2-ms.mu2*ms.mu2-ms.d2);
         }
         else
         {
            d1+=fabs(ms.mu1-ms0.mu1-ms0.d1);
            d1+=fabs(ms0.mu2-ms.mu2-ms.d2);
         }
         d1+=fabs(ms0.m1-ms.m1);
         d1+=fabs(ms0.m2-ms.m2);
      }
      else
      {
         d1+=fabs(ms.m1-ms0.m1-ms0.d1);
         d1+=fabs(ms0.m2-ms.m2-ms.d2);
         d1+=fabs(ms0.mu1-ms.mu1);
         d1+=fabs(ms0.mu2-ms.mu2);
      }
      
      if (rw.pwr==0)
         (*ie)|=(ms0.d1!=ms.d1);
      if (rw.pwr>0)
         (*ie)|=(fabs(ms0.d1)>=fabs(ms.d1));
      
      delta1+=ms.d1;
      delta2+=ms.d2;         
      
      ms0=ms;
   }
   
   if (rw.mrwfact<MRW)
   {
      if ((rw.mrwfact==TMRW1)||(rw.mrwfact==TMRW2)||(rw.mrwfact==TMRW4)||
          (rw.mrwfact==TMRW1_EO)||(rw.mrwfact==TMRW2_EO)||(rw.mrwfact==TMRW4_EO))
         d0+=fabs(x0[0]*x0[0]-ms.mu1*ms.mu1-ms.d1);
      else
         d0+=fabs(x0[0]-ms.mu1-ms.d1);
      d0+=fabs(x[1]-ms.mu2);
   }
   else
   {
      d0+=fabs(x0[0]-ms.m1-ms.d1);
      d0+=fabs(x[1]-ms.m2);
   }

   if ((rw.mrwfact==TMRW1)||(rw.mrwfact==TMRW2)||(rw.mrwfact==TMRW4)||
         (rw.mrwfact==TMRW1_EO)||(rw.mrwfact==TMRW2_EO)||(rw.mrwfact==TMRW4_EO))
   {
      d2+=fabs(delta1-(x0[0]*x0[0]-x[0]*x[0]));
      d2+=fabs(delta2-(x0[1]*x0[1]-x[1]*x[1]));
   }
   else
   {
      d2+=fabs(delta1-(x0[0]-x[0]));
      d2+=fabs(delta2-(x0[1]-x[1]));
   }
      
   if (my_rank==0)
   {          
      if (*ie)
         printf("\nReweighting distances not constant (p=0) or increasing (p>0).\n");
      if (d0>DBL_EPSILON)
      {
         (*ie)=1;
         printf("\nStart/end not correct, (diff: %.1e)\n",d0);
      }
      if (d1>sqrt((double)(4*5*rw.nm))*DBL_EPSILON)
      {
         (*ie)=1;
         printf("\nReweighting distances not correct, (diff: %.1e)\n",d1);
      }
      if (d2>sqrt((double)(2*rw.nm))*DBL_EPSILON)
      {
         (*ie)=1;
         printf("\nReweighting distances do not sum up correctly, (diff: %.1e)\n",d2);
      }
   } 
   
   return (d0+d1+d2);
}


int main(int argc,char *argv[])
{
   int my_rank,irw,nm,ie,iea,p,nrw,nmx;
   double kappa0,kappa,kappa2,mu,mu0,gamma;
   double d,dmx;
   FILE *flog=NULL,*fin=NULL;
   mrw_parms_t rw;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check11.log","w",stdout);
      fin=freopen("check11.in","r",stdin);
      
      printf("\n");
      printf("Check interpolations\n");
      printf("----------------------\n\n");
   }
   
   if (my_rank==0)
   {
      find_section("Configurations");
      read_line("nrw","%d",&nrw);
   }

   MPI_Bcast(&nrw,1,MPI_INT,0,MPI_COMM_WORLD);   

   for (irw=0;irw<nrw;irw++)
      read_mrw_parms(irw);
      
   if (my_rank==0)
      fclose(fin);

   print_mrw_parms();

   dmx=0.0;
   nmx=0;
   iea=0;

   for (irw=0;irw<nrw;irw++)
   {
      rw=mrw_parms(irw);
      
      if (my_rank==0)
      {            
         printf("Reweighting factor %d, %s",irw,mrwfacts[rw.mrwfact]);
      }   

      d=check_mrw_inter(irw,&ie);
      
      if (d>dmx)
         dmx=d;
      
      if (rw.nm>nmx)
         nmx=rw.nm;
      
      iea+=ie;

      if (my_rank==0)
         printf("\n");
   }

   if (my_rank==0)
   {
      printf("\n");
      if (iea)
         printf("%d errors found\n",iea);
      else
         printf("all ok\n");
      printf("max diff = %.1e\n",dmx);
      printf("(should be smaller than %.1e)\n\n",sqrt((double)(4*(4*5+2)*nmx))*DBL_EPSILON);
   }
   
   if (my_rank==0)
      printf("Generic test of all types and powers:\n");

   kappa0=0.1315;
   kappa=0.131;
   mu0=0.05;
   mu=0.0;
   gamma=0.8;
   kappa2=0.1316;
   nm=24;
   
   dmx=0.0;
   iea=0;
  
   for (irw=0;irw<MRWFACTS;irw++)
   {
      if (my_rank==0)
      {            
         printf("%s, p: ",mrwfacts[irw]);
      }   
      
      for (p=0;p<=4;p++)
      {
         if (my_rank==0)
         {            
            printf("%d ",p);
         }   

         init_mrw();
         set_mrw_parms(irw,mrwfact[irw],kappa0,kappa,mu0,mu,gamma,kappa2,0,0,nm,p,1,0);
         
         d=check_mrw_inter(irw,&ie);
         
         if (d>dmx)
            dmx=d;
         
         iea+=ie;
      }

      if (my_rank==0)
         printf("\n");
   }
   
   if (my_rank==0)
   {
      printf("\n");
      if (iea)
         printf("%d errors found\n",iea);
      else
         printf("all ok\n");
      printf("max diff = %.1e\n",dmx);
      printf("(should be smaller than %.1e)\n\n",sqrt((double)(4*(4*5+2)*nm))*DBL_EPSILON);
      fclose(flog);
   }
   
   MPI_Finalize();    
   exit(0);
}
