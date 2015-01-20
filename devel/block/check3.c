
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of assign_ud2ubgr() and assign_ud2udblk()
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "block.h"
#include "global.h"

typedef union
{
   su3 u;
   float r[18];
} umat_t;

typedef union
{
   su3_dble u;
   double r[18];
} umat_dble_t;


static void set_ud(void)
{
   int x0,x1,x2,x3,ix;
   int y0,y1,y2,y3,ifc;
   su3_dble *udb,*ud;

   random_ud();
   udb=udfld();

   for (x0=0;x0<L0;x0++)
   {
      for (x1=0;x1<L1;x1++)
      {
         for (x2=0;x2<L2;x2++)
         {
            for (x3=0;x3<L3;x3++)
            {
               if ((x0+x1+x2+x3)&0x1)
               {
                  y0=cpr[0]*L0+x0;
                  y1=cpr[1]*L1+x1;
                  y2=cpr[2]*L2+x2;
                  y3=cpr[3]*L3+x3;                  

                  ix=ipt[x3+L3*x2+L2*L3*x1+L1*L2*L3*x0];
                  ud=udb+8*(ix-(VOLUME/2));

                  for (ifc=0;ifc<8;ifc++)
                  {
                     (*ud).c11.re=(double)(y0);
                     (*ud).c11.im=(double)(y1);

                     (*ud).c22.re=(double)(y2);
                     (*ud).c22.im=(double)(y3);                     

                     (*ud).c33.re=(double)(ifc);
                     (*ud).c33.im=0.0;

                     ud+=1;
                  }
               }
            }
         }
      }
   }

   set_flags(UPDATED_UD);
}


static int check_u(int y0,int y1,int y2,int y3,int ifc,su3 *u)
{
   int ie;

   ie =((*u).c11.re!=(float)(y0));
   ie|=((*u).c11.im!=(float)(y1));   
   ie|=((*u).c22.re!=(float)(y2));   
   ie|=((*u).c22.im!=(float)(y3));   
   ie|=((*u).c33.re!=(float)(ifc));   
   ie|=((*u).c33.im!=0.0f);   
   
   return ie;
}


static int check_ublk(block_t *b)
{
   int *bo,*bs,ie;
   int x0,x1,x2,x3,ix;
   int y0,y1,y2,y3,ifc;
   su3 *ub,*u;

   bo=(*b).bo;
   bs=(*b).bs;
   ub=(*b).u;   
   ie=0;
   
   for (x0=0;x0<bs[0];x0++)
   {
      for (x1=0;x1<bs[1];x1++)
      {
         for (x2=0;x2<bs[2];x2++)
         {
            for (x3=0;x3<bs[3];x3++)
            {
               if ((x0+x1+x2+x3)&0x1)
               {
                  y0=cpr[0]*L0+bo[0]+x0;
                  y1=cpr[1]*L1+bo[1]+x1;
                  y2=cpr[2]*L2+bo[2]+x2;
                  y3=cpr[3]*L3+bo[3]+x3;       

                  ix=(*b).ipt[x3+bs[3]*x2+bs[2]*bs[3]*x1+bs[1]*bs[2]*bs[3]*x0];
                  u=ub+8*(ix-((*b).vol/2));

                  for (ifc=0;ifc<8;ifc++)
                  {
                     ie|=check_u(y0,y1,y2,y3,ifc,u);
                     u+=1;
                  }
               }
            }
         }
      }
   }

   return ie;
}


static int check_ud(int y0,int y1,int y2,int y3,int ifc,su3_dble *ud)
{
   int ie;

   ie =((*ud).c11.re!=(double)(y0));
   ie|=((*ud).c11.im!=(double)(y1));   
   ie|=((*ud).c22.re!=(double)(y2));   
   ie|=((*ud).c22.im!=(double)(y3));   
   ie|=((*ud).c33.re!=(double)(ifc));   
   ie|=((*ud).c33.im!=0.0);   
   
   return ie;
}


static int check_udblk(block_t *b)
{
   int *bo,*bs,ie;
   int x0,x1,x2,x3,ix;
   int y0,y1,y2,y3,ifc;
   su3_dble *udb,*ud;

   bo=(*b).bo;
   bs=(*b).bs;
   udb=(*b).ud;   
   ie=0;
   
   for (x0=0;x0<bs[0];x0++)
   {
      for (x1=0;x1<bs[1];x1++)
      {
         for (x2=0;x2<bs[2];x2++)
         {
            for (x3=0;x3<bs[3];x3++)
            {
               if ((x0+x1+x2+x3)&0x1)
               {
                  y0=cpr[0]*L0+bo[0]+x0;
                  y1=cpr[1]*L1+bo[1]+x1;
                  y2=cpr[2]*L2+bo[2]+x2;
                  y3=cpr[3]*L3+bo[3]+x3;       

                  ix=(*b).ipt[x3+bs[3]*x2+bs[2]*bs[3]*x1+bs[1]*bs[2]*bs[3]*x0];
                  ud=udb+8*(ix-((*b).vol/2));

                  for (ifc=0;ifc<8;ifc++)
                  {
                     ie|=check_ud(y0,y1,y2,y3,ifc,ud);
                     ud+=1;
                  }
               }
            }
         }
      }
   }

   return ie;
}


static void fnd_coord(block_t *b,int ix,int *x0,int *x1,int *x2,int *x3)
{
   int iz;

   (*x0)=0;   
   iz=(*b).idn[ix][0];

   while (iz<(*b).vol)
   {
      (*x0)+=1;
      iz=(*b).idn[iz][0];
   }

   (*x1)=0;   
   iz=(*b).idn[ix][1];

   while (iz<(*b).vol)
   {
      (*x1)+=1;
      iz=(*b).idn[iz][1];
   }

   (*x2)=0;   
   iz=(*b).idn[ix][2];

   while (iz<(*b).vol)
   {
      (*x2)+=1;
      iz=(*b).idn[iz][2];
   }   

   (*x3)=0;   
   iz=(*b).idn[ix][3];

   while (iz<(*b).vol)
   {
      (*x3)+=1;
      iz=(*b).idn[iz][3];
   }   
}


static int cmp_u(su3 *u,su3 *v)
{
   int i;
   umat_t *uu,*uv;

   uu=(umat_t*)(u);
   uv=(umat_t*)(v);

   for (i=0;i<18;i++)
   {
      if ((*uu).r[i]!=(*uv).r[i])
         return 1;
   }

   return 0;
}


static int check_ubnd(block_t *b)
{
   int *bo,ie,ifc,iz,ix;
   int x0,x1,x2,x3;
   int y0,y1,y2,y3;
   su3 *u;
   bndry_t *bb;

   bo=(*b).bo;
   bb=(*b).bb;
   ie=0;

   if ((*bb).u==NULL)
      return 0;

   for (ifc=0;ifc<8;ifc++)
   {
      u=(*bb).u;
      
      for (iz=0;iz<(*bb).vol;iz++)
      {
         ix=(*bb).ipp[iz];

         if (iz<((*bb).vol/2))
            ie|=cmp_u(u,(*b).u+8*(ix-((*b).vol/2))+(ifc^0x1));
         
         fnd_coord(b,ix,&x0,&x1,&x2,&x3);

         y0=cpr[0]*L0+bo[0]+x0;
         y1=cpr[1]*L1+bo[1]+x1;
         y2=cpr[2]*L2+bo[2]+x2;
         y3=cpr[3]*L3+bo[3]+x3;         

         if (iz<((*bb).vol/2))         
            ie|=check_u(y0,y1,y2,y3,ifc^0x1,u);         
         else
         {
            if (ifc==0)
               y0=safe_mod(y0-1,NPROC0*L0);
            if (ifc==1)
               y0=safe_mod(y0+1,NPROC0*L0);

            if (ifc==2)
               y1=safe_mod(y1-1,NPROC1*L1);
            if (ifc==3)
               y1=safe_mod(y1+1,NPROC1*L1);

            if (ifc==4)
               y2=safe_mod(y2-1,NPROC2*L2);
            if (ifc==5)
               y2=safe_mod(y2+1,NPROC2*L2);

            if (ifc==6)
               y3=safe_mod(y3-1,NPROC3*L3);
            if (ifc==7)
               y3=safe_mod(y3+1,NPROC3*L3);

            ie|=check_u(y0,y1,y2,y3,ifc,u);            
         }

         u+=1;
      }

      bb+=1;
   }

   return ie;
}


static int check_ubgr(blk_grid_t grid)
{
   int ie,nb,isw,n;
   block_t *b;

   b=blk_list(grid,&nb,&isw);
   ie=0;

   for (n=0;n<nb;n++)
   {
      ie|=check_ublk(b+n);
      ie|=check_ubnd(b+n);
   }

   return ie;
}


int main(int argc,char *argv[])
{
   int my_rank,bs[4];
   int nb,isw,n,ie;
   block_t *b;
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check1.in","r",stdin);

      printf("\n");
      printf("Check of assign_ud2ubgr() and assign_ud2udblk()\n");
      printf("-----------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      fclose(fin);

      printf("bs = %d %d %d %d\n\n",bs[0],bs[1],bs[2],bs[3]);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);

   start_ranlux(0,1234);
   geometry();
   set_sap_parms(bs,0,1,1);
   set_dfl_parms(bs,2);
   alloc_bgr(SAP_BLOCKS);
   alloc_bgr(DFL_BLOCKS);

   random_ud();
   assign_ud2ubgr(SAP_BLOCKS);
   set_ud();
   assign_ud2ubgr(SAP_BLOCKS);
   assign_ud2u();
   print_flags();
   print_grid_flags(SAP_BLOCKS);
   
   error(check_ubgr(SAP_BLOCKS),1,"main [check3.c]",
         "assign_ud2ubgr() is incorrect");      

   b=blk_list(DFL_BLOCKS,&nb,&isw);
   random_ud();
   assign_ud2udblk(DFL_BLOCKS,0);
   set_ud();
   ie=0;

   for (n=0;n<nb;n++)
   {
      assign_ud2udblk(DFL_BLOCKS,n);
      ie|=check_udblk(b+n);
   }

   print_flags();   
   print_grid_flags(DFL_BLOCKS);
   
   error(ie,1,"main [check3.c]",
         "assign_ud2udblk() is incorrect");
   error_chk();

   if (my_rank==0)
   {
      printf("No errors detected\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
