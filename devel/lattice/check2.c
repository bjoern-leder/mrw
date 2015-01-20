
/*******************************************************************************
*
* File check2.c
*
* Copyright (C) 2010, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Check of openbc() and openbcd()
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
#include "su3fcts.h"
#include "utils.h"
#include "uflds.h"
#include "lattice.h"
#include "global.h"

#define N0 (NPROC0*L0)


static void new_flds(void)
{
   su3 *u,*um;
   su3_dble *ud,*udm;

   u=ufld();
   um=u+4*VOLUME;

   for (;u<um;u++)
      random_su3(u);

   ud=udfld();
   udm=ud+4*VOLUME;

   for (;ud<udm;ud++)
      random_su3_dble(ud);   

   set_flags(UPDATED_U);
   set_flags(UPDATED_UD);
}

   
static int is_zero(su3 *u)
{
   int i;
   float r[18];

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

   for (i=0;i<18;i++)
   {
      if (r[i]!=0.0f)
         return 0;
   }

   return 1;
}


static int is_zero_dble(su3_dble *ud)
{
   int i;
   double r[18];

   r[ 0]=(*ud).c11.re;
   r[ 1]=(*ud).c11.im;
   r[ 2]=(*ud).c12.re;
   r[ 3]=(*ud).c12.im;
   r[ 4]=(*ud).c13.re;
   r[ 5]=(*ud).c13.im;

   r[ 6]=(*ud).c21.re;
   r[ 7]=(*ud).c21.im;
   r[ 8]=(*ud).c22.re;
   r[ 9]=(*ud).c22.im;
   r[10]=(*ud).c23.re;
   r[11]=(*ud).c23.im;

   r[12]=(*ud).c31.re;
   r[13]=(*ud).c31.im;
   r[14]=(*ud).c32.re;
   r[15]=(*ud).c32.im;
   r[16]=(*ud).c33.re;
   r[17]=(*ud).c33.im;

   for (i=0;i<18;i++)
   {
      if (r[i]!=0.0)
         return 0;
   }

   return 1;
}


static void save_u(su3 *usv)
{
   su3 *u,*um,*v;

   u=ufld();
   um=u+4*VOLUME;
   v=usv;

   for (;u<um;u++)
   {
      *v=*u;
      v+=1;
   }
}


static void save_ud(su3_dble *udsv)
{
   su3_dble *ud,*udm,*vd;

   ud=udfld();
   udm=ud+4*VOLUME;
   vd=udsv;

   for (;ud<udm;ud++)
   {
      *vd=*ud;
      vd+=1;
   }
}


static int cmp_u(su3 *u,su3 *v)
{
   int i;
   float r[18];

   r[ 0]=(*u).c11.re-(*v).c11.re;
   r[ 1]=(*u).c11.im-(*v).c11.im;
   r[ 2]=(*u).c12.re-(*v).c12.re;
   r[ 3]=(*u).c12.im-(*v).c12.im;
   r[ 4]=(*u).c13.re-(*v).c13.re;
   r[ 5]=(*u).c13.im-(*v).c13.im;

   r[ 6]=(*u).c21.re-(*v).c21.re;
   r[ 7]=(*u).c21.im-(*v).c21.im;
   r[ 8]=(*u).c22.re-(*v).c22.re;
   r[ 9]=(*u).c22.im-(*v).c22.im;
   r[10]=(*u).c23.re-(*v).c23.re;
   r[11]=(*u).c23.im-(*v).c23.im;

   r[12]=(*u).c31.re-(*v).c31.re;
   r[13]=(*u).c31.im-(*v).c31.im;
   r[14]=(*u).c32.re-(*v).c32.re;
   r[15]=(*u).c32.im-(*v).c32.im;
   r[16]=(*u).c33.re-(*v).c33.re;
   r[17]=(*u).c33.im-(*v).c33.im;   

   for (i=0;i<18;i++)
   {
      if (r[i]!=0.0f)
         return 1;
   }

   return 0;
}


static int cmp_ud(su3_dble *u,su3_dble *v)
{
   int i;
   double r[18];

   r[ 0]=(*u).c11.re-(*v).c11.re;
   r[ 1]=(*u).c11.im-(*v).c11.im;
   r[ 2]=(*u).c12.re-(*v).c12.re;
   r[ 3]=(*u).c12.im-(*v).c12.im;
   r[ 4]=(*u).c13.re-(*v).c13.re;
   r[ 5]=(*u).c13.im-(*v).c13.im;

   r[ 6]=(*u).c21.re-(*v).c21.re;
   r[ 7]=(*u).c21.im-(*v).c21.im;
   r[ 8]=(*u).c22.re-(*v).c22.re;
   r[ 9]=(*u).c22.im-(*v).c22.im;
   r[10]=(*u).c23.re-(*v).c23.re;
   r[11]=(*u).c23.im-(*v).c23.im;

   r[12]=(*u).c31.re-(*v).c31.re;
   r[13]=(*u).c31.im-(*v).c31.im;
   r[14]=(*u).c32.re-(*v).c32.re;
   r[15]=(*u).c32.im-(*v).c32.im;
   r[16]=(*u).c33.re-(*v).c33.re;
   r[17]=(*u).c33.im-(*v).c33.im;   

   for (i=0;i<18;i++)
   {
      if (r[i]!=0.0)
         return 1;
   }

   return 0;
}


static void cmp_flds(int my_rank,su3 *usv,su3_dble *udsv,int *iu,int *iud)
{
   int x0,x1,x2,x3,x[4];
   int ip,ix,mu,it,itd;
   su3 *ub,*u,*v;
   su3_dble *udb,*ud,*vd;

   ub=ufld();
   udb=udfld();
   it=0;
   itd=0;

   for (x0=0;x0<(L0*NPROC0);x0++)
   {
      for (x1=0;x1<(L1*NPROC1);x1++)
      {
         for (x2=0;x2<(L2*NPROC2);x2++)
         {
            for (x3=0;x3<(L3*NPROC3);x3++)
            {
               x[0]=x0;
               x[1]=x1;
               x[2]=x2;
               x[3]=x3;

               ipt_global(x,&ip,&ix);

               if ((my_rank==ip)&&(ix>=(VOLUME/2)))
               {
                  u=ub+8*(ix-(VOLUME/2));
                  v=usv+8*(ix-(VOLUME/2));

                  ud=udb+8*(ix-(VOLUME/2));
                  vd=udsv+8*(ix-(VOLUME/2));

                  for (mu=0;mu<4;mu++)
                  {
                     if ((mu==0)&&(x0==(L0*NPROC0-1)))
                     {
                        it+=(is_zero(u)==0);
                        itd+=(is_zero_dble(ud)==0);
                     }
                     else
                     {
                        it+=cmp_u(u,v);
                        itd+=cmp_ud(ud,vd);
                     }

                     u+=1;
                     v+=1;
                     ud+=1;
                     vd+=1;

                     if ((mu==0)&&(x0==0))
                     {
                        it+=(is_zero(u)==0);
                        itd+=(is_zero_dble(ud)==0);
                     }
                     else
                     {
                        it+=cmp_u(u,v);
                        itd+=cmp_ud(ud,vd);
                     }

                     u+=1;
                     v+=1;
                     ud+=1;
                     vd+=1;
                  }
               }
            }
         }
      }
   }

   (*iu)=it;
   (*iud)=itd;
}
   


int main(int argc,char *argv[])
{
   int my_rank,iu,iud;
   double phi[2],phi_prime[2];
   su3 *usv;
   su3_dble *udsv;
   FILE *flog=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check2.log","w",stdout);
      printf("\n");
      printf("Check of openbc(), openbcd(), sfbc() and sfbcd()\n");
      printf("------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);
   }

   start_ranlux(0,12345);
   geometry();

   usv=amalloc(4*VOLUME*sizeof(*usv),ALIGN);
   udsv=amalloc(4*VOLUME*sizeof(*udsv),ALIGN);
   error((usv==NULL)||(udsv==NULL),1,"main [check2.c]",
         "Unable to allocate auxiliary arrays");

   new_flds();
   save_u(usv);
   save_ud(udsv);

   if (my_rank==0)
   {
      printf("Before imposing open boundary conditions:\n");
      print_flags();
   }
      
   openbc();
   openbcd();

   if (my_rank==0)
   {
      printf("After imposing open boundary conditions:\n");
      print_flags();
   }

   cmp_flds(my_rank,usv,udsv,&iu,&iud);
      
   error(iu!=0,1,"main [check2.c]","Action of openbc() is incorrect");
   error(iud!=0,1,"main [check2.c]","Action of openbcd() is incorrect");

   iu=check_bc();
   iud=check_bcd();

   error(iu!=1,1,"main [check2.c]","Action of openbc() is incorrect");
   error(iud!=1,1,"main [check2.c]","Action of openbcd() is incorrect");   

   new_flds();
   save_u(usv);
   save_ud(udsv);

   if (my_rank==0)
   {
      printf("Before imposing SF boundary conditions:\n");
      print_flags();
   }

   phi[0]=0.1;
   phi[1]=0.2;
   phi_prime[0]=-0.3;
   phi_prime[1]=0.4;
   set_sf_parms(phi,phi_prime);
   
   sfbc();
   sfbcd();

   if (my_rank==0)
   {
      printf("After imposing SF boundary conditions:\n");
      print_flags();
   }

   iu=check_sfbc();
   iud=check_sfbcd();

   error(iu!=1,1,"main [check2.c]","Action of sfbc() is incorrect");
   error(iud!=1,1,"main [check2.c]","Action of sfbcd() is incorrect");   
   
   error_chk();

   if (my_rank==0)
   {
      printf("No errors detected --- all programs work correctly\n\n");  
      fclose(flog);
   }
   
   MPI_Finalize();
   exit(0);
}
