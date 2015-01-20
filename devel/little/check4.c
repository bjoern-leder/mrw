
/*******************************************************************************
*
* File check4.c
*
* Copyright (C) 2007, 2011, 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Consistency checks on Aw_dble(),..,Awhat()
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
#include "sflds.h"
#include "vflds.h"
#include "linalg.h"
#include "dirac.h"
#include "dfl.h"
#include "little.h"
#include "global.h"


static void random_basis(int Ns)
{
   int i;
   spinor **ws;

   ws=reserve_ws(Ns);

   for (i=0;i<Ns;i++)
   {
      random_s(VOLUME,ws[i],1.0f);
      bnd_s2zero(ALL_PTS,ws[i]);
   }
   
   dfl_subspace(ws);
   release_ws();
}


int main(int argc,char *argv[])
{
   int my_rank,iop,ifail;
   int bs[4],Ns,nb,nv,nvh;   
   double mu,d;
   complex **wv,z;
   complex_dble **wvd,zd;
   void (*op)(complex *v,complex *w);   
   void (*op_dble)(complex_dble *v,complex_dble *w);
   char *pr,*prd;
   FILE *fin=NULL,*flog=NULL;   

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check4.log","w",stdout);
      fin=freopen("check3.in","r",stdin);
      
      printf("\n");
      printf("Consistency checks on Aw_dble(),..,Awhat()\n");
      printf("------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("bs","%d %d %d %d",&bs[0],&bs[1],&bs[2],&bs[3]);
      read_line("Ns","%d",&Ns);
      fclose(fin);

      printf("bs = %d %d %d %d\n",bs[0],bs[1],bs[2],bs[3]);
      printf("Ns = %d\n\n",Ns);
      fflush(flog);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);   

   set_lat_parms(5.5,1.0,0.0,0.0,0.0,0.456,1.0,1.234);
   set_sw_parms(0.125);
   set_dfl_parms(bs,Ns);   
   mu=0.0376;
   
   start_ranlux(0,123456);   
   geometry();

   alloc_ws(Ns);
   alloc_wv(4);
   alloc_wvd(6);

   wv=reserve_wv(4);
   wvd=reserve_wvd(4);
   nb=VOLUME/(bs[0]*bs[1]*bs[2]*bs[3]);
   nv=Ns*nb;
   nvh=nv/2;
   
   random_ud();
   random_basis(Ns);

   ifail=set_Awhat(mu);
   error(ifail!=0,1,"main [check4.c]","Inversion of Aee or Aoo failed");

   zd.re=-1.0;
   zd.im=0.0;
   z.re=-1.0f;
   z.im=0.0f;

   for (iop=0;iop<6;iop++)
   {
      if (iop==0)
      {
         op=Awhat;
         pr= "Awhat()  ";
         op_dble=Awhat_dble;
         prd="Awhat_dble()  ";
      }
      else if (iop==1)
      {
         op=Aweeinv;
         pr= "Aweeinv()";         
         op_dble=Aweeinv_dble;
         prd="Aweeinv_dble()";
      }
      else if (iop==2)
      {
         op=Awooinv;
         pr= "Awooinv()";         
         op_dble=Awooinv_dble;
         prd="Awooinv_dble()";
      }
      else if (iop==3)
      {
         op=Awoe;
         pr= "Awoe()   ";         
         op_dble=Awoe_dble;
         prd="Awoe_dble()   ";
      }      
      else if (iop==4)
      {
         op=Aweo;
         pr= "Aweo()   ";         
         op_dble=Aweo_dble;
         prd="Aweo_dble()   ";
      }      
      else
      {
         op=Aw;
         pr= "Aw()     ";         
         op_dble=Aw_dble;
         prd="Aw_dble()     ";
      }
      
      random_vd(nv,wvd[0],1.0);
      random_vd(nv,wvd[1],1.0);      
      assign_vd2vd(nv,wvd[0],wvd[2]);
      assign_vd2vd(nv,wvd[1],wvd[3]);

      assign_vd2v(nv,wvd[0],wv[0]);
      assign_vd2v(nv,wvd[1],wv[1]);
      assign_v2v(nv,wv[0],wv[2]);
      assign_v2v(nv,wv[1],wv[3]);

      op_dble(wvd[0],wvd[1]);
      op(wv[0],wv[1]);

      mulc_vadd_dble(nv,wvd[2],wvd[0],zd);
      d=vnorm_square_dble(nv,0,wvd[2]);
      error(d!=0.0,1,"main [check4.c]",
            "%s modifies the input field",prd);

      mulc_vadd(nv,wv[2],wv[0],z);
      d=(double)(vnorm_square(nv,0,wv[2]));
      error(d!=0.0,1,"main [check4.c]",
            "%s modifies the input field",pr);      

      if ((iop<2)||(iop==4))
      {
         mulc_vadd_dble(nvh,wvd[3]+nvh,wvd[1]+nvh,zd);
         d=vnorm_square_dble(nvh,0,wvd[3]+nvh);
         error(d!=0.0,1,"main [check4.c]",
               "%s modifies the odd components of the output field",prd);

         mulc_vadd(nvh,wv[3]+nvh,wv[1]+nvh,z);
         d=(double)(vnorm_square(nvh,0,wv[3]+nvh));
         error(d!=0.0,1,"main [check4.c]",
               "%s modifies the odd components of the output field",pr);

         assign_vd2v(nvh,wvd[1],wv[0]);
         mulc_vadd(nvh,wv[0],wv[1],z);
         d=(double)(vnorm_square(nvh,1,wv[0])/
                    vnorm_square(nvh,1,wv[1]));
         if (my_rank==0)
            printf("Deviation of %s from %s: %.1e\n",pr,prd,sqrt(d));
      }

      if ((iop==2)||(iop==3))
      {
         mulc_vadd_dble(nvh,wvd[3],wvd[1],zd);
         d=vnorm_square_dble(nvh,0,wvd[3]);
         error(d!=0.0,1,"main [check4.c]",
               "%s modifies the even components of the output field",prd);

         mulc_vadd(nvh,wv[3],wv[1],z);
         d=(double)(vnorm_square(nvh,0,wv[3]));
         error(d!=0.0,1,"main [check4.c]",
               "%s modifies the even components of the output field",pr);

         assign_vd2v(nvh,wvd[1]+nvh,wv[0]+nvh);
         mulc_vadd(nvh,wv[0]+nvh,wv[1]+nvh,z);
         d=(double)(vnorm_square(nvh,1,wv[0]+nvh)/
                    vnorm_square(nvh,1,wv[1]+nvh));

         if (my_rank==0)
            printf("Deviation of %s from %s: %.1e\n",pr,prd,sqrt(d));         
      }

      if (iop==5)
      {
         assign_vd2v(nv,wvd[1],wv[0]);
         mulc_vadd(nv,wv[0],wv[1],z);
         d=(double)(vnorm_square(nv,1,wv[0])/
                    vnorm_square(nv,1,wv[1]));
         if (my_rank==0)
            printf("Deviation of %s from %s: %.1e\n",pr,prd,sqrt(d));
      }
   }

   ifail=set_Awhat(-mu);
   error(ifail!=0,1,"main [check4.c]","Inversion of Aee or Aoo failed");
   
   random_vd(nvh,wvd[0],1.0);
   set_vd2zero(nvh,wvd[0]+nvh);
   Aw_dble(wvd[0],wvd[1]);

   Aweeinv_dble(wvd[1],wvd[2]);
   mulc_vadd_dble(nvh,wvd[2],wvd[0],zd);
   d=vnorm_square_dble(nvh,1,wvd[2])/vnorm_square_dble(nvh,1,wvd[0]);

   if (my_rank==0)
   {
      printf("\n");
      printf("Comparison of Aweeinv_dble() and Aw_dble(): %.1e\n",sqrt(d));
   }
   
   Awoe_dble(wvd[0],wvd[2]);
   mulc_vadd_dble(nvh,wvd[2]+nvh,wvd[1]+nvh,zd);
   d=vnorm_square_dble(nvh,1,wvd[2]+nvh)/vnorm_square_dble(nvh,1,wvd[1]+nvh);

   if (my_rank==0)
      printf("Comparison of Awoe_dble()    and Aw_dble(): %.1e\n",sqrt(d));

   random_vd(nvh,wvd[0]+nvh,1.0);
   set_vd2zero(nvh,wvd[0]);
   Aw_dble(wvd[0],wvd[1]);

   Awooinv_dble(wvd[1],wvd[2]);
   mulc_vadd_dble(nvh,wvd[2]+nvh,wvd[0]+nvh,zd);
   d=vnorm_square_dble(nvh,1,wvd[2]+nvh)/vnorm_square_dble(nvh,1,wvd[0]+nvh);

   if (my_rank==0)
      printf("Comparison of Awooinv_dble() and Aw_dble(): %.1e\n",sqrt(d));

   random_vd(nvh,wvd[2],1.0);
   assign_vd2vd(nvh,wvd[2],wvd[3]);
   Aweo_dble(wvd[0],wvd[2]);
   mulc_vadd_dble(nvh,wvd[3],wvd[2],zd);   
   mulc_vadd_dble(nvh,wvd[3],wvd[1],zd);
   d=vnorm_square_dble(nvh,1,wvd[3])/vnorm_square_dble(nvh,1,wvd[1]);

   if (my_rank==0)
      printf("Comparison of Aweo_dble()    and Aw_dble(): %.1e\n",sqrt(d));   

   random_vd(nv,wvd[0],1.0);
   Awhat_dble(wvd[0],wvd[1]);
   Awoe_dble(wvd[0],wvd[2]);
   Awooinv_dble(wvd[2],wvd[3]);
   set_vd2zero(nvh,wvd[0]+nvh);
   Aw_dble(wvd[0],wvd[2]);
   Aweo_dble(wvd[3],wvd[2]);
   Aweeinv_dble(wvd[2],wvd[3]);
   
   mulc_vadd_dble(nvh,wvd[3],wvd[1],zd);
   d=vnorm_square_dble(nvh,1,wvd[3])/vnorm_square_dble(nvh,1,wvd[1]);
   error_chk();
   
   if (my_rank==0)
   {
      printf("Comparison of Aweeinv_dble(), Awooinv_dble(), \n");
      printf(" Awoe_dble(), Aweo_dble() and Awhat_dble(): %.1e\n\n",
             sqrt(d));
      fclose(flog);
   }
   
   MPI_Finalize();   
   exit(0);
}
