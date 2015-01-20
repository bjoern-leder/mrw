
/*******************************************************************************
*
* File check1.c
*
* Copyright (C) 2005, 2007, 2010, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Writing and reading gauge and momentum configurations
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "su3fcts.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "mdflds.h"
#include "linalg.h"
#include "archive.h"
#include "global.h"


static void save_flds(su3_dble *usv,su3_alg_dble *fsv)
{
   su3_dble *udb;
   mdflds_t *mdfs;

   udb=udfld();
   mdfs=mdflds();
   cm3x3_assign(4*VOLUME,udb,usv);
   assign_alg2alg(4*VOLUME,(*mdfs).mom,fsv);
}


static int cmp_ud(su3_dble *u,su3_dble *v)
{
   int it;

   it =((*u).c11.re!=(*v).c11.re);
   it|=((*u).c11.im!=(*v).c11.im);
   it|=((*u).c12.re!=(*v).c12.re);
   it|=((*u).c12.im!=(*v).c12.im);
   it|=((*u).c13.re!=(*v).c13.re);
   it|=((*u).c13.im!=(*v).c13.im);

   it|=((*u).c21.re!=(*v).c21.re);
   it|=((*u).c21.im!=(*v).c21.im);
   it|=((*u).c22.re!=(*v).c22.re);
   it|=((*u).c22.im!=(*v).c22.im);
   it|=((*u).c23.re!=(*v).c23.re);
   it|=((*u).c23.im!=(*v).c23.im);

   it|=((*u).c31.re!=(*v).c31.re);
   it|=((*u).c31.im!=(*v).c31.im);
   it|=((*u).c32.re!=(*v).c32.re);
   it|=((*u).c32.im!=(*v).c32.im);
   it|=((*u).c33.re!=(*v).c33.re);
   it|=((*u).c33.im!=(*v).c33.im);
   
   return it;
}


static int check_ud(su3_dble *usv)
{
   int it;
   su3_dble *u,*um;

   u=udfld();
   um=u+4*VOLUME;
   it=0;
   
   for (;u<um;u++)
   {
      it|=cmp_ud(u,usv);
      usv+=1;
   }

   return it;
}


static int cmp_fd(su3_alg_dble *ma,su3_alg_dble *mb)
{
   int it;

   it =((*ma).c1!=(*mb).c1);
   it|=((*ma).c2!=(*mb).c2);
   it|=((*ma).c3!=(*mb).c3);
   it|=((*ma).c4!=(*mb).c4);
   it|=((*ma).c5!=(*mb).c5);
   it|=((*ma).c6!=(*mb).c6);
   it|=((*ma).c7!=(*mb).c7);
   it|=((*ma).c8!=(*mb).c8);
   
   return it;
}


static int check_fd(su3_alg_dble *fsv)
{
   int it;
   su3_alg_dble *m,*mm;
   mdflds_t *mdfs;

   mdfs=mdflds();   
   m=(*mdfs).mom;
   mm=m+4*VOLUME;
   it=0;
   
   for (;m<mm;m++)
   {
      it|=cmp_fd(m,fsv);
      fsv+=1;
   }

   return it;
}


int main(int argc,char *argv[])
{
   int my_rank,nsize;
   su3_dble **usv;
   su3_alg_dble **fsv;
   char loc_dir[NAME_SIZE],cnfg[NAME_SIZE],mfld[NAME_SIZE];
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      flog=freopen("check1.log","w",stdout);
      fin=freopen("check1.in","r",stdin);
      
      printf("\n");
      printf("Writing and reading gauge and momentum configurations\n");
      printf("-----------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("loc_dir","%s\n",loc_dir);
      fclose(fin);
      
      fflush(flog);
   }

   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);   

   start_ranlux(0,123456);
   geometry();
   alloc_wud(1);
   alloc_wfd(1);

   check_dir(loc_dir);   
   nsize=name_size("%s/testcnfg_%d",loc_dir,NPROC);
   error_root(nsize>=NAME_SIZE,1,"main [check1.c]","loc_dir name is too long");
   sprintf(cnfg,"%s/testcnfg_%d",loc_dir,my_rank);   
   sprintf(mfld,"%s/testmfld_%d",loc_dir,my_rank);   
   
   if (my_rank==0)
   {
      printf("Write random field configurations to the files %s/testcnfg_*\n"
             "and %s/testmfld_* on the local disks.\n\n",loc_dir,loc_dir);
      printf("Then read the fields from there, compare with the saved fields\n"
             "and remove all files.\n\n");
   }

   usv=reserve_wud(1);
   fsv=reserve_wfd(1);

   random_ud();
   random_mom();
   save_flds(usv[0],fsv[0]);

   write_cnfg(cnfg);
   write_mfld(mfld);

   random_ud();
   random_mom();
   
   read_cnfg(cnfg);
   read_mfld(mfld);
   
   error_chk();
   remove(cnfg);
   remove(mfld);
   
   error(check_ud(usv[0])!=0,1,"main [check1.c]",
         "The gauge field is not properly restored");
   error(check_fd(fsv[0])!=0,1,"main [check1.c]",
         "The momentum field is not properly restored");

   print_flags();
   
   if (my_rank==0)
   {   
      printf("No errors detected --- the fields are correctly written\n\n");
      fclose(flog);
   }

   MPI_Finalize();
   exit(0);
}
