
/*******************************************************************************
*
* File check3.c
*
* Copyright (C) 2005, 2007, 2008, 2010, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Importing a configuration previously exported by check3, possibly with
* periodic extension
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


static double avg_plaq(void)
{
   double plaq;

   plaq=plaq_sum_dble(1);
   
   return plaq/((double)(6*NPROC)*(double)(VOLUME));
}


static double avg_norm(void)
{
   double norm;
   mdflds_t *mdfs;

   mdfs=mdflds();
   norm=norm_square_alg(4*VOLUME,1,(*mdfs).mom);

   return norm/((double)(4*NPROC)*(double)(VOLUME));
}


int main(int argc,char *argv[])
{
   int my_rank,nsize,ir,l0;
   stdint_t l[4];
   double plaq0,plaq1,plaq2;
   double norm0,norm1,norm2;
   char cnfg_dir[NAME_SIZE],cnfg[NAME_SIZE],mfld[NAME_SIZE];
   FILE *flog=NULL,*fin=NULL;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   
   if (my_rank==0)
   {
      flog=freopen("check3.log","w",stdout);
      fin=freopen("check3.in","r",stdin);
      
      printf("\n");
      printf("Importing gauge and momentum fields exported by check3\n");
      printf("------------------------------------------------------\n\n");

      printf("%dx%dx%dx%d lattice, ",NPROC0*L0,NPROC1*L1,NPROC2*L2,NPROC3*L3);
      printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
      printf("%dx%dx%dx%d local lattice\n\n",L0,L1,L2,L3);

      read_line("cnfg_dir","%s\n",cnfg_dir);
      fclose(fin);

      fflush(flog);
   }

   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   
   start_ranlux(0,9876);
   geometry();
   random_ud();
   random_mom();
   
   plaq0=avg_plaq();
   norm0=avg_norm();

   check_dir_root(cnfg_dir);   
   nsize=name_size("%s/testcnfg0",cnfg_dir);
   error_root(nsize>=NAME_SIZE,1,"main [check3.c]","cnfg_dir name is too long");

   sprintf(cnfg,"%s/testcnfg",cnfg_dir);
   sprintf(mfld,"%s/testmfld",cnfg_dir);
   
   if (my_rank==0)
   {
      fin=fopen(cnfg,"rb");
      error_root(fin==NULL,1,"main [check3.c]","Unable to open input file");

      ir=fread(l,sizeof(stdint_t),4,fin);
      ir+=fread(&plaq1,sizeof(double),1,fin);
      error_root(ir!=5,1,"main [check3.c]","Incorrect read count");
      fclose(fin);

      if (endianness()==BIG_ENDIAN)
      {
         bswap_int(4,l);
         bswap_double(1,&plaq1);
      }
      
      printf("Random gauge field, average plaquette = %.12f\n\n",plaq0);
      printf("Now read gauge field from file %s:\n",cnfg);
      printf("%dx%dx%dx%d lattice\n",
             (int)(l[0]),(int)(l[1]),(int)(l[2]),(int)(l[3]));
      printf("Average plaquette = %.12f\n",plaq1);

      l0=(int)(l[0]);
   }
   else
      l0=0;

   MPI_Bcast(&l0,1,MPI_INT,0,MPI_COMM_WORLD);

   import_cnfg(cnfg);
   plaq2=avg_plaq();
   error_chk();

   if (my_rank==0)
   {
      printf("Should be         = %.12f\n\n",plaq2);
      remove(cnfg);
   }

   print_flags();

   error(check_bcd()!=1,1,"main [check3.c]",
         "Open boundary conditions are not correctly set");
         
   if (my_rank==0)
   {
      fin=fopen(mfld,"rb");
      error_root(fin==NULL,1,"main [check3.c]",
                 "Unable to open input file");

      ir=fread(l,sizeof(stdint_t),4,fin);
      ir+=fread(&norm1,sizeof(double),1,fin);
      error_root(ir!=5,1,"main [check3.c]","Incorrect read count");
      fclose(fin);

      if (endianness()==BIG_ENDIAN)
      {
         bswap_int(4,l);
         bswap_double(1,&norm1);
      }
      
      printf("Random momentum field, average norm = %.12f\n\n",norm0);
      printf("Now read momentum field from file %s:\n",mfld);
      printf("%dx%dx%dx%d lattice\n",
             (int)(l[0]),(int)(l[1]),(int)(l[2]),(int)(l[3]));
      printf("Average norm = %.12f\n",norm1);
   }

   import_mfld(mfld);
   norm2=avg_norm();
   error_chk();

   if (my_rank==0)
   {
      printf("Should be    = %.12f\n\n",norm2);
      remove(mfld);
      fclose(flog);
   }
   
   MPI_Finalize();
   exit(0);
}
