
/*******************************************************************************
*
* File sarchive.c
*
* Copyright (C) 2007, 2008, 2011, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs to read and write global double-precision spinor fields
*
* The externally accessible functions are
*
*   void write_sfld(char *out,spinor_dble *sd)
*     Writes the lattice sizes, the processor grid, the rank of the
*     calling process, the size of the spinor_dble structure, the square 
*     of the norm of the spinor field sd and the local part of the latter 
*     to the file "out".
*
*   void read_sfld(char *in,spinor_dble *sd)
*     Reads the local part of the spinor field sd from the file "in",
*     assuming the field was previously stored on this file by the
*     program write_sfld().
*
*   void export_sfld(char *out,spinor_dble *sd)
*     Writes the lattice sizes and the spinor field sd to the file "out"
*     from process 0 in the universal format specified below.
*
*   void import_sfld(char *in,spinor_dble *sd)
*     Reads the spinor field sd from the file "in". The file is read by
*     process 0 only and it is assumed that the field is stored on the
*     file in the universal format.
*
* Notes:
*
* The spinor fields are assumed to be global quark fields of local size
* equal to VOLUME or larger. Only these components are written and read.
* All programs involve communications and must be called simultaneously
* on all processes.
*
* The program export_sfld() first writes the lattice sizes and the square
* of the norm of the spinor field sd. Then follow the spinor at the first
* point, the second point, and so on. The order of the point (x0,x1,x2,x3)
* with coordinates in the range 0<=x0<N0,...,0<=x3<N3 is determined by the
* index
*
*   ix=x3+N3*x2+N2*N3*x1+N1*N2*N3*x0
*
* where N0,N1,N2,N3 are the (global) lattice sizes.
*
* Independently of the machine, the export function writes the data to the
* output file in little-endian byte order. Integers and double-precision
* numbers on the output file occupy 4 and 8 bytes, respectively, the latter
* being formatted according to the IEEE-754 standard. The import function
* assumes the data on the input file to be little endian and converts them
* to big-endian order if the machine is big endian. Exported fields can
* thus be safely exchanged between different machines.
*
* In the case of the write and read functions, no byte reordering is applied
* and the data are written and read respecting the endianness of the machine.
*
*******************************************************************************/

#define SARCHIVE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "lattice.h"
#include "linalg.h"
#include "archive.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static spinor_dble *sbuf=NULL;


void write_sfld(char *out,spinor_dble *sd)
{
   int ldat[9],iw;
   double norm;
   FILE *fout=NULL;

   error(sd==NULL,1,"write_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");   
   
   fout=fopen(out,"wb");
   error_loc(fout==NULL,1,"write_sfld [sarchive.c]",
             "Unable to open output file");
   error_chk();

   ldat[0]=NPROC0;
   ldat[1]=NPROC1;
   ldat[2]=NPROC2;
   ldat[3]=NPROC3;

   ldat[4]=L0;
   ldat[5]=L1;
   ldat[6]=L2;
   ldat[7]=L3;

   MPI_Comm_rank(MPI_COMM_WORLD,ldat+8);
   norm=norm_square_dble(VOLUME,0,sd);

   iw=fwrite(ldat,sizeof(int),9,fout);
   iw+=fwrite(&norm,sizeof(double),1,fout);
   iw+=fwrite(sd,sizeof(spinor_dble),VOLUME,fout);

   error_loc((iw!=(10+VOLUME))||(ferror(fout)!=0),1,
             "write_sfld [sarchive.c]","Incorrect write count or write error");
   error_chk();
   fclose(fout);
}


void read_sfld(char *in,spinor_dble *sd)
{
   int ldat[9],ir,n;
   double norm0,norm1,eps;
   FILE *fin=NULL;

   error(sd==NULL,1,"read_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");
   
   fin=fopen(in,"rb");
   error_loc(fin==NULL,1,"read_sfld [sarchive.c]",
             "Unable to open input file");
   error_chk();

   ir=fread(ldat,sizeof(int),9,fin);
   MPI_Comm_rank(MPI_COMM_WORLD,&n);

   error((ldat[0]!=NPROC0)||(ldat[1]!=NPROC1)||
         (ldat[2]!=NPROC2)||(ldat[3]!=NPROC3)||
         (ldat[4]!=L0)||(ldat[5]!=L1)||(ldat[6]!=L2)||(ldat[7]!=L3)||
         (ldat[8]!=n),1,"read_sfld [sarchive.c]","Unexpected lattice data");

   ir+=fread(&norm0,sizeof(double),1,fin);
   ir+=fread(sd,sizeof(spinor_dble),VOLUME,fin);

   error_loc((ir!=(10+VOLUME))||(ferror(fin)!=0),1,
             "read_sfld [sarchive.c]","Incorrect read count or read error");   
   error_chk();
   fclose(fin);
   
   norm1=norm_square_dble(VOLUME,0,sd);
   eps=sqrt((double)(VOLUME))*DBL_EPSILON;
   error_loc(fabs(norm1-norm0)>(eps*norm0),1,"read_sfld [sarchive.c]",
             "Norm test failed");
   error_chk();
}


static int check_machine(void)
{
   int ie;
   
   error_root(sizeof(stdint_t)!=4,1,"check_machine [sarchive.c]",
              "Size of a stdint_t integer is not 4");
   error_root(sizeof(double)!=8,1,"check_machine [sarchive.c]",
              "Size of a double is not 8");   

   ie=endianness();
   error_root(ie==UNKNOWN_ENDIAN,1,"check_machine [sarchive.c]",
              "Unkown endianness");

   return ie;
}


static void alloc_sbuf(void)
{
   sbuf=amalloc(L3*sizeof(spinor_dble),ALIGN);
   
   error(sbuf==NULL,1,"alloc_sbuf [sarchive.c]",
         "Unable to allocate auxiliary array");
}


static void get_spinors(int iy,spinor_dble *sd)
{
   int y3,iz;
   spinor_dble *sb;

   sb=sbuf;
   iy*=L3;

   for (y3=0;y3<L3;y3++)
   {
      iz=ipt[iy+y3];
      (*sb)=sd[iz];
      sb+=1;
   }
}


static void set_spinors(int iy,spinor_dble *sd)
{
   int y3,iz;
   spinor_dble *sb;

   sb=sbuf;
   iy*=L3;

   for (y3=0;y3<L3;y3++)
   {
      iz=ipt[iy+y3];
      sd[iz]=(*sb);
      sb+=1;
   }
}


void export_sfld(char *out,spinor_dble *sd)
{
   int my_rank,np[4],n,iw,ie;
   int iwa,dmy,tag0,tag1;
   int x0,x1,x2,x3,y0,y1,y2,ix,iy;
   stdint_t lsize[4];
   double norm;
   MPI_Status stat;
   FILE *fout=NULL;

   error(sd==NULL,1,"export_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");   
   
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (sbuf==NULL)
      alloc_sbuf();

   dmy=1;
   tag0=mpi_tag();
   tag1=mpi_tag();
   ie=check_machine();
   norm=norm_square_dble(VOLUME,1,sd);   

   if (my_rank==0)
   {
      fout=fopen(out,"wb");
      error_root(fout==NULL,1,"export_sfld [sarchive.c]",
                 "Unable to open output file");

      lsize[0]=N0;
      lsize[1]=N1;
      lsize[2]=N2;
      lsize[3]=N3;

      if (ie==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&norm);
      }
      
      iw=fwrite(lsize,sizeof(stdint_t),4,fout);
      iw+=fwrite(&norm,sizeof(double),1,fout);
      error_root(iw!=5,1,"export_sfld [sarchive.c]","Incorrect write count");
   }

   iwa=0;
   
   for (ix=0;ix<(N0*N1*N2);ix++)
   {
      x0=(ix/(N1*N2));
      x1=(ix/N2)%N1;
      x2=ix%N2;

      y0=x0%L0;
      y1=x1%L1;
      y2=x2%L2;
      iy=y2+L2*y1+L1*L2*y0;

      np[0]=x0/L0;
      np[1]=x1/L1;
      np[2]=x2/L2;

      for (x3=0;x3<N3;x3+=L3)
      {
         np[3]=x3/L3;
         n=ipr_global(np);
         if (my_rank==n)
            get_spinors(iy,sd);

         if (n>0)
         {
            if (my_rank==0)
            {
               MPI_Send(&dmy,1,MPI_INT,n,tag0,
                        MPI_COMM_WORLD);                  
               MPI_Recv((double*)(sbuf),L3*24,MPI_DOUBLE,n,
                        tag1,MPI_COMM_WORLD,&stat);
            }
            else if (my_rank==n)
            {
               MPI_Recv(&dmy,1,MPI_INT,0,tag0,
                        MPI_COMM_WORLD,&stat);               
               MPI_Send((double*)(sbuf),L3*24,MPI_DOUBLE,0,
                        tag1,MPI_COMM_WORLD);
            }
         }

         if (my_rank==0)
         {
            if (ie==BIG_ENDIAN)
               bswap_double(L3*24,(double*)(sbuf));
            iw=fwrite(sbuf,sizeof(spinor_dble),L3,fout);
            iwa|=(iw!=L3);
         }
      }
   }
   
   if (my_rank==0)
   {
      error_root(iwa!=0,1,"export_sfld [sarchive.c]","Incorrect write count");
      fclose(fout);
   }
}


void import_sfld(char *in,spinor_dble *sd)
{
   int my_rank,np[4],n,ir,ie;
   int ira,dmy,tag0,tag1;
   int x0,x1,x2,x3,y0,y1,y2,ix,iy;
   stdint_t lsize[4];
   double norm0,norm1,eps;
   MPI_Status stat;
   FILE *fin=NULL;

   error(sd==NULL,1,"import_sfld [sarchive.c]",
         "Attempt to access unallocated memory space");  
   
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (sbuf==NULL)
      alloc_sbuf();

   dmy=1;
   tag0=mpi_tag();
   tag1=mpi_tag();
   ie=check_machine();
   
   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"import_sfld [sarchive.c]",
                 "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),4,fin);
      ir+=fread(&norm0,sizeof(double),1,fin);
      error_root(ir!=5,1,"import_sfld [sarchive.c]","Incorrect read count");

      if (ie==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&norm0);
      }

      error_root((lsize[0]!=N0)||(lsize[1]!=N1)||(lsize[2]!=N2)||
                 (lsize[3]!=N3),1,"import_sfld [sarchive.c]",
                 "Lattice sizes do not match");
   }

   MPI_Bcast(&norm0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   ira=0;
   
   for (ix=0;ix<(N0*N1*N2);ix++)
   {
      x0=(ix/(N1*N2));
      x1=(ix/N2)%N1;
      x2=ix%N2;

      y0=x0%L0;
      y1=x1%L1;
      y2=x2%L2;
      iy=y2+L2*y1+L1*L2*y0;

      np[0]=x0/L0;
      np[1]=x1/L1;
      np[2]=x2/L2;

      for (x3=0;x3<N3;x3+=L3)
      {
         np[3]=x3/L3;
         n=ipr_global(np);

         if (my_rank==0)
         {
            ir=fread(sbuf,sizeof(spinor_dble),L3,fin);
            ira|=(ir!=L3);
            
            if (ie==BIG_ENDIAN)
               bswap_double(L3*24,(double*)(sbuf));
         }

         if (n>0)
         {
            if (my_rank==0)
            {
               MPI_Send((double*)(sbuf),L3*24,MPI_DOUBLE,n,
                        tag1,MPI_COMM_WORLD);
               MPI_Recv(&dmy,1,MPI_INT,n,tag0,
                        MPI_COMM_WORLD,&stat);
            }
            else if (my_rank==n)
            {
               MPI_Recv((double*)(sbuf),L3*24,MPI_DOUBLE,0,
                        tag1,MPI_COMM_WORLD,&stat);
               MPI_Send(&dmy,1,MPI_INT,0,tag0,
                        MPI_COMM_WORLD);               
            }
         }

         if (my_rank==n)
            set_spinors(iy,sd);
      }
   }

   if (my_rank==0)
   {
      error_root(ira!=0,1,"import_sfld [sarchive.c]","Incorrect read count");
      fclose(fin);
   }

   norm1=norm_square_dble(VOLUME,1,sd);
   eps=sqrt((double)(N0)*(double)(N1*N2*N3))*DBL_EPSILON;
   error(fabs(norm1-norm0)>(eps*norm0),1,"import_sfld [sarchive.c]",
         "Norm test failed");
}
