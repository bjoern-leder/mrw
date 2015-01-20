
/*******************************************************************************
*
* File marchive.c
*
* Copyright (C) 2010, 2011, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs to read and write momentum-field configurations
*
* The externally accessible functions are
*
*   void write_mfld(char *out)
*     Writes the lattice sizes, the processor grid, the rank of the
*     calling process and the local part of the field to the file "out".
*
*   void read_mfld(char *in)
*     Reads the local part of the momentum field from the file "in",
*     assuming the field was previously stored on this file by the program
*     write_mfld().
*
*   void export_mfld(char *out)
*     Writes the lattice sizes and the momentum field to the file "out"
*     from process 0 in the universal format specified below.
*
*   void import_mfld(char *in)
*     Reads the momentum field from the file "in" and extends the field
*     periodically if needed. The file is read by process 0 only and it is
*     assumed that the field is stored on the file in the universal format.
*
* Notes:
*
* The programs in this module involve communications and must be called
* simultaneously on all processes.
*
* The momentum field is the one pointed to by the structure returned by
* mdflds() [mdflds/mdflds.c].
*
* The program export_mfld first writes the lattice sizes and the average
* square norm of the field components to the output file. Then follow the
* field variables in the directions +0,-0,...,+3,-3 at the first odd point,
* the second odd point, and so on. The order of the point (x0,x1,x2,x3)
* with coordinates in the range 0<=x0<N0,...,0<=x3<N3 is determined by the
* index
*
*   ix=x3+N3*x2+N2*N3*x1+N1*N2*N3*x0
*
* where N0,N1,N2,N3 are the lattice sizes.
*
* Independently of the machine, the export function writes the data to the
* output file in little-endian byte order. Integers and double-precision
* numbers on the output file occupy 4 and 8 bytes, respectively, the latter
* being formatted according to the IEEE-754 standard. The import function
* assumes the data on the input file to be little endian and converts them
* to big-endian order if the machine is big endian. Exported fields can thus
* be safely exchanged between different machines.
*
* In the case of the write and read functions, no byte reordering is applied
* and the data are written and read respecting the endianness of the machine.
*
*******************************************************************************/

#define MARCHIVE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "lattice.h"
#include "mdflds.h"
#include "linalg.h"
#include "archive.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static su3_alg_dble *mbuf=NULL,*nbuf;
static mdflds_t *mdfs;


void write_mfld(char *out)
{
   int ldat[9],iw;
   FILE *fout=NULL;

   fout=fopen(out,"wb");
   error_loc(fout==NULL,1,"write_mfld [marchive.c]",
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

   mdfs=mdflds();
   iw=fwrite(ldat,sizeof(int),9,fout);
   iw+=fwrite((*mdfs).mom,sizeof(su3_alg_dble),4*VOLUME,fout);

   error_loc(iw!=(9+4*VOLUME),1,"write_mfld [marchive.c]",
             "Incorrect write count");
   error_chk();
   fclose(fout);
}


void read_mfld(char *in)
{
   int n,ldat[9],ir;
   FILE *fin;

   fin=fopen(in,"rb");
   error_loc(fin==NULL,1,"read_mfld [marchive.c]",
             "Unable to open input file");
   error_chk();

   ir=fread(ldat,sizeof(int),9,fin);
   MPI_Comm_rank(MPI_COMM_WORLD,&n);

   error((ldat[0]!=NPROC0)||(ldat[1]!=NPROC1)||
         (ldat[2]!=NPROC2)||(ldat[3]!=NPROC3)||
         (ldat[4]!=L0)||(ldat[5]!=L1)||(ldat[6]!=L2)||(ldat[7]!=L3)||
         (ldat[8]!=n),1,"read_mfld [marchive.c]","Unexpected lattice data");

   mdfs=mdflds();
   ir+=fread((*mdfs).mom,sizeof(su3_alg_dble),4*VOLUME,fin);

   error_loc(ir!=(9+4*VOLUME),1,"read_mfld [marchive.c]",
             "Incorrect read count");
   error_chk();
   fclose(fin);
}


static int check_machine(void)
{
   int ie;
   
   error_root(sizeof(stdint_t)!=4,1,"check_machine [marchive.c]",
              "Size of a stdint_t integer is not 4");
   error_root(sizeof(double)!=8,1,"check_machine [marchive.c]",
              "Size of a double is not 8");   

   ie=endianness();
   error_root(ie==UNKNOWN_ENDIAN,1,"check_machine [marchive.c]",
              "Unkown endianness");

   return ie;
}


static void alloc_mbuf(int my_rank)
{
   if (my_rank==0)
   {
      mbuf=amalloc(4*(L3+N3)*sizeof(su3_alg_dble),ALIGN);
      nbuf=mbuf+4*L3;
   }
   else
      mbuf=amalloc(4*L3*sizeof(su3_alg_dble),ALIGN);

   error(mbuf==NULL,1,"alloc_mbuf [marchive.c]",
         "Unable to allocate auxiliary array");
}


static void get_links(int iy)
{
   int y3,iz,mu;
   su3_alg_dble *m,*n,*mom;

   mom=(*mdfs).mom;
   n=mbuf;
   iy*=L3;

   if (ipt[iy]<(VOLUME/2))
      iy+=1;

   for (y3=0;y3<L3;y3+=2)
   {
      iz=ipt[iy+y3];
      m=mom+8*(iz-(VOLUME/2));

      for (mu=0;mu<8;mu++)
      {
         *n=*m;
         n+=1;
         m+=1;
      }
   }
}


static void set_links(int iy)
{
   int y3,iz,mu;
   su3_alg_dble *m,*n,*mom;

   mom=(*mdfs).mom;
   n=mbuf;
   iy*=L3;

   if (ipt[iy]<(VOLUME/2))
      iy+=1;

   for (y3=0;y3<L3;y3+=2)
   {
      iz=ipt[iy+y3];
      m=mom+8*(iz-(VOLUME/2));

      for (mu=0;mu<8;mu++)
      {
         *m=*n;
         n+=1;
         m+=1;
      }
   }
}


static double avg_norm(void)
{
   double norm;

   norm=norm_square_alg(4*VOLUME,1,(*mdfs).mom);

   return norm/((double)(4*N0)*(double)(N1*N2*N3));
}


void export_mfld(char *out)
{
   int my_rank,np[4],n,iw,ie;
   int iwa,dmy,tag0,tag1;
   int x0,x1,x2,x3,y0,y1,y2,ix,iy;
   stdint_t lsize[4];
   double norm;
   MPI_Status stat;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (mbuf==NULL)
      alloc_mbuf(my_rank);

   dmy=1;
   tag0=mpi_tag();
   tag1=mpi_tag();
   ie=check_machine();
   mdfs=mdflds();
   norm=avg_norm();

   if (my_rank==0)
   {
      fout=fopen(out,"wb");
      error_root(fout==NULL,1,"export_mfld [marchive.c]",
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

      error_root(iw!=5,1,"export_mfld [marchive.c]","Incorrect write count");
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
            get_links(iy);

         if (n>0)
         {
            if (my_rank==0)
            {
               MPI_Send(&dmy,1,MPI_INT,n,tag0,
                        MPI_COMM_WORLD);               
               MPI_Recv(mbuf,4*L3*8,MPI_DOUBLE,n,tag1,
                        MPI_COMM_WORLD,&stat);
            }
            else if (my_rank==n)
            {
               MPI_Recv(&dmy,1,MPI_INT,0,tag0,
                        MPI_COMM_WORLD,&stat);
               MPI_Send(mbuf,4*L3*8,MPI_DOUBLE,0,tag1,
                        MPI_COMM_WORLD);
            }
         }

         if (my_rank==0)
         {
            if (ie==BIG_ENDIAN)
               bswap_double(4*L3*8,mbuf);
            iw=fwrite(mbuf,sizeof(su3_alg_dble),4*L3,fout);
            iwa|=(iw!=(4*L3));
         }
      }
   }

   if (my_rank==0)
   {
      error_root(iwa!=0,1,"export_mfld [marchive.c]",
                 "Incorrect write count");
      fclose(fout);
   }
}


void import_mfld(char *in)
{
   int my_rank,np[4],n,ir,ie;
   int ira,dmy,tag0,tag1;
   int k,l,x0,x1,x2,y0,y1,y2,y3,c0,c1,c2,ix,iy,ic;
   int n0,n1,n2,n3,nc0,nc1,nc2,nc3;
   stdint_t lsize[4];
   double norm0,norm1,eps;
   MPI_Status stat;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (mbuf==NULL)
      alloc_mbuf(my_rank);

   dmy=1;
   tag0=mpi_tag();
   tag1=mpi_tag();
   ie=check_machine();
   mdfs=mdflds();
   
   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"import_mfld [marchive.c]",
                 "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),4,fin);
      ir+=fread(&norm0,sizeof(double),1,fin);
      error_root(ir!=5,1,"import_mfld [marchive.c]","Incorrect read count");

      if (ie==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&norm0);
      }

      np[0]=(int)(lsize[0]);
      np[1]=(int)(lsize[1]);
      np[2]=(int)(lsize[2]);
      np[3]=(int)(lsize[3]);      

      error_root((np[0]<1)||((N0%np[0])!=0)||
                 (np[1]<1)||((N1%np[1])!=0)||
                 (np[2]<1)||((N2%np[2])!=0)||
                 (np[3]<1)||((N3%np[3])!=0),1,"import_mfld [marchive.c]",
                 "Unexpected or incompatible lattice sizes");
   }

   MPI_Bcast(np,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&norm0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

   n0=np[0];
   n1=np[1];
   n2=np[2];
   n3=np[3];

   nc0=N0/n0;
   nc1=N1/n1;
   nc2=N2/n2;
   nc3=N3/n3;
   ira=0;

   for (ix=0;ix<(n0*n1*n2);ix++)
   {
      x0=(ix/(n1*n2));
      x1=(ix/n2)%n1;
      x2=ix%n2;

      if (my_rank==0)
      {
         n=4*n3;
         ir=fread(nbuf,sizeof(su3_alg_dble),n,fin);
         ira|=(ir!=n);

         if (ie==BIG_ENDIAN)
            bswap_double(n*8,nbuf);
         
         for (k=1;k<nc3;k++)
         {
            for (l=0;l<n;l++)
               nbuf[k*n+l]=nbuf[l];
         }
      }

      for (ic=0;ic<(nc0*nc1*nc2);ic++)
      {
         c0=(ic/(nc1*nc2));
         c1=(ic/nc2)%nc1;
         c2=ic%nc2;

         y0=x0+c0*n0;
         y1=x1+c1*n1;
         y2=x2+c2*n2;
         iy=(y2%L2)+L2*(y1%L1)+L1*L2*(y0%L0);

         np[0]=y0/L0;
         np[1]=y1/L1;
         np[2]=y2/L2;

         for (y3=0;y3<N3;y3+=L3)
         {
            np[3]=y3/L3;
            n=ipr_global(np);

            if (n>0)
            {
               if (my_rank==0)
               {
                  MPI_Send(nbuf+4*y3,4*L3*8,MPI_DOUBLE,n,tag1,
                           MPI_COMM_WORLD);
                  MPI_Recv(&dmy,1,MPI_INT,n,tag0,
                           MPI_COMM_WORLD,&stat);
               }
               else if (my_rank==n)
               {
                  MPI_Recv(mbuf,4*L3*8,MPI_DOUBLE,0,tag1,
                           MPI_COMM_WORLD,&stat);
                  MPI_Send(&dmy,1,MPI_INT,0,tag0,
                           MPI_COMM_WORLD);
               }
            }
            else if (my_rank==0)
               for (l=0;l<(4*L3);l++)
                  mbuf[l]=nbuf[4*y3+l];

            if (my_rank==n)
               set_links(iy);
         }
      }
   }

   if (my_rank==0)
   {
      error_root(ira!=0,1,"import_mfld [marchive.c]",
                 "Incorrect read count");
      fclose(fin);
   }
   
   norm1=avg_norm();
   eps=sqrt((double)(4*N0)*(double)(N1*N2*N3))*DBL_EPSILON;
   error(fabs(norm1-norm0)>(eps*norm0),1,"import_mfld [marchive.c]",
         "Norm test failed");
}
