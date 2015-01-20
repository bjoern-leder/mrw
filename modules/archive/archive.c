
/*******************************************************************************
*
* File archive.c
*
* Copyright (C) 2005, 2007, 2009, 2010, 2011, 2012, 2013 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Programs to read and write gauge-field configurations
*
* The externally accessible functions are
*
*   void write_cnfg(char *out)
*     Writes the lattice sizes, the processor grid, the rank of the
*     calling process, the state of the random number generator and the
*     local double-precision gauge field to the file "out".
*
*   void read_cnfg(char *in)
*     Reads the data previously written by the program write_cnfg from
*     the file "in" and resets the random number generator and the local
*     double-precision gauge field accordingly. The program checks that
*     the configuration satisfies open boundary conditions.
*
*   void export_cnfg(char *out)
*     Writes the lattice sizes and the global double-precision gauge field to
*     the file "out" from process 0 in the universal format specified below.
*
*   void import_cnfg(char *in)
*     Reads the global double-precision gauge field from the file "in" from
*     process 0, assuming that the field is stored in the universal format.
*     The field is periodically extended if needed and the program imposes 
*     open boundary conditions when they are not already satisfied.
*
* Notes:
*
* All programs in this module may involve global communications and must be
* called simultaneously on all processes.
*
* The program export_cnfg() first writes the lattice sizes and the average of
* the plaquette Re tr{U_p} to the output file. Then follow the 8 link variables
* in the directions +0,-0,...,+3,-3 at the first odd point, the second odd
* point, and so on. The order of the point (x0,x1,x2,x3) with coordinates in
* the range 0<=x0<N0,...,0<=x3<N3 is determined by the index
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
* to big-endian order if the machine is big endian. Exported configurations
* can thus be safely exchanged between different machines.
*
* In the case of the write and read functions, no byte reordering is applied
* and the data are written and read respecting the endianness of the machine.
* The copy_file() program copies characters one by one and therefore preserves
* the byte ordering.
*
* It is permissible to import field configurations that do not satisfy open
* boundary conditions. Fields satisfying open boundary conditions cannot be
* periodically extended in the time direction (an error occurs in this case).
*
*******************************************************************************/

#define ARCHIVE_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "global.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

static int ns,nd,*state=NULL;
static su3_dble *ubuf=NULL,*vbuf,*udb;


static void alloc_state(void)
{
   int n;

   ns=rlxs_size();
   nd=rlxd_size();

   if (ns<nd)
      n=nd;
   else
      n=ns;

   state=amalloc(n*sizeof(int),3);
   error(state==NULL,1,"alloc_state [archive.c]",
         "Unable to allocate auxiliary array");
}


void write_cnfg(char *out)
{
   int ldat[9],iw;
   FILE *fout;

   if (state==NULL)
      alloc_state();

   fout=fopen(out,"wb");
   error_loc(fout==NULL,1,"write_cnfg [archive.c]",
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

   iw=fwrite(ldat,sizeof(int),9,fout);
   rlxs_get(state);
   iw+=fwrite(state,sizeof(int),ns,fout);
   rlxd_get(state);
   iw+=fwrite(state,sizeof(int),nd,fout);
   udb=udfld();
   iw+=fwrite(udb,sizeof(su3_dble),4*VOLUME,fout);

   error_loc(iw!=(9+ns+nd+4*VOLUME),1,"write_cnfg [archive.c]",
             "Incorrect write count");
   error_chk();
   fclose(fout);
}


void read_cnfg(char *in)
{
   int n,ldat[9],ir;
   FILE *fin;

   if (state==NULL)
      alloc_state();

   fin=fopen(in,"rb");
   error_loc(fin==NULL,1,"read_cnfg [archive.c]",
             "Unable to open input file");
   error_chk();

   ir=fread(ldat,sizeof(int),9,fin);
   MPI_Comm_rank(MPI_COMM_WORLD,&n);

   error((ldat[0]!=NPROC0)||(ldat[1]!=NPROC1)||
         (ldat[2]!=NPROC2)||(ldat[3]!=NPROC3)||
         (ldat[4]!=L0)||(ldat[5]!=L1)||(ldat[6]!=L2)||(ldat[7]!=L3)||
         (ldat[8]!=n),1,"read_cnfg [archive.c]","Unexpected lattice data");

   ir+=fread(state,sizeof(int),ns,fin);
   rlxs_reset(state);
   ir+=fread(state,sizeof(int),nd,fin);
   rlxd_reset(state);
   udb=udfld();
   ir+=fread(udb,sizeof(su3_dble),4*VOLUME,fin);

   error_loc(ir!=(9+ns+nd+4*VOLUME),1,"read_cnfg [archive.c]",
             "Incorrect read count");
   error_chk();
   fclose(fin);
   
   error(check_bcd()!=1,1,"read_cnfg [archive.c]",
         "Field does not satisfy open bcd");
   set_flags(UPDATED_UD);
}


static int check_machine(void)
{
   int ie;
   
   error_root(sizeof(stdint_t)!=4,1,"check_machine [archive.c]",
              "Size of a stdint_t integer is not 4");
   error_root(sizeof(double)!=8,1,"check_machine [archive.c]",
              "Size of a double is not 8");   

   ie=endianness();
   error_root(ie==UNKNOWN_ENDIAN,1,"check_machine [archive.c]",
              "Unkown endianness");

   return ie;
}


static void alloc_ubuf(int my_rank)
{
   if (my_rank==0)
   {
      ubuf=amalloc(4*(L3+N3)*sizeof(su3_dble),ALIGN);
      vbuf=ubuf+4*L3;
   }
   else
      ubuf=amalloc(4*L3*sizeof(su3_dble),ALIGN);

   error(ubuf==NULL,1,"alloc_ubuf [archive.c]",
         "Unable to allocate auxiliary array");
}


static void get_links(int iy)
{
   int y3,iz,mu;
   su3_dble *u,*v;

   v=ubuf;
   iy*=L3;

   if (ipt[iy]<(VOLUME/2))
      iy+=1;

   for (y3=0;y3<L3;y3+=2)
   {
      iz=ipt[iy+y3];
      u=udb+8*(iz-(VOLUME/2));

      for (mu=0;mu<8;mu++)
      {
         *v=*u;
         v+=1;
         u+=1;
      }
   }
}


static void set_links(int iy)
{
   int y3,iz,mu;
   su3_dble *u,*v;

   v=ubuf;
   iy*=L3;

   if (ipt[iy]<(VOLUME/2))
      iy+=1;

   for (y3=0;y3<L3;y3+=2)
   {
      iz=ipt[iy+y3];
      u=udb+8*(iz-(VOLUME/2));

      for (mu=0;mu<8;mu++)
      {
         *u=*v;
         v+=1;
         u+=1;
      }
   }
}


void export_cnfg(char *out)
{
   int my_rank,np[4],n,iw,ie;
   int iwa,dmy,tag0,tag1;
   int x0,x1,x2,x3,y0,y1,y2,ix,iy;
   stdint_t lsize[4];
   double plaq;
   MPI_Status stat;
   FILE *fout=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (ubuf==NULL)
      alloc_ubuf(my_rank);

   dmy=1;
   tag0=mpi_tag();
   tag1=mpi_tag();
   ie=check_machine();
   udb=udfld();
   plaq=plaq_sum_dble(1)/((double)(6*N0)*(double)(N1*N2*N3));

   if (my_rank==0)
   {
      fout=fopen(out,"wb");
      error_root(fout==NULL,1,"export_cnfg [archive.c]",
                 "Unable to open output file");

      lsize[0]=N0;
      lsize[1]=N1;
      lsize[2]=N2;
      lsize[3]=N3;

      if (ie==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&plaq);
      }
      
      iw=fwrite(lsize,sizeof(stdint_t),4,fout);
      iw+=fwrite(&plaq,sizeof(double),1,fout);

      error_root(iw!=5,1,"export_cnfg [archive.c]","Incorrect write count");
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
               MPI_Recv(ubuf,4*L3*18,MPI_DOUBLE,n,tag1,
                        MPI_COMM_WORLD,&stat);
            }
            else if (my_rank==n)
            {
               MPI_Recv(&dmy,1,MPI_INT,0,tag0,
                        MPI_COMM_WORLD,&stat);
               MPI_Send(ubuf,4*L3*18,MPI_DOUBLE,0,tag1,
                        MPI_COMM_WORLD);
            }
         }

         if (my_rank==0)
         {
            if (ie==BIG_ENDIAN)
               bswap_double(4*L3*18,ubuf);
            iw=fwrite(ubuf,sizeof(su3_dble),4*L3,fout);
            iwa|=(iw!=(4*L3));
         }
      }
   }

   if (my_rank==0)
   {
      error_root(iwa!=0,1,"export_cnfg [archive.c]",
                 "Incorrect write count");      
      fclose(fout);
   }
}


void import_cnfg(char *in)
{
   int my_rank,np[4],n,ir,ie,ibc;
   int ira,dmy,tag0,tag1;
   int k,l,x0,x1,x2,y0,y1,y2,y3,c0,c1,c2,ix,iy,ic;
   int n0,n1,n2,n3,nc0,nc1,nc2,nc3;
   stdint_t lsize[4];
   double plaq0,plaq1,eps;
   MPI_Status stat;
   FILE *fin=NULL;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (ubuf==NULL)
      alloc_ubuf(my_rank);

   dmy=1;
   tag0=mpi_tag();
   tag1=mpi_tag();
   ie=check_machine();
   udb=udfld();
   
   if (my_rank==0)
   {
      fin=fopen(in,"rb");
      error_root(fin==NULL,1,"import_cnfg [archive.c]",
                 "Unable to open input file");

      ir=fread(lsize,sizeof(stdint_t),4,fin);
      ir+=fread(&plaq0,sizeof(double),1,fin);
      error_root(ir!=5,1,"import_cnfg [archive.c]","Incorrect read count");

      if (ie==BIG_ENDIAN)
      {
         bswap_int(4,lsize);
         bswap_double(1,&plaq0);
      }

      np[0]=(int)(lsize[0]);
      np[1]=(int)(lsize[1]);
      np[2]=(int)(lsize[2]);
      np[3]=(int)(lsize[3]);      
      
      error_root((np[0]<1)||((N0%np[0])!=0)||
                 (np[1]<1)||((N1%np[1])!=0)||
                 (np[2]<1)||((N2%np[2])!=0)||
                 (np[3]<1)||((N3%np[3])!=0),1,"import_cnfg [archive.c]",
                 "Unexpected or incompatible lattice sizes");
   }

   MPI_Bcast(np,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&plaq0,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

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
         ir=fread(vbuf,sizeof(su3_dble),n,fin);
         ira|=(ir!=n);

         if (ie==BIG_ENDIAN)
            bswap_double(n*18,vbuf);
         
         for (k=1;k<nc3;k++)
         {
            for (l=0;l<n;l++)
               vbuf[k*n+l]=vbuf[l];
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
                  MPI_Send(vbuf+4*y3,4*L3*18,MPI_DOUBLE,n,tag1,
                           MPI_COMM_WORLD);
                  MPI_Recv(&dmy,1,MPI_INT,n,tag0,
                           MPI_COMM_WORLD,&stat);                  
               }
               else if (my_rank==n)
               {
                  MPI_Recv(ubuf,4*L3*18,MPI_DOUBLE,0,tag1,
                           MPI_COMM_WORLD,&stat);
                  MPI_Send(&dmy,1,MPI_INT,0,tag0,
                           MPI_COMM_WORLD);                  
               }
            }
            else if (my_rank==0)
               for (l=0;l<(4*L3);l++)
                  ubuf[l]=vbuf[4*y3+l];

            if (my_rank==n)
               set_links(iy);
         }
      }
   }

   if (my_rank==0)
   {
      error_root(ira!=0,1,"import_cnfg [archive.c]","Incorrect read count");
      fclose(fin);
   }      

   set_flags(UPDATED_UD);
   plaq1=plaq_sum_dble(1)/((double)(6*N0)*(double)(N1*N2*N3));
   eps=sqrt((double)(6*N0)*(double)(N1*N2*N3))*DBL_EPSILON;
   error(fabs(plaq1-plaq0)>eps,1,"import_cnfg [archive.c]",
         "Plaquette test failed");

   ibc=check_bcd();
   error_root((ibc==1)&&(n0!=N0),1,"import_cnfg [archive.c]",
              "Attempt to periodically extend a field with open bcd");

   if (ibc==0)
      openbcd();
}
