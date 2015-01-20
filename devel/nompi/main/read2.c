
/*******************************************************************************
*
* File read1.c
*
* Copyright (C) 2012 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Reads and evaluates data from the data files created by the program ms1.
* The file to be read has to be specified on the command line.
*
* This program writes the history of the measured normalized reweighting
* factors to the file <run name>.run.dat in the plots directory. The
* associated integrated  autocorrelation times are estimated and printed
* to stdout.
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "utils.h"
#include "extras.h"

static struct
{
   int nrw;
   int *nsrc;
} file_head;

static struct
{
   int nc;
   double **sqn,**lnr;
} data;

static int endian;
static int first,last,step,nms;
static double **avrw,**lnrw;


static void read_file_head(FILE *fdat)
{
   int nrw,*nsrc;
   int ir,l;
   stdint_t istd[1];

   ir=fread(istd,sizeof(stdint_t),1,fdat);
   error_root(ir!=1,1,"read_file_head [read1.c]",
              "Incorrect read count");
   
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   nrw=(int)(istd[0]);
   nsrc=malloc(nrw*sizeof(*nsrc));
   error(nsrc==NULL,1,"read_file_head [read1.c]",
         "Unable to allocate nsrc array");

   for (l=0;l<nrw;l++)
   {
      ir+=fread(istd,sizeof(stdint_t),1,fdat);

      if (endian==BIG_ENDIAN)
         bswap_int(1,istd);

      nsrc[l]=(int)(istd[0]);
   }
   
   error_root(ir!=(1+nrw),1,"read_file_head [read1.c]",
              "Incorrect read count");

   file_head.nrw=nrw;
   file_head.nsrc=nsrc;
}


static void alloc_data(void)
{
   int nrw,*nsrc;
   int n,l;
   double **pp,*p;

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   n=0;

   for (l=0;l<nrw;l++)
      n+=nsrc[l];

   pp=malloc(2*nrw*sizeof(*pp));
   p=malloc(2*n*sizeof(*p));
   error((pp==NULL)||(p==NULL),1,"alloc_data [read1.c]",
         "Unable to allocate data arrays");

   data.sqn=pp;
   data.lnr=pp+nrw;
   
   for (l=0;l<nrw;l++)
   {
      data.sqn[l]=p;
      p+=nsrc[l];
      data.lnr[l]=p;
      p+=nsrc[l];
   }
}


static int read_data(FILE *fdat)
{
   int ir,n;
   int nrw,*nsrc,irw,isrc;
   stdint_t istd[1];
   double dstd[1];
   
   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   
   data.nc=(int)(istd[0]);

   nrw=file_head.nrw;      
   nsrc=file_head.nsrc;
   n=0;

   for (irw=0;irw<nrw;irw++)
   {
      for (isrc=0;isrc<nsrc[irw];isrc++)
      {
         ir+=fread(dstd,sizeof(double),1,fdat);

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
            
         data.sqn[irw][isrc]=dstd[0];
      }

      for (isrc=0;isrc<nsrc[irw];isrc++)
      {
         ir+=fread(dstd,sizeof(double),1,fdat);

         if (endian==BIG_ENDIAN)
            bswap_double(1,dstd);
            
         data.lnr[irw][isrc]=dstd[0];
      }

      n+=nsrc[irw];
   }

   error_root(ir!=(1+2*n),1,"read_data [read1.c]",
              "Read error or incomplete data record");   

   return 1;
}


static void cnfg_range(FILE *fdat,int *fst,int *lst,int *stp)
{
   int nc;
   
   (*fst)=0;
   (*lst)=0;
   (*stp)=1;
   nc=0;

   while (read_data(fdat))
   {
      nc+=1;
      (*lst)=data.nc;

      if (nc==1)
         (*fst)=data.nc;
      else if (nc==2)
         (*stp)=data.nc-(*fst);
   }

   error(nc==0,1,"cnfg_range [read1.c]","No data records on data file");
}


static void select_cnfg_range(FILE *fdat)
{
   int fst,lst,stp;
   
   cnfg_range(fdat,&fst,&lst,&stp);

   printf("Available configuration range: %d - %d by %d\n",
          fst,lst,stp);
   printf("Select first,last,step: ");
   scanf("%d",&first);
   scanf(",");
   scanf("%d",&last);
   scanf(",");
   scanf("%d",&step);
   printf("\n");

   error((step%stp)!=0,1,"select_cnfg_range [read1.c]",
         "Step must be a multiple of the configuration separation");
   
   if (first<fst)
   {
      first=first+((fst-first)/step)*step;
      if (first<fst)
         first+=step;
   }
   
   if (last>lst)
   {
      last=last-((last-lst)/step)*step;
      if (last>lst)
         last-=step;
   } 

   error((last<first)||(((last-first)%step)!=0)||(((first-fst)%stp)!=0),1,
         "select_cnfg_range [read1.c]","Improper configuration range");

   printf("Selected configuration range: %d - %d by %d\n\n",
          first,last,step);

   nms=(last-first)/step+1;
}


static void alloc_avrw(void)
{
   int nrw,irw,ims;
   double **pp,*p;

   nrw=file_head.nrw;
   pp=malloc((2*nrw+2)*sizeof(*pp));
   p=malloc((2*nrw+2)*nms*sizeof(*p));
   error((pp==NULL)||(p==NULL),1,"alloc_avrw [read1.c]",
         "Unable to allocate data arrays");
   avrw=pp;
   lnrw=pp+nrw+1;
   
   for (irw=0;irw<(2*nrw+2);irw++)
   {
      pp[irw]=p;
      p+=nms;
   }

   for (ims=0;ims<nms;ims++)
   {
      lnrw[nrw][ims]=0.0;
      avrw[nrw][ims]=1.0;
   }
}


static void data2avrw(int ims)
{
   int nrw,irw;
   int nsrc,isrc;
   double lnm,rw,*lnr;

   nrw=file_head.nrw;

   for (irw=0;irw<nrw;irw++)
   {
      nsrc=file_head.nsrc[irw];
      lnr=data.lnr[irw];
      lnm=lnr[0];
   
      for (isrc=1;isrc<nsrc;isrc++)
      {
         if (lnr[isrc]<lnm)
            lnm=lnr[isrc];
      }

      lnrw[irw][ims]=lnm;
      lnrw[nrw][ims]+=lnm;
      rw=0.0;

      for (isrc=0;isrc<nsrc;isrc++)
         rw+=exp(lnm-lnr[isrc]);

      rw/=(double)(nsrc);
      avrw[irw][ims]=rw;
      avrw[nrw][ims]*=rw;
   }
}


static void normalize_avrw(void)
{
   int nrw,irw,ims;
   double lnma,rwa,*lnm,*rw;

   nrw=file_head.nrw;

   for (irw=0;irw<=nrw;irw++)
   {
      lnm=lnrw[irw];
      lnma=lnm[0];
   
      for (ims=0;ims<nms;ims++)
      {
         if (lnm[ims]<lnma)
            lnma=lnm[ims];
      }

      rw=avrw[irw];
      
      for (ims=0;ims<nms;ims++)
         rw[ims]*=exp(lnma-lnm[ims]);

      rwa=0.0;
   
      for (ims=0;ims<nms;ims++)
         rwa+=rw[ims];

      rwa/=(double)(nms);

      for (ims=0;ims<nms;ims++)
         rw[ims]/=rwa;
   }
}


static void read_file(char *fin)
{
   int nc,ims;
   long ipos;
   FILE *fdat;

   fdat=fopen(fin,"rb");
   error(fdat==NULL,1,"read_file [read1.c]","Unable to open data file");
   printf("Read data from file %s\n\n",fin);

   read_file_head(fdat);
   alloc_data();

   ipos=ftell(fdat);   
   select_cnfg_range(fdat);
   fseek(fdat,ipos,SEEK_SET);
   alloc_avrw();
   ims=0;

   while (read_data(fdat))
   {
      nc=data.nc;

      if ((nc>=first)&&(nc<=last)&&(((nc-first)%step)==0)&&(ims<nms))
      {
         data2avrw(ims);
         ims+=1;
      }
   }

   fclose(fdat);   
   error(ims!=nms,1,"read_file [read1.c]","Incorrect read count");

   normalize_avrw();
}


static double f(int nx,double x[])
{
   return x[0];
}


static void print_plot(char *fin)
{
   int n,nrw,irw,ims;
   char base[NAME_SIZE],plt_file[NAME_SIZE],*p;
   FILE *fout;

   p=strstr(fin,".ms1.dat");
   error(p==NULL,1,"print_plot [read1.c]","Unexpected data file name");
   n=p-fin;

   p=strrchr(fin,'/');
   if (p==NULL)
      p=fin;
   else
      p+=1;
   n-=(p-fin);
   
   error(n>=NAME_SIZE,1,"print_plot [read1.c]","File name is too long");
   strncpy(base,p,n);
   base[n]='\0';
   
   error(name_size("plots/%s.run.dat",base)>=NAME_SIZE,1,
         "print_plot [read1.c]","File name is too long");
   sprintf(plt_file,"plots/%s.run.dat",base);
   fout=fopen(plt_file,"w");
   error(fout==NULL,1,"print_plot [read1.c]",
         "Unable to open output file");   

   nrw=file_head.nrw;
   
   fprintf(fout,"#\n");
   fprintf(fout,"# Data written by the program ms1\n");
   fprintf(fout,"# -------------------------------\n");
   fprintf(fout,"#\n");
   fprintf(fout,"# Number of measurements = %d\n",nms);
   fprintf(fout,"#\n");   
   fprintf(fout,"# nc:   Configuration number\n");
   fprintf(fout,"# W:    Normalized reweighting factors\n");
   fprintf(fout,"#\n");
   fprintf(fout,"#  nc");

   for (irw=0;irw<nrw;irw++)
      fprintf(fout,"       W[%d] ",irw);

   if (nrw==1)
      fprintf(fout,"\n");
   else
      fprintf(fout,"       W[all]\n");
   
   fprintf(fout,"#\n");

   for (ims=0;ims<nms;ims++)
   {
      fprintf(fout," %5d  ",first+ims*step);

      for (irw=0;irw<nrw;irw++)
         fprintf(fout,"  %.4e",avrw[irw][ims]);

      if (nrw==1)
         fprintf(fout,"\n");
      else
         fprintf(fout,"  %.4e\n",avrw[nrw][ims]);
   }

   fclose(fout);

   printf("Data printed to file %s\n\n",plt_file);
}


int main(int argc,char *argv[])
{
   int nrw,irw,*nsrc;
   
   error(argc!=2,1,"main [read1.c]","Syntax: read1 <filename>");

   printf("\n");
   printf("History of reweighting factors\n");
   printf("------------------------------\n\n");

   read_file(argv[1]);
   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   
   printf("The total number of measurements is %d\n",nms);
   printf("Integrated autocorrelation times and associated errors are ");
   printf("estimated\n");
   
   if (nms>100)
      printf("using the numerically determined autocorrelation function\n");
   else
      printf("by binning and calculating jackknife errors\n");
   
   printf("Autocorrelation times are given in numbers of measurements\n\n");
   
   for (irw=0;irw<=nrw;irw++)
   {
      if (irw<nrw)
      {
         printf("Reweighting factor no %d\n",irw);
         printf("Number of random sources = %d\n\n",nsrc[irw]);
      }
      else if (nrw==1)
         continue;
      else
         printf("Product of all reweighting factors\n\n");
      
      if (nms>=100)
         print_auto(nms,avrw[irw]);
      else
         print_jack(1,nms,avrw+irw,f);

      printf("\n");
   }
   
   print_plot(argv[1]);
   exit(0);
}
