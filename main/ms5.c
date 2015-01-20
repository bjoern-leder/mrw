
/*******************************************************************************
*
* File ms5.c
*
* Copyright (C) 2012, 2013 Martin Luescher, 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Measurement of mass reweighting factors
*
* Syntax: ms5 -i <input file> [-noexp] [-a [-norng]]
*
* For usage instructions see the file README.ms1 and doc/mrw.pdf
*
*******************************************************************************/

#define MAIN_PROGRAM

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "flags.h"
#include "random.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "archive.h"
#include "dfl.h"
#include "mrw.h"
#include "version.h"
#include "global.h"
#include "su3.h"

#define N0 (NPROC0*L0)
#define N1 (NPROC1*L1)
#define N2 (NPROC2*L2)
#define N3 (NPROC3*L3)

#define MAX(n,m) \
   if ((n)<(m)) \
      (n)=(m)

static struct
{
   int nrw;
   int *nsrc;
   int *nm;
   int *ndbl;
} file_head;

static struct
{
   int nc;
   double **dbl;
} data;

static int my_rank,noexp,append,norng,endian;
static int first,last,step,level,seed;
static int ipgrd[2],*rlxs_state=NULL,*rlxd_state=NULL;

static char line[NAME_SIZE];
static char log_dir[NAME_SIZE],dat_dir[NAME_SIZE];
static char loc_dir[NAME_SIZE],cnfg_dir[NAME_SIZE];
static char log_file[NAME_SIZE],log_save[NAME_SIZE],end_file[NAME_SIZE];
static char par_file[NAME_SIZE],par_save[NAME_SIZE];
static char dat_file[NAME_SIZE],dat_save[NAME_SIZE];
static char rng_file[NAME_SIZE],rng_save[NAME_SIZE];
static char cnfg_file[NAME_SIZE],nbase[NAME_SIZE];
static FILE *fin=NULL,*flog=NULL,*fdat=NULL,*fend=NULL;


static void alloc_data(void)
{
   int nrw,*nsrc,*nm,*ndbl;
   int n,l;
   double **pp,*p;

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   nm=file_head.nm;
   ndbl=file_head.ndbl;
   n=0;

   for (l=0;l<nrw;l++)
      n+=(nm[l]*nsrc[l]*ndbl[l]);

   pp=malloc(nrw*sizeof(*pp));
   p=malloc(n*sizeof(*p));
   error((pp==NULL)||(p==NULL),1,"alloc_data [ms5.c]",
         "Unable to allocate data arrays");

   data.dbl=pp;
   
   for (l=0;l<nrw;l++)
   {
      data.dbl[l]=p;
      p+=(nm[l]*nsrc[l]*ndbl[l]);
   }
}


static void write_file_head(void)
{
   int nrw,*nsrc,*nm,*ndbl;
   int iw,l;
   stdint_t istd[3];

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   nm=file_head.nm;
   ndbl=file_head.ndbl;
   
   istd[0]=(stdint_t)(nrw);
   
   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   for (l=0;l<nrw;l++)
   {
      istd[0]=(stdint_t)(nsrc[l]);
      istd[1]=(stdint_t)(nm[l]);
      istd[2]=(stdint_t)(ndbl[l]);
   
      if (endian==BIG_ENDIAN)
         bswap_int(3,istd);

      iw+=fwrite(istd,sizeof(stdint_t),3,fdat);
   }

   error_root(iw!=(1+3*nrw),1,"write_file_head [ms5.c]",
              "Incorrect write count");
}


static void check_file_head(void)
{
   int nrw,*nsrc,*nm,*ndbl;
   int ir,ie,l;
   stdint_t istd[3];

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   nm=file_head.nm;
   ndbl=file_head.ndbl;

   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   ie=(istd[0]!=(stdint_t)(nrw));

   for (l=0;l<nrw;l++)
   {
      ir+=fread(istd,sizeof(stdint_t),3,fdat);

      if (endian==BIG_ENDIAN)
         bswap_int(3,istd);

      ie|=(istd[0]!=(stdint_t)(nsrc[l]));
      ie|=(istd[1]!=(stdint_t)(nm[l]));
      ie|=(istd[2]!=(stdint_t)(ndbl[l]));
   }
   
   error_root(ir!=(1+3*nrw),1,"check_file_head [ms5.c]",
              "Incorrect read count");
   
   error_root(ie!=0,1,"check_file_head [ms5.c]",
              "Unexpected value of nrw or nsrc");
}


static void print_rw(int irw)
{
   int isrc,nsrc,nm,ndbl;
   double *lnr;
   mrw_parms_t rwp;

   if (my_rank==0)
   {
      rwp=mrw_parms(irw);
      nsrc=rwp.nsrc;
      nm=rwp.nm;
      lnr=data.dbl[irw];
      ndbl=file_head.ndbl[irw];

      printf("RWF %d: -ln(r) = %.4e + i%.4e",irw,lnr[0],lnr[1]);

      if (nsrc*nm<=4)
      {
         for (isrc=ndbl;isrc<ndbl*nsrc*nm;isrc+=ndbl)
            printf(",%.4e + i%.4e",lnr[isrc],lnr[isrc+1]);
      }
      else
      {
         printf(",%.4e + i%.4e,...",lnr[ndbl],lnr[ndbl+1]);

         for (isrc=ndbl*(nsrc*nm-2);isrc<ndbl*nsrc*nm;isrc+=ndbl)
            printf(",%.4e + i%.4e",lnr[isrc],lnr[isrc+1]);
      }   

      printf("\n");
   }
}


static void write_data(void)
{
   int iw,n,i,idbl;
   int nrw,*nsrc,*nm,*ndbl,ndblm,irw,isrc,im;
   stdint_t istd[1];
   double *dstd;   

   istd[0]=(stdint_t)(data.nc);

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);

   iw=fwrite(istd,sizeof(stdint_t),1,fdat);

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   nm=file_head.nm;
   ndbl=file_head.ndbl;
   n=0;

   ndblm=ndbl[0];
   for (irw=1;irw<nrw;irw++)
   {
      if (ndbl[irw]>ndblm)
         ndblm=ndbl[irw];
   }
   
   dstd=malloc(ndblm*sizeof(*dstd));
   error_root(dstd==NULL,1,"write_data [ms5.c]",
         "Unable to allocate data arrays");
   
   for (irw=0;irw<nrw;irw++)
   {
      for (im=0;im<nm[irw];im++)
      {
         for (isrc=0;isrc<nsrc[irw];isrc++)
         {
            idbl=im*nsrc[irw]*ndbl[irw]+isrc*ndbl[irw];
            for (i=0;i<ndbl[irw];i++)
               dstd[i]=data.dbl[irw][idbl+i];
 
            if (endian==BIG_ENDIAN)
               bswap_double(ndbl[irw],dstd);

            iw+=fwrite(dstd,sizeof(double),ndbl[irw],fdat);
         }
      }
      
      n+=(nm[irw]*nsrc[irw]*ndbl[irw]);
   }
   
   free(dstd);
   
   error_root(iw!=n+1,1,"write_data [ms5.c]",
              "Incorrect write count");
}


static int read_data(void)
{
   int ir,n,i,idbl;
   int nrw,*nsrc,*nm,*ndbl,ndblm,irw,isrc,im;
   stdint_t istd[1];
   double *dstd;
   
   ir=fread(istd,sizeof(stdint_t),1,fdat);

   if (ir!=1)
      return 0;

   if (endian==BIG_ENDIAN)
      bswap_int(1,istd);
   
   data.nc=(int)(istd[0]);

   nrw=file_head.nrw;      
   nsrc=file_head.nsrc;
   nm=file_head.nm;
   ndbl=file_head.ndbl;
   n=0;

   ndblm=ndbl[0];
   for (irw=1;irw<nrw;irw++)
   {
      if (ndbl[irw]>ndblm)
         ndblm=ndbl[irw];
   }
   
   dstd=malloc(ndblm*sizeof(*dstd));
   error_root(dstd==NULL,1,"read_data [ms5.c]",
         "Unable to allocate data arrays");

   for (irw=0;irw<nrw;irw++)
   {
      for (im=0;im<nm[irw];im++)
      {
         for (isrc=0;isrc<nsrc[irw];isrc++)
         {
            ir+=fread(dstd,sizeof(double),ndbl[irw],fdat);

            if (endian==BIG_ENDIAN)
               bswap_double(ndbl[irw],dstd);
            
            idbl=im*nsrc[irw]*ndbl[irw]+isrc*ndbl[irw];
            for (i=0;i<ndbl[irw];i++)
               data.dbl[irw][idbl+i]=dstd[i];
         }
      }
      
      n+=(nm[irw]*nsrc[irw]*ndbl[irw]);
   }

   free(dstd);

   error_root(ir!=n+1,1,"read_data [ms5.c]",
              "Read error or incomplete data record");

   return 1;
}


static void read_dirs(void)
{
   int nrw,*nsrc;
   
   if (my_rank==0)
   {
      find_section("Run name");
      read_line("name","%s",nbase);

      find_section("Directories");
      read_line("log_dir","%s",log_dir);
      read_line("dat_dir","%s",dat_dir);

      if (noexp)
      {
         read_line("loc_dir","%s",loc_dir);
         cnfg_dir[0]='\0';
      }
      else
      {
         read_line("cnfg_dir","%s",cnfg_dir);         
         loc_dir[0]='\0';
      }

      find_section("Configurations");
      read_line("first","%d",&first);
      read_line("last","%d",&last);
      read_line("step","%d",&step);
      read_line("nrw","%d",&nrw);
      
      find_section("Random number generator");
      read_line("level","%d",&level);
      read_line("seed","%d",&seed);     

      error_root((last<first)||(step<1)||(((last-first)%step)!=0),1,
                 "read_dirs [ms5.c]","Improper configuration range");
      error_root(nrw<1,1,"read_dirs [ms5.c]",
                 "The number nrw or reweighting factors must be at least 1");
   }

   MPI_Bcast(nbase,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);

   MPI_Bcast(log_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(dat_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(loc_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   MPI_Bcast(cnfg_dir,NAME_SIZE,MPI_CHAR,0,MPI_COMM_WORLD);
   
   MPI_Bcast(&first,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&last,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&step,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nrw,1,MPI_INT,0,MPI_COMM_WORLD);
   
   MPI_Bcast(&level,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&seed,1,MPI_INT,0,MPI_COMM_WORLD);

   nsrc=malloc(3*nrw*sizeof(*nsrc));
   error(nsrc==NULL,1,"read_dirs [ms5.c]",
         "Unable to allocate data array");
   file_head.nrw=nrw;
   file_head.nsrc=nsrc;   
   file_head.nm=nsrc+nrw;   
   file_head.ndbl=file_head.nm+nrw;   
}


static void setup_files(void)
{
   if (noexp)
      error_root(name_size("%s/%sn%d_%d",loc_dir,nbase,last,NPROC-1)>=NAME_SIZE,
                 1,"setup_files [ms5.c]","loc_dir name is too long");
   else
      error_root(name_size("%s/%sn%d",cnfg_dir,nbase,last)>=NAME_SIZE,
                 1,"setup_files [ms5.c]","cnfg_dir name is too long");

   check_dir_root(log_dir);
   check_dir_root(dat_dir);
   error_root(name_size("%s/%s.ms5.log~",log_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms5.c]","log_dir name is too long");
   error_root(name_size("%s/%s.ms5.dat~",dat_dir,nbase)>=NAME_SIZE,
              1,"setup_files [ms5.c]","dat_dir name is too long");   
      
   sprintf(log_file,"%s/%s.ms5.log",log_dir,nbase);
   sprintf(par_file,"%s/%s.ms5.par",dat_dir,nbase);   
   sprintf(dat_file,"%s/%s.ms5.dat",dat_dir,nbase);
   sprintf(rng_file,"%s/%s.ms5.rng",dat_dir,nbase);   
   sprintf(end_file,"%s/%s.ms5.end",log_dir,nbase);
   sprintf(log_save,"%s~",log_file);
   sprintf(par_save,"%s~",par_file);   
   sprintf(dat_save,"%s~",dat_file);
   sprintf(rng_save,"%s~",rng_file);   
}


static void read_lat_parms(void)
{
   double kappa_u,kappa_s,kappa_c,csw,cF;

   if (my_rank==0)
   {
      find_section("Lattice parameters");
      read_line("kappa_u","%lf",&kappa_u);
      read_line("kappa_s","%lf",&kappa_s);
      read_line("kappa_c","%lf",&kappa_c);      
      read_line("csw","%lf",&csw);
      read_line("cF","%lf",&cF);   
   }

   MPI_Bcast(&kappa_u,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&kappa_s,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&kappa_c,1,MPI_DOUBLE,0,MPI_COMM_WORLD);   
   MPI_Bcast(&csw,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&cF,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   
   set_lat_parms(0.0,1.0,kappa_u,kappa_s,kappa_c,csw,1.0,cF);

   if (append)
      check_lat_parms(fdat);
   else
      write_lat_parms(fdat);
}


static void read_mrw_factors(void)
{
   int nrw,*nsrc,*nm,*ndbl,irw;
   mrw_parms_t rwp;

   nrw=file_head.nrw;
   nsrc=file_head.nsrc;
   nm=file_head.nm;
   ndbl=file_head.ndbl;

   for (irw=0;irw<nrw;irw++)
   {
      read_mrw_parms(irw);
      rwp=mrw_parms(irw);
      nsrc[irw]=rwp.nsrc;
      nm[irw]=rwp.nm;
      
      if ((rwp.mrwfact==TMRW2)||(rwp.mrwfact==TMRW3)||(rwp.mrwfact==TMRW4)||
          (rwp.mrwfact==TMRW2_EO)||(rwp.mrwfact==TMRW3_EO)||(rwp.mrwfact==TMRW4_EO)||
          (rwp.mrwfact==MRW_ISO)||(rwp.mrwfact==MRW_TF))
         ndbl[irw]=9;
      else
         ndbl[irw]=4;
   }

   if (append)
      check_mrw_parms(fdat);
   else
      write_mrw_parms(fdat);
}


static void read_sap_parms(void)
{
   int bs[4];

   if (my_rank==0)
   {
      find_section("SAP");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   set_sap_parms(bs,1,4,5);

   if (append)
      check_sap_parms(fdat);
   else
      write_sap_parms(fdat);
}


static void read_dfl_parms(void)
{
   int bs[4],Ns;
   int ninv,nmr,ncy,nkv,nmx;
   double kappa,mu,res;

   if (my_rank==0)
   {
      find_section("Deflation subspace");
      read_line("bs","%d %d %d %d",bs,bs+1,bs+2,bs+3);
      read_line("Ns","%d",&Ns);
   }

   MPI_Bcast(bs,4,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&Ns,1,MPI_INT,0,MPI_COMM_WORLD);   
   set_dfl_parms(bs,Ns);
   
   if (my_rank==0)
   {
      find_section("Deflation subspace generation");
      read_line("kappa","%lf",&kappa);
      read_line("mu","%lf",&mu);
      read_line("ninv","%d",&ninv);     
      read_line("nmr","%d",&nmr);
      read_line("ncy","%d",&ncy);
   }

   MPI_Bcast(&kappa,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&mu,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   MPI_Bcast(&ninv,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&nmr,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&ncy,1,MPI_INT,0,MPI_COMM_WORLD);
   set_dfl_gen_parms(kappa,mu,ninv,nmr,ncy);
   
   if (my_rank==0)
   {
      find_section("Deflation projection");
      read_line("nkv","%d",&nkv);
      read_line("nmx","%d",&nmx);           
      read_line("res","%lf",&res);
   }

   MPI_Bcast(&nkv,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&nmx,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&res,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
   set_dfl_pro_parms(nkv,nmx,res);
   
   if (append)
      check_dfl_parms(fdat);
   else
      write_dfl_parms(fdat);
}


static void read_solvers(void)
{
   int nrw,irw,l;
   int isap,idfl;
   mrw_parms_t rwp;
   solver_parms_t sp;

   nrw=file_head.nrw;
   isap=0;
   idfl=0;
   
   for (irw=0;irw<nrw;irw++)
   {
      rwp=mrw_parms(irw);

      for (l=0;l<2;l++)
      {
         sp=solver_parms(rwp.isp[l]);

         if (sp.solver==SOLVERS)
         {
            read_solver_parms(rwp.isp[l]);
            sp=solver_parms(rwp.isp[l]);

            if (sp.solver==SAP_GCR)
               isap=1;
            else if (sp.solver==DFL_SAP_GCR)
            {
               isap=1;
               idfl=1;
            }
         }
      }
   }

   if (append)
      check_solver_parms(fdat);
   else
      write_solver_parms(fdat);
   
   if (isap)
      read_sap_parms();

   if (idfl)
      read_dfl_parms();
}


static void read_infile(int argc,char *argv[])
{
   int ifile;

   if (my_rank==0)
   {
      flog=freopen("STARTUP_ERROR","w",stdout);
 
      ifile=find_opt(argc,argv,"-i");      
      endian=endianness();

      error_root((ifile==0)||(ifile==(argc-1)),1,"read_infile [ms5.c]",
                 "Syntax: ms5 -i <input file> [-noexp] [-a [-norng]]");

      error_root(endian==UNKNOWN_ENDIAN,1,"read_infile [ms5.c]",
                 "Machine has unknown endianness");

      noexp=find_opt(argc,argv,"-noexp");      
      append=find_opt(argc,argv,"-a");
      norng=find_opt(argc,argv,"-norng");
      
      fin=freopen(argv[ifile+1],"r",stdin);
      error_root(fin==NULL,1,"read_infile [ms5.c]",
                 "Unable to open input file");
   }

   MPI_Bcast(&endian,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&noexp,1,MPI_INT,0,MPI_COMM_WORLD);   
   MPI_Bcast(&append,1,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(&norng,1,MPI_INT,0,MPI_COMM_WORLD);
   
   read_dirs();
   setup_files();

   if (my_rank==0)
   {
      if (append)
         fdat=fopen(par_file,"rb");
      else
         fdat=fopen(par_file,"wb");

      error_root(fdat==NULL,1,"read_infile [ms5.c]",
                 "Unable to open parameter file");
   }

   read_lat_parms();
   read_mrw_factors();
   read_solvers();

   if (my_rank==0)
   {
      fclose(fin);
      fclose(fdat);

      if (append==0)
         copy_file(par_file,par_save);
   }
}


static void check_old_log(int *fst,int *lst,int *stp)
{
   int ie,ic,isv;
   int fc,lc,dc,pc;
   int np[4],bp[4];
   
   fend=fopen(log_file,"r");
   error_root(fend==NULL,1,"check_old_log [ms5.c]",
              "Unable to open log file");

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;      
   isv=0;
         
   while (fgets(line,NAME_SIZE,fend)!=NULL)
   {
      if (strstr(line,"process grid")!=NULL)
      {
         if (sscanf(line,"%dx%dx%dx%d process grid, %dx%dx%dx%d",
                    np,np+1,np+2,np+3,bp,bp+1,bp+2,bp+3)==8)
         {
            ipgrd[0]=((np[0]!=NPROC0)||(np[1]!=NPROC1)||
                      (np[2]!=NPROC2)||(np[3]!=NPROC3));
            ipgrd[1]=((bp[0]!=NPROC0_BLK)||(bp[1]!=NPROC1_BLK)||
                      (bp[2]!=NPROC2_BLK)||(bp[3]!=NPROC3_BLK));
         }
         else
            ie|=0x1;
      }
      else if (strstr(line,"fully processed")!=NULL)
      {
         pc=lc;
         
         if (sscanf(line,"Configuration no %d",&lc)==1)
         {
            ic+=1;
            isv=1;
         }
         else
            ie|=0x1;
         
         if (ic==1)
            fc=lc;
         else if (ic==2)
            dc=lc-fc;
         else if ((ic>2)&&(lc!=(pc+dc)))
            ie|=0x2;
      }
      else if (strstr(line,"Configuration no")!=NULL)
         isv=0;
   }

   fclose(fend);

   error_root((ie&0x1)!=0x0,1,"check_old_log [ms5.c]",
              "Incorrect read count");   
   error_root((ie&0x2)!=0x0,1,"check_old_log [ms5.c]",
              "Configuration numbers are not equally spaced");
   error_root(isv==0,1,"check_old_log [ms5.c]",
              "Log file extends beyond the last configuration save");

   (*fst)=fc;
   (*lst)=lc;
   (*stp)=dc;
}


static void check_old_dat(int fst,int lst,int stp)
{
   int ie,ic;
   int fc,lc,dc,pc;
   
   fdat=fopen(dat_file,"rb");
   error_root(fdat==NULL,1,"check_old_dat [ms5.c]",
              "Unable to open data file");

   check_file_head();

   fc=0;
   lc=0;
   dc=0;
   pc=0;

   ie=0x0;
   ic=0;

   while (read_data()==1)
   {
      pc=lc;
      lc=data.nc;
      ic+=1;
      
      if (ic==1)
         fc=lc;
      else if (ic==2)
         dc=lc-fc;
      else if ((ic>2)&&(lc!=(pc+dc)))
         ie|=0x1;
   }
   
   fclose(fdat);

   error_root(ic==0,1,"check_old_dat [ms5.c]",
              "No data records found");
   error_root((ie&0x1)!=0x0,1,"check_old_dat [ms5.c]",
              "Configuration numbers are not equally spaced");
   error_root((fst!=fc)||(lst!=lc)||(stp!=dc),1,"check_old_dat [ms5.c]",
              "Configuration range is not as reported in the log file");
}


static void check_files(void)
{
   int fst,lst,stp;

   ipgrd[0]=0;
   ipgrd[1]=0;
   
   if (my_rank==0)
   {
      if (append)
      {
         check_old_log(&fst,&lst,&stp);
         check_old_dat(fst,lst,stp);

         error_root((fst!=lst)&&(stp!=step),1,"check_files [ms5.c]",
                    "Continuation run:\n"
                    "Previous run had a different configuration separation");
         error_root(first!=lst+step,1,"check_files [ms5.c]",
                    "Continuation run:\n"
                    "Configuration range does not continue the previous one");
      }
      else
      {
         fin=fopen(log_file,"r");
         fdat=fopen(dat_file,"rb");

         error_root((fin!=NULL)||(fdat!=NULL),1,"check_files [ms5.c]",
                    "Attempt to overwrite old *.log or *.dat file");

         fdat=fopen(dat_file,"wb");
         error_root(fdat==NULL,1,"check_files [ms5.c]",
                    "Unable to open data file");
         write_file_head();
         fclose(fdat);
      }
   }
}


static void print_info(void)
{
   int isap,idfl,n;
   long ip;   
   lat_parms_t lat;
   
   if (my_rank==0)
   {
      ip=ftell(flog);
      fclose(flog);

      if (ip==0L)
         remove("STARTUP_ERROR");
      
      if (append)
         flog=freopen(log_file,"a",stdout);
      else
         flog=freopen(log_file,"w",stdout);

      error_root(flog==NULL,1,"print_info [ms5.c]","Unable to open log file");
      printf("\n");

      if (append)
         printf("Continuation run\n\n");
      else
      {
         printf("Measurement of reweighting factors\n");
         printf("----------------------------------\n\n");
      }

      printf("Program version %s\n",openQCD_RELEASE);         

      if (endian==LITTLE_ENDIAN)
         printf("The machine is little endian\n");
      else
         printf("The machine is big endian\n");
      if (noexp)
         printf("Configurations are read in imported file format\n\n");
      else
         printf("Configurations are read in exported file format\n\n");

      if ((ipgrd[0]!=0)&&(ipgrd[1]!=0))
         printf("Process grid and process block size changed:\n");            
      else if (ipgrd[0]!=0)
         printf("Process grid changed:\n");
      else if (ipgrd[1]!=0)
         printf("Process block size changed:\n");
      
      if ((append==0)||(ipgrd[0]!=0)||(ipgrd[1]!=0))
      {
         printf("%dx%dx%dx%d lattice, ",N0,N1,N2,N3);
         printf("%dx%dx%dx%d local lattice\n",L0,L1,L2,L3);         
         printf("%dx%dx%dx%d process grid, ",NPROC0,NPROC1,NPROC2,NPROC3);
         printf("%dx%dx%dx%d process block size\n",
                NPROC0_BLK,NPROC1_BLK,NPROC2_BLK,NPROC3_BLK);

         if (append)
            printf("\n");
         else
            printf("SF boundary conditions on the quark fields\n\n");
      }

      if (append)
      {
         printf("Random number generator:\n");

         if (norng)
            printf("level = %d, seed = %d, effective seed = %d\n\n",
                   level,seed,seed^(first-step));
         else
         {
            printf("State of ranlxs and ranlxd reset to the\n");
            printf("last exported state\n\n");
         }
      }
      else
      {
         printf("Random number generator:\n");
         printf("level = %d, seed = %d\n\n",level,seed);
         
         lat=lat_parms();
         printf("Lattice parameters:\n");
         n=fdigits(lat.kappa_u);
         printf("kappa_u = %.*f\n",IMAX(n,6),lat.kappa_u);      
         n=fdigits(lat.kappa_s);
         printf("kappa_s = %.*f\n",IMAX(n,6),lat.kappa_s);
         n=fdigits(lat.kappa_c);
         printf("kappa_c = %.*f\n",IMAX(n,6),lat.kappa_c);               
         n=fdigits(lat.csw);
         printf("csw = %.*f\n",IMAX(n,1),lat.csw);      
         n=fdigits(lat.cF);
         printf("cF = %.*f\n\n",IMAX(n,1),lat.cF);

         print_mrw_parms();
         print_rat_parms();
         print_solver_parms(&isap,&idfl);

         if (isap)
            print_sap_parms(0);

         if (idfl)
            print_dfl_parms(0);
      }

      printf("Configurations no %d -> %d in steps of %d\n\n",
             first,last,step);      
      fflush(flog);
   }
}


static void dfl_wsize(int *nws,int *nwv,int *nwvd)
{
   dfl_parms_t dp;
   dfl_pro_parms_t dpp;

   dp=dfl_parms();
   dpp=dfl_pro_parms();

   MAX(*nws,dp.Ns+2);
   MAX(*nwv,2*dpp.nkv+2);
   MAX(*nwvd,4);
}


static void solver_wsize(int isp,int nsd,int np,
                         int *nws,int *nwsd,int *nwv,int *nwvd)
{
   solver_parms_t sp;

   sp=solver_parms(isp);

   if (sp.solver==CGNE)
   {
      MAX(*nws,5);
      MAX(*nwsd,nsd+3);
   }
   else if (sp.solver==SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+1);
      MAX(*nwsd,nsd+2);
   }
   else if (sp.solver==DFL_SAP_GCR)
   {
      MAX(*nws,2*sp.nkv+2);      
      MAX(*nwsd,nsd+4);
      dfl_wsize(nws,nwv,nwvd);
   }         
}


static void reweight_wsize(int *nws,int *nwsd,int *nwv,int *nwvd)
{
   int nrw,irw,nsd,l;
   int isp;
   mrw_parms_t rwp;

   (*nws)=0;
   (*nwsd)=0;
   (*nwv)=0;
   (*nwvd)=0;
   nrw=file_head.nrw;

   for (irw=0;irw<nrw;irw++)
   {
      rwp=mrw_parms(irw);
      for (l=0;l<2;l++)
      {
         isp=rwp.isp[l];

         nsd=2;
         solver_wsize(isp,nsd,0,nws,nwsd,nwv,nwvd);
      }
   }
}


static void print_status(int irw,int ninv,int *status)
{
   int n,nrs,l,i;
   mrw_parms_t rwp;
   solver_parms_t sp;   

   if (my_rank==0)
   {
      rwp=mrw_parms(irw);
      n=rwp.nsrc*rwp.nm;

      printf("RWF %d: status = ",irw);

      for (i=0;i<ninv;i++)
      {
         if (i>0)
            printf(",  ");
            
         for (l=0;l<2;l++)
            status[3*i+l]=(status[3*i+l]+(n/2))/n;

         nrs=status[3*i+2];
         if (i>0)
            sp=solver_parms(rwp.isp[1]);
         else
            sp=solver_parms(rwp.isp[0]);
         
         if (sp.solver==DFL_SAP_GCR)
            printf("%d,%d",status[3*i+0],status[3*i+1]);
         else
            printf("%d",status[3*i+0]);

         if (nrs)
            printf(" (no of subspace regenerations = %d)",nrs);
      }

      printf("\n");   
   }
}


static void set_data(int nc)
{
   int nrw,nsrc,irw,isrc,itm,ninv,idbl,nm,ndbl,ieo;
   int l,j,i,status[9],stat[9];
   double sqne,*lnr,sqnp[2];
   complex_dble z,lnw1[2];
   mrw_parms_t rwp;
   mrw_masses_t ms;
   tm_parms_t tmp;

   nrw=file_head.nrw;
   data.nc=nc;   

   for (irw=0;irw<nrw;irw++)
   {
      lnr=data.dbl[irw];
      rwp=mrw_parms(irw);
      nsrc=rwp.nsrc;
      nm=rwp.nm;
      ndbl=file_head.ndbl[irw];
      
      if (rwp.mrwfact<MRW)
      {
         itm=1;
         ieo=(rwp.mrwfact)%2;
      }
      else
      {
         itm=0;
         ieo=0;
         tmp=tm_parms();
         if (tmp.eoflg!=rwp.tmeo)
            set_tm_parms(rwp.tmeo);
      }

      ninv=1;
      if ((rwp.mrwfact==TMRW)||(rwp.mrwfact==TMRW1)||(rwp.mrwfact==MRW)||
          (rwp.mrwfact==TMRW_EO)||(rwp.mrwfact==TMRW1_EO))
         ninv=1;
      else if ((rwp.mrwfact==TMRW3)||(rwp.mrwfact==TMRW3_EO)||
               (rwp.mrwfact==MRW_ISO)||(rwp.mrwfact==MRW_TF))
         ninv=2;
      else if ((rwp.mrwfact==TMRW2)||(rwp.mrwfact==TMRW4)||
               (rwp.mrwfact==TMRW2_EO)||(rwp.mrwfact==TMRW4_EO))
         ninv=3;
      else
         error_root(1,1,"set_data [ms5.c]","Unknown reweighting factor");
      
      for (l=0;l<9;l++)
      {
         status[l]=0;
         stat[l]=0;
      }
               
      /*printf("itm: %d, ione: %d\n",itm,ione);*/
      for (j=0;j<nm;j++)
      {
         ms=get_mrw_masses(irw,j);
         /*printf("m1: %.6f, mu1: %.6f, d1: %.6f\n",ms.m1,ms.mu1,ms.d1);
         printf("m2: %.6f, mu2: %.6f, d2: %.6f\n",ms.m2,ms.mu2,ms.d2);*/
         for (isrc=0;isrc<nsrc;isrc++)
         {
            if (ieo)
            {
               if (ninv==1)
                  z=mrw1eo(ms,itm,rwp.isp[0],sqnp,&sqne,stat);
               else if (ninv==2)
                  z=mrw2eo(ms,itm,rwp.isp,lnw1,sqnp,&sqne,stat);
               else
               {
                  z.re=mrw3eo(ms,rwp.isp,lnw1,sqnp,&sqne,stat);
                  z.im=0.0;
               }
            }
            else
            {
               if (ninv==1)
                  z=mrw1(ms,itm,rwp.isp[0],sqnp,&sqne,stat);
               else if (ninv==2)
                  z=mrw2(ms,itm,rwp.isp,lnw1,sqnp,&sqne,stat);
               else
               {
                  z.re=mrw3(ms,rwp.isp,lnw1,sqnp,&sqne,stat);
                  z.im=0.0;
               }
            }
            
            if ((rwp.mrwfact==TMRW1)||(rwp.mrwfact==TMRW1_EO))
            {
               z.re=ms.d1*sqnp[0];
               z.im=0;
            }
            
            idbl=j*nsrc*ndbl+isrc*ndbl;

            lnr[idbl  ]=z.re;
            lnr[idbl+1]=z.im;

            if (ninv==1)
            {
               lnr[idbl+2]=sqnp[0];
               lnr[idbl+3]=sqne;
            }
            else
            {
               lnr[idbl+2]=lnw1[0].re;
               lnr[idbl+3]=lnw1[0].im;
               lnr[idbl+4]=lnw1[1].re;
               lnr[idbl+5]=lnw1[1].im;
               lnr[idbl+6]=sqnp[0];
               lnr[idbl+7]=sqnp[1];
               lnr[idbl+8]=sqne;
            }

            for (i=0;i<ninv;i++)
            {
               for (l=0;l<2;l++)
                  status[3*i+l]+=stat[3*i+l];
               status[3*i+2]+=(stat[3*i+2]!=0);
            }
         }
      }

      print_status(irw,ninv,status);
      print_rw(irw);
   }
}


static void init_rng(void)
{
   int ic;
   
   if (append)
   {
      if (norng)
         start_ranlux(level,seed^(first-step));
      else
      {
         ic=import_ranlux(rng_file);
         error_root(ic!=(first-step),1,"init_rng [ms5.c]",
                    "Configuration number mismatch (*.rng file)");
      }
   }
   else
      start_ranlux(level,seed);
}


static void save_ranlux(void)
{
   int nlxs,nlxd;

   if (rlxs_state==NULL)
   {
      nlxs=rlxs_size();
      nlxd=rlxd_size();

      rlxs_state=malloc((nlxs+nlxd)*sizeof(int));
      rlxd_state=rlxs_state+nlxs;

      error(rlxs_state==NULL,1,"save_ranlux [ms5.c]",
            "Unable to allocate state arrays");
   }

   rlxs_get(rlxs_state);
   rlxd_get(rlxd_state);
}


static void restore_ranlux(void)
{
   rlxs_reset(rlxs_state);
   rlxd_reset(rlxd_state);
}


static void check_endflag(int *iend)
{
   if (my_rank==0)
   {
      fend=fopen(end_file,"r");

      if (fend!=NULL)
      {
         fclose(fend);
         remove(end_file);
         (*iend)=1;
         printf("End flag set, run stopped\n\n");
      }
      else
         (*iend)=0;
   }

   MPI_Bcast(iend,1,MPI_INT,0,MPI_COMM_WORLD);
}


int main(int argc,char *argv[])
{
   int nc,iend,status;
   int nws,nwsd,nwv,nwvd;
   double wt1,wt2,wtavg;
   dfl_parms_t dfl;
   
   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   read_infile(argc,argv);
   alloc_data();
   check_files();
   print_info();
   dfl=dfl_parms();

   geometry();
   init_rng();

   reweight_wsize(&nws,&nwsd,&nwv,&nwvd);
   alloc_ws(nws);
   alloc_wsd(nwsd+1);
   alloc_wv(nwv);
   alloc_wvd(nwvd);
   
   iend=0;   
   wtavg=0.0;
   
   for (nc=first;(iend==0)&&(nc<=last);nc+=step)
   {
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      
      if (my_rank==0)
         printf("Configuration no %d\n",nc);

      if (noexp)
      {
         save_ranlux();
         sprintf(cnfg_file,"%s/%sn%d_%d",loc_dir,nbase,nc,my_rank);
         read_cnfg(cnfg_file);
         restore_ranlux();
      }
      else
      {
         sprintf(cnfg_file,"%s/%sn%d",cnfg_dir,nbase,nc);
/*         random_ud();
         export_cnfg(cnfg_file);*/
         import_cnfg(cnfg_file);
      }

      if (dfl.Ns)
      {
         dfl_modes(&status);
         error_root(status<0,1,"main [ms5.c]",
                    "Deflation subspace generation failed (status = %d)",
                    status);
      }
      
      set_data(nc);
      
      if (my_rank==0)
      {
         fdat=fopen(dat_file,"ab");
         error_root(fdat==NULL,1,"main [ms5.c]",
                    "Unable to open dat file");
         write_data();
         fclose(fdat);
      }

      export_ranlux(nc,rng_file);
      error_chk();
   
      MPI_Barrier(MPI_COMM_WORLD);
      wt2=MPI_Wtime();
      wtavg+=(wt2-wt1);

      if (my_rank==0)
      {
         printf("Configuration no %d fully processed in %.2e sec ",
                nc,wt2-wt1);
         printf("(average = %.2e sec)\n\n",
                wtavg/(double)((nc-first)/step+1));
      }

      check_endflag(&iend);      

      if (my_rank==0)
      {
         fflush(flog);         
         copy_file(log_file,log_save);
         copy_file(dat_file,dat_save);
         copy_file(rng_file,rng_save);
      }
   }
      
   if (my_rank==0)
      fclose(flog);
   
   MPI_Finalize();    
   exit(0);
}
