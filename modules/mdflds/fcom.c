
/*******************************************************************************
*
* File fcom.c
*
* Copyright (C) 2010, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Communication of the force variables residing at the boundaries of the
* local lattices
*
* The externally accessible functions are
*
*   void copy_bnd_frc(void)
*     Fetches the force variables on the boundaries of the local lattice
*     from the neighbouring processes.
*
*   void add_bnd_frc(void)
*     Adds the values of the force variables on the boundaries of the  
*     local lattice to the force field on the neighbouring processes.
*
*   void free_fcom_bufs(void)
*     Frees the communication buffers used in this module.
*
* Notes:
*
* The force field is the one returned by mdflds(). Its elements are ordered
* in the same way as those of the global gauge fields (see main/README.global
* and lattice/README.uidx).
*
* The programs in this module perform global communications and must be
* called simultaneously on all MPI processes.
*
*******************************************************************************/

#define FCOM_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "utils.h"
#include "lattice.h"
#include "mdflds.h"
#include "global.h"

static su3_alg_dble *sbuf_f0=NULL,*sbuf_fk,*rbuf_f0,*rbuf_fk;
static mdflds_t *mdfs;
static uidx_t *idx;


static void alloc_frcbufs(void)
{
   mdfs=mdflds();
   idx=uidx();

   if (BNDRY>0)
   {
      sbuf_f0=amalloc(7*(BNDRY/4)*sizeof(*sbuf_f0),ALIGN);
      error(sbuf_f0==NULL,1,"alloc_frcbufs [fcom.c]",
            "Unable to allocate communication buffers");
   
      sbuf_fk=sbuf_f0+(BNDRY/4);
      rbuf_f0=(*mdfs).frc+4*VOLUME;
      rbuf_fk=rbuf_f0+(BNDRY/4);
   }
}


static void pack_f0(void)
{
   int mu,nu0,snu0;
   int *iu,*ium;
   su3_alg_dble *f,*fb,*frc;

   fb=(*mdfs).frc;
   snu0=0;

   for (mu=0;mu<4;mu++)
   {
      nu0=idx[mu].nu0;

      if (nu0>0)
      {
         f=sbuf_f0+snu0;
         iu=idx[mu].iu0;
         ium=iu+nu0;

         for (;iu<ium;iu++)
         {
            frc=fb+(*iu);

            (*f).c1=(*frc).c1;
            (*f).c2=(*frc).c2;
            (*f).c3=(*frc).c3;
            (*f).c4=(*frc).c4;
            (*f).c5=(*frc).c5;
            (*f).c6=(*frc).c6;
            (*f).c7=(*frc).c7;
            (*f).c8=(*frc).c8;            
            
            f+=1;
         }

         snu0+=nu0;
      }
   }
}


static void pack_fk(void)
{
   int mu,nuk,snuk;
   int *iu,*ium;
   su3_alg_dble *f,*fb,*frc;

   fb=(*mdfs).frc;
   snuk=0;

   for (mu=0;mu<4;mu++)
   {
      nuk=idx[mu].nuk;

      if (nuk>0)
      {
         f=sbuf_fk+snuk;
         iu=idx[mu].iuk;
         ium=iu+nuk;

         for (;iu<ium;iu++)
         {
            frc=fb+(*iu);

            (*f).c1=(*frc).c1;
            (*f).c2=(*frc).c2;
            (*f).c3=(*frc).c3;
            (*f).c4=(*frc).c4;
            (*f).c5=(*frc).c5;
            (*f).c6=(*frc).c6;
            (*f).c7=(*frc).c7;
            (*f).c8=(*frc).c8;            
            
            f+=1;            
         }

         snuk+=nuk;
      }
   }
}


static void fwd_send_f0(void)
{
   int mu,nu0,snu0,nbf;
   int tag,saddr,raddr,np;
   su3_alg_dble *sbuf,*rbuf;
   MPI_Status stat;

   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;
   snu0=0;

   for (mu=0;mu<4;mu++)
   {
      nu0=idx[mu].nu0;

      if (nu0>0)
      {
         tag=mpi_tag();
         saddr=npr[2*mu];
         raddr=npr[2*mu+1];
         sbuf=sbuf_f0+snu0;
         rbuf=rbuf_f0+snu0;
         nbf=8*nu0;
         snu0+=nu0;

         if (np==0)
         {
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
         }
         else
         {
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}


static void fwd_send_fk(void)
{
   int mu,nuk,snuk,nbf;
   int tag,saddr,raddr,np;
   su3_alg_dble *sbuf,*rbuf;
   MPI_Status stat;

   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;
   snuk=0;

   for (mu=0;mu<4;mu++)
   {
      nuk=idx[mu].nuk;

      if (nuk>0)
      {
         tag=mpi_tag();
         saddr=npr[2*mu];
         raddr=npr[2*mu+1];
         sbuf=sbuf_fk+snuk;
         rbuf=rbuf_fk+snuk;
         nbf=8*nuk;
         snuk+=nuk;

         if (np==0)
         {
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
         }
         else
         {
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}


void copy_bnd_frc(void)
{
   if (NPROC>1)
   {
      if (sbuf_f0==NULL)
         alloc_frcbufs();
      
      pack_f0();
      fwd_send_f0();
      pack_fk();
      fwd_send_fk();
   }
}


static void bck_send_f0(void)
{
   int mu,nu0,snu0,nbf;
   int tag,saddr,raddr,np;
   su3_alg_dble *sbuf,*rbuf;
   MPI_Status stat;

   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;
   snu0=0;

   for (mu=0;mu<4;mu++)
   {
      nu0=idx[mu].nu0;

      if (nu0>0)
      {
         tag=mpi_tag();
         saddr=npr[2*mu+1];
         raddr=npr[2*mu];
         sbuf=rbuf_f0+snu0;
         rbuf=sbuf_f0+snu0;
         nbf=8*nu0;
         snu0+=nu0;

         if (np==0)
         {
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
         }
         else
         {
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}


static void bck_send_fk(void)
{
   int mu,nuk,snuk,nbf;
   int tag,saddr,raddr,np;
   su3_alg_dble *sbuf,*rbuf;
   MPI_Status stat;

   np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;
   snuk=0;

   for (mu=0;mu<4;mu++)
   {
      nuk=idx[mu].nuk;

      if (nuk>0)
      {
         tag=mpi_tag();
         saddr=npr[2*mu+1];
         raddr=npr[2*mu];
         sbuf=rbuf_fk+snuk;
         rbuf=sbuf_fk+snuk;
         nbf=8*nuk;
         snuk+=nuk;

         if (np==0)
         {
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
         }
         else
         {
            MPI_Recv(rbuf,nbf,MPI_DOUBLE,raddr,tag,MPI_COMM_WORLD,&stat);
            MPI_Send(sbuf,nbf,MPI_DOUBLE,saddr,tag,MPI_COMM_WORLD);
         }
      }
   }
}


static void add_f0(void)
{
   int mu,nu0,snu0;
   int *iu,*ium;
   su3_alg_dble *f,*fb,*frc;

   fb=(*mdfs).frc;
   snu0=0;

   for (mu=0;mu<4;mu++)
   {
      nu0=idx[mu].nu0;

      if (nu0>0)
      {
         f=sbuf_f0+snu0;
         iu=idx[mu].iu0;
         ium=iu+nu0;

         for (;iu<ium;iu++)
         {
            frc=fb+(*iu);

            (*frc).c1+=(*f).c1;
            (*frc).c2+=(*f).c2;
            (*frc).c3+=(*f).c3;
            (*frc).c4+=(*f).c4;
            (*frc).c5+=(*f).c5;
            (*frc).c6+=(*f).c6;
            (*frc).c7+=(*f).c7;
            (*frc).c8+=(*f).c8;            
            
            f+=1;
         }

         snu0+=nu0;
      }
   }
}


static void add_fk(void)
{
   int mu,nuk,snuk;
   int *iu,*ium;
   su3_alg_dble *f,*fb,*frc;

   fb=(*mdfs).frc;
   snuk=0;

   for (mu=0;mu<4;mu++)
   {
      nuk=idx[mu].nuk;

      if (nuk>0)
      {
         f=sbuf_fk+snuk;
         iu=idx[mu].iuk;
         ium=iu+nuk;

         for (;iu<ium;iu++)
         {
            frc=fb+(*iu);

            (*frc).c1+=(*f).c1;
            (*frc).c2+=(*f).c2;
            (*frc).c3+=(*f).c3;
            (*frc).c4+=(*f).c4;
            (*frc).c5+=(*f).c5;
            (*frc).c6+=(*f).c6;
            (*frc).c7+=(*f).c7;
            (*frc).c8+=(*f).c8;            
            
            f+=1;            
         }

         snuk+=nuk;
      }
   }
}


void add_bnd_frc(void)
{
   if (NPROC>1)
   {
      if (sbuf_f0==NULL)
         alloc_frcbufs();         
   
      bck_send_fk();
      add_fk();
      bck_send_f0();
      add_f0();
   }
}


void free_fcom_bufs(void)
{
   if (sbuf_f0!=NULL)
   {
      afree(sbuf_f0);
      sbuf_f0=NULL;
   }
}
