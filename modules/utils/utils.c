
/*******************************************************************************
*
* File utils.c
*
* Copyright (C) 2005, 2008, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Collection of basic utility programs
*
* The externally accessible functions are
*
*   int safe_mod(int x,int y)
*     Returns x mod y, where y is assumed positive and x can have any
*     sign. The return value is in the interval [0,y)
*
*   void *amalloc(size_t size,int p)
*     Allocates an aligned memory area of "size" bytes, with a starting
*     address (the return value) that is an integer multiple of 2^p. A
*     NULL pointer is returned if the allocation was not successful
*
*   void afree(void *addr)
*     Frees the aligned memory area at address "addr" that was previously
*     allocated using amalloc. If the memory space at this address was
*     already freed using afree, or if the address does not match an
*     address previously returned by amalloc, the program does not do
*     anything
*
*   int mpi_permanent_tag(void)
*     Returns a new send tag that is guaranteed to be unique and which 
*     is therefore suitable for use in permanent communication requests.
*     The available number of tags of this kind is 16384
*
*   int mpi_tag(void)
*     Returns a new send tag for use in non-permanent communications.
*     Note that the counter for these tags wraps around after 16384
*     tags have been delivered
*
*   void error(int test,int no,char *name,char *format,...)
*     Checks whether "test"=0 on all processes and, if not, aborts the
*     program gracefully with error number "no" after printing the "name"
*     of the calling program and an error message to stdout from process 0.
*     The message is formed on process 0 using the "format" string and any
*     additional arguments, exactly as in a printf statement
*
*   void error_root(int test,int no,char *name,char *format,...)
*     Same as the error() function except that "test" is examined on
*     process 0 only
*
*   int error_loc(int test,int no,char *name,char *message)
*     Checks whether "test"=0 on the local process and, if not, writes
*     the error number "no", the program "name" and the error "message" 
*     to an internal buffer. Only the data of the first instance where
*     this happens are recorded. Note that saved program names and error
*     messages are truncated to 127 and 511 bytes, respectively. In all
*     cases, the program returns the value of "test"
*
*   void error_chk(void)
*     Checks the status of the data saved by error_loc() and aborts the
*     program gracefully, with error number 1, if an error is recorded on
*     some of the processes. Before abortion the error numbers, program
*     names and error messages saved on these processes are printed to
*     stdout from process 0
*
*   void message(char *format,...)
*     Prints a message from process 0 to stdout. The usage and argument
*     list is the same as in the case of the printf function
*
* Notes:
*
* If an error is detected in a locally operating program, it is not possible
* to stop the global program immediately in a decent manner. Such errors
* should first be recorded by the error_loc() function, and the main program
* may later be aborted by calling error_chk() outside the program where the
* error was detected.
*
* An important point to note is that the programs error() and error_chk()
* require a global communication. They must therefore be called on all
* processes simultaneously.
*
*******************************************************************************/

#define UTILS_C

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <string.h>
#include "mpi.h"
#include "utils.h"
#include "global.h"

#define MAX_TAG 32767
#define MAX_PERMANENT_TAG MAX_TAG/2

static int pcmn_cnt=-1,cmn_cnt=MAX_TAG;
static int err_no,err_flg=0;
static char prog_name[128],err_msg[512];

struct addr_t
{
   char *addr;
   char *true_addr;
   struct addr_t *next;
};

static struct addr_t *first=NULL;


int safe_mod(int x,int y)
{
   if (x>=0)
      return x%y;
   else
      return (y-(abs(x)%y))%y;
}


void *amalloc(size_t size,int p)
{
   int shift;
   char *true_addr,*addr;
   unsigned long mask;
   struct addr_t *new;

   if ((size<=0)||(p<0))
      return(NULL);

   shift=1<<p;
   mask=(unsigned long)(shift-1);

   true_addr=malloc(size+shift);
   new=malloc(sizeof(*first));

   if ((true_addr==NULL)||(new==NULL))
   {
      free(true_addr);
      free(new);
      return(NULL);
   }

   addr=(char*)(((unsigned long)(true_addr+shift))&(~mask));
   (*new).addr=addr;
   (*new).true_addr=true_addr;
   (*new).next=first;
   first=new;

   return (void*)(addr);
}


void afree(void *addr)
{
   struct addr_t *p,*q;

   q=NULL;

   for (p=first;p!=NULL;p=(*p).next)
   {
      if ((*p).addr==addr)
      {
         if (q!=NULL)
            (*q).next=(*p).next;
         else
            first=(*p).next;

         free((*p).true_addr);
         free(p);
         return;
      }

      q=p;
   }
}


void error(int test,int no,char *name,char *format,...)
{
   int i,all,my_rank;
   va_list args;

   i=(test!=0);
   MPI_Allreduce(&i,&all,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   if (all==0)
      return;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      printf("\nError in %s:\n",name);
      va_start(args,format);
      vprintf(format,args);
      va_end(args);
      printf("\nProgram aborted\n\n");
      fflush(stdout);

      MPI_Abort(MPI_COMM_WORLD,no);
   }
   else
      for (i=1;i<2;i=safe_mod(i,2));
}


void error_root(int test,int no,char *name,char *format,...)
{
   int my_rank;
   va_list args;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if ((my_rank==0)&&(test!=0))
   {
      printf("\nError in %s:\n",name);
      va_start(args,format);
      vprintf(format,args);
      va_end(args);
      printf("\nProgram aborted\n\n");
      fflush(stdout);

      MPI_Abort(MPI_COMM_WORLD,no);
   }
}


int error_loc(int test,int no,char *name,char *message)
{
   if ((test!=0)&&(err_flg==0))
   {
      err_no=no;
      strncpy(prog_name,name,128);
      strncpy(err_msg,message,512);
      prog_name[127]='\0';
      err_msg[511]='\0';
      err_flg=1;
   }

   return test;
}


int mpi_permanent_tag(void)
{
   if (pcmn_cnt<MAX_PERMANENT_TAG)
      pcmn_cnt+=1;
   else
      error_loc(1,1,"mpi_permanent_tag [utils.c]",
                "Requested more than 16384 tags");

   return pcmn_cnt;
}


int mpi_tag(void)
{
   if (cmn_cnt==MAX_TAG)
      cmn_cnt=MAX_PERMANENT_TAG;

   cmn_cnt+=1;   

   return cmn_cnt;
}


void error_chk(void)
{
   int i,n,my_rank,tag1,tag2,tag3;
   int err[2];
   MPI_Status stat;

   MPI_Allreduce(&err_flg,&i,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

   if (i==0)
      return;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
   err[0]=err_flg;
   err[1]=err_no;

   if (my_rank==0)
      printf("Errors were detected on the following processes:\n\n");

   for (n=0;n<NPROC;n++)
   {
      if (n>0)
      {
         MPI_Barrier(MPI_COMM_WORLD);

         tag1=mpi_tag();
         tag2=mpi_tag();
         tag3=mpi_tag();

         if (my_rank==n)
         {
            MPI_Send(err,2,MPI_INT,0,tag1,MPI_COMM_WORLD);
            MPI_Send(prog_name,127,MPI_CHAR,0,tag2,MPI_COMM_WORLD);
            MPI_Send(err_msg,511,MPI_CHAR,0,tag3,MPI_COMM_WORLD);
         }

         if (my_rank==0)
         {
            MPI_Recv(err,2,MPI_INT,n,tag1,MPI_COMM_WORLD,&stat);
            MPI_Recv(prog_name,127,MPI_CHAR,n,tag2,MPI_COMM_WORLD,&stat);
            MPI_Recv(err_msg,511,MPI_CHAR,n,tag3,MPI_COMM_WORLD,&stat);
         }
      }

      if ((err[0]==1)&&(my_rank==0))
      {
         printf("%3d: in %s:\n",n,prog_name);
         printf("     %s (error number %d)\n",err_msg,err[1]);
      }
   }

   if (my_rank==0)
   {
      printf("\nProgram aborted\n\n");
      fflush(stdout);

      MPI_Abort(MPI_COMM_WORLD,1);
   }
   else
      for (i=1;i<2;i=safe_mod(i,2));
}


void message(char *format,...)
{
   int my_rank;
   va_list args;

   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

   if (my_rank==0)
   {
      va_start(args,format);
      vprintf(format,args);
      va_end(args);
   }
}
