
/*******************************************************************************
*
* File map_u2blk.c
*
* Copyright (C) 2006, 2011 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
* Copying of the gauge fields to the blocks in a block grid
*
* The externally accessible functions are
*
*   void assign_ud2ubgr(blk_grid_t grid)
*     Assigns the global double-precision gauge field to the corresponding
*     single-precision fields in the specified block grid (see the notes).
*
*   void assign_ud2udblk(blk_grid_t grid,int n)
*     Assigns the global double-precision gauge field to the corresponding
*     double-precision field on the n'th block of the specified block grid
*     (see the notes).
*
* Notes:
*
* The program assign_ud2ubgr() copies the gauge field to all blocks and their
* exterior boundaries (if the field is allocated there). An error occurs if
* the single-precision gauge field on the blocks is shared. 
*
* The program assign_ud2udblk() does *not* copy the link variables to the
* boundaries of the block. The double-precision gauge field on the blocks
* must be shared in this case.
*
* As explained in README.block, the field arrays on the blocks reserve space
* for all 8 link variables at the odd points, including those on the links
* that "stick out" of the block. While the latter are used for technical
* purposes only, the programs in this module copy these too.
*
* Both programs in this module may involve communications and must be called
* on all processes simultaneously. 
*
*******************************************************************************/

#define MAP_U2BLK_C

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include "su3.h"
#include "flags.h"
#include "utils.h"
#include "lattice.h"
#include "uflds.h"
#include "block.h"
#include "global.h"

static int init=0,np,nbf[8],nmu[8],ofs[8],tags[8];
static su3 *ubuf;


static void alloc_ubuf(void)
{
   int ifc;

   if (NPROC>1)
   {
      np=(cpr[0]+cpr[1]+cpr[2]+cpr[3])&0x1;
   
      nbf[0]=FACE0/2;
      nbf[1]=FACE0/2;
      nbf[2]=FACE1/2;
      nbf[3]=FACE1/2;
      nbf[4]=FACE2/2;
      nbf[5]=FACE2/2;   
      nbf[6]=FACE3/2;
      nbf[7]=FACE3/2;

      ofs[0]=0;

      for (ifc=0;ifc<8;ifc++)
      {
         nmu[ifc]=cpr[ifc/2]&0x1;

         if (ifc>0)
            ofs[ifc]=ofs[ifc-1]+nbf[ifc-1];
      
         tags[ifc]=mpi_permanent_tag();
      }

      ubuf=amalloc(BNDRY*sizeof(*ubuf),ALIGN);
      error(ubuf==NULL,1,"alloc_ubuf [map_u2blk.c]",
            "Unable to allocate communication buffer");
   }
   
   init=1;
}


static void set_ud2u(su3_dble *ud,su3 *u)
{
   (*u).c11.re=(float)((*ud).c11.re);
   (*u).c11.im=(float)((*ud).c11.im);
   (*u).c12.re=(float)((*ud).c12.re);
   (*u).c12.im=(float)((*ud).c12.im);
   (*u).c13.re=(float)((*ud).c13.re);
   (*u).c13.im=(float)((*ud).c13.im);

   (*u).c21.re=(float)((*ud).c21.re);
   (*u).c21.im=(float)((*ud).c21.im);
   (*u).c22.re=(float)((*ud).c22.re);
   (*u).c22.im=(float)((*ud).c22.im);
   (*u).c23.re=(float)((*ud).c23.re);
   (*u).c23.im=(float)((*ud).c23.im);

   (*u).c31.re=(float)((*ud).c31.re);
   (*u).c31.im=(float)((*ud).c31.im);
   (*u).c32.re=(float)((*ud).c32.re);
   (*u).c32.im=(float)((*ud).c32.im);
   (*u).c33.re=(float)((*ud).c33.re);
   (*u).c33.im=(float)((*ud).c33.im);
}


static void fetch_bnd_u(void)
{
   int ifc,*pt,*pm;
   su3 *u;
   su3_dble *udb;

   if (init==0)
      alloc_ubuf();
   
   udb=udfld();
   u=ubuf;
   pt=map+(BNDRY/2);

   for (ifc=0;ifc<8;ifc++)
   {
      pm=pt+nbf[ifc];

      for (;pt<pm;pt++)
      {
         set_ud2u(udb+8*((*pt)-VOLUME/2)+ifc,u);
         u+=1;
      }
   }
}


static void send_bnd_u(void)
{
   int ifc,io,n;
   int saddr,raddr,tag;
   su3 *sbuf0,*rbuf0,*sbuf,*rbuf;
   MPI_Status stat;

   sbuf0=ubuf;
   rbuf0=sbuf0+(BNDRY/2);
   
   for (ifc=0;ifc<8;ifc++)
   {
      if (nbf[ifc]>0)
      {
         io=(ifc^nmu[ifc])^0x1;

         sbuf=sbuf0+ofs[io^0x1];
         rbuf=rbuf0+ofs[io];
         saddr=npr[io];
         raddr=saddr;

         n=18*nbf[ifc];
         tag=tags[ifc];
         
         if (np==0)
         {
            MPI_Send((float*)(sbuf),n,MPI_FLOAT,saddr,tag,
                     MPI_COMM_WORLD);
            MPI_Recv((float*)(rbuf),n,MPI_FLOAT,raddr,tag,
                     MPI_COMM_WORLD,&stat);
         }
         else
         {
            MPI_Recv((float*)(rbuf),n,MPI_FLOAT,raddr,tag,
                     MPI_COMM_WORLD,&stat);
            MPI_Send((float*)(sbuf),n,MPI_FLOAT,saddr,tag,
                     MPI_COMM_WORLD);
         }
      }
   }
}


static void assign_ud2ub(block_t *b)
{
   int vol,ifc;
   int ix,iy,*imb,*ipp,*imbb;
   su3 *u;
   su3_dble *udb,*vd;
   bndry_t *bb;

   vol=(*b).vol;
   imb=(*b).imb;

   udb=udfld();
   u=(*b).u;
   
   for (ix=(vol/2);ix<vol;ix++)
   {
      iy=imb[ix];
      vd=udb+8*(iy-(VOLUME/2));

      for (ifc=0;ifc<8;ifc++)
      {
         set_ud2u(vd,u);
         u+=1;
         vd+=1;
      }
   }

   bb=(*b).bb;

   if ((*bb).u!=NULL)
   {   
      for (ifc=0;ifc<8;ifc++)
      {
         vol=(*bb).vol;
         ipp=(*bb).ipp;
         u=(*bb).u;

         for (ix=0;ix<(vol/2);ix++)
         {
            iy=imb[ipp[ix]];
            set_ud2u(udb+8*(iy-(VOLUME/2))+(ifc^0x1),u);         
            u+=1;
         }

         imbb=(*bb).imb;
      
         for (;ix<vol;ix++)
         {
            iy=imbb[ix];

            if (iy<VOLUME)
               set_ud2u(udb+8*(iy-(VOLUME/2))+ifc,u);
            else
               (*u)=ubuf[iy-VOLUME];

            u+=1;
         }

         bb+=1;
      }
   }
}


void assign_ud2ubgr(blk_grid_t grid)
{
   int iprms[1],nb,isw;
   block_t *b,*bm;

   if (NPROC>1)
   {
      iprms[0]=(int)(grid);
      
      MPI_Bcast(iprms,1,MPI_INT,0,MPI_COMM_WORLD);      

      error(iprms[0]!=(int)(grid),1,"assign_u2ubgr [map_u2blk.c]",
            "Parameter is not global");
   }
   
   b=blk_list(grid,&nb,&isw);

   if (nb==0)
   {
      error_root(1,1,"assign_ud2ubgr [map_u2blk.c]",
                 "Block grid is not allocated");
      return;
   }

   if (((*b).u==NULL)||((*b).shf&0x4))
   {
      error_root(1,1,"assign_ud2ubgr [map_u2blk.c]",
                 "Gauge field on the grid is not allocated or shared");
      return;
   }
   
   if (NPROC>1)
   {
      fetch_bnd_u();
      send_bnd_u();
   }
   
   bm=b+nb;

   for (;b<bm;b++)
      assign_ud2ub(b);

   set_grid_flags(grid,ASSIGNED_UD2UBGR);
}


void assign_ud2udblk(blk_grid_t grid,int n)
{
   int nb,isw,vol,ifc;
   int ix,iy,*imb;
   su3_dble *udb,*ud,*vd;
   block_t *b;

   b=blk_list(grid,&nb,&isw);

   if ((n<0)||(n>=nb))
   {
      error_loc(1,1,"assign_ud2udblk [map_u2blk.c]",
                "Block grid is not allocated or block number out of range");
      return;
   }

   b+=n;
   ud=(*b).ud;
   
   if ((ud==NULL)||(!((*b).shf&0x8)))
   {
      error_loc(1,1,"assign_ud2udblk [map_u2blk.c]",
                "Block field is not allocated or not shared");
      return;
   }      

   vol=(*b).vol;
   imb=(*b).imb;
   udb=udfld();
   
   for (ix=(vol/2);ix<vol;ix++)
   {
      iy=imb[ix];
      vd=udb+8*(iy-(VOLUME/2));

      for (ifc=0;ifc<8;ifc++)
      {
         (*ud)=(*vd);
         ud+=1;
         vd+=1;
      }
   }
}
