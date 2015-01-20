
/*******************************************************************************
*
* File mrw.h
*
* Copyright (C) 2013 Bjoern Leder, Jacob Finkenrath
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************/

#ifndef MRW_H
#define MRW_H

#include "su3.h"

typedef enum
{
   TMRW,TMRW_EO,TMRW1,TMRW1_EO,TMRW2,TMRW2_EO,TMRW3,TMRW3_EO,TMRW4,TMRW4_EO,MRW,MRW_EO,MRW_ISO,MRW_TF,
   MRWFACTS
} mrwfact_t;

typedef struct
{
   mrwfact_t mrwfact;
   int nsrc,isp[2],nm,pwr,tmeo;
   double kappa0,kappa,m0,m,mu0,mu,gamma,kappa2;
} mrw_parms_t;

typedef struct
{
   double m1,mu1,d1;
   double m2,mu2,d2;
} mrw_masses_t;


/* MRW_C */
extern complex_dble mrw1(mrw_masses_t ms,int tm,int isp,double *sqnp,
                         double *sqne,int *status);
extern complex_dble mrw2(mrw_masses_t ms,int tm,int *isp,complex_dble *lnw1,
                         double *sqnp,double *sqne,int *status);
extern double mrw3(mrw_masses_t ms,int *isp,complex_dble *lnw1,double *sqnp,
                   double *sqne,int *status);
/* MRWEO_C */
extern complex_dble mrw1eo(mrw_masses_t ms,int tm,int isp,double *sqnp,
                         double *sqne,int *status);
extern complex_dble mrw2eo(mrw_masses_t ms,int tm,int *isp,complex_dble *lnw1,
                         double *sqnp,double *sqne,int *status);
extern double mrw3eo(mrw_masses_t ms,int *isp,complex_dble *lnw1,double *sqnp,
                   double *sqne,int *status);
/* MRW_PARMS_C */
extern void init_mrw(void);
extern mrw_parms_t set_mrw_parms(int irw,mrwfact_t mrwfact,double kappa0,double kappa,
                                 double mu0,double mu,double gamma,double kappa2,
                                 int isp1,int isp2,int nm,int pwr,int nsrc,int tmeo);
extern mrw_parms_t mrw_parms(int irw);
extern void read_mrw_parms(int irw);
extern void print_mrw_parms(void);
extern void write_mrw_parms(FILE *fdat);
extern void check_mrw_parms(FILE *fdat);
extern mrw_masses_t get_mrw_masses(int irw,int k);

#endif
