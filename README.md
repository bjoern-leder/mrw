
********************************************************************************
      
            mrw - Mass and twisted-mass reweighting based on openQCD

********************************************************************************

mrw is a module for the openQCD software package 
(http://luscher.web.cern.ch/luscher/openQCD/)
providing additional reweighting factors for mass and twisted-mass
reweighting. Details on the stochastic estimator used in the module can
be found in 

      J. Finkenrath, F. Knechtli and B. Leder, "One flavor mass reweighting in lattice QCD,"
      Nucl. Phys. B 877, 441 (2013), arXiv:1306.3962 [hep-lat]

The latest release is based on openQCD-1.2.


      FEATURES

- openQCD reweighting type I and II (even-odd preconditioned) with interpolation
- factorization with non-equidistant interpolations
- mass reweighting with twisted-mass term only on even sites
- isospin mass reweighting
- strange quark mass reweighting (simultaneous reweighting of light and strange quark mass)
- several check routines for all parts of the module


      DESCRIPTION OF THE MODULE

The module adds two new directories (modules/mrw and devel/mrw) and a new main 
program (main/ms5.c) to the original
openQCD package. A full list of the added and changed files:

modules/mrw
devel/mrw
main/ms5.c
main/ms5.in
main/ms5_test_*.in
main/Makefile
doc/mrw.pdf
include/mrw.h
CHANGELOG.mrw

Details on the implementation of the reweighting factors can be found
in the documentation included in the package (doc/mrw.pdf). For
compilation and linking with MPI see the openQCD part of the package.


      KNOWN ISSUES OF THE LATEST RELEASE

- some check routines (check4 and check9) are statistical with limited 
significance
- no even-odd preconditioning for mass reweighting yet
- file format of ms5.dat differs from the one of ms1.dat


      OCTAVE ANALYSIS SCRIPTS

Octave/Matlab scripts for importing the output of the programs can be downloaded from 
http://www-ai.math.uni-wuppertal.de/~leder/mrw/octave-mrw-1.0.tar.gz.
Scripts for plotting and simple analysis are included as
well. The scripts assume Octave version 3.8 or newer.


      AUTHORS

The mrw module was written by Björn Leder (leder(at)math.uni-wuppertal.de) and 
Jacob Finkenrath. The
Octave/Matlab scripts have been written by Björn Leder. For the authors
of openQCD see its webpage http://luscher.web.cern.ch/luscher/openQCD/.


      LICENSE

The software may be used under the terms of the GNU General Public License (GPL)