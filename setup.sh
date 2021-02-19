#!/bin/bash
currentdir=$PWD
wget https://ftp.gnu.org/gnu/gsl/gsl-2.6.tar.gz
tar xvf gsl-2.6.tar.gz
cd gsl-2.6
./configure --prefix=$currentdir/gsl
make -j10
make install
cd ..
rm -f gsl-2.6.tar.gz
rm -rf gsl-2.6
