# NESAP-Kernels

Michael Klemm the two FFT kernels are FFT0-fftpack and FFT1-fftpack.  FFT0-fftpack is implemented as a global parallel region with the mpi calls contained within a master region with barriers around it.  FFT1-fftpack the paralallel regions are localized to the do loops.


LagrangeMultiplier1 - Lagrange multiplier code where only the plane-wave basis set is distributed, and the overlap matrices are data replicated across CPUs.  This kernel is called in nwpw for smaller numbers of CPUs or when the number of orbitals in the calculation is fairly small.  To check the numerics of the code the print out of <psi2|psi2> needs to be uncommented in lmbda_test.F
