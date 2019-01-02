PlummerPlus
===========

Generates anisotropic and rotation Plummer models, see Breen, Varri & Heggie (2019) for details.

Example:

```sh
$./PlummerPlus.py -n 10000 -q -6 -o outputfile

 Dejonghe (1987) anisotropic q=-6.0 Plummer model with N = 10000  (random seed 101)
 rh = 7.722e-01 K.E. = 2.482e-01 vt^2 = 4.138e-01 vr^2 = 8.267e-02 
 1-0.5*<vt^2>/<vr^2> = -1.503e+00 
 2.0Tr/Tp = 0.4 (Polyachenko and Shukhman 1981, crit value 1.7 +/- 0.25) 
 L = [1.568e-03,-1.123e-03,1.566e-03]  |L|=2.484e-03 nf=0
 T_phi/|pot| = 2.642e-03, (assuming pot = 0.5, see ostriker & peebles 1973, 0.14 +/- 0.03)

```

### Features
 - Ansiotropic models of Dejonghe (1987), tangential and radial velocity anisotropy
 - Osipkov–Merritt model radial velocity anisotropy
 - Rotation introduced via the Lyden-Bell trick (and generalisations)
 - use -h for full list of options

### Todos

 - include embedded Plummer models using Eddington Formula 
 - Kroupa IMF (plus evolution with SSE)
