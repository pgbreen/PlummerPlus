#!/usr/bin/env python
#
# PROGRAM: AniPlummer.py 
#
# Description: Generates realisation of anisotropic and rotating models with plummer model density distribution
# 		Note on units G=M=a=1 for generation but values are rescales to Henon units for output 
#		(see scale factors: lfact=3.0*pi/16.0, vfact=1/lfact. default output file fort.10)
#
# AUTHOR: Phil Breen, University of Edinburgh
# 
# LICENSE:	This program is free software under the GNU General Public
#               License v3 and comes with ABSOLUTELY NO WARRANTY.
#               See https://www.gnu.org/licenses/gpl.html for details.
# 
# CITATION: Please use Breen, Varri, Heggie 2018 (https://arxiv.org/abs/)
#
# EXAMPLES:
#       use "chmod a+x AniPlummer.py" to make script executable, alternativaly use "python AniPlummer.py ...."
#
# 	isotropic plummmer with 8K particles  
#	./AniPlummer.py -n 8192
#
#	8k anisotropic plummer with Dejonghe with q=-2 (see http://adsabs.harvard.edu/full/1987MNRAS.224...13D) 
#	./AniPlummer.py -n 8192 -q -2
#
# 	8k Osipkov-Merritt  radially anisotropic plummer with anisotropic radius ra=1.0 (see e.g. merritt, d. 1985. aj, 90, 1027)
#	./AniPlummer.py -n 8192 -ra 1.0
#
#	8k Einstien shpere i.e. plummer model with only ciruclar orbits
#	./AniPlummer.py -n 8192 -e
#
#	8K isotropic plummmer with rotation via Lyden-Bell trick i.e. reverse velocities of 50% particles with L_z < 0
#       ./AniPlummer.py -n 8192 -a 0.5
# 
#	8K isotropic plummmer with offset rotation, 50% of the most bound stars rotating about z-axis 
# 	least bound 50% rotating is offset by 90 degres (i.e. about y-axis). 
#       Note most stars use -a for the fraction of stars reverse where as least bount stars use second value in -oa flag (i.e. -oa angle flipfraction)
#       ./AniPlummer.py -n 8192 -a 1.0  -oa 90.0 1.0 0.5
#
# 	for a high shear model, set offset to 180 degree (i.e. -z) and used mass fraction 0.67 (i.e. rotatin model with 0 net L!) 
# 	./AniPlummer.py -n 10000  -a 1.0 -oa 180.0 1.0 0.67
#
#	create mass segrated model most bound 50% of the mass consisting of particles 10.0 times more massive then the least
#	./AniPlummer.py -n 10000 -ms 0.5 10.0
#
""" AniPlummer.py: generates realisation of anisotropic and rotating models with plummer model density distribution """

import argparse
import numpy as np
import scipy.special as sci
from scipy.interpolate import interp1d
from math import pi, sqrt, cos, sin

parser = argparse.ArgumentParser(description="Generates anisotropic and rotating models with plummer model density distribution  ")

parser.add_argument("-n", help="number of particles (default: 1000)",
                    type=int,default=1000,metavar="")

parser.add_argument("-rs", help="random number generator seed (default: 101)",
                    type=int,default=101,metavar="rand_seed")

parser.add_argument("-o", help="name of output file (default: \"fort.10\")",
                    type=str,default="fort.10",metavar="")



# rotation parameters 
parser.add_argument("-a", help="Lynden-Bell trick flip fraction",
                    type=float,default=0,metavar="")

parser.add_argument("-oa", help="Offset angle in degrees for particles below energy cut off and flip fraction for offset stars (e.g. -oa 90.0 0.5): OA offset angle, A2 flip fraction, MF mass fraction above which the offset is applied ",
                    type=float,default=[0,0,-1],metavar=("OA","A2","MF"),nargs=3)


parser.add_argument("-ms", help="Mass segregated model, set fraction of total mass (MF) and mass ratio (MR). This will reduce number of particles see Readme file. ",type=float,default=[0,0],metavar=("MF","MR"),nargs=2)


# exclusice arguments for velocity space
group = parser.add_mutually_exclusive_group()
group.add_argument("-q", help="q values of Dejonghe (1987) anisotropic plummer models, q<=+2 ",
                    type=float,default=0,metavar="")

group.add_argument("-ra", help="anisotropic radius of Osipkov-Merritt radially anisotropic plummer model, ra >= 0.75",
                    type=float,default=0,metavar="")

group.add_argument("-e", help=" Einstein sphere, plummer model with only circular orbits ",
                    action="store_true")

args = parser.parse_args()

np.random.seed(args.rs)

w = np.zeros((args.n, 7))

# w[:,0] = mass, w[:,1:4] = x,y,z, w[:,4:] = vx,vy,zx
w[:,0] = 1.0/float(args.n) 
 

#---------------------------------generate positions------------------------------------

x = np.random.rand(args.n)
r = np.reciprocal(np.sqrt(np.power(x,-2.0/3.0)-1.0))

ctheta = np.random.uniform(-1.,1.0,args.n)
stheta = np.sin(np.arccos(ctheta))
phi = 2.0*pi*np.random.rand(args.n) 

w[:,1] = stheta*np.cos(phi)
w[:,2] = stheta*np.sin(phi)
w[:,3] = ctheta

w[:,1:4] = w[:,1:4]*r[:,None]


#---------------------------------generate velocities------------------------------------

#calucates orthogonal basis using r and returns random vr,vt units (needed for anisotorpic models)
if args.q != 0 	or args.ra != 0 or args.e:
	sign = [-1.0,1.0]
	theta = 2.0*pi*np.random.rand(args.n)
	ctheta = np.cos(theta)
	stheta = np.sin(theta)
	rbit = np.random.randint(2, size=args.n)
	ui = 0
	def unitv(ri):
		ru = ri/np.linalg.norm(ri)
		e1 = [ri[1],-ri[0],0.0]
		e1 /= np.linalg.norm(e1)
		e2 = np.cross(ru,e1)
		#theta = 2.0*pi*np.random.rand()
		#vr = ru*sign[np.random.randint(2)] 
		#vt = (e1*cos(theta) + e2*sin(theta))
		vr = ru*sign[rbit[unitv.i]] 
		vt = (e1*ctheta[unitv.i] + e2*stheta[unitv.i])
		unitv.i += 1
		return vr,vt
	unitv.i = 0


#anisotropoic plummer Dejonghe
if args.q != 0:
	assert args.q <= +2, " q value needs to be in range (-inf,+2) " 
	#
	# Define distribution function as in Dejonghe (1987)
	#
	# sci.gamma(a+b) replaced with 1.0 as cancels with sci.gamma(0.5*q) in Fq
	# for x <= 1, sci.gamma(a+d) = 1 
	# cost1 coeff for x=<1, cost2 coeff for x>1 
	cost1 = 1.0/( sci.gamma(4.5-args.q)) 
	cost2 = 1.0/( sci.gamma(1.0-0.5*args.q) * sci.gamma(4.5-0.5*args.q))

	def H(a, b, c, d, x ):	
		if x  <= 1:
			return cost1*(x**a)*sci.hyp2f1(a+b, 1+a-c, a+d, x)
		else:
			y = 1.0 / x 
			return cost2*(y**b)*sci.hyp2f1(a+b, 1+b-d, b+c, y)

	#remove factor sci.gamma(0.5*q) cancels with term in H
	const3 = ( 3.0*sci.gamma(6-args.q)/(2.0*( (2.0*pi)**2.5 ) ) )
	def Fq(E, L):
		assert E > 0
		return const3  * (E**(3.5-args.q)) * H(0.0, 0.5*args.q, 4.5-args.q, 1.0, (L**2.0) / (2.0*E))

 	sf = 1.1 	# increase fmax found on grid by sf
	steps = 100.0 	# step size in velocity space vmax/steps
	def maxfq(psi,r):
		maxfq = 0.0
		vmax = sqrt(2.0*psi)
		incr = vmax/steps
		for ev in np.arange(0,vmax,incr):
			E=psi-0.5*ev**2
			if E <= 0 and abs(E) < 1e-15:
				continue
			for jv in np.arange(0,vmax,incr):
				if ( jv > ev ): #not realisitc
					continue 	
				L=jv*r
				val = Fq(E,L)*jv			
				if (maxfq < val):
					maxfq = val
		return sf*maxfq 
	
	psirange = np.linspace(0.000, 1.000, num=100, endpoint=True)
	maxfqarray = np.zeros(len(psirange))	
	for i,psi in enumerate(psirange) :
		if psi == 0.0:
			maxfqarray[i] = 0.0
			continue
		ri = sqrt((1.0/psi)**2.0-1.0)
		maxfqarray[i] = maxfq(psi,ri)
	
	# linear interpolation function to calculate bound
	psimax = interp1d(psirange, maxfqarray)

	# accept reject sampling
	r2 = np.power(r,2)
	psi = np.reciprocal(np.sqrt(r2 + 1.0))
	fmaxv = psimax(psi)
	vmax = np.sqrt(2.0*psi)
	v = np.zeros(args.n)	
	rvf =  np.random.rand(int(round(50*args.n)))
	rvc = 0
	for i in xrange(args.n):
		fmax = fmaxv[i]
		loopc = 0
		while True:
			#rv =  np.random.rand(3)
			vr =  rvf[rvc+0]*vmax[i]
			vt =  rvf[rvc+1]*vmax[i]
			l = r[i]*vt 
			vsq = vr**2 + vt**2
			E =  psi[i] - 0.5*vsq

			if E < 0:
				rvc+=2
				continue			

			f1 =  rvf[rvc+2]*fmax
			f = Fq(E,l)*vt

			rvc += 3
			
			if f >= f1:  
			#	print i,loopc
				vrv,vtv = unitv(w[i,1:4])
				w[i,4:] = vrv*vr+vtv*vt
				break

			if rvc == len(rvf):
				rvf = np.random.rand(int(round(50*args.n)))
				rvc = 0		

			loopc += 1
			if loopc > 10000:
				print r[i], fmax, E, l, 
				raise NameError('Failed to sample')

	#print float(rvc)/float(len(rvf)),rvc,len(rvf),i

			
# anisotropoic plummer Osipkov-Merritt radial only Osipkov 1979; Merritt 1985
elif  args.ra != 0:
	assert args.ra >= +0.75, " ra value needs to be in range (+0.75,+inf) " 
	def df(psi,r,vr,vt):
		E = psi - 0.5*(vr**2+vt**2)
		if E < 0:
			return 0.0
		q = -E + (r**2.0)*(vt**2.0)/(2.0*args.ra**2)
		if q >= 0:
			return 0.0
		sig0 = 1.0/6.0
		fi = (sqrt(2.0)/(378.0*(pi**3)*sig0))*((-q/sig0)**(7.0/2.0))*( 1.0-(args.ra**-2)+(63.0/4.0)*(args.ra**-2)*(-q/sig0)**(-2))

		assert fi >= 0, " DF negative! {0} r={1} vr,vt={2},{3} E,q={4},{5}".format(fi,r,vr,vt,E,q)		
		return fi

	sf = 1.1 	# increase fmax found on grid by sf
	steps = 100.0 	# step size in velocity space vmax/steps
	def maxfq(psi,r):
		maxfq = 0.0
		vmax = sqrt(2.0*psi)
		incr = vmax/steps
		for ev in np.arange(0,vmax,incr):
			E=psi-0.5*ev**2
			if E <= 0 and abs(E) < 1e-15:
				continue
			for jv in np.arange(0,vmax,incr):
				if ( jv > ev ): #not realisitc
					continue 	
				L=jv*r
				val = df(psi,r,ev,jv)*jv			
				if (maxfq < val):
					maxfq = val
		return sf*maxfq 
	
	psirange = np.linspace(0.000, 1.000, num=100, endpoint=True)
	maxfqarray = np.zeros(len(psirange))	
	for i,psi in enumerate(psirange) :
		if psi == 0.0:
			maxfqarray[i] = 0.0
			continue
		ri = sqrt((1.0/psi)**2.0-1.0)
		maxfqarray[i] = maxfq(psi,ri)
	
	# linear interpolation function to calculate bound
	psimax = interp1d(psirange, maxfqarray)

	# accept reject sampling
	r2 = np.power(r,2)
	psi = np.reciprocal(np.sqrt(r2 + 1.0))
	fmaxv = psimax(psi)
	vmax = np.sqrt(2.0*psi)
	v = np.zeros(args.n)	
	rvf =  np.random.rand(int(round(40*args.n)))
	rvc = 0
	for i in xrange(args.n):
		fmax = fmaxv[i]
		loopc = 0
		while True:
			rvc+=3
			vr = rvf[rvc]*vmax[i]
			vt = rvf[rvc+1]*vmax[i]
			l = r[i]*vt 
			vsq = vr**2 + vt**2
			E =  psi[i] - 0.5*vsq

			if E < 0:
				continue
			
			f1 =  rvf[rvc+2]*fmax
			f = df(psi[i],r[i],vr,vt)*vt

			if f >= f1:  
				# vrv,vtv random vr vt unit vectors 
				vrv,vtv = unitv(w[i,1:4])
				w[i,4:] = vrv*vr+vtv*vt
				break
			loopc += 1
			if loopc > 10000:
				print r[i], fmax, E, l, 
				raise NameError('Failed to sample')
	#print float(rvc)/float(len(rvf))

#Einstein sphere
elif args.e:
	r2 = np.power(r,2)
	v = np.sqrt( np.divide(r2, np.power( np.sqrt( r2+1.0 ),3.0 ) ))

	for i in xrange(args.n):
		# create unit vectors e1,e2 in plane normal to r
		# assuming ri[0], ri[1] non-zero 
		vr,vt = unitv(w[i,1:4])
		w[i,4:] = vt*v[i]

	
#isotropoic plummer 
else:
	r2 = np.power(r,2)
	vmax = np.sqrt(2.0*np.reciprocal(np.sqrt(r2 + 1.0)))
	v = np.zeros(args.n)
	for i in xrange(args.n):
		while True:
			xi = np.random.rand(2)
			f1 =  xi[1]*0.1
			f = xi[0]*xi[0]*(1.0 - xi[0]*xi[0])**3.5
			if f >=  f1:
				break
		v[i] = vmax[i]*xi[0]

	ctheta = 2.0*np.random.rand(args.n)-1.0
	stheta = np.sin(np.arccos(ctheta))
	phi = 2.0*pi*np.random.rand(args.n) 

	w[:,4] = stheta*np.cos(phi)
	w[:,5] = stheta*np.sin(phi)
	w[:,6] = ctheta

	w[:,4:] = w[:,4:]*v[:,None]

#---------------------------------add rotation------------------------------------

countflip = 0
if args.oa[2] >= 0.0:
	# general trick energy cut and 
	E = 0.5*np.power(v,2.0) - np.reciprocal(np.sqrt(np.power(x,2.0) + 1.0))
	ecut = np.percentile(E, 100.0*args.oa[2])

#if args.oa[0] > 0:
	theta = pi*args.oa[0]/180.0
	ov = np.array([0.0,sin(theta ),cos(theta )])
	L = np.cross(w[:,1:4],w[:,4:])	
	
	for i in xrange(args.n):	
		if L[i,2] < 0.0 and E[i] < ecut:
			if args.a > np.random.rand():
				w[i,4:] *= -1.0
				countflip+=1
		if E[i] > ecut:
			if np.dot(L[i,:],ov) < 0.0:
				if args.oa[1] > np.random.rand():
					w[i,4:] *= -1.0
					countflip+=1

elif args.a > 0:
	#basic LB trick 
	countflip = 0
	L = np.cross(w[:,1:4],w[:,4:])
	for i in xrange(args.n):	
		if L[i,2] < 0.0:
			if args.a > np.random.rand():
				w[i,4:] *= -1.0
				countflip += 1


#-----------------------------handle mass segregation---

if args.ms[0] > 0 and args.ms[1] > 0:
	if args.oa[2] == -1:
		E = 0.5*np.power(v,2.0) - np.reciprocal(np.sqrt(np.power(x,2.0) + 1.0))
	ecut = np.percentile(E, 100.0*args.ms[0])
		
	nfold = args.ms[0]*args.n
	nkeep  = int(round(nfold/args.ms[1]))	
	nc = 0
	
	rl = []
	for i in xrange(args.n):
		if E[i] < ecut:
			nc+=1 
			if nc<=nkeep:
				w[i,0] *= args.ms[1]
			else:
				
				rl.append(i)
	print "\n Warning mass segregation particle number reduced n = {}!".format(args.n-len(rl))
		
	w=np.delete(w,rl,axis=0)
	# reset n and renormlise mass to account for round error
	w[:,0] *= 1.0/sum(w[:,0])
	args.n = args.n-len(rl)
#
	


#--------------------------------scale to Henon units and save data--------------------------------

# scale to henon units and save data to output file "fort.10" (use -o to rename output)
lfact=(3.0*pi)/16.0		 
vfact = 1.0/sqrt(lfact)
w[:,1:4] *= lfact
w[:,4:] *= vfact

np.savetxt(args.o, w)

# statistics 
if args.e:
	name = " Einstein sphere"
elif args.ra > 0:
	name = " Osipkov-Merritt radially anisotropic (ra={})".format(args.ra)
elif args.q != 0:
	name = " Dejonghe (1987) anisotropic q={}".format(args.q)
else:
 	name = " Isotropic"


print "\n{} Plummer model with N = {}  ".format(name, args.n )


svr2 = 0.0
svt2 = 0.0
for i in xrange(args.n):
	vi = w[i,4:]
	xi = w[i,1:4]
	xiu = xi/np.linalg.norm(xi)
	vr = np.inner(xiu,vi)
	vt = np.linalg.norm(vi - vr*xiu)
	svr2 += w[i,0]*vr**2
	svt2 += w[i,0]*vt**2

if args.ms[0] == 0:
	r2 = np.sum(np.power(w[:,1:4],2), axis=1)
	rh = sqrt(np.median(r2))
else:
	r2 = np.sum(np.power(w[:,1:4],2), axis=1)
	indx = sorted(range(args.n),key=lambda k: r2[k])
	mc = 0.0 
	for i in xrange(args.n):
		mc += w[indx[i],0]
		if mc >= 0.5:
			rh = sqrt(r2[indx[i]])
			break

print " rh = {:.3e} K.E. = {:.3e} vt^2 = {:.3e} vr^2 = {:.3e} ".format(rh,0.5*(svt2+svr2) ,svt2,svr2)
if True:
#if not args.e:
	print " 1-0.5*<vt^2>/<vr^2> = {:.3e} ".format(1.0 - 0.5*svt2/svr2)
	print " 2.0Tr/Tp = {:.3} (Polyachenko and Shukhman 1973, crit value 1.7 +/- 0.25) ".format(2.0*svr2/svt2)

#if args.a > 0 or args.oa[0] > 0:
	L = np.multiply(w[:,0,None],np.cross(w[:,1:4],w[:,4:]))
	L = np.sum(L, axis=0)
	print " L = [{:.3e},{:.3e},{:.3e}]  |L|={:.3e} nf={}".format(L[0],L[1],L[2],np.linalg.norm(L),countflip )

#
# bins size, number of rings nbin^2, for nbin=10 each ring contains 1% mass
#
def sign(x):
	if x >= 0:
		return +1.0
	else:
		return -1.0
#hack use pre cal R2,|z| bins as rho fixed
rindex = [0.0385891298799, 0.0866921555876, 0.148588985051, 0.230883647572, 0.345272808206,
0.517942199325,0.807197126028, 1.38667629541, 3.10347289646]

nbin = 10
R2 = np.sum(np.power(w[:,1:3],2), axis=1)
z = np.abs(w[:,3])
indx = sorted(range(args.n),key=lambda k: R2[k])
Rbins = np.array_split(indx,nbin)
vi = w[:,4:]
xi = np.zeros((args.n,3))
xi[:,0] = w[:,2]
xi[:,1] = -w[:,1]
xi = np.divide(xi,np.sqrt(R2)[:,None])
vphi = xi[:,0]*vi[:,0] + xi[:,1]*vi[:,1]
vphike = 0.0
for rl in Rbins:
	zi =  z[rl]
	indxi = sorted(range(len(zi)),key=lambda k: zi[k])
	zbins = np.array_split(indxi,nbin)
	#print R2[rl[0]],R2[rl[-1]]
	for zl in zbins:
		
		#print "{:.2e}".format(zi[zl[-1]]),
		vphiavg = np.sum(vphi[rl[zl]])
		vsign = sign(vphiavg)
		#mass = np.sum(w[rl[zl],0])
		vphike += vsign*(vphiavg**2)/len(zl)
	#print ""


print " T_phi/|pot| = {:.3e}, (assuming pot = 0.5)".format(-1.0*vphike/args.n)
print "\n"


