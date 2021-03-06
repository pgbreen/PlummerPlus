import scipy.special as sci
import scipy.integrate as inte 
import numpy as np
from math import pi,sqrt

# Warnings:
# using convention E>0
# units G=M=scale_radius=1 (output converted to Henon units)

#anisotropoic plummer Dejonghe 1986
def genDf(q):
	cost1 = 1.0/( sci.gamma(4.5-q)) 
	cost2 = 1.0/( sci.gamma(1.0-0.5*q) * sci.gamma(4.5-0.5*q))

	def H(a, b, c, d, x ):	
		if x  <= 1:
			return cost1*(x**a)*sci.hyp2f1(a+b, 1+a-c, a+d, x)
		else:
			y = 1.0 / x 
			return cost2*(y**b)*sci.hyp2f1(a+b, 1+b-d, b+c, y)

	#remove factor sci.gamma(0.5*q) cancels with term in H
	const3 = ( 3.0*sci.gamma(6-q)/(2.0*( (2.0*pi)**2.5 ) ) )
	def Fq(E, L):
		assert E > 0
		return const3  * (E**(3.5-q)) * H(0.0, 0.5*q, 4.5-q, 1.0, (L**2.0) / (2.0*E))

	return Fq	
# q=2 df 
def Fq2(E,L):
	assert E>0
	return (6./(2.*pi)**3)*(2.*E-L**2)**1.5


# OM df
def genOMdf(ra):
	def df(E,L):
		if E < 0:
			return 0.0
		q = -E + (L**2.0)/(2.0*ra**2)
		if q >= 0:
			return 0.0
		sig0 = 1.0/6.0
		fi = (sqrt(2.0)/(378.0*(pi**3)*sig0))*((-q/sig0)**(7.0/2.0))*( 1.0-(ra**-2)+(63.0/4.0)*(ra**-2)*(-q/sig0)**(-2))
		#assert fi >= 0, " DF negative! {0} r={1} vr,vt={2},{3} E,q={4},{5}".format(fi,r,vr,vt,E,q)		
		return fi
	return df




# potentail
def pot(r):
	return 1./sqrt(1.+r**2)

def dens(r):
	return 3./(4.0*pi*sqrt(1.+r**2)**5)

def surDens(R):
	return (1./pi)*(1.+R**2)**-2


# set alpha (a) and q
a=1.0
q=-6.0

#radius range
ra = np.logspace(-1,2,num=20)

#Henon units scale factors
lfact=3.0*pi/16.0
vfact = 1./sqrt(lfact)
for r in ra:
	vesc = sqrt(2.*pot(r))
	
	Fq = genDf(q)

	def wDF(vphi,vth,vr):
		vi2 = vphi**2+vth**2+vr**2
		vt = sqrt(vphi**2+vth**2)
		return Fq( pot(r)-0.5*vi2, vt*r )*vphi

	vphiavg = inte.tplquad(wDF, 0, vesc, lambda x: 0, lambda x: sqrt(vesc**2-x**2),lambda x,y: 0, lambda x,y: sqrt(vesc**2-x**2-y**2))

	def wDF(vt,vr):
		vi2 = vt**2+vr**2
		return 4.*pi*Fq( pot(r)-0.5*vi2, vt*r )*vt**2
	
	vphiavg2 = inte.dblquad(wDF, 0, vesc, lambda x: 0, lambda x: sqrt(vesc**2-x**2))
	vtavg = vfact*vphiavg2[0]/dens(r)
	
	def wDF(vr,vth,vphi,z):
		vi2 = vphi**2+vth**2+vr**2
		vt = sqrt(vphi**2+vth**2)
		return Fq( pot(sqrt(r**2+z**2))-0.5*vi2, vt*sqrt(r**2+z**2) )*vphi
		#Fq = genDf(q)

	def lim0(vth,vphi,z):
		vesc = sqrt(2.*pot(sqrt(r**2+z**2)))		
		return [0.0, sqrt(vesc**2 - vphi**2 - vth**2)]
	
	def lim1(vphi,z):
		vesc = sqrt(2.*pot(sqrt(r**2+z**2)))		
		return [0.0, sqrt(vesc**2 - vphi**2)]

	def lim2(z):
		vesc = sqrt(2.*pot(sqrt(r**2+z**2)))		
		return [0.0, vesc]
	
	
	def lim3():	
		return [0, np.inf]
	
	#project int
	#vphiavg2 = inte.nquad(wDF, [lim0, lim1, lim2, lim3])

	# vfact scaling to Henon untis
	# symmetry -> 2.0*a*int over Lz+
	# 4.0 from int over vphi,vtheta,vr +.+.+  
	# c2 fact 2 project z = (0,inf)
	c1 = vfact*8.0*a/dens(r)
	c2 = vfact*2.*8.0*a/surDens(r)
		
	print r*lfact, 0.5*pi*c1*vphiavg[0],vtavg, vphiavg[1] #, c2*vphiavg2[0]
	#print r*lfact, c1*vphiavg2[0], vphiavg2[1]

