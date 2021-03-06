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
		fi = (sqrt(2.0)/(378.0*(pi**3)*sqrt(sig0)))*((-q/sig0)**(7.0/2.0))*( 1.0-(ra**-2)+(63.0/4.0)*(ra**-2)*(-q/sig0)**(-2))
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
#a=1.0
#q=-6.0

#radius range
#ra = np.logspace(-1,2,num=20)
#Fq = genDf(q)
#Henon units scale factors
lfact=3.0*pi/16.0
vfact = 1./sqrt(lfact)


#def wDF(vr, vt, r):
#	vi2 = vt**2+vr**2
#	return Fq( pot(r)-0.5*vi2, vt*r )*vt*r**2

#vphiavg = inte.tplquad(wDF, 0, np.inf, lambda x: 0, lambda x: sqrt(2.*pot(x)),lambda x,y: 0, lambda x,y: sqrt(2.*pot(x)-y**2) )
#print  (16.0*pi*pi)*vphiavg[0]

Fq = genDf(2.)
def vphir2(r):
	def wDF(vr,vt):
		vi2 = vt**2+vr**2
		return Fq( pot(r)-0.5*vi2, vt*r )*vt*vt
	vphiavg3 = inte.dblquad(wDF, 0.0, sqrt(2.*pot(r)), lambda x: 0.0, lambda x: sqrt(2.*pot(r)-x**2))
	return 8.0*vphiavg3[0]/dens(r)

vphiavg3 = inte.quad(lambda r: r**2*dens(r)*vphir2(r)**2, 0, np.inf)
print vfact*vfact*4.0*pi*vphiavg3[0]

#def wDF(vr, vt, r):
#	vi2 = vt**2+vr**2
#	return Fq( pot(r)-0.5*vi2, vt*r )*vt*r**2*vi2

#vphiavg = inte.tplquad(wDF, 0, np.inf, lambda x: 0, lambda x: sqrt(2.*pot(x)),lambda x,y: 0, lambda x,y: sqrt(2.*pot(x)-y**2) )
#print  0.5*(vfact**2)*(16.0*pi*pi)*vphiavg[0]

#print  q, sigr, 0.5*sigtheta, 0.5*(4.-q)/(2.*(6.-q)) ,  (1./(6.-q)), sigr+sigtheta,	vavg**2
# 0.5*(4.-q)/(2.*(6.-q)) ,  (1./(6.-q)),
print "#  ra sigr^2 sigtheta^2 vphi^2 <vphi>^2  "
for q in [-16, -12, -6, -2, 0, 1,2]:
#for q in [1.0,0.95,0.9,0.8, 0.75]:
	Fq = genDf(q)
#	Fq = genOMdf(q)


	def wDF(vr, vt, r):
		vi2 = vt**2+vr**2
		return Fq( pot(r)-0.5*vi2, vt*r )*vt*r**2*vr**2
	vr2 = inte.tplquad(wDF, 0, np.inf, lambda x: 0, lambda x: sqrt(2.*pot(x)),lambda x,y: 0, lambda x,y: sqrt(2.*pot(x)-y**2) )


	def wDF(vr, vt, r):
		vi2 = vt**2+vr**2
		return Fq( pot(r)-0.5*vi2, vt*r )*vt*r**2*(vt**2)
	vt2 = inte.tplquad(wDF, 0, np.inf, lambda x: 0, lambda x: sqrt(2.*pot(x)),lambda x,y: 0, lambda x,y: sqrt(2.*pot(x)-y**2) )

	def wDF(vr, vt, r):
		vi2 = vt**2+vr**2
		return Fq( pot(r)-0.5*vi2, vt*r )*vt*r**2*vt
	vphiavg3 = inte.tplquad(wDF, 0, np.inf, lambda x: 0, lambda x: sqrt(2.*pot(x)),lambda x,y: 0, lambda x,y: sqrt(2.*pot(x)-y**2) )

	sigr =  (vfact**2)*(16.0*pi*pi)*vr2[0]
	sigtheta = (vfact**2)*(16.0*pi*pi)*vt2[0]
	vavg = (vfact)*(32.0*pi)*vphiavg3[0]
	
	#aval =  (vfact**2)*(3.*pi/16.)*(1./(6.-q))
	#, sigtheta^2 = 0.5*(4.-q)/(2.*(6.-q)) ,  sigr2 = (1./(6.-q)),
	#f = 1.0 #0.25/sigr
	print  q, sigr, 0.5*sigtheta, 0.5*sigtheta,  vavg**2



