import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal, getcontext

# Set the desired precision (number of decimal places)
getcontext().prec = 24  # You can adjust this value based on your needs

# pot parameters
# C0      = Decimal('-67.583747')
# r_star  = Decimal('0.77187385')  # fm e MeV

# in a real program i woul dmake a module for the constants, but here... not worth the time
r_star = Decimal('1')/Decimal('4')
C0	   = Decimal('-505.1500')

#physical parameters, same as before
m           = Decimal('938.919') / Decimal('2.0')  # reduced mass in MeV
hbarc       = Decimal('197.3269804')
twomu_on_h2 = Decimal('2') * m / (hbarc ** Decimal('2'))     # it has the resuced mass if you want the readial equation (check the reduced radial problem o fQM1)

#print(1./twomu_on_h2)
        
def V(r):
        return C0 * np.exp(-Decimal('0.25') * r**Decimal('2') / (r_star**Decimal('2')))

def get_wavefunction(E_guess,a,b,n):
# add here explanation of input.
# solving the second order sh eq. for fixed energy from left
# E_guess - energy for the evaluation
# a - initial point (0, for radial functions)
# b - final point
# n - number of points

        h = (b-a)/(n-1)
        xsa = a + (np.arange(n)+0.5)*h
        xs = [Decimal(str(x)) for x in xsa]
        psi_sa = np.zeros(n)
        psi_s = [Decimal(str(x)) for x in psi_sa] 

        E = E_guess
        
        def k(r):
            return twomu_on_h2 * (E - V(r))
        
        def f(r,psi_s):
            return -k(r)*psi_s


        psi_s[0] = 0
        psi_s[1] = 1
        for j,x in enumerate(xs):
            if j < 2:
                pass
            elif j < n-1:
                a = Decimal(str((1+((h**2)/12))))*k(xs[j+1])
                b = Decimal(str(2*(1-((5*(h**2))/12))))*k(xs[j])
                c = Decimal(str((1+((h**2)/12))))*k(xs[j-1])
                psi_s[j+1] = (Decimal('1')/a)*(b* psi_s[j]-c* psi_s[j-1])
            
        return psi_s, xs
    
    
    
def numerov(E_start, E_stop, Error_fun, max_iter):
# numerov algorith to find boundstate energy
# E_start and E_stop are the bounderies of research
# Error_fun is the error wanted for the wavefunction to be zero at Rmax --> Psi(Rmax) = 0 +- Error_fun
# max_iter is the maximum of iterations used
# add here reference to paper or website of the method is any

    E_midpoint = E_start
    first_cyc  = True
    for i in range(max_iter):
        #print(f"E_midpoint = {E_midpoint}")
        if first_cyc:
            E_midpoint=E_start
            first_cyc=False
        else:
            E_midpoint = E_start - (E_start - E_stop)/2 #L/(2**(i+1))
        psi, xs = get_wavefunction(E_midpoint,Rmin,Rmax, nsteps) # tra -10 e -20 psi cambia segno
        print("E = ",E_midpoint,"  Psi[Rmax] = ",psi[-1])
        if abs(psi[-1]) < Error_fun: #this is already the last point of psi
            break
        if psi[-1] > Decimal('0'):  
            E_start = E_midpoint
            E_stop  = E_stop
            print(f"psi[{len(psi)}] = {psi[len(psi)-1]} -> choose right")
        elif psi[-1] < Decimal('0'):
            E_start = E_start
            E_stop  = E_midpoint
            print(f"psi[{len(psi)}] = {psi[len(psi)-1]} -> choose left")
        #E_midpoint = E_start - (E_start - E_stop)/2 #L/(2**(i+1))
        
        
    # add here the check of how many nodes there are in psi
    # add here psi normalization (int psi^2 = 1)
    return E_midpoint, psi, xs
        
        
    
    
    
# Try between 10000 and 1000000
nsteps  = 10000
Rmax    = 50
Rmin    = 0


# ----------- little debug to check the method
debug = True
if (debug):
    Etest   = Decimal('-0')
    psi, xs = get_wavefunction(Etest,Rmin,Rmax, nsteps)
    #psii = [float(x) for x in psi]
    #integ = np.trapz(psii)
    #print(type(integ))
    # integral = Decimal(str(integ))
    plt.plot(xs, psi, label='Initial guess, E = '+str(Etest))
    plt.xlabel('r')
    plt.ylabel('psi(r)')
    plt.legend()
    plt.grid(True)
    plt.show()

    Etest   = Decimal('-3')
    psi, xs = get_wavefunction(Etest,Rmin,Rmax, nsteps)
    plt.plot(xs, psi, label='Initial guess, E = '+str(Etest))
    plt.xlabel('r')
    plt.ylabel('psi(r)')
    plt.legend()
    plt.grid(True)
    plt.show()
# -------------




# Main numerov algorith to search boundstats
# Eerror is the required accuracy for the energy of the bound states
# Since E_error = L/2^{N} the number of steps required for a prescripted energy are given by Maxiter = np.log2(L/R) + 1 as an integer
E_stop   = Decimal('-10')
E_start  = Decimal('-0')
E_error = 0.000001
Error_fun   = 0.001
L =  E_start - E_stop
# max_iter = int(np.log2(L/E_error)//1)+1
max_iter = 100
E_midpoint, psi, xs = numerov(E_start,E_stop,E_error,max_iter)



plt.plot(xs, psi/np.trapz(psi), label=f'Solution at E = {E_midpoint:.6f}+-{E_error}')
plt.xlabel('r')
plt.ylabel('psi(r)')
plt.legend()
plt.grid(True)
plt.show()

print(f"E_start = {E_start},  E_midpoint = {E_midpoint},  E_stop = {E_stop}")