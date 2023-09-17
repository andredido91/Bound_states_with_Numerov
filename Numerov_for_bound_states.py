import numpy as np
import matplotlib.pyplot as plt

#pot parameters
#C0 = -67.583747
#r_star = 0.77187385 # fm e MeV

# in a real program i woul dmake a module for the constants, but here... not worth the time
r_star = 1./4.
C0	   = -505.1500

#physical parameters, same as before
m           = 938.919/2.0        # reduced mass in MeV
hbarc       = 197.327            # (solita costante di struttura)
twomu_on_h2 = 2*m/(hbarc**2)     # it has the resuced mass if you want the readial equation (check the reduced radial problem o fQM1)

#print(1./twomu_on_h2)



def V(r):  # I kept the potential outside to change it more handly
        return C0 * np.exp(-0.25*r**2/r_star**2)
        



def standard_numerov(psi_s, xs, n, h, k):

    psi_s[0] = 0
    psi_s[1] = 1

    for j in range(n):
        if j < 2:
            pass
        elif j < n-1: #< n_mid:
            a = (1+((h**2)/12)*k(xs[j+1]))
            b = 2*(1-((5*(h**2))/12)*k(xs[j]))
            c = (1+((h**2)/12)*k(xs[j-1]))
            psi_s[j+1] = (1/a)*(b* psi_s[j]-c* psi_s[j-1])

    return psi_s, xs




def both_extreme_numerov(psi_s, xs, b, h, k):
    """two point along the x axis are shared by the left and right integrations in 
    order to compute the log derivative in the same point"""

    xs_L = xs[:b+1]
    psi_s_L = np.zeros(len(xs_L-1))
    psi_s_L[0] = 0
    psi_s_L[1] = 1

    xs_R = xs[b-1:]   # è più lungo di uno rispetto a xs_L

    psi_s_R = np.zeros(len(xs_R-1))

    psi_s_R[-1] = 0
    psi_s_R[-2] = 1
    for j in range(len(psi_s_L)-1):    # psi_s_L -1 since the last point of psi is determined at x[j_last-1]
        if j < 2:
            pass
        else:
            a           = (1+((h**2)/12)*k(xs[j+1]))
            b           = 2*(1-((5*(h**2))/12)*k(xs[j]))
            c           = (1+((h**2)/12)*k(xs[j-1]))
            psi_s_L[j+1]  = (1/a)*(b* psi_s_L[j]-c* psi_s_L[j-1])
    for j in range(len(psi_s_R)-1):
        if j < 2:
            pass
        else:
            inverse_j = len(psi_s_R)-1-j
            a_inv       = (1+((h**2)/12)*k(xs[inverse_j-1]))
            b_inv       = 2*(1-((5*(h**2))/12)*k(xs[inverse_j]))
            c_inv       = (1+((h**2)/12)*k(xs[inverse_j+1]))
            psi_s_R[inverse_j - 1] = (1/a_inv)*(b_inv* psi_s_R[inverse_j]-c_inv* psi_s_R[inverse_j+1])

    #psi_s_L = psi_s_L/np.trapz(psi_s_L)
    #psi_s_R = psi_s_R/np.trapz(psi_s_R)

    dlogpsi_L = (psi_s_L[-1]-psi_s_L[-2])/(h*psi_s_L[-2]) #*(1/h)
    dlogpsi_R = (psi_s_R[0]-psi_s_R[1])/(h*psi_s_R[0]) #*(1/h)

    diff = dlogpsi_L - dlogpsi_R
    psi_s_L = psi_s_L/np.trapz(psi_s_L)
    psi_s_R = psi_s_R/np.trapz(psi_s_R)
    plt.plot(xs_L, psi_s_L, label='Initial guess, E =' )
    plt.xlabel('r')
    plt.ylabel('psi(r)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.plot(xs_R, psi_s_R, label='Initial guess, E =' )
    plt.xlabel('r')
    plt.ylabel('psi(r)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #print(str(min(psi_s[1000:999000])))
    #print(str(psi_s))
    return psi_s, xs, diff

def get_wavefunction(E_guess,a,b,n, both_extreme, i_x_mid):
# add here explanation of input.
# solving the second order sh eq. for fixed energy from left
# E_guess - energy for the evaluation
# a - initial point (0, for radial functions)
# b - final point
# n - number of points
# i_x_mid - index of the cenvergence point applied in integration from both sids
        h = (b-a)/(n-1)
        xs = a + (np.arange(n)+0.5)*h
        psi_s = np.zeros(n)
        if both_extreme == False:
            E = E_guess           # to be used with standard_numerov
        if both_extreme == True:
            E = V(xs[i_x_mid])     # to be used with both_extreme_numerov
                
        def k(r):
            return twomu_on_h2*(E-V(r))
        
        def f(r,psi_s):
            return -k(r)*psi_s

        if both_extreme == False:
            psi_s, xs = standard_numerov(psi_s, xs, n, h, k)
            return psi_s, xs
        if both_extreme == True:
            psi_s, xs, diff = both_extreme_numerov(psi_s, xs, i_x_mid, h, k)
            return psi_s, xs, diff
    
    
    
def numerov(E_start, E_stop, Error_fun, max_iter, both_extreme):
# numerov algorith to find boundstate energy
# E_start and E_stop are the bounderies of research
# Error_fun is the error wanted for the wavefunction to be zero at Rmax --> Psi(Rmax) = 0 +- Error_fun
# max_iter is the maximum of iterations used
# add here reference to paper or website of the method is any
    E_midpoint = E_start
    first_cyc  = True
    diff_list =  []
    for i in range(max_iter):
        #print(f"E_midpoint = {E_midpoint}")
        if first_cyc:
            E_midpoint=E_start
            first_cyc=False
            i_x_mid = int(nsteps/2)
            i_x_min = 0
            i_x_max = nsteps-1
        else:
            E_midpoint = E_start - (E_start - E_stop)/2 #L/(2**(i+1))
            i_x_mid = int(i_x_min + (i_x_max-i_x_min)/2)
            
        if both_extreme == True:
            psi, xs, differ = get_wavefunction(E_midpoint,Rmin,Rmax, nsteps, both_extreme, i_x_mid) # tra -10 e -20 psi cambia segno
            print("E = ",{V(xs[i_x_mid])},"  diff = ",differ, "xs[", i_x_mid, "] =", xs[i_x_mid], "i_x_mid = ", i_x_mid)
            if abs(differ) < Error_fun:
                break
            if differ > 0:  
                i_x_min = i_x_mid
                i_x_max  = i_x_max
                print(f"dlog diff = {differ} -> choose right")       # causa inconveniente in psi_s_L[last] -> genera i_x_mid[j] > i_x_mid[j-1] 
            elif differ < 0:
                i_x_min = i_x_min
                i_x_max  = i_x_mid
                print(f"dlog diff = {differ} -> choose left")
            i_x_mid = int(i_x_min + (i_x_max-i_x_min)/2)
            diff_list = diff_list + [differ]
            print(f"{diff_list} \n \n")

        elif both_extreme == False:
            psi, xs = get_wavefunction(E_midpoint,Rmin,Rmax, nsteps, both_extreme, i_x_mid) # tra -10 e -20 psi cambia segno
            print("E = ",E_midpoint,"  Psi[Rmax] = ",psi[-1])
            if abs(psi[-1]) < Error_fun:
                break
            if psi[-1]>0:  
                E_start = E_midpoint
                E_stop  = E_stop
                #print(f"psi[{len(psi)}] = {psi[len(psi)-1]} -> choose lower")
            elif psi[-1]<0:
                E_start = E_start
                E_stop  = E_midpoint
                #print(f"psi[{len(psi)}] = {psi[len(psi)-1]} -> choose upper")
            E_midpoint = E_start - (E_start - E_stop)/2 #L/(2**(i+1))
        
        
    # add here the check of how many nodes there are in psi
    # add here psi normalization (int psi^2 = 1)
    return E_midpoint, psi, xs
        
        
    
    
    
# Try between 10000 and 1000000
nsteps  = 2000000
Rmax    = 20
Rmin    = 0


# ----------- little debug to check the method
debug = False
if (debug):
    Etest   = -0.
    psi, xs = get_wavefunction(Etest,Rmin,Rmax, nsteps)
    plt.plot(xs, psi, label='Initial guess, E = '+str(Etest))
    plt.xlabel('r')
    plt.ylabel('psi(r)')
    plt.legend()
    plt.grid(True)
    plt.show()

    Etest   = -3
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
# Error_fun is the 
E_stop   = -10
E_start  = -0
E_error = 0.000001
Error_fun   = 0.001           # Error_fun is the error wanted for the wavefunction to be zero at Rmax --> Psi(Rmax) = 0 +- Error_fun
L =  E_start - E_stop
# max_iter = int(np.log2(L/E_error)//1)+1
max_iter = 100
both_extreme = True
E_midpoint, psi, xs = numerov( E_start, E_stop, Error_fun ,max_iter , both_extreme)



plt.plot(xs, psi/np.trapz(psi), label=f'Solution at E = {E_midpoint:.6f}+-{E_error}')
plt.xlabel('r')
plt.ylabel('psi(r)')
plt.legend()
plt.grid(True)
plt.show()

print(f"E_start = {E_start},  E_midpoint = {E_midpoint},  E_stop = {E_stop}")