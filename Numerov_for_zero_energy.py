import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#pot parameters
#C0 = -67.583747
#r_star = 0.77187385 # fm e MeV

# in a real program i would make a module for the constants, but here not worth the time
#r_star = 1./4.
#C0	   = -505.1500

#pot parameters
C0 = -67.583747
r_star = 0.77187385 # fm e MeV

# Physical parameters
m           = 938.919/2.0        # reduced mass in MeV
hbarc       = 197.327            # (solita costante di struttura)
twomu_on_h2 = 2*m/(hbarc**2)     # it has the resuced mass if you want the readial equation (check the reduced radial problem o fQM1)

# Define the linear function to fit
def linear_function(x, a, b):
    return a * x + b

def V(r):
    return C0 * np.exp(-0.25*(r)**2/r_star**2)     # 7.7 -> 7.95

#def V(r):
#        return C0 * np.exp(-0.25*(r)**2/r_star**2) + 7.99/r**2     # 7.7 -> 7.95

def find_index_with_min_difference(vector, x_mid):
    """
    Find the index j of the vector for which the difference between the j-th entry and x_mid is minimal.

    Parameters:
    vector (list or numpy.array): Input vector.
    x_mid (float): Scalar value.

    Returns:
    int: Index j for which the difference is minimal.
    """
    min_difference = float('inf')  # Initialize with infinity
    min_index = None

    for i, element in enumerate(vector):
        difference = abs(element - x_mid)
        if difference < min_difference:
            min_difference = difference
            min_index = i

    return min_index

def bisection_method(V, E_guess, a, b, tol=1e-12, max_iterations=1000):
    """
    Find the root of the equation V(x) - E_guess = 0 using the bisection method.

    Parameters:
    V (callable): The function V(x) for which we want to find the root.
    E_guess (float): Initial guess for the root.
    a (float): Left endpoint of the initial interval.
    b (float): Right endpoint of the initial interval.
    tol (float): Tolerance (stop when the interval size is less than this).
    max_iterations (int): Maximum number of iterations.

    Returns:
    float: Approximation of the root.
    """

    # Check if the initial endpoints have opposite signs
    #if V(a) * V(b) >= 0:
    #    raise ValueError("The function values at the endpoints must have opposite signs.")

    iteration = 0
    while (b - a) / 2 > tol and iteration < max_iterations:
        c = (a + b) / 2  # Compute the midpoint
        V_c = V(c) - E_guess

        # Update the interval
        if V_c == 0:
            return c  # Found the exact root
        elif V_c * V(a) < 0:
            b = c
        else:
            a = c

        iteration += 1

    return (a + b) / 2  # Return the approximation of the root

def standard_numerov(psi_s, xs, n, h, k):
    """
    Perform the Numerov algorithm to solve a differential equation using the given parameters.

    Parameters:
    psi_s (numpy.array): Array to store the solution.
    xs (numpy.array): Array of x values.
    n (int): Number of steps for the iteration.
    h (float): Step size.
    k (function): Function representing the coefficient.

    Returns:
    tuple: Tuple containing the updated psi_s array and xs array.
    """

    # Initialize initial conditions
    psi_s[0] = 0
    psi_s[1] = 1

    # Iterate through the range of n
    for j in range(n):
        if j < 2:
            pass
        elif j < n-1:    # Ensure we're within bounds
            # Compute coefficients for the Numerov algorithm
            a = (1+((h**2)/12)*k(xs[j+1]))
            b = 2*(1-((5*(h**2))/12)*k(xs[j]))
            c = (1+((h**2)/12)*k(xs[j-1]))
            
            # Update psi_s using Numerov algorithm
            psi_s[j+1] = (1/a)*(b* psi_s[j]-c* psi_s[j-1])

    return psi_s, xs  # Return updated psi_s and xs arrays



def both_extreme_numerov(psi_s, xs, b, h, k):
    """
    Perform the Numerov algorithm for both extreme regions of the differential equation and plot the results.

    Parameters:
    psi_s (numpy.array): Array to store the solution.
    xs (numpy.array): Array of x values.
    b (int): Index to split the array xs into left and right regions.
    h (float): Step size.
    k (function): Function representing the coefficient.

    Returns:
    tuple: Tuple containing the updated psi_s array, xs array, and the difference in logarithmic derivatives.
    """

    # Split and initialize xs and psi_s into left and right regions
    xs_L = xs[:b+1]
    psi_s_L = np.zeros(len(xs_L))
    psi_s_L[0] = 0
    psi_s_L[1] = 0.5

    xs_R = xs[b-1:]
    psi_s_R = np.zeros(len(xs_R))
    psi_s_R[-1] = 0.5
    psi_s_R[-2] = 1

    # Perform Numerov algorithm for the left region
    for j in range(len(psi_s_L)-1):
        if j < 2:
            pass
        else:
            a           = (1+((h**2)/12)*k(xs[j+1]))
            b           = 2*(1-((5*(h**2))/12)*k(xs[j]))
            c           = (1+((h**2)/12)*k(xs[j-1]))
            psi_s_L[j+1]  = (1/a)*(b* psi_s_L[j]-c* psi_s_L[j-1])

    # Perform Numerov algorithm for the right region
    for j in range(len(psi_s_R)-1):
        if j < 2:
            pass
        else:
            inverse_j = len(psi_s_R)-1-j
            a_inv       = (1+((h**2)/12)*k(xs[inverse_j-1]))
            b_inv       = 2*(1-((5*(h**2))/12)*k(xs[inverse_j]))
            c_inv       = (1+((h**2)/12)*k(xs[inverse_j+1]))
            psi_s_R[inverse_j - 1] = (1/a_inv)*(b_inv* psi_s_R[inverse_j]-c_inv* psi_s_R[inverse_j+1])

    # Compute the difference in logarithmic derivatives and normalize the wave functions
    dlogpsi_L = (psi_s_L[-1]-psi_s_L[-2])/(h*psi_s_L[-2])
    dlogpsi_R = (psi_s_R[1]-psi_s_R[0])/(h*psi_s_R[0])
    alpha = psi_s_L[-1]/psi_s_R[1]
    psi_s_L = psi_s_L*alpha
    diff = dlogpsi_L - dlogpsi_R
    psi_s = np.concatenate((psi_s_L[:-2],psi_s_R), axis = 0)
    psi_s_L = psi_s_L/np.trapz(psi_s_L)
    psi_s_R = psi_s_R/np.trapz(psi_s_R)
    psi_s = psi_s/np.trapz(psi_s)

    # Plot the wave functions
    plt.plot(xs, psi_s, label=f'psi_s' )
    plt.plot(xs_L, psi_s_L, label=f'psi_s_L' )
    plt.plot(xs_R, psi_s_R, label=f'psi_s_R' )
    plt.xlabel('r')
    plt.ylabel('psi(r)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return psi_s, xs, diff


def get_wavefunction(E_guess,a,b,n, both_extreme, i_x_mid):
    """
    Compute the wavefunction using the Numerov algorithm based on the provided parameters.

    Parameters:
    E_guess (float): Initial energy guess.
    a (float): Lower bound of the region.
    b (float): Upper bound of the region.
    n (int): Number of steps for the iteration.
    both_extreme (bool): Whether to use both extreme Numerov algorithm or standard Numerov.
    i_x_mid (int): Index to use for the both extreme Numerov algorithm.

    Returns:
    tuple: Tuple containing the wavefunction psi_s array and xs array for standard Numerov,
           or tuple containing the wavefunction psi_s array, xs array, and the difference in logarithmic derivatives
           for both extreme Numerov.
    """

    h = (b-a)/(n-1)
    xs = a + (np.arange(n))*h    # or (np.arange(n)+0.5 )*h
    psi_s = np.zeros(n)

    # Set the energy value based on the algorithm choice
    if both_extreme == False:
        E = E_guess             # to be used with standard_numerov
    if both_extreme == True:
        E = V(xs[i_x_mid])      # to be used with both_extreme_numerov
    

    # Define the k and f functions based on the provided potential
    def k(r):
        return twomu_on_h2*(E-V(r))
    def f(r,psi_s):
        return -k(r)*psi_s

    # Use the appropriate Numerov algorithm based on the choice
    if both_extreme == False:
        psi_s, xs = standard_numerov(psi_s, xs, n, h, k)
        return psi_s, xs
    if both_extreme == True:
        psi_s, xs, diff = both_extreme_numerov(psi_s, xs, i_x_mid, h, k)
        return psi_s, xs, diff
    

def numerov(E_start, E_stop, Error_fun, max_iter, both_extreme):
    """
    Perform the Numerov algorithm with energy bisection to find the wavefunction.

    Parameters:
    E_start (float): Initial guess for the energy (lower bound).
    E_stop (float): Stopping criterion for the energy (upper bound).
    Error_fun (float): Error tolerance for stopping the iterations.
    max_iter (int): Maximum number of iterations.
    both_extreme (bool): Whether to use or not the both extreme Numerov algorithm.

    Returns:
    tuple: Tuple containing the final energy E_midpoint, the wavefunction psi, and the xs array.

    # add here reference to paper or website of the method is any
    """

    # Initialize variables and arrays
    E_midpoint = E_start
    first_cyc  = True
    diff_list =  []

    # Iterate through a maximum number of iterations
    for i in range(max_iter):

        # Setup initial conditions for the iteration
        if first_cyc:
            # E_midpoint=E_start
            first_cyc=False
            h = (Rmax-Rmin)/(nsteps-1)
            xs = Rmin + (np.arange(nsteps))*h
        else:
            E_midpoint = E_start - (E_start - E_stop)/2

        # Energy bisection to find the right energy level with both extreme Numerov algorithm
        if both_extreme == True:
            E_midpoint = E_start - (E_start - E_stop)/2

            x_mid = bisection_method(V, E_midpoint, Rmin, Rmax, tol=1e-12, max_iterations=1000)
            #print(f"x_mid = {x_mid}, E_midpoint = {E_midpoint}")
            i_x_mid = find_index_with_min_difference(xs,x_mid)
            psi, xs, differ = get_wavefunction(E_midpoint, Rmin, Rmax, nsteps, both_extreme, i_x_mid) # tra -10 e -20 psi cambia segno
            print("E = ",{V(xs[i_x_mid])},"  diff = ",differ, "xs[", i_x_mid, "] =", xs[i_x_mid], "i_x_mid = ", i_x_mid)
            
            # Check the difference in logarithmic derivatives to determine the next energy guess
            if abs(differ) < Error_fun:
                break
            if differ > 0:
                E_start = E_midpoint
                E_stop  = E_stop
                print(f"differ = {differ} >0 -> choose lower")
            elif differ < 0:
                E_start = E_start
                E_stop  = E_midpoint
                print(f"differ = {differ} < 0 -> choose upper")
            E_midpoint = E_start - (E_start - E_stop)/2
            diff_list = diff_list + [differ]
            print(f"{diff_list} \n \n")
        
        # Standard Numerov algorithm
        elif both_extreme == False:
            i_x_mid = 0
            psi, xs = get_wavefunction(E_midpoint,Rmin,Rmax, nsteps, both_extreme, i_x_mid) # tra -10 e -20 psi cambia segno
            print("E = ",E_midpoint,"  Psi[Rmax] = ",psi[-1])
            if abs(psi[-1]) < Error_fun:
                break

            # Check the wavefunction at Rmax to determine the next energy guess
            if psi[-1]>0:  
                E_start = E_midpoint
                E_stop  = E_stop
                print(f"psi[{len(psi)}] = {psi[len(psi)-1]} -> choose lower")
            elif psi[-1]<0:
                E_start = E_start
                E_stop  = E_midpoint
                print(f"psi[{len(psi)}] = {psi[len(psi)-1]} -> choose upper")
            E_midpoint = E_start - (E_start - E_stop)/2
    
    # Add further checks and normalization of the wavefunction
    # add here the check of how many nodes there are in psi
    # add here psi normalization (int psi^2 = 1)
    return E_midpoint, psi, xs
        
        
    
    
    
# Try between 10000 and 1000000
nsteps  = 2000000
Rmax    = 40
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


# Main Numerov algorithm to search for bound states or the zero energy wavefunction
# E_error is the required accuracy for the energy of the bound states
# Since E_error = L/2^{N}, the number of steps required for a prescribed energy are given by Maxiter = np.log2(L/R) + 1 as an integer
# Error_fun is the error wanted for the wavefunction to be zero at Rmax --> Psi(Rmax) = 0 +- Error_fun

E_stop          =   -10
E_start         =   -0
E_error         =   0.00000002
Error_fun       =   0.0001           # Error_fun is the error wanted for the wavefunction to be zero at Rmax --> Psi(Rmax) = 0 +- Error_fun
L               =   E_start - E_stop
max_iter        =   int(np.log2(L/E_error)//1)+1
#max_iter       =   100
both_extreme    =   False

# Calculate the initial energy midpoint for the Numerov algorithm
E_midpoint      =   -2.221390027552843          # Fixed to a specified value since already calculated


# Set the scattering energy
E_scatt         =   0.

# Run Numerov algorithm to calculate the scattering state wavefunction
psi_scatt, xs   =   get_wavefunction(E_scatt, Rmin, Rmax, nsteps, both_extreme, 0)

# Select data points after x = R for fitting the asymptotical behaviour of psi
i_node          =   find_index_with_min_difference(psi_scatt[100:],0)
#psi_scatt       =   psi_scatt/np.sqrt(np.trapz(psi_scatt[:i_node]*psi_scatt[:i_node]))
R               =   20 #xs[i_node+100]
xs_fit          =   xs[xs >= R]
psi_fit         =   psi_scatt[xs >= R]

print(f"the area inside the node of psi scatt is {np.sqrt(np.trapz(psi_scatt[:i_node]*psi_scatt[:i_node], xs[:i_node]))}")
# Fit the linear function to the selected data points
params, covariance = curve_fit(linear_function, xs_fit, psi_fit)

# Extract the fitting parameters
a_fit, b_fit    =   params

# Implement the normalization factor that send psi_outer (the linear_funtion) to 1 at x = 0
alpha           =   1/(b_fit)
psi_outer       =   alpha * linear_function(xs, a_fit, b_fit)
psi_scatt       =   alpha * psi_scatt
print(psi_outer)
print(psi_scatt)

plot_potential = True

if plot_potential == True:
    plt.plot(xs, V(xs), label='V(r) [MeV]')
    plt.xlabel(r'$r$ [$fm$]')
    plt.ylabel(r'$V(r)$ $MeV$')
    plt.ylim(np.min(V(xs)-2), 20)
    plt.legend()
    plt.grid(True)
    plt.savefig('zero_energy/V(r).pdf')
    plt.close()
scattering_length = -b_fit/a_fit
latex_equation1 = r"$a_0 =$" + str(round(scattering_length,4))

# Plot the original psi and the fitted linear function
plt.plot(xs, psi_scatt, label=r'$\Psi(r)$ at $E = 0$')
plt.plot(xs, psi_outer, color='red', label= r'Asymptotic $\Psi(r)$')
plt.scatter(scattering_length,0, label= latex_equation1)
plt.xlabel(r'$r$ [$fm$]')
plt.ylabel(r'$\Psi(r)$')
#plt.text(0.05, 0.05, f'{latex_equation1}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5)) # add box with parameters
plt.legend()
plt.grid(True)
plt.savefig('zero_energy/Psi_E=0.pdf')
plt.close()

# Calculate the effective range and print relevant information
eff_r           =   psi_outer*psi_outer - psi_scatt*psi_scatt
print(f"max of eff_r is {np.max(eff_r)}, minimum = {np.min(eff_r)}, eff_r = {eff_r}")
eff_range       =   2*np.trapz(eff_r, xs)
print(f"Scattering length is {-b_fit/a_fit} and the Effective range r_0 is {eff_range} \n")
print(f"Prediction of the bound state energy from the scattering length: E_bs = {1/((-b_fit/a_fit)**2 * twomu_on_h2)}")

# Plot the original data and the fitted linear function
plt.plot(xs[::1000], psi_scatt[::1000]*psi_scatt[::1000], label= r'$\Psi_{ scat }^2(r)$')
plt.plot(xs[::1000], psi_outer[::1000]*psi_outer[::1000], color='red', label='$\Phi^2_{ out }(r)$ ')
plt.fill_between(xs[::1000],  psi_scatt[::1000]*psi_scatt[::1000], psi_outer[::1000]*psi_outer[::1000], 
                 #where= (psi_scatt[::1000]*psi_scatt[::1000] < psi_outer[::1000]*psi_outer[::1000]), 
                 interpolate=True, 
                 color='gray', 
                 alpha=0.5, 
                 label=r'$r_0 = 2 \int_{ 0 }^{\infty} \left(\Phi^2_{ out }(r) -\Psi_{ scat }^2(r)\right) \, dr$')

plt.xlabel(r'$r$ [$fm$]')
plt.ylabel(r'$\Psi(r)$')
latex_equation2 = r"$r_0 =$" + str(round(eff_range,4))
plt.text(0.75, 0.05, f'{latex_equation2}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5)) # add box with parameters
plt.legend()
plt.grid(True)
plt.savefig('zero_energy/Squared_Psi_E=0.pdf')
plt.close()


