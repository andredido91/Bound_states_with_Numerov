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
#C0 = -67.583747
r_star = 0.77187385 # fm e MeV

# Physical parameters
m           = 938.919/2.0        # reduced mass in MeV
hbarc       = 197.327            # (solita costante di struttura)
twomu_on_h2 = 2*m/(hbarc**2)     # it has the resuced mass if you want the readial equation (check the reduced radial problem o fQM1)

# Define the linear function to fit
def linear_function(x, a, b):
    return a * x + b

def V(r, C0):
    return C0 * np.exp(-0.25*(r)**2/r_star**2)

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

def fit_and_compute_r0_a0(xs, psi_scatt, C0):
    # Select data points after x = R for fitting the asymptotical behaviour of psi
    #i_node          =   find_index_with_min_difference(psi_scatt[100:],0)
    R               =   20 #xs[i_node+100]
    xs_fit          =   xs[xs >= R]
    psi_fit         =   psi_scatt[xs >= R]

    # Fit the linear function to the selected data points
    params, covariance = curve_fit(linear_function, xs_fit, psi_fit)

    # Extract the fitting parameters
    a_fit, b_fit    =   params

    # Implement the normalization factor that send psi_outer (the linear_funtion) to 1 at x = 0
    alpha           =   1/(b_fit)
    psi_outer       =   alpha * linear_function(xs, a_fit, b_fit)
    psi_scatt       =   alpha * psi_scatt

    print('Fitted parameters:')
    print('a:', a_fit)
    print('b:', b_fit)

    scattering_length = -b_fit/a_fit
    # Calculate the effective range and print relevant information
    eff_r           =   psi_outer*psi_outer - psi_scatt*psi_scatt
    eff_range       =   2*np.trapz(eff_r, xs)
    print(f"Scattering length: {scattering_length} \nEffective range: {eff_range} \n")
    print(f"Prediction of the Bound State energy from the scattering length: E_bs = {1/((scattering_length)**2 * twomu_on_h2)} \n \n")

    # Plot the original psi and the fitted linear function
    plt.plot(xs, psi_scatt, label=f'Psi at E = 0, C0 = {C0:.5f}, a_0 = {scattering_length:.5f}, r_0 = {eff_range:.5f}')
    plt.plot(xs, psi_outer, color='red', label='Asymptotic behaviour of psi')
    plt.xlabel('x [fm^{-1}]')
    plt.ylabel('Psi(x)')
    plt.legend()
    plt.grid(True)
    plt.savefig('Psi_E=0.pdf')
    plt.close()

    return eff_range, scattering_length

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


def get_wavefunction(E_set,C0, a,b,n):
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
    E = E_set

    V_lambda = lambda r : V(r, C0)
    # Define the k and f functions based on the provided potential
    def k(r):
        return twomu_on_h2*(E-V_lambda(r))
    def f(r,psi_s):
        return -k(r)*psi_s

    # Use the Numerov algorithm
    psi_scatt, xs = standard_numerov(psi_s, xs, n, h, k)
    eff_range, scattering_length = fit_and_compute_r0_a0(xs, psi_scatt, C0)
    return eff_range, scattering_length, psi_scatt, xs

    

def numerov(E_set, C0_start, C0_stop, Error_scatt, max_iter):
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
    E           = E_set
    first_cyc   = True
    # Iterate through a maximum number of iterations
    for i in range(max_iter):
        # Setup initial conditions for the iteration
        if first_cyc:
            C0_midpoint = C0_start - (C0_start-C0_stop)/2
            first_cyc   = False
            h           = (Rmax-Rmin)/(nsteps-1)
            xs          = Rmin + (np.arange(nsteps))*h
        else:
            C0_midpoint = C0_start - (C0_start-C0_stop)/2

        # Standard Numerov algorithm
        eff_range, scattering_length, psi, xs = get_wavefunction(E, C0_midpoint, Rmin,Rmax, nsteps) # tra -10 e -20 psi cambia segno
        if abs(scattering_length) < Error_scatt:
            break
        # Check the wavefunction at Rmax to determine the next energy guess
        if scattering_length<0:  
            C0_start    = C0_midpoint
            C0_stop     = C0_stop
            print(f"a_0 = {scattering_length} > 0 --> Choose lower")
        elif scattering_length > 0:
            C0_start    = C0_start
            C0_stop     = C0_midpoint
            print(f"a_0 = {scattering_length} < 0 --> Choose upper")
        C0_midpoint = C0_start - (C0_start - C0_stop)/2
    
    return C0_midpoint, eff_range, scattering_length, psi, xs

    
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

C0_stop          =   -100
C0_start         =   -0.01
Error_scatt       =   0.00001           # Error_scatt is the error wanted tollerance on the scattering length error
L               =   C0_start - C0_stop
max_iter        =   int(np.log2(L/Error_scatt)//1)+1
print(f"max iter = {max_iter}")

# Set the scattering energy
E_process         =   0.

# Run Numerov algorithm to calculate the scattering state wavefunction, C0, a_0, r_0
C0_fitted, eff_range, scattering_length, psi, xs   =   numerov(E_process, C0_start, C0_stop, Error_scatt, max_iter)


plot_potential = True
if plot_potential == True:
    plt.title(f"C0 = {C0_fitted:.8f}, r_0 = {eff_range:.8f}, a_0 = {scattering_length:.8f}")
    plt.plot(xs, V(xs , C0_fitted), label='V(r) [MeV]')
    plt.xlabel('x [fm^{-1}]')
    plt.ylabel('V(r) MeV')
    plt.ylim(np.min(V(xs , C0_fitted)-2), 20)
    plt.legend()
    plt.grid(True)
    plt.savefig('V(r).pdf')
    plt.show()
