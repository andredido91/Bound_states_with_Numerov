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
#r_star = 0.77187385 # fm e MeV

# Deuteron LECs lambda = 2 = 1/r*
#C0 = -142.36782836914
#Dineutron scattering length fit of C0
C0 = -104.970684051513 
r_star = 1/2 # fm e MeV
C0 = -68.37466431002274
r_star = 0.7662130373761021

# Physical parameters
m           = 938.919/2.0        # reduced mass in MeV
#m           = 938.95/2.0        # reduced mass in MeV

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

    psi_s, xs = standard_numerov(psi_s, xs, n, h, k)
    return psi_s, xs

# Try between 10000 and 1000000
nsteps  = 4000000
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
R               =   15 #xs[i_node+100]
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
latex_equation1 = r"$a_0 =$" + str(round(scattering_length,4))+ f"\n$a_0^{{target}} = -18.6300$"

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
latex_equation2 = r"$r_0 =$" + str(round(eff_range,4)) + f"\n$r_0^{{target}}= 1.4910$"
plt.text(0.55, 0.05, f'{latex_equation2}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5)) # add box with parameters
plt.legend()
plt.grid(True)
plt.savefig('zero_energy/Squared_Psi_E=0.pdf')
plt.close()


