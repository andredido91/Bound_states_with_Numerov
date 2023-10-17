import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import cycler
import a0_module as a0_mod
from scipy.optimize import root

pgf_with_latex =  {
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots 
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 14,               # LaTeX default is 10pt font.
    "font.size": 15,
    "legend.fontsize": 14,               # Make the legend/label fonts 
    "xtick.labelsize": 14,               # a little smaller
    "ytick.labelsize": 14,
    "pgf.preamble": "\n".join([ # plots will use this preamble
        r"\usepackage{amsmath}",
        r"\usepackage{amsfonts}",
        r"\usepackage{dsfont}",
        #r"\usepackage{amssymb}",
        ])}

mpl.rcParams.update(pgf_with_latex)

plt.rcParams.update({
    'font.size': 15,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'text.latex.preamble': r'\usepackage{dsfont}'
})

#print(lc.LECs)
#exit()
#pot parameters
#C0 = -67.583747
#r_star = 0.77187385 # fm e MeV

# in a real program i would make a module for the constants, but here not worth the time
#r_star = 1./4.
#C0	   = -505.1500

#pot parameters
#C0 = -67.583747
#r_star = 0.77187385 # fm e MeV

#r_star = [10, 5, 2, 1, 0.5, 0.1, 0.05, 0.01]
# Physical parameters
m_N           = 938.919
#m_N         = 938.95
m_L         = 1115.683
m_LN        = (m_N*m_L)/(m_N+m_L)

#m           = m_N/2.0        # as used in NNQS
m = m_LN
hbarc       = 197.327            # (solita costante di struttura)
twomu_on_h2 = 2*m/(hbarc**2)     # it has the resuced mass if you want the readial equation (check the reduced radial problem o fQM1)

cutoff_MeV = 600.0
r_star = [hbarc/cutoff_MeV]   # in fermi

# Define the linear function to fit
def linear_function(x, a, b):
    return a * x + b

def node_number(psi):
    n_nodes_psi = 0
    for i, psi_i in enumerate(psi):
        if i == 0:
            sign_i_minus1 = np.sign(psi_i)
        if i > 0:
            sign_i = np.sign(psi_i)
            if sign_i == - sign_i_minus1:
                #print(f"il segno è {psi_i} != {psi[i-1]}")
                #print(i)
                n_nodes_psi += 1 
            sign_i_minus1 = sign_i
    return n_nodes_psi

def V(r, C0, r_star):
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

def fit_and_compute_r0_a0(xs, psi_scatt, C0,j):
    # Select data points after x = R for fitting the asymptotical behaviour of psi
    #i_node          =   find_index_with_min_difference(psi_scatt[100:],0)
    R               =   0.9*Rmax #xs[i_node+100]
    xs_fit          =   xs[xs >= R][::5]
    psi_fit         =   psi_scatt[xs >= R][::5]

    # Fit the linear function to the selected data points
    params, covariance = curve_fit(linear_function, xs_fit, psi_fit)

    # Extract the fitting parameters
    a_fit, b_fit    =   params

    # Implement the normalization factor that send psi_outer (the linear_funtion) to 1 at x = 0
    alpha           =   1/(b_fit)
    psi_outer       =   alpha * linear_function(xs, a_fit, b_fit)
    psi_scatt       =   alpha * psi_scatt

    #print('Fitted parameters:')
    #print('a:', a_fit)
    #print('b:', b_fit)

    scattering_length = -b_fit/a_fit
    # Calculate the effective range and print relevant information
    eff_r           =   psi_outer*psi_outer - psi_scatt*psi_scatt
    eff_range       =   2*np.trapz(eff_r, xs)
    eff_r = None
    print(f"Scattering length: {scattering_length} \nEffective range: {eff_range} \n")
    #print(f"Prediction of the Bound State energy from the scattering length: E_bs = {1/((scattering_length)**2 * twomu_on_h2)} \n \n")

    # Plot the original psi and the fitted linear function
    N = 5
    cmap = plt.cm.coolwarm
    mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
    plt.figure(figsize=(10, 8)) # set figure size
    plt.title(f'$r^* =$ {r_star_i}')
    plt.plot(xs[::10000], psi_scatt[::10000])
    plt.plot(xs[::10000], psi_outer[::10000])
    psi_outer   = None                  # Necessario liberare la memoria!!!!
    psi_scatt   = None
    plt.xlabel(r'$r \quad [fm]$')
    plt.ylabel(r'$u(r)$')
    #plt.legend()
    plt.text(0.05, 0.05, f'$C_0 =$ {C0:.8f}\n$a_0 =$ {scattering_length:.8f}\n$r_0 =$ {eff_range:.8f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5)) # add box with parameters
        
    legend_elements = [Line2D([0], [0], color=cmap(0.), lw=2, label= r'$u(r)$'),
                       Line2D([0], [0], color=cmap(0.2), lw=2, label= r'Asymptotic $u(r)$')
                       ]    
        
    plt.grid(True, linestyle=':') #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
    plt.legend(fontsize=25, handles=legend_elements, loc='lower right') # set legend position
    plt.grid(True)
    plt.savefig(f'fit_potential/{name_set}_{name_a_0_i}_r_star={r_star_i}_step={j}.pdf')
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

    return psi_s  # Return updated psi_s and xs arrays


def get_wavefunction(E_set,C0, r_star_i, a,b,n,j):
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


    psi_s = np.zeros(n)

    # Set the energy value based on the algorithm choice
    E = E_set

    V_lambda = lambda r : V(r, C0, r_star_i)

    # Define the k and f functions based on the provided potential
    def k(r):
        return twomu_on_h2*(E-V_lambda(r))

    # Use the Numerov algorithm
    psi_scatt = standard_numerov(psi_s, xs, n, h, k)
    node_num = node_number(psi_scatt)
    psi_s = None
    eff_range, scattering_length = fit_and_compute_r0_a0(xs, psi_scatt, C0,j)
    psi_scatt = None
    return eff_range, scattering_length, node_num

def defineC0range(E, r_star_i, Rmin, Rmax, nsteps ,j, C0_start,C0_stop, n_node_target):
    C0_max_old = C0_stop
    C0_min_old = C0_start
    escape_min = False
    escape_max = False
    def found_n_node_correct(sign,C_0_n_node,C_0_old):
        return 0
    
    eff_range, scattering_length, n_C0_max_old = get_wavefunction(E, C0_stop, r_star_i, Rmin, Rmax, nsteps,j)
    ind = 1
    delta_init =  (C0_stop-C0_start)/2
    
    return 0 #C0_min_inn, C0_min_ext, C0_max_inn, C0_max_ext
        
def numerov(E_set, C0_start, C0_stop, r_star_i, Error_scatt, max_iter, target_scatt, infinite, n_node_target):
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
    a_zeroes_rstar = []
    C_zeroes_rstar = []
    # Initialize variables and arrays
    E           = E_set
    first_cyc   = True
    # Iterate through a maximum number of iterations
    for j in range(max_iter):
        # Setup initial conditions for the iteration
        if first_cyc:
            first_cyc   = False
            h           = (Rmax-Rmin)/(nsteps-1)
            xs          = Rmin + (np.arange(nsteps))*h
            #C0_start, C0_start_ext, C0_stop, C0_stop_ext = defineC0range(E, r_star_i, Rmin, Rmax, nsteps ,j ,C0_start, C0_stop, n_node_target)
            choice = "lower"
            while choice == "lower":
                eff_range, scattering_length, node_number = get_wavefunction(E, C0_stop, r_star_i, Rmin, Rmax, nsteps,j) # tra -10 e -20 psi cambia segno
                print(f"The number of nodes in psi is: {node_number}")
                if scattering_length < target_scatt and node_number == n_node_target: 
                    choice = "upper"
                    print(f"C0 stop fixed to --> C0_stop = {C0_stop} !! \n \n")
                    C0_save_start = C0_start
                    C0_save_stop = C0_stop
                elif scattering_length > target_scatt and node_number == n_node_target: 
                    choice = "lower"
                    C0_start = C0_stop
                    C0_stop = 2*C0_stop
                    print(f"C0 boundary enlarged --> C0 stop = {C0_stop} !! \n \n ")
                elif node_number != n_node_target:
                    print(f"current node number {node_number} is not in target {n_node_target}")
                    exit()
            C0_midpoint = C0_start - (C0_start-C0_stop)/2

        else:
            C0_midpoint = C0_start - (C0_start-C0_stop)/2
        C_zeroes_rstar += [C0_midpoint]
        # Standard Numerov algorithm
        eff_range, scattering_length , __= get_wavefunction(E, C0_midpoint, r_star_i, Rmin, Rmax, nsteps,j) # tra -10 e -20 psi cambia segno
        a_zeroes_rstar += [scattering_length]
        if infinite == "True":
            if abs(scattering_length-target_scatt) < Error_scatt:
                break
        if infinite == "False":
            if abs(target_scatt-scattering_length) < Error_scatt:
                break
        # Check the scattering length to determine the next C0 guess
        if scattering_length > target_scatt:                     # > zero for matching a finite scattering length value, <0 for infinite scatt length
            C0_start    = C0_midpoint
            C0_stop     = C0_stop
            print(f"a_0 = {scattering_length} < 0 --> Choose right interval")
        elif scattering_length < target_scatt:                   # < zero for matching a finite scattering length value, >0 for infinite scatt length
            C0_start    = C0_start
            C0_stop     = C0_midpoint
            print(f"a_0 = {scattering_length} > 0 --> Choose left interval")
        C0_midpoint = C0_start - (C0_start - C0_stop)/2

    
    N = 5
    cmap = plt.cm.coolwarm
    mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
    #inverted_a_zeroes_rstar = [1 / x  for x in a_zeroes_rstar]
    #indices = np.argsort(C_zeroes_rstar)
    #print(f"indices = {indices}")
    #C_zeroes_rstar = np.sort(C_zeroes_rstar)
    #inverted_a_zeroes_rstar = [inverted_a_zeroes_rstar[k] for k in indices]
    #print(f"{inverted_a_zeroes_rstar}, {C_zeroes_rstar}")

    #print(f"_______________________________ {inverted_a_zeroes_rstar}")
    plt.figure(figsize=(10, 8)) # set figure size
    plt.scatter( C_zeroes_rstar,a_zeroes_rstar, color=cmap(0.))
    #plt.semilogy(C_zeroes_rstar, inverted_a_zeroes_rstar, color=cmap(0.))
    plt.ylabel(r'$a_0$ [$fm$]')
    plt.xlabel(r'$C_0$ [MeV]')
    #plt.ylim(np.min(inverted_a_zeroes_rstar)-1, np.max(inverted_a_zeroes_rstar)+2)
    plt.legend()

    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=1)
                        ]
    legend_elements = [Line2D([0], [0], color=cmap(0.), lw=2, label= r'$a_0(C_0)$')
                        ]    
    plt.grid(True, linestyle=':') #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
    plt.legend(fontsize=25)
    plt.legend(handles=legend_elements)
    plt.savefig(f'fit_potential/a0(C0)_rstar={r_star_i}.pdf')
    plt.close()

    return C0_midpoint, eff_range, scattering_length, C0_save_start, C0_save_stop

    
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


# Main algorithm
nsteps  = 2000000

C0_stop          =   -190
C0_start         =   -187
n_node_target    =    0
Error_scatt      =    0.00001           # Error_scatt is the error wanted tollerance on the scattering length error
L                =    C0_start - C0_stop
max_iter         =    int(np.log2(L/Error_scatt)//1)+1
print(f"max iter = {max_iter}")

# Set the scattering energy
E_process         =   0.

C0s             = []
eff_ranges      = []
scat_lengths    = []
# Run Numerov algorithm to calculate the scattering state wavefunction, C0, a_0, r_0
for l, (name_set,set_a_0) in enumerate(a0_mod.a_zeroes_dict.items()):
    print(f"Working on a_0 from {name_set}\n\n")
    print(f"a_0 are: \n {set_a_0} \n\n")
    for m, (name_a_0_i,a_0_i) in enumerate(set_a_0.items()):
        print(name_a_0_i)
        print(a_0_i)
        for i, r_star_i in enumerate(r_star):
            infinite = False
            target_scatt = a_0_i
            Rmax            = 40 + 4 * r_star_i
            Rmin            = 0
            h = (Rmax-Rmin)/(nsteps-1)
            xs = Rmin + (np.arange(nsteps))*h    # or (np.arange(n)+0.5 )*h
            C0_fitted, eff_range, scattering_length, C0_save_start, C0_save_stop =   numerov(E_process, C0_start, C0_stop, r_star_i, Error_scatt, max_iter, target_scatt, infinite, n_node_target)
            C0_start        = C0_save_start
            C0_stop         = C0_save_stop
            C0s             += [C0_fitted]
            eff_ranges      += [eff_range]
            scat_lengths    += [scattering_length]

            plot_potential = True
            if plot_potential == True:
                N = 5
                cmap = plt.cm.coolwarm
                mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
                #plt.title(f"C0 = {C0_fitted:.8f}, r_0 = {eff_range:.8f}, a_0 = {scattering_length:.8f}")
                plt.figure(figsize=(10, 8)) # set figure size
                plt.plot(xs, V(xs , C0_fitted, r_star_i))
                plt.text(0.60, 0.05, f'$C_0 =$ {C0_fitted:.5f}\n$a_0 =$ {scattering_length:.5f}\n$r_0 =$ {eff_range:.5f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5)) # add box with parameters

                plt.xlabel(r'$r$ [fm]')
                plt.ylabel(r'$V(r)$ MeV')
                plt.ylim(np.min(V(xs , C0_fitted, r_star_i)-2), 5)
                plt.legend()
                custom_lines = [Line2D([0], [0], color=cmap(0.), lw=1)
                                ]            
                legend_elements = [Line2D([0], [0], color=cmap(0.), lw=2, label= r'$V(r)$')
                                ]                
                plt.grid(True, linestyle=':') #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
                plt.legend(fontsize=25)
                plt.legend(handles=legend_elements)
                plt.savefig(f'fit_potential/Fitted_Potential_rstar={r_star_i}_set={name_set}_a0={a_0_i}.pdf')
                plt.close()

        plot_eff_ranges_and_scatt_lengths = False
        if plot_eff_ranges_and_scatt_lengths == True:
            N = 5
            cmap = plt.cm.coolwarm
            mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
            plt.figure(figsize=(10, 8)) # set figure size
            plt.scatter(r_star, eff_ranges, color=cmap(0.))
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel(r'$r_0$ [fm]')
            plt.xlabel(r'$r^{*}$ [fm]')
            plt.ylim(np.min(eff_ranges)-1, np.max(eff_ranges)+2)
            plt.legend()

            custom_lines = [Line2D([0], [0], color=cmap(0.), lw=1)
                            ]
            legend_elements = [Line2D([0], [0], color=cmap(0.), lw=2, label= r'$r_0(r^{*})$')
                            ]    
            plt.grid(True, linestyle=':') #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
            plt.legend(fontsize=25)
            plt.legend(handles=legend_elements)
            plt.savefig('fit_potential/r0(rstar).pdf')
            plt.close()

            
            
            N = 5
            cmap = plt.cm.coolwarm
            mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
            plt.figure(figsize=(10, 8)) # set figure size
            plt.semilogy(r_star, C0s, color=cmap(0.))
            plt.ylabel(r'$C_0$ [MeV]')
            plt.xlabel(r'$r^*$ [fm]')
            plt.ylim(np.min(C0s)-1, np.max(C0s)+2)
            plt.legend()

            custom_lines = [Line2D([0], [0], color=cmap(0.), lw=1)
                            ]
            legend_elements = [Line2D([0], [0], color=cmap(0.), lw=2, label= r'$C_0(r^{*})$')
                            ]    
            plt.grid(True, linestyle=':') #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
            plt.legend(fontsize=25)
            plt.legend(handles=legend_elements)
            plt.savefig('fit_potential/C0(rstar).pdf')
            plt.close()

    print("ENDED")