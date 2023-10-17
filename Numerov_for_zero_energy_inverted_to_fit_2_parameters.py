import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib import cycler
import a0_module as a0_mod
from scipy.optimize import minimize, root, fsolve, bisect

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

# Physical parameters
m_N           = 938.919
m_L         = 1115.683
m_LN        = (m_N*m_L)/(m_N+m_L)
m_NN           = m_N/2.0        # as used in NNQS
scattering_length = None
eff_range = None
m = m_NN
hbarc       = 197.327            # (solita costante di struttura)
twomu_on_h2 = 2*m/(hbarc**2)     # it has the resuced mass if you want the readial equation (check the reduced radial problem o fQM1)

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

def fit_and_compute_r0_a0(xs, psi_scatt, C0, r_star):
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
    plot = True
    if plot == True:
        # Plot the original psi and the fitted linear function
        N = 5
        cmap = plt.cm.coolwarm
        mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
        plt.figure(figsize=(10, 8)) # set figure size
        plt.title(f'$r^* =$ {r_star}')
        plt.plot(xs[::1000], psi_scatt[::1000])
        plt.plot(xs[::1000], psi_outer[::1000])
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
        plt.savefig(f'fit_potential/{name_set}_{name_a_0_i}.pdf')
        plt.close()

    return eff_range, scattering_length

def cost_function(a_0, a_0_target, r_0, r_0_target):
    print(f"a0 test = {1/a_0} \t a0_obj = {1/a_0_target}\t \t diff = {1/a_0 - 1/a_0_target} \nr0 test = {r_0} \t r_0_obj = {r_0_target} \t diff = {r_0 - r_0_target}")
    return 10000* (((a_0-a_0_target)**2)/np.abs(a_0_target) + (r_0-r_0_target)**2)

def error_function(C0_rs, r_0_target, a_0_target, E_process):
    C0 = C0_rs[0]
    r_star = C0_rs[1]
    r_0, a_0, node_num = get_wavefunction(E_process, C0, r_star)
    if node_num != n_node_target:
        print(f"current node number {node_number} is not in target {n_node_target}")
        #exit()
    cost = cost_function(a_0, a_0_target, r_0, r_0_target)
    print(f"C_0 = {C0}, r_star = {r_star}")
    print(f"cost = {cost}\n")

    return cost

def root_finder(C0_rs, r_0_target, a_0_target, E_process):
    C0 = C0_rs[0]
    r_star = C0_rs[1]
    print(f"C_0 = {C0}, r_star = {r_star}")
    r_0, a_0, node_num = get_wavefunction(E_process, C0, r_star)
    if node_num != n_node_target:
        print(f"current node number {node_number} is not in target {n_node_target}")
        #exit()
    return [a_0-a_0_target, r_0 - r_0_target]

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


def get_wavefunction(E_set,C0_i, r_star_i):
    psi_s = np.zeros(nsteps)
    global eff_range
    global scattering_length

    # Set the energy value based on the algorithm choice
    E = E_set

    V_lambda = lambda r : V(r, C0_i, r_star_i)

    # Define the k and f functions based on the provided potential
    def k(r):
        return twomu_on_h2*(E-V_lambda(r))

    # Use the Numerov algorithm
    psi_scatt = standard_numerov(psi_s, xs, nsteps, h, k)
    node_num = node_number(psi_scatt)
    psi_s = None
    eff_range, scattering_length = fit_and_compute_r0_a0(xs, psi_scatt, C0_i, r_star_i)
    #print(f"computed things XXXXXXXXXX {scattering_length}, {eff_range}")
    psi_scatt = None
    return eff_range, 1/scattering_length, node_num


# Main algorithm
nsteps  = 2000000

cutoff_MeV = 600.0
r_star_guess = hbarc/cutoff_MeV   # in fermi
r_star_guess = 0.771874
#C0_stop          =   -67.5836-100
C0_stop          =   -200
#C0_start         =   -67.5836+50
C0_start         =   -10

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
    print(f"a_0 in the set are: \n {set_a_0} \n\n")
    if l == 0:
        for m, (name_a_0_i,a_0_i) in enumerate(set_a_0.items()):
            if m == 0:
                print(f"name_a_0_i = {name_a_0_i}= {a_0_i[0]}, r_0 = {a_0_i[1]} \n \n \n \n \n ")
                a_0_target  = a_0_i[0]
                r_0_target  = a_0_i[1]
                Rmax        = 20 # + 4 * r_star_guess
                Rmin        = 0
                h           = (Rmax-Rmin)/(nsteps-1)
                xs          = Rmin + (np.arange(nsteps))*h    # or (np.arange(n)+0.5 )*h

                C0_guess    = C0_start+(C0_stop-C0_start)/2
                C0_rs       = [C0_guess, r_star_guess]
                bnds        = ((C0_stop, C0_start), (r_star_guess-0.3, r_star_guess+0.3))

                ## goodmethods:    lm   ,     excitingmixing
                root_finder_lam = lambda C0_rs : root_finder(C0_rs, r_0_target, a_0_target, E_process)

                #res_0_ = root(root_finder_lam, (C0_guess, r_star_guess), method='df-sane', tol=0.0001, callback=None, options=None)
                #res_0_ = fsolve(root_finder_lam, (C0_guess, r_star_guess))
                #####res_0_ = bisect(root_finder_lam)
                
                #C0_fitted, r_star_fitted = res_0_.x
                
                minimizer = True

                if minimizer == True:
                    #bnds       = ((C0_stop, C0_start), (0.15, 0.7))
                    bnds        = ((C0_stop, C0_start), (r_star_guess-0.45, r_star_guess+0.35))
                    #eps        =  0.00001 ## the best value is eps = 0.00001 (10^-5) for nsteps = 800000

                    eps         =  0.0001
                    tolerance   = 0.0001
                    options     = {'eps': eps, 'ftol': tolerance}  
                    err_fun     = lambda C0_rs: error_function(C0_rs, r_0_target, 1/a_0_target, E_process)
                    res_0       = minimize(err_fun, (C0_guess, r_star_guess), jac = '3-point', method='L-BFGS-B', bounds=bnds, options=options)
                    C0_fitted, r_star_fitted = res_0.x

                    print(f"DONE\n \n \n C_0 = {C0_fitted}, r_star = {r_star_fitted}")

                    #nsteps = 2000000
                    #h = (Rmax-Rmin)/(nsteps-1)
                    #xs = Rmin + (np.arange(nsteps))*h    # or (np.arange(n)+0.5 )*h
                    bnds        = ((C0_fitted-0.5, C0_fitted+0.5), (r_star_fitted-0.01, r_star_fitted+0.01))
                    eps         =  0.0000001
                    tolerance   = 0.000001

                    options     = {'eps': eps, 'ftol': tolerance, 'xtol': 0.000001}  
                    #C0_rs      = [C0_guess, r_star_guess]    
                    err_fun     = lambda C0_rs: error_function(C0_rs, r_0_target, 1/a_0_target, E_process)
                    res         = minimize(err_fun, (C0_fitted, r_star_fitted), jac = '3-point', method='L-BFGS-B', bounds=bnds, options=options)
                    # x the solution array, success a Boolean flag indicating if the optimizer exited successfully and 
                    # message which describes the cause of the termination. 
                    # See OptimizeResult for a description of other attributes at https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
                    
                    C0_fitted, r_star_fitted = res.x
                                              
                print(f"C0_fitted = {C0_fitted}, r_star_fitted = {r_star_fitted}")
                plot_potential = True
                if plot_potential == True:
                    N = 5
                    cmap = plt.cm.coolwarm
                    mpl.rcParams['axes.prop_cycle'] = cycler(color=cmap(np.linspace(0, 1, N)))
                    #plt.title(f"C0 = {C0_fitted:.8f}, r_0 = {eff_range:.8f}, a_0 = {scattering_length:.8f}")
                    plt.figure(figsize=(10, 8)) # set figure size
                    plt.plot(xs, V(xs , C0_fitted, r_star_fitted))
                    plt.text(0.60, 0.05, f'$C_0 =$ {C0_fitted:.5f}\n$a_0 =$ {scattering_length:.5f}\n$r_0 =$ {eff_range:.5f}', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5)) # add box with parameters

                    plt.xlabel(r'$r$ [fm]')
                    plt.ylabel(r'$V(r)$ MeV')
                    plt.ylim(np.min(V(xs , C0_fitted, r_star_fitted)-2), 5)
                    plt.legend()
                    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=1)
                                ]       
                    legend_elements = [Line2D([0], [0], color=cmap(0.), lw=2, label= r'$V(r)$')
                                ]           
                    plt.grid(True, linestyle=':') #https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D.set_linestyle
                    plt.legend(fontsize=25)
                    plt.legend(handles=legend_elements)
                    plt.savefig(f'fit_potential/Fitted_Potential_rstar={r_star_fitted}_set={name_set}_a0={a_0_i}.pdf')
                    plt.close()