import numpy as np
import matplotlib.pyplot as plt
class RungeKuttaMethods():
    """Class that groups methods to perform numerical integration"""
    
    @staticmethod
    def euler(f,a,b,n,yinit):
        """Euler method"""        
        h = (b-a)/(n-1)
        xs = a + np.arange(n)*h
      
        #arange is a np function that creates a row array 
        #with n values from 0 to n-1 (dimension is nx1)
        #Mind that it takes floats but often return errors of
        #evaluation (including, non including endpoint) due to finite precision
        
        #np.arange(n) gives a row with n entries:
        #[0 1 2 3 4 5 ... n-1]
        #then it is multiplied by h, that is, for a fixed n,
        #the lenght of every n-1 interval that divides band a.
        #xs is a vector whose minimum value is a (a+0) and every
        #entry represents one marker for the integration (h, 2h, 3h,...)
     
        ys = np.zeros(n)    #array of same dimension of xs (nx1) filled with zeros
                    
        y = yinit
        for j,x in enumerate(xs): #for each marker set by xs
            ys[j] = y   #setting initial value [j=0] of ys to yinit, then
                        #as y grows ys[j] moves with j
            y += h*f(x, y)  
            #y is equal to y plus the step, that is the value of
            #the function at the marker to which y and x correspond
            #(the minimum value in the interval of lenght h on the x axis)
        return xs, ys
    
    # @staticmethod
    # def rk2(f,a,b,n,yinit):
    #    h = (b-a)/(n-1)
    #    xs = a + np.arange(n)*h  
    #    ys = np.zeros(n)
    #    y = yinit
    #    for j,x in enumerate(xs):
    #        ys[j] = y
    #        y += h*f(x, y)  
    #    return xs, ys

    @staticmethod
    def rk4(f,a,b,n,yinit):
        """Runge-Kutta method, order 4"""        
        h = (b-a)/(n-1)
        xs = a + np.arange(n)*h
        ys = np.zeros(n)
    
        y = yinit
        for j,x in enumerate(xs):
            ys[j] = y
            k0 = h*f(x, y)
            k1 = h*f(x+h/2, y+k0*h/2)
            k2 = h*f(x+h/2, y+k1*h/2)
            k3 = h*f(x+h, y+k2*h)
            y += (k0 + 2*k1 + 2*k2 + k3)/6
        return xs, ys
    
    
    @staticmethod
    def numerov(E_guess,a,b,n,yinit, y_prime_init):
        h = (b-a)/(n-1)
        xs = a + (np.arange(n)+0.5)*h
        psi_s = np.zeros(n)
                
        C0 = -67.583747

        r_star = 0.77187385 # fm e  MeV
        m =  938.919        # MeV
        hbarc = 197.327       # (solita costante di struttura)
        E = E_guess
        
        def k(r):
            def V(r):
                return C0 * np.exp(-0.25*r**2/r_star**2)
            return 2*m*(E-V(r))/hbarc**2
        
        def f(r,psi_s):
            return -k(r)*psi_s

        def second_order_rk4(f,a,b,n,yinit, y_prime_init):     # a = t_0, b = t_0+dt
            """Second order ODE Runge-Kutta method, order 4"""        
            h = (b-a)/(n-1)
            xs_rk = a + np.arange(n)*h
            psi_s_rk = np.zeros(n)
            u_s = np.zeros(n)

            psi_s_rk[0] = yinit
            u_s[0] = y_prime_init

            for j,x in enumerate(xs):
                if j < n-1:

                    k0 = h * psi_s_rk[j]
                    l0 = h * f(xs_rk[j], psi_s_rk[j])

                    k1 = h * (psi_s_rk[j] + 0.5 * l0)
                    l1 = h * f(xs_rk[j] + 0.5 * h, psi_s_rk[j] + 0.5 * k0)

                    k2 = h * (psi_s_rk[j] + 0.5 * l1)
                    l2 = h * f(xs_rk[j] + 0.5 * h, psi_s_rk[j] + 0.5 * k1)

                    k3 = h * (psi_s_rk[j] + l2)
                    l3 = h * f(xs_rk[j] + h, psi_s_rk[j] + k2)
                    psi_s_rk[j+1] = psi_s_rk[j] + (1.0 / 6.0) * (k0 + 2 * k1 + 2 * k2 + k3)
                    u_s[j+1] = u_s[j] + (1.0 / 6.0) * (l0 + 2 * l1 + 2 * l2 + l3)
                    #print(psi_s_rk[j])
                
                else:
                    return xs_rk, psi_s_rk
    
        start_xs, start_psi_s = second_order_rk4(f,a,a+h,3,yinit, y_prime_init)
        #xs[0]=start_xs[0]
        psi_s[0] = 0
        #xs[1]=start_xs[1]
        psi_s[1] = 1
        #xs[2]=start_xs[2]
        #psi_s[2] = start_psi_s[2]
        
        # Plot the solution
        #plt.plot(start_xs, start_psi_s, label='Solution')
        #plt.xlabel('x')
        #plt.ylabel('y(x)')
        #plt.legend()
        #plt.grid(True)
        #plt.show()
        
        for j,x in enumerate(xs):
            if j < 2:
                pass
            elif j < n-1:
                a = (1+((h**2)/12)*k(xs[j+1]))
                b = 2*(1-((5*(h**2))/12)*k(xs[j]))
                c = (1+((h**2)/12)*k(xs[j-1]))
                #print(k(xs[j]))
                psi_s[j+1] = (1/a)*(b* psi_s[j]-c* psi_s[j-1])
            
        return psi_s, xs
    
psi, xs = RungeKuttaMethods.numerov(-10.0,0,10, 1000,0.01, -1.1) # tra -10 e -20
# Plot the solution
plt.plot(xs, psi, label='Initial guess')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.grid(True)
plt.show()


R = 0.000005    # Errore sulla valutazione dell'energia
E_stop = -20
E_start = -10
L =  E_start - E_stop

num_iter = int(np.log2(L/R)//1+1)

for i in range(num_iter):
    E_midpoint = E_start - (E_start - E_stop)/2 #L/(2**(i+1))
    print(f"E_midpoint = {E_midpoint}")
    psi, xs = RungeKuttaMethods.numerov(E_midpoint,0,10, 10000,0.01, -1.1) # tra -10 e -20 psi cambia segno
    if psi[len(psi)-1]>0:
        E_start = E_midpoint
        E_stop = E_stop
        print(f"psi[{len(psi)}] = {psi[len(psi)-1]} -> choose right")
    elif psi[len(psi)-1]<0:
        E_start = E_start
        E_stop = E_midpoint
        print(f"psi[{len(psi)}] = {psi[len(psi)-1]} -> choose left")
    print(i)
    
    
plt.plot(xs, psi/np.trapz(psi), label=f'Solution at E = {E_midpoint:.6f}+-{R}')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.legend()
plt.grid(True)
plt.show()

print(f"E_start = {E_start},E_midpoint = {E_midpoint},E_stop = {E_stop}")

