import numpy as np
import matplotlib.pyplot as plt

MU0 = 1.0
EPS0 = 1.0
C0 = 1 / np.sqrt(MU0*EPS0)

# Constants for permittivity regions test
EPS1 = 2.0
C1 = 1 / np.sqrt(MU0*EPS1)
R = (np.sqrt(EPS0)-np.sqrt(EPS1))/(np.sqrt(EPS0)+np.sqrt(EPS1))
T = 2*np.sqrt(EPS0)/(np.sqrt(EPS0)+np.sqrt(EPS1))
# Constants for dispersive materal
c_silver = np.array([
    5.987e-1 + 4.195e3j,
    -2.211e-1 + 2.680e-1j,
    -4.240 + 7.324e2j,
    6.391e-1 - 7.186e-2j,
    1.806 + 4.563j,
    1.443 - 8.219e1j
])*1.60218e-19 # Conversion to Jules

a_silver = np.array([
   -2.502e-2 - 8.626e-3j,
   -2.021e-1 - 9.407e-1j,
   -1.467e1 - 1.338j,
   -2.997e-1 - 4.034j,
   -1.896 - 4.808j,
   -9.396 - 6.477j
])*1.60218e-19 # Conversion to Jules

def gaussian_pulse(x, x0, sigma):
    return np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


class FDTD1D:
    def __init__(self, xE, bounds=('pec', 'pec')):
        self.xE = np.array(xE)
        self.xH = (self.xE[:-1] + self.xE[1:]) / 2.0
        self.dx = self.xE[1] - self.xE[0]
        self.bounds = bounds
        self.e = np.zeros_like(self.xE)
        self.h = np.zeros_like(self.xH)
        self.h_old = np.zeros_like(self.h)
        self.eps = np.ones_like(self.xE)  # Default permittivity is 1 everywhere
        self.cond = np.zeros_like(self.xE)  # Default conductivity is 0 everywheree
        
        self.energyE = []
        self.energyH = []
        self.energy = []

        self.time = 0
        self.total_field = []

        self.material_regions = [] # Region with dispersive material
        self.J = np.zeros((6, len(self.xE)), dtype=np.complex128) # Current in dispersive material
        self.material_coefficients = [] # Coefficientes of the dispersive material

    def set_initial_condition(self, initial_condition):
        self.e[:] = initial_condition[:]

    def set_permittivity_regions(self, regions):
        """Set different permittivity regions in the grid.
        
        Args:
            regions: List of tuples (start_x, end_x, eps_value) defining regions
                    with different permittivity values.
        """
        for start_x, end_x, eps_value in regions:
            start_idx = np.searchsorted(self.xE, start_x)
            end_idx = np.searchsorted(self.xE, end_x)
            self.eps[start_idx:end_idx] = eps_value

    def set_conductivity_regions(self, regions):
        """Set different conductivity regions in the grid.
        
        Args:
            regions: List of tuples (start_x, end_x, cond_a_value) defining regions
                    with different conductivity values.
        """
        for start_x, end_x, cond_value in regions:
            start_idx = np.searchsorted(self.xE, start_x)
            end_idx = np.searchsorted(self.xE, end_x)
            self.cond[start_idx:end_idx] = cond_value


    def set_material_region(self, regions,dt):
        # Set a region with dispersive material.
        # Args:
             # regions: List of tuples (start_x, end_x, Einf, cond_value) defining regions
                        # with different infinite-frequency permitivity and conductivity values
             # dt: Temporal step. Necessary for the estimation of coefficients inside
             #      the dispersive material

        self.material_regions = regions
        for start_x, end_x, Einf, cond_value in regions:
            start_idx = np.searchsorted(self.xE, start_x)
            end_idx = np.searchsorted(self.xE, end_x)
            self.eps[start_idx:end_idx] = Einf
            self.cond[start_idx:end_idx] = cond_value
        # We define the main parameters of the region
        k_mat = (1 + a_silver * dt / 2) / (1 - a_silver * dt / 2)
        beta_mat = (EPS0 * c_silver * dt) / (1 - a_silver * dt / 2)
        aux = 2 * EPS0 * Einf + np.sum(2 * np.real(beta_mat))
        den_mat = aux + cond_value * dt
        num_mat = aux - cond_value * dt
        coef_mat = num_mat / den_mat
        self.material_coefficients = [k_mat,beta_mat,den_mat,coef_mat] # We upload the coefficents

    def step(self, dt):
        self.e_old_left = self.e[1]
        self.e_old_right = self.e[-2]

        self.h[:] = self.h[:] - dt / self.dx / MU0 * (self.e[1:] - self.e[:-1])
        # Total field
        if self.total_field:
            isource = self.total_field[0]
            sourcefunction = self.total_field[1]
            self.h[isource] = self.h[isource] + sourcefunction(self.xH[isource],self.time)

        self.time += dt/2 # First half time step upload

        if self.material_regions:
            start_x, end_x, *_ = self.material_regions[0]
            iE_in = np.searchsorted(self.xE, start_x)
            iE_out = np.searchsorted(self.xE, end_x)
            k_mat = self.material_coefficients[0]
            beta_mat = self.material_coefficients[1]
            den_mat = self.material_coefficients[2]
            coef_mat = self.material_coefficients[3]

            e_old = np.copy(self.e)
         # Left-side of the dispersive material
            self.e[1:iE_in] = (1 / ((self.eps[1:iE_in] / dt) + (self.cond[1:iE_in] / 2))) * (
            ((self.eps[1:iE_in] / dt) - (self.cond[1:iE_in] / 2)) * self.e[1:iE_in]
            - 1 / self.dx * (self.h[1:iE_in] - self.h[0:iE_in - 1]))
         # Inside the dispersive material
            Jsum = 0
            for p in range(6):
                Jsum += np.real((1+k_mat[p])*self.J[p,:])
        
            self.e[iE_in:iE_out] = coef_mat*self.e[iE_in:iE_out] + 2*dt/den_mat*(-(self.h[iE_in:iE_out] - self.h[(iE_in-1):(iE_out-1)])/self.dx - Jsum[iE_in:iE_out])

            for p in range(6):
                self.J[p,:] = k_mat[p]*self.J[p,:] + beta_mat[p]*(self.e-e_old)/dt

             # Right-side of the material
            self.e[iE_out:-1] = (1 / ((self.eps[iE_out:-1] / dt) + (self.cond[iE_out:-1] / 2))) * (
            ((self.eps[iE_out:-1] / dt) - (self.cond[iE_out:-1] / 2)) * self.e[iE_out:-1]
            - 1 / self.dx * (self.h[iE_out:] - self.h[iE_out - 1:-1])
             )
        else:
            self.e[1:-1] = ( 1 / ((self.eps[1:-1] / dt) + (self.cond[1:-1] / 2)) ) * ( ( (self.eps[1:-1]/dt) - (self.cond[1:-1]/2) ) * self.e[1:-1] - 1 / self.dx * (self.h[1:] - self.h[:-1]) )
        # Total field
        if self.total_field:
            self.e[isource] = self.e[isource] + sourcefunction(self.xE[isource],self.time)
        
        self.time += dt/2 #  Second half time step upload
        # Boundry Conditions
        if self.bounds[0] == 'pec':
            self.e[0] = 0.0
        elif self.bounds[0] == 'mur':
            self.e[0] = self.e_old_left + (C0*dt - self.dx) / \
                (C0*dt + self.dx)*(self.e[1] - self.e[0])
        elif self.bounds[0] == 'pmc':
            self.e[0] = self.e[0] - 2 * dt/ self.dx/ EPS0*(self.h[0])
        elif self.bounds[0] == 'periodic':
            self.e[0] = self.e[-2]
        else:
            raise ValueError(f"Unknown boundary condition: {self.bounds[0]}")

        if self.bounds[1] == 'pec':
            self.e[-1] = 0.0
        elif self.bounds[1] == 'mur':
            self.e[-1] = self.e_old_right + (C0*dt - self.dx) / \
                (C0*dt + self.dx)*(self.e[-2] - self.e[-1])
        elif self.bounds[1] == 'pmc':
            self.e[-1] = self.e[-1] + 2 * dt/self.dx / EPS0*(self.h[-1])
        elif self.bounds[1] == 'periodic':
            self.e[-1] = self.e[1]
        else:
            raise ValueError(f"Unknown boundary condition: {self.bounds[1]}")
        
        # Energy calculation
        self.energyE.append(0.5 * np.dot(self.e, self.dx * self.eps * self.e))
        self.energyH.append(0.5 * np.dot(self.h_old, self.dx * MU0 * self.h))
        self.energy.append(0.5 * np.dot(self.e, self.dx * self.eps * self.e) + 0.5 * np.dot(self.h_old, self.dx * MU0 * self.h))
        self.h_old[:] = self.h[:]

        # For debugging and visualization
        plt.plot(self.xE, self.e, label='Electric Field')
        plt.plot(self.xH, self.h, label='Magnetic Field')
        plt.ylim(-1,1)
        if self.material_regions:
            for start_x, end_x, *_ in self.material_regions:
                plt.axvspan(start_x, end_x, color='gray', alpha=0.3)
        plt.pause(0.01)
        plt.grid()
        plt.cla()
    
    def add_totalfield(self,xs,sourceFunction):
        isource = np.where(self.xE > xs)[0][0] # Index in xE and xH of the location of the source
        self.total_field = self.total_field + [isource, sourceFunction]



    def run_until(self, Tf, dt):
        n_steps = int(Tf / dt)
        for _ in range(n_steps):
            self.step(dt)

        return self.e 

######################################################################
# ----------- Ejecución -------------------

x = np.linspace(0, 300e-9, 600)
sim = FDTD1D(x, bounds=('pmc', 'pmc'))

def source_test(x,t):
    return 0.5*np.exp(-(x-2*C0*t)**2 / (20e-9)**2 )

sim.set_initial_condition(np.zeros_like(sim.xE)) #asi solo esta la fuente de total_field

#pulse = gaussian_pulse(sim.xE, 25, 5.0)
#sim.set_initial_condition(pulse)

# Fuente total-field (pulso gaussiano en el tiempo)
sim.add_totalfield(
    100e-9,
    source_test)

# Región de material (plata)
sim.set_material_region([
    (150e-9, 200e-9, 5.0, 6.14e7)
],dt=0.5 * sim.dx / C0)

# Ejecutar simulación
sim.run_until(Tf=100, dt=0.5 * sim.dx / C0)