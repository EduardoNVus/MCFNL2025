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

    def step(self, dt):
        self.e_old_left = self.e[1]
        self.e_old_right = self.e[-2]

        self.h[:] = self.h[:] - dt / self.dx / MU0 * (self.e[1:] - self.e[:-1])
        # Total field
        isource = self.total_field[0]
        sourcefunction = self.total_field[1]
        self.h[isource] = self.h[isource] + sourcefunction(self.xH[isource],self.time)
        self.time += dt/2 # First half time step upload


        self.e[1:-1] = ( 1 / ((self.eps[1:-1] / dt) + (self.cond[1:-1] / 2)) ) * ( ( (self.eps[1:-1]/dt) - (self.cond[1:-1]/2) ) * self.e[1:-1] - 1 / self.dx * (self.h[1:] - self.h[:-1]) )
        # Total field
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
        #plt.ylim(-1,1)
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

