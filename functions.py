import numpy as np
import matplotlib.pyplot as plt 
import math

from scipy.optimize import newton


from matplotlib.animation import FuncAnimation

## NUMERIC CALCULATION FUNCTIONS
def U_init(xnode, numnp):
    '''Initial condition of Burgers Equation
    (input) xnode arr: Array with x values stored 
    (input) numnp int: Number of nodes

    (output) U tuple: U tuple stores initial conditions for each of the three variables investigated.
    '''

    U = (np.zeros(numnp), np.zeros(numnp), np.zeros(numnp))

    for i in range(numnp):
        if xnode[i] < 0.5:
            U[0][i] = 1.0
            U[1][i] = 0
            U[2][i] = 2.5
        elif xnode[i] >= 0.5:
            U[0][i] = 0.125
            U[1][i] = 0
            U[2][i] = 0.25
    return U

def calc_p (gamma, rho_E, m, rho):
    '''
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) rho_E float: Value of density*Energy 
    (input) m float: Value of momentum 
    (input) rho float: Value of density 

    (output) p float: Return output of calculated pressure
    ''' 

    p = (gamma - 1) * (rho_E - (m**2)/(2*rho))
    return p

def assemble_mass_matrix(numel, xnode, wpg, N_mef):
    '''
    (input) numel int: Number of elements
    (input) xnode arr: Array with x values 
    (input) wpg array: Array with weights 
    (input) N_mef arr: Array with the shape function 

    (output) M: Return output with assembled M matrix
    ''' 
    numnp = numel + 1
    M_rho = np.zeros((numnp, numnp))  # Mass matrix for rho
    M_m = np.zeros((numnp, numnp))    # Mass matrix for m
    M_rho_E = np.zeros((numnp, numnp))# Mass matrix for rho_E

    M = (M_rho, M_m, M_rho_E)  # Tuple of mass matrices for rho, m, rho_E

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        ngaus = wpg.shape[0]
        for ig in range(ngaus):
            N = N_mef[ig, :]
            w_ig = weight[ig]

            M_rho[np.ix_(isp, isp)] += w_ig * (np.outer(N, N))
            M_m[np.ix_(isp, isp)] += w_ig * (np.outer(N, N))
            M_rho_E[np.ix_(isp, isp)] += w_ig * (np.outer(N, N))

    M_rho[0,0] = 1
    M_m[0,0] = 1
    M_rho_E[0,0] = 1

    M_rho[-1, -1] = 1
    M_m[-1, -1] = 1
    M_rho_E[-1, -1] = 1   

    return M

def assemble_flux_vector(U_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 

    (output) F arr: Flux value returned by function 
    '''
    F = (np.zeros(len(U_current[0])), np.zeros(len(U_current[1])), np.zeros(len(U_current[2])))

    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]

        # Get value of each variable at current element  
        rho_el =  U_current[0][isp]
        m_el =  U_current[1][isp]
        rho_E_el = U_current[2][isp] 

        p_el = calc_p(gamma, rho_E_el, m_el, rho_el)
        
        ngaus = wpg.shape[0]

        # Loop over the gaussian points
        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            # Calculate values of rho, m, rho_E and p at gaussian points.
            rho_gp = np.dot(N, rho_el)
            m_gp = np.dot(N, m_el)
            rho_E_gp = np.dot(N, rho_E_el)
            p_gp = np.dot(N, p_el)


            # # Calculate flux using gaussian values. OPTION 1 I DID AND IT WORKS
            F_rho_gp = m_gp
            F_m_gp = m_gp**2/rho_gp + p_gp
            F_rho_E_gp = (m_gp * (rho_E_gp + p_gp)/ rho_gp)

            F[0][isp] += w_ig * (Nx * F_rho_gp)
            F[1][isp] += w_ig * (Nx * F_m_gp)
            F[2][isp] += w_ig * (Nx * F_rho_E_gp)
    return F

def apply_boundary_conditions(U_temp, numnp):
    '''
    (input) U tuple: List of solution arrays at specific timestep
    (input) numnp int: Number of nodes

    (output) U_temp tuple: U_temp list with applied boundary conditions  
    '''
    U_temp[0][0] = 1.0  # Homogeneous inflow boundary condition at the first node of each variable
    U_temp[1][0] = 0.0
    U_temp[2][0] = 2.5

    U_temp[0][numnp-1] = 0.125  # Homogeneous outflow boundary condition at the last node of each variable
    U_temp[1][numnp-1] = 0
    U_temp[2][numnp-1] = 0.25

    return U_temp

def compute_jacobian (rho, m, rho_E, gamma):
    '''
    (input) rho float: Value of density 
    (input) m float: Value of momentum 
    (input) rho_E float: Value of density*Energy 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)

    (output) A arr: Array of Jacobian matrix
    ''' 

    A = np.array([
                 [0, 1, 0],
                 [(gamma-3)/2 * m**2/rho**2, (3-gamma)*(m/rho), (gamma-1)],
                 [-gamma * ((m * rho_E)/rho**2) + (gamma-1)*(m**3/rho**3), gamma*(rho_E / rho) - 1.5*(gamma-1)* (m**2/rho**2), gamma * m / rho]
                 ] 
                 )
    return A

def assemble_TG_one_step(U_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma, dt):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) dt float: timestep


    (output) M: Mass matrix tuple returned by function 
    '''
    numnp = numel + 1

    M_rho = np.zeros((numnp, numnp))
    M_m = np.zeros((numnp, numnp))
    M_rho_E = np.zeros((numnp, numnp))
    M = (M_rho, M_m, M_rho_E)

    F_rho = np.zeros(numnp)
    F_m = np.zeros(numnp)
    F_rho_E = np.zeros(numnp)
    F = (F_rho, F_m, F_rho_E)

    K_rho = np.zeros((numnp, numnp))
    K_m = np.zeros((numnp, numnp))
    K_rho_E = np.zeros((numnp, numnp))
    K = (K_rho, K_m, K_rho_E) # Tuple of stifness matrices for rho, m, rho_E



    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el =  U_current[0][isp]
        m_el =  U_current[1][isp]
        rho_E_el = U_current[2][isp] 
        p_el = calc_p(gamma, rho_E_el, m_el, rho_el)
        
        ngaus = wpg.shape[0]

        F_rho_el = m_el
        F_m_el = m_el**2/rho_el + p_el
        F_rho_E_el = (m_el * (rho_E_el + p_el)/ rho_el)

        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            # # Calculate values of rho, m, rho_E and p and their derivatives at the gaussian points.
            rho_gp = np.dot(N, rho_el)
            m_gp = np.dot(N, m_el)
            rho_E_gp = np.dot(N, rho_E_el)
            p_gp = np.dot(N, p_el)
            # p_gp = calc_p(gamma, rho_E_gp, m_gp, rho_gp) # pressure straight at gaussian point


            # Calculate flux using gaussian values
            vel_gp = m_gp/rho_gp


            # if, elif used to check if velocity is positive or negative. For compression regions dv/dx < 0 we take the linear approximation.
            if vel_gp >= 0:
                F_rho_gp = m_gp
                F_m_gp = m_gp**2/rho_gp + p_gp
                F_rho_E_gp = (m_gp * (rho_E_gp + p_gp)/ rho_gp)

            elif vel_gp <0:
                F_rho_gp = np.dot(N, F_rho_el)
                F_m_gp = np.dot(N, F_m_el)
                F_rho_E_gp = np.dot(N, F_rho_E_el)

            M_rho[np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M_m[np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M_rho_E[np.ix_(isp, isp)] += w_ig * np.outer(N, N)

            F_rho[isp] += w_ig * ( F_rho_gp * Nx )
            F_m[isp] += w_ig * ( F_m_gp * Nx )
            F_rho_E[isp] += w_ig * ( F_rho_E_gp * Nx )


            A = compute_jacobian(rho_gp, m_gp, rho_E_gp, gamma)
            A_squared = A**2

            K_rho[np.ix_(isp, isp)] += - 0.5 * dt * w_ig * (A_squared[0, 0] *(np.outer(Nx, Nx)) + 
                                        A_squared[1, 0] *(np.outer(Nx, Nx)) +  
                                        A_squared[2, 0] *(np.outer(Nx, Nx)))
            
            K_m[np.ix_(isp, isp)] += - 0.5 * dt * w_ig * (A_squared[0, 1] * (np.outer(Nx, Nx)) + 
                                      A_squared[1, 1] *(np.outer(Nx, Nx)) +  
                                      A_squared[2, 1] *(np.outer(Nx, Nx)))
            
            K_rho_E[np.ix_(isp, isp)] += - 0.5 * dt * w_ig * (A_squared[0, 2] *(np.outer(Nx, Nx)) + 
                                          A_squared[1, 2] *(np.outer(Nx, Nx)) +  
                                          A_squared[2, 2] *(np.outer(Nx, Nx)))
            
    M_rho[0,0] = 1
    M_m[0,0] = 1
    M_rho_E[0,0] = 1

    M_rho[-1, -1] = 1
    M_m[-1, -1] = 1
    M_rho_E[-1, -1] = 1  
            
    return M, F, K

def assemble_TG_two_step(U_current, numel, xnode, N_mef, Nxi_mef, wpg, gamma, dt):
    '''
    (input) U_current tuple: current solution tuple
    (input) numel int: number of elements
    (input) xnode arr: Array with x values stored
    (input) N_mef arr: Array with the shape function 
    (input) Nxi_mef arr: Array with shape function derivatives
    (input) wpg arr: Array with weights 
    (input) gamma float: Ratio of Cv/Cp (7/5 for ideal gas)
    (input) dt float: timestep

    (output) M: Mass matrix tuple returned by function 
    '''
    numnp = numel + 1

    M_rho = np.zeros((numnp, numnp))
    M_m = np.zeros((numnp, numnp))
    M_rho_E = np.zeros((numnp, numnp))
    M = (M_rho, M_m, M_rho_E)

    F_rho = np.zeros(numnp)
    F_m = np.zeros(numnp)
    F_rho_E = np.zeros(numnp)
    F = (F_rho, F_m, F_rho_E)



    for i in range(numel):
        h = xnode[i + 1] - xnode[i]
        weight = wpg * h / 2
        isp = [i, i + 1]  # Global number of the nodes of the current element

        # Get value of each variable at current element  
        rho_el =  U_current[0][isp]
        m_el =  U_current[1][isp]
        rho_E_el = U_current[2][isp] 
        p_el = calc_p(gamma, rho_E_el, m_el, rho_el)

        ngaus = wpg.shape[0]

        F_rho_el = m_el
        F_m_el = m_el**2/rho_el + p_el
        F_rho_E_el = (m_el * (rho_E_el + p_el)/ rho_el)

        for ig in range(ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]

            # Intermediate value at integration(Gaussian) point:
            rho_gp = np.dot(N, rho_el)
            m_gp = np.dot(N, m_el)
            rho_E_gp = np.dot(N, rho_E_el)

            F_rho_gpx = np.dot(Nx, F_rho_el)
            F_m_gpx = np.dot(Nx, F_m_el)
            F_rho_E_gpx = np.dot(Nx, F_rho_E_el)

            rho_inter = rho_gp - 0.5 * dt * F_rho_gpx
            m_inter = m_gp - 0.5 * dt * F_m_gpx
            rho_E_inter = rho_E_gp - 0.5 * dt * F_rho_E_gpx
            p_inter = calc_p(gamma, rho_E_inter, m_inter, rho_inter)

            F_rho_inter = m_inter
            F_m_inter = m_inter**2/ rho_inter + p_inter
            F_rho_E_inter = (m_inter * (rho_E_inter + p_inter)/ rho_inter)


            M_rho[np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M_m[np.ix_(isp, isp)] += w_ig * np.outer(N, N)
            M_rho_E[np.ix_(isp, isp)] += w_ig * np.outer(N, N)

            F_rho[isp] += w_ig * (Nx * F_rho_inter)
            F_m[isp] += w_ig * (Nx * F_m_inter)
            F_rho_E[isp] += w_ig * (Nx * F_rho_E_inter)



    M_rho[0,0] = 1
    M_m[0,0] = 1
    M_rho_E[0,0] = 1

    M_rho[-1, -1] = 1
    M_m[-1, -1] = 1
    M_rho_E[-1, -1] = 1  

    return M, F


## ANALYTIC CALCULATION FUNCTIONS
def f(P, pL, pR, cL, cR, gamma):
    a = (gamma-1)*(cR/cL)*(P-1) 
    b = np.sqrt( 2*gamma*(2*gamma + (gamma+1)*(P-1) ) )
    return P - pL/pR*( 1 - a/b )**(2.*gamma/(gamma-1.))

def SodShockAnalytic(config, t_end):

    h = config['xnode'][1]
    Nx = len(config['xnode'])
    v_analytic = np.zeros((3,Nx),dtype='float64')

    # compute speed of sound
    cL = np.sqrt(config['gamma']*config['U_init_analytical_left'][2]/config['U_init_analytical_left'][0]) 
    cR = np.sqrt(config['gamma']*config['U_init_analytical_right'][2]/config['U_init_analytical_right'][0])
    # compute P
    P = newton(f, 0.5, args=(config['U_init_analytical_left'][2], config['U_init_analytical_right'][2], cL, cR, config['gamma']), tol=1e-12)

    # compute region positions right to left
    # region R
    c_shock = config['U_init_analytical_right'][1] + cR*np.sqrt( (config['gamma']-1+P*(config['gamma']+1)) / (2*config['gamma']) )
    x_shock = config['x0_analytical'] + int(np.floor(c_shock*t_end/h))
    v_analytic[0,x_shock-1:] = config['U_init_analytical_right'][0]
    v_analytic[1,x_shock-1:] = config['U_init_analytical_right'][1]
    v_analytic[2,x_shock-1:] = config['U_init_analytical_right'][2]
    
    # region 2
    alpha = (config['gamma']+1)/(config['gamma']-1)
    c_contact = config['U_init_analytical_left'][1] + 2*cL/(config['gamma']-1)*( 1-(P*config['U_init_analytical_right'][2]/config['U_init_analytical_left'][2])**((config['gamma']-1.)/2/config['gamma']) )
    x_contact = config['x0_analytical'] + int(np.floor(c_contact*t_end/h))
    v_analytic[0,x_contact:x_shock-1] = (1 + alpha*P)/(alpha+P)*config['U_init_analytical_right'][0]
    v_analytic[1,x_contact:x_shock-1] = c_contact
    v_analytic[2,x_contact:x_shock-1] = P*config['U_init_analytical_right'][2]
    
    # region 3
    r3 = config['U_init_analytical_left'][0]*(P*config['U_init_analytical_right'][2]/config['U_init_analytical_left'][2])**(1/config['gamma'])
    p3 = P*config['U_init_analytical_right'][2]
    c_fanright = c_contact - np.sqrt(config['gamma']*p3/r3)
    x_fanright = config['x0_analytical'] + int(np.ceil(c_fanright*t_end/h))
    v_analytic[0,x_fanright:x_contact] = r3
    v_analytic[1,x_fanright:x_contact] = c_contact
    v_analytic[2,x_fanright:x_contact] = P*config['U_init_analytical_right'][2]
    
    # region 4
    c_fanleft = -cL
    x_fanleft = config['x0_analytical'] + int(np.ceil(c_fanleft*t_end/h))
    u4 = 2 / (config['gamma']+1) * (cL + (config['xnode'][x_fanleft:x_fanright]-config['xnode'][config['x0_analytical']])/t_end )
    v_analytic[0,x_fanleft:x_fanright] = config['U_init_analytical_left'][0]*(1 - (config['gamma']-1)/2.*u4/cL)**(2/(config['gamma']-1));
    v_analytic[1,x_fanleft:x_fanright] = u4;
    v_analytic[2,x_fanleft:x_fanright] = config['U_init_analytical_left'][2]*(1 - (config['gamma']-1)/2.*u4/cL)**(2*config['gamma']/(config['gamma']-1));

    # region L
    v_analytic[0,:x_fanleft] = config['U_init_analytical_left'][0]
    v_analytic[1,:x_fanleft] = config['U_init_analytical_left'][1]
    v_analytic[2,:x_fanleft] = config['U_init_analytical_left'][2]

    rho_energy = v_analytic[2]/(config['gamma']-1) + (v_analytic[0] * v_analytic[1]**2)/2

    return v_analytic, rho_energy


## Plotting Functions
def plot_solution(t_end, variables_tuple , config, analytic, rho_energy_analytic):
    fig, axs = plt.subplots(2,2,figsize=(8,8), layout='constrained')
    # First row
    axs[0, 0].set_title(f"Density - t = {t_end}s")
    axs[0, 0].plot(config['xnode'], variables_tuple[0][:, config['nstep']], linestyle ="", marker="x")
    axs[0, 0].plot(config['xnode'], analytic[0].T)
    axs[0, 0].set_ylabel('rho', fontweight='bold')
    axs[0, 0].set_xlabel('x', fontweight='bold')
    axs[0, 0].set_xlim([0.0, 1.05])
    axs[0, 0].set_xticks([i * 0.1 for i in range(11)])
    axs[0, 0].set_ylim([-0.05,1.05])


    axs[0, 1].set_title(f"Velocity - t = {t_end}s")
    axs[0, 1].plot(config['xnode'], variables_tuple[1][:, config['nstep']], linestyle ="", marker="x")
    axs[0, 1].plot(config['xnode'], analytic[1].T)
    axs[0, 1].set_ylabel('v', fontweight='bold')
    axs[0, 1].set_xlabel('x', fontweight='bold')
    axs[0, 1].set_xlim([0.0, 1.05])
    axs[0, 1].set_xticks([i * 0.1 for i in range(11)])
    axs[0, 1].set_ylim([-0.05,1.05])

    # Second row
    axs[1, 0].set_title(f"Pressure - t = {t_end}s")
    axs[1, 0].plot(config['xnode'], variables_tuple[2][:, config['nstep']], linestyle ="", marker="x")
    axs[1, 0].plot(config['xnode'], analytic[2].T)
    axs[1, 0].set_ylabel('p', fontweight='bold')
    axs[1, 0].set_xlabel('x', fontweight='bold')
    axs[1, 0].set_xlim([0.0, 1.05])
    axs[1, 0].set_xticks([i * 0.1 for i in range(11)])
    axs[1, 0].set_ylim([-0.05,1.05])

    axs[1, 1].set_title(f"rho_Energy - t = {t_end}s")
    axs[1, 1].plot(config['xnode'], variables_tuple[3][:, config['nstep']], linestyle ="", marker="x")
    axs[1, 1].plot(config['xnode'], rho_energy_analytic.T)
    axs[1, 1].set_ylabel('rho_E', fontweight='bold')
    axs[1, 1].set_xlabel('x', fontweight='bold')
    axs[1, 1].set_xlim([0.0, 1.05])
    axs[1, 1].set_xticks([i * 0.1 for i in range(11)])
    axs[1, 1].set_ylim([-0.10, 3.05])

    plt.savefig(f"./{config['folder_path']}/{config['method_file_name']}_t_end={t_end}.png")

def plot_animation(xnode, U, nstep, ylabel, variable_title, stabilization_graph_title, folder_path, file_name, t_end, dt):
    fig, ax = plt.subplots()
    line, = ax.plot(xnode, U[:, 0], label=f't = 0.0', linestyle ="", marker="x")
    ax.set_xlabel('x')
    ax.set_ylabel(ylabel)
    ax.set_title(stabilization_graph_title + variable_title)
    ax.set_xlim(0, 1)
    ax.legend()

    def update(frame):
        line.set_ydata(U[:, frame])
        ax.legend([f't = {frame * dt:.2f}'])
        return line,

    ani = FuncAnimation(fig, update, frames=range(0, nstep + 1), blit=True)
    ani.save(f'{folder_path}/{file_name}_{ylabel}_t={t_end}.gif', writer='imagemagick')
    plt.close(fig)


    