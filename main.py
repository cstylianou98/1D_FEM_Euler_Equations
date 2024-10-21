import numpy as np
from functions import *
import os
import matplotlib.pyplot as plt

from scipy.linalg import solve


def configure_simulation():
     t_end = float(input("\n>>> Input the last timestep (s) -----> "))
     stabilization_choice = int(input(
          "\n>>> Please choose your numerical scheme. \n"
          " 1. RK4 with Standard Galerkin \n"
          " 2. Taylor Galerkin (TG2) One-Step \n"
          " 3. Taylor Galerkin (TG2) Two-Step \n"
          " 4. RK4 with Taylor Galerkin Two-Step \n"
          " 5. RK4 with Taylor Galerkin Two-Step and Entropy Viscosity \n"
          "\nType your choice here -----> "
     ))

     while stabilization_choice not in [1, 2, 3, 4, 5]:
          print ("\n>>> Invalid choice. Please type an appropriate integer (1, 2, 3, 4) for the relevant numerical scheme choice.")
          stabilization_choice = int(input(
          "1. RK4 with Standard Galerkin \n"
          "2. Taylor Galerkin (TG2) One-Step \n"
          "3. Taylor Galerkin (TG2) Two-Step \n"
          "4. RK4 with Taylor Galerkin (TG2) Two-Step \n"
          " 5. RK4 with Taylor Galerkin Two-Step and Entropy Viscosity \n"
          "\nType your choice here -----> "
          ))
     return t_end, stabilization_choice

# Setup and initialization
def setup_simulation(t_end, stabilization_choice):
    variables_titles = ['- Density', '- Velocity', '- Pressure', '- Energy']
    y_axis_labels = ['rho', 'v', 'p', 'E']
    stabilization_graph_titles = ['1D Shock Tube (RK4-Standard Galerkin) ', '1D Shock Tube (TG2-One step)', '1D Shock Tube (TG2-Two step)', '1D Shock Tube (RK4-TG2 Two step)', 'RK4-TG2 Two-step and EV']
    folder_paths = ['./sod_tube_RK4_standard_galerkin', './sod_tube_TG2_one_step', './sod_tube_TG2_two_step', './sod_tube_RK4_TG2_two_step', './RK4_TG2_two_step_EV']
    file_names = ['sod_tube_RK4_standard_galerkin', 'sod_tube_TG2_one_step', './sod_tube_TG2_two_step', './sod_tube_RK4_TG2_two_step', './RK4_TG2_two_step_EV']
    methods_file_name = ['RK4_standard_galerkin', 'TG2_one_step', 'TG2_two_step', 'RK4_TG2_two_step', 'RK4_TG2_two_step_EV']

    if stabilization_choice == 1:
         stabilization_graph_title = stabilization_graph_titles[0]
         folder_path = folder_paths[0]
         file_name = file_names[0]
         method_file_name = methods_file_name[0]
         dt = 1.5*10**(-3) 


    elif stabilization_choice == 2:
         stabilization_graph_title = stabilization_graph_titles[1]
         folder_path = folder_paths[1]
         file_name = file_names[1]
         method_file_name = methods_file_name[1]
         dt = 1.5*10**(-3) 
   
    elif stabilization_choice == 3:
         stabilization_graph_title = stabilization_graph_titles[2]
         folder_path = folder_paths[2]
         file_name = file_names[2]
         method_file_name = methods_file_name[2]
         dt = 0.002 

    elif stabilization_choice == 4:
         stabilization_graph_title = stabilization_graph_titles[3]
         folder_path = folder_paths[3]
         file_name = file_names[3]
         method_file_name = methods_file_name[3]
         dt = 0.002 

    elif stabilization_choice == 5:
         stabilization_graph_title = stabilization_graph_titles[4]
         folder_path = folder_paths[4]
         file_name = file_names[4]
         method_file_name = methods_file_name[4]
         dt = 0.002 

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Parameters 
    L = 1.0
    numel = 100
    gamma = 1.4
    h = L/numel
    numnp = numel + 1
    xnode = np.linspace(0, L, numnp)

    nstep = int(t_end / dt) 

    # Gauss points and weights for [-1, 1]
    xipg = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    wpg = np.array([1, 1])

    # Shape functions and their derivatives on reference element
    N_mef = np.array([(1 - xipg) / 2, (1 + xipg) / 2])
    Nxi_mef = np.array([[-1/2, 1/2], [-1/2, 1/2]])

    #initializing solution array
    U = (np.zeros((numnp, nstep + 1)), 
        np.zeros((numnp, nstep + 1)), 
        np.zeros((numnp, nstep + 1)))

    # Get initial conditions and set to first timestep of tuple
    rho_init, m_init, rho_E_init = U_init(xnode, numnp)

    U[0][:,0] = rho_init
    U[1][:,0] = m_init
    U[2][:,0] = rho_E_init

    # Get analytical solutions inputs
    U_init_analytical_left = np.array([1.0, 0.0, 1])
    U_init_analytical_right = np.array([0.125, 0.0, 0.1])
    x0_analytical = numel // 2



    return {
         't_end': t_end,
         'variables_titles': variables_titles,
         'y_axis_labels': y_axis_labels,
         'stabilization_choice': stabilization_choice,
         'stabilization_graph_title': stabilization_graph_title,
         'folder_path': folder_path,
         'file_name': file_name,
         'method_file_name': method_file_name,
         'L': L,
         'xnode': xnode,
         'numel': numel,
         'numnp': numnp,
         'gamma': gamma,
         'dt': dt,
         'nstep': nstep,
         'N_mef': N_mef,
         'Nxi_mef': Nxi_mef,
         'wpg': wpg,
         'U': U,
         'U_init_analytical_left': U_init_analytical_left,
         'U_init_analytical_right': U_init_analytical_right,
         'x0_analytical': x0_analytical

    }


# Main time-stepping loop
def run_simulation(config):
    if config['stabilization_choice'] == 1:
        M_tuple = assemble_mass_matrix(config['numel'], config['xnode'], config['wpg'], config['N_mef'])
        for n in range(config['nstep']):
            U_temp = [config['U'][0][:, n], config['U'][1][:, n], config['U'][2][:, n]]

            # k1 step
            F_tuple = assemble_flux_vector(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'])
            k1 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # k2 step
            U_temp = [config['U'][var][:, n] + 0.5 * k1[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            F_tuple = assemble_flux_vector(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'])
            k2 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # k3 step
            U_temp = [config['U'][var][:, n] + 0.5 * k2[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            F_tuple = assemble_flux_vector(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'])
            k3 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # k4 step
            U_temp = [config['U'][var][:, n] + k3[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            F_tuple = assemble_flux_vector(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'])
            k4 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # Update solution
            for var in range(len(config['U'])):
                config['U'][var][:, n + 1] = config['U'][var][:, n] + (1.0 / 6.0) * (k1[var] + 2 * k2[var] + 2 * k3[var] + k4[var])

            # Apply boundary conditions again
            config['U'][0][0, n+1] = 1.0
            config['U'][0][config['numnp']-1, n+1] = 0.125

            config['U'][1][0, n+1] = 0.0
            config['U'][1][config['numnp']-1, n+1] = 0.0

            config['U'][2][0, n+1] = 2.5
            config['U'][2][config['numnp']-1, n+1] = 0.25     

        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        variables_tuple = (rho, vel, final_p, energy)

    elif config['stabilization_choice'] == 2:
        for n in range (config['nstep']):
            U_n = (config['U'][0][:, n], 
                      config['U'][1][:, n], 
                      config['U'][2][:, n])

            U_n = apply_boundary_conditions(U_n, config['numnp'])

            M_tuple, F_tuple, K_tuple = assemble_TG_one_step(U_n, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])

            for var in range(len(config['U'])):
                M = M_tuple[var]
                F = F_tuple[var]
                K = K_tuple[var]
                
                b = config['dt'] * (F + K @ config['U'][var][:, n]) + M @ config['U'][var][:, n]
                U_next = np.linalg.solve(M, b)
                config['U'][var][:, n + 1] = U_next

            # Apply boundary conditions again
            config['U'][0][0, n+1] = 1.0
            config['U'][0][config['numnp']-1, n+1] = 0.125

            config['U'][1][0, n+1] = 0.0
            config['U'][1][config['numnp']-1, n+1] = 0.0

            config['U'][2][0, n+1] = 2.5
            config['U'][2][config['numnp']-1, n+1] = 0.25     


        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        variables_tuple = (rho, vel, final_p, energy)
 
    elif config['stabilization_choice'] == 3:
        for n in range (config['nstep']):
            U_n = (config['U'][0][:, n], 
                      config['U'][1][:, n], 
                      config['U'][2][:, n])

            U_n = apply_boundary_conditions(U_n, config['numnp'])

            M_tuple, F_tuple= assemble_TG_two_step(U_n, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])

            for var in range(len(config['U'])):
                M = M_tuple[var]
                F = F_tuple[var]
                
                b = config['dt'] * (F) + M @ config['U'][var][:, n]
                U_next = np.linalg.solve(M, b)
                config['U'][var][:, n + 1] = U_next

            # Apply boundary conditions again
            config['U'][0][0, n+1] = 1.0
            config['U'][0][config['numnp']-1, n+1] = 0.125

            config['U'][1][0, n+1] = 0.0
            config['U'][1][config['numnp']-1, n+1] = 0.0

            config['U'][2][0, n+1] = 2.5
            config['U'][2][config['numnp']-1, n+1] = 0.25     


        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        variables_tuple = (rho, vel, final_p, energy)

    if config['stabilization_choice'] == 4:
        for n in range(config['nstep']):
            print('Timestep is:', n)
            U_temp = [config['U'][0][:, n], config['U'][1][:, n], config['U'][2][:, n]]

            # k1 step
            M_tuple, F_tuple = assemble_TG_two_step(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])
            k1 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # k2 step
            U_temp = [config['U'][var][:, n] + 0.5 * k1[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            M_tuple, F_tuple = assemble_TG_two_step(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])
            k2 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # k3 step
            U_temp = [config['U'][var][:, n] + 0.5 * k2[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            M_tuple, F_tuple = assemble_TG_two_step(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])
            k3 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # k4 step
            U_temp = [config['U'][var][:, n] + k3[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            M_tuple, F_tuple = assemble_TG_two_step(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'])
            k4 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # Update solution
            for var in range(len(config['U'])):
                config['U'][var][:, n + 1] = config['U'][var][:, n] + (1.0 / 6.0) * (k1[var] + 2 * k2[var] + 2 * k3[var] + k4[var])

            # Apply boundary conditions again
            config['U'][0][0, n+1] = 1.0
            config['U'][0][config['numnp']-1, n+1] = 0.125

            config['U'][1][0, n+1] = 0.0
            config['U'][1][config['numnp']-1, n+1] = 0.0

            config['U'][2][0, n+1] = 2.5
            config['U'][2][config['numnp']-1, n+1] = 0.25     

        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        variables_tuple = (rho, vel, final_p, energy)

    if config['stabilization_choice'] == 5:
        
        viscosity_e = np.zeros((config['numnp'], config['numnp']))
        
        for n in range(config['nstep']):
            print('timestep is', n)
            U_temp = [config['U'][0][:, n], config['U'][1][:, n], config['U'][2][:, n]]

            # k1 step
            M_tuple, F_tuple, viscosity_term_tuple = assemble_TG_two_step(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'], viscosity_e)
            k1 = tuple(config['dt'] * solve(M_tuple[var],  F_tuple[var] + viscosity_term_tuple) for var in range(len(U_temp)))

            # k2 step
            U_temp = [config['U'][var][:, n] + 0.5 * k1[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            M_tuple, F_tuple, viscosity_term_tuple = assemble_TG_two_step(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'], viscosity_e)
            k2 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # k3 step
            U_temp = [config['U'][var][:, n] + 0.5 * k2[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            M_tuple, F_tuple, viscosity_term_tuple = assemble_TG_two_step(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'], viscosity_e)
            k3 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # k4 step
            U_temp = [config['U'][var][:, n] + k3[var] for var in range(len(U_temp))]
            U_temp = apply_boundary_conditions(U_temp, config['numnp'])
            M_tuple, F_tuple, viscosity_term_tuple = assemble_TG_two_step(U_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'], config['dt'], viscosity_e)
            k4 = tuple(config['dt'] * solve(M_tuple[var], F_tuple[var]) for var in range(len(U_temp)))

            # Update solution
            for var in range(len(config['U'])):
                config['U'][var][:, n + 1] = config['U'][var][:, n] + (1.0 / 6.0) * (k1[var] + 2 * k2[var] + 2 * k3[var] + k4[var])

            # Apply boundary conditions again
            config['U'][0][0, n+1] = 1.0
            config['U'][0][config['numnp']-1, n+1] = 0.125

            config['U'][1][0, n+1] = 0.0
            config['U'][1][config['numnp']-1, n+1] = 0.0

            config['U'][2][0, n+1] = 2.5
            config['U'][2][config['numnp']-1, n+1] = 0.25 

            U_entropy_res_temp = [config['U'][var][:, n] for var in range(len(U_temp))]
            entropy_res, viscosity_e = assemble_entropy_res(U_entropy_res_temp, config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['gamma'])

        rho = config['U'][0]
        vel = config['U'][1] / config['U'][0]
        final_p = calc_p(config['gamma'], config['U'][2], config['U'][1], config['U'][0])
        energy = config['U'][2]
        variables_tuple = (rho, vel, final_p, energy, entropy_res, viscosity_e)

    return variables_tuple

    
# Main Execution
def main():
    t_end, stabilization_choice = configure_simulation()
    config = setup_simulation(t_end, stabilization_choice)
    variables_tuple = run_simulation(config)
    analytic, rho_energy_analytic = SodShockAnalytic(config, t_end)  
    plot_solution(t_end, variables_tuple, config, analytic, rho_energy_analytic)
    plot_entropy_res(variables_tuple, config)

    # for var in range (len(variables_tuple)):

        # plot_animation(
        #     config['xnode'], 
        #     variables_tuple[var], 
        #     config['nstep'],
        #     config['y_axis_labels'][var], 
        #     config['variables_titles'][var], 
        #     config['stabilization_graph_title'], 
        #     config['folder_path'], 
        #     config['file_name'], 
        #     config['t_end'],
        #     config['dt']

        # )

if __name__ == "__main__":
    main()
