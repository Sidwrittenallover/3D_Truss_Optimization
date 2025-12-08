import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import lazy_3d_1 as tc

def pso_optimization(
    n_particles, n_iterations, c1, c2, E, ρ, xFac, nodes, members, restrainedDoF, forceVectors,
    A_min, A_max, sigma_max, delta_max, f_min_list,
    symmetry_groups=None, stress=True, displacement=True, buckling=True, tolerance=1e-6
):
    # Determine if symmetry groups are provided
    if symmetry_groups is None:
        symmetry_groups = [[i + 1] for i in range(len(members))]  # Treat each member independently

    n_symmetry_groups = len(symmetry_groups)  # Number of symmetry groups

    # Initialize particles for symmetry groups with integers in cm²
    positions = np.random.randint(A_min, A_max + 1, (n_particles, n_symmetry_groups)).astype(float)
    velocities = np.zeros((n_particles, n_symmetry_groups))
    personal_best_positions = np.copy(positions)
    personal_best_scores = [np.inf] * n_particles
    global_best_position = np.zeros(n_symmetry_groups)
    global_best_score = np.inf

    v_max = (A_max - A_min) * 0.01  # Reduced for finer search
    v_min = -v_max

    best_scores_over_time = []

    for iteration in tqdm(range(n_iterations), desc="PSO Optimization Progress"):
        # Adaptive inertia weight: starts high (exploration) -> decreases (exploitation)
        w = 0.9 - 0.5 * (iteration / n_iterations)      
        for i in range(n_particles):
            member_areas = np.zeros(len(members))
            for group_idx, group in enumerate(symmetry_groups):
                for member_idx in group:
                    member_areas[member_idx - 1] = positions[i, group_idx] / 1e4  # Convert cm² to m²

            truss = tc.TrussAnalysis3D(E, member_areas, ρ, xFac, nodes, members,
                                   restrainedDoF, forceVectors)

            #if not truss.stable:
            #    fitness = np.inf
            #else:
            
            weight = truss.calculate_weight()

            
            # Get maximum stress if needed
            max_stress = truss.get_max_stress() if stress else 0
            
            # Get maximum displacement if needed
            max_displacement = truss.get_max_displacement() if displacement else 0
            
            
            # Get buckling constraints if needed
            #buckling_constraints = truss.get_buckling_constraints() if buckling else None

            # Check constraint violations
            constraint_violation = 0

            #if stress and max_stress > sigma_max:
            #    constraint_violation += (max_stress - sigma_max) / sigma_max


            if stress or buckling:
                aisc_violation = truss.calculate_aisc_constraints(400e6)
                if aisc_violation > 0:
                    constraint_violation += aisc_violation / (0.6 * 400e6 )

            if displacement and max_displacement > delta_max:
                    constraint_violation += (max_displacement - delta_max) / delta_max

            # if buckling and buckling_constraints is not None and np.any(buckling_constraints > 0):
            #     maximum_buckling_violations = np.max(buckling_constraints)
            #     constraint_violation += maximum_buckling_violations / sigma_max

            # Feasibility-based fitness evaluation
            if constraint_violation == 0:
                fitness = weight  # Feasible solutions evaluated by weight
            else:
                penalty_factor = 1e6 * (1+ iteration/n_iterations)  # Adjust this value as necessary
                fitness = weight + penalty_factor * (constraint_violation)

            # Update personal bests
            if fitness < personal_best_scores[i]:
                personal_best_scores[i] = fitness
                personal_best_positions[i] = np.copy(positions[i])

            # Update global best
            if fitness < global_best_score:
                global_best_score = fitness
                global_best_position = np.copy(positions[i])

        best_scores_over_time.append(global_best_score)

        # Particle Swarm Optimizer updates
        for i in range(n_particles):
            r1 = np.random.rand(n_symmetry_groups)
            r2 = np.random.rand(n_symmetry_groups)
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best_positions[i] - positions[i]) +
                           c2 * r2 * (global_best_position - positions[i]))

            # Apply velocity limits
            velocities[i] = np.clip(velocities[i], v_min, v_max)

            # Update positions
            positions[i] = positions[i] + velocities[i]

            # Ensure positions remain integers within bounds
            positions[i] = np.clip(positions[i], A_min, A_max).astype(float)
            #positions[i] = np.clip(np.round(positions[i]), A_min, A_max).astype(float)

            # Only round the final best solution
            #global_best_position = np.round(global_best_position)


        # Reinitialize stagnant particles every 50 iterations
        if iteration > 0 and iteration % 50 == 0:
            for i in range(n_particles):
                if personal_best_scores[i] == np.inf:  # Never found feasible solution
                    positions[i] = np.random.randint(A_min, A_max + 1, n_symmetry_groups)
                    velocities[i] = np.zeros(n_symmetry_groups)

    return global_best_position, global_best_score, best_scores_over_time