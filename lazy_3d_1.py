import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from tqdm import tqdm

class TrussAnalysis3D:
    def __init__(self, E, A, ρ, xFac, nodes, members, restrainedDoF, forceVectors):
        """
        Initializes the 3D Static Truss Analysis.
        Removed: rho (density), nodal_masses, and method (lumped/consistent) as they are for dynamic analysis.
        """
        self.E = E
        self.A = A if isinstance(A, np.ndarray) else np.full(len(members), A)
        self.xFac = xFac
        self.nodes = np.array(nodes, dtype="float")  # Shape (N, 3)
        self.members = np.array(members)
        self.ρ = ρ

        
        # Adjust restrained DoF to 0-based index
        self.restrainedDoF = [x - 1 for x in restrainedDoF]
        self.forceVectors = forceVectors
        
        # 3 Degrees of Freedom per node (x, y, z)
        self.nDoF = np.amax(members) * 3 

        # Initialize cached properties
        self._direction_cosines = None
        self._lengths = None
        self._Kp = None
        self._Ks = None
        self._U_all = None
        self._UG_all = None
        self._FG_all = None
        self._mbrForces_all = None
        self._stresses_all = None
        self._buckling_constraints = None

    @property
    def direction_cosines(self):
        if self._direction_cosines is None:
            self._direction_cosines, self._lengths = self.calculate_member_properties()
        return self._direction_cosines

    @property
    def lengths(self):
        if self._lengths is None:
            self._direction_cosines, self._lengths = self.calculate_member_properties()
        return self._lengths

    @property
    def Kp(self):
        if self._Kp is None:
            self._Kp = self.build_primary_stiffness_matrix()
        return self._Kp

    @property
    def Ks(self):
        if self._Ks is None:
            self._Ks = self.extract_structure_stiffness_matrix()
        return self._Ks

    def calculate_member_properties(self):
        # Vectorized 3D calculation
        node_i_indices = self.members[:, 0] - 1
        node_j_indices = self.members[:, 1] - 1
        
        # Calculate differences in x, y, z
        dx = self.nodes[node_j_indices, 0] - self.nodes[node_i_indices, 0]
        dy = self.nodes[node_j_indices, 1] - self.nodes[node_i_indices, 1]
        dz = self.nodes[node_j_indices, 2] - self.nodes[node_i_indices, 2]
        
        lengths = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Calculate Direction Cosines (Cx, Cy, Cz)
        Cx = dx / lengths
        Cy = dy / lengths
        Cz = dz / lengths
        
        # Stack them into a (Num_Members, 3) array
        direction_cosines = np.stack((Cx, Cy, Cz), axis=1)
        
        return direction_cosines, lengths

    def calculateKg(self, memberNo):
        """
        Builds the 6x6 Global Stiffness Matrix for a 3D Truss Element
        """
        idx = memberNo - 1
        Cx, Cy, Cz = self.direction_cosines[idx]
        L = self.lengths[idx]
        A = self.A[idx]
        coeff = (self.E * A / L)

        # Transformation matrix Lambda (3x3)
        lambda_matrix = np.outer([Cx, Cy, Cz], [Cx, Cy, Cz])
        
        # Construct 6x6 matrix
        # [  Lambda   -Lambda ]
        # [ -Lambda    Lambda ]
        K_local = np.zeros((6, 6))
        K_local[0:3, 0:3] = lambda_matrix
        K_local[0:3, 3:6] = -lambda_matrix
        K_local[3:6, 0:3] = -lambda_matrix
        K_local[3:6, 3:6] = lambda_matrix
        
        return coeff * K_local



    def build_primary_stiffness_matrix(self):
        Kp = np.zeros([self.nDoF, self.nDoF])
        
        for n, mbr in enumerate(self.members):
            k_elem = self.calculateKg(n + 1)
            node_i = mbr[0]
            node_j = mbr[1]
            
            # Global indices for 3D (3 DoF per node)
            # Node i indices: 3*i-3, 3*i-2, 3*i-1
            i_indices = [3 * node_i - 3, 3 * node_i - 2, 3 * node_i - 1]
            j_indices = [3 * node_j - 3, 3 * node_j - 2, 3 * node_j - 1]
            
            indices = i_indices + j_indices 
            
            # Add to global matrix
            for row_local, row_global in enumerate(indices):
                for col_local, col_global in enumerate(indices):
                    Kp[row_global, col_global] += k_elem[row_local, col_local]
                    
        return Kp

    def extract_structure_stiffness_matrix(self):
        Ks = np.delete(self.Kp, self.restrainedDoF, 0)
        Ks = np.delete(Ks, self.restrainedDoF, 1)
        return np.matrix(Ks)

    def solve_displacements(self, forceVector):
        forceVectorRed = np.delete(forceVector, self.restrainedDoF, 0)
        # Use solve instead of inv for numerical stability
        return np.linalg.solve(self.Ks, forceVectorRed)

    def construct_global_displacement_vector(self, U):
        UG = np.zeros(self.nDoF)
        c = 0
        for i in range(self.nDoF):
            if i in self.restrainedDoF:
                UG[i] = 0
            else:
                UG[i] = U[c].item()
                c += 1
        return np.array([UG]).T

    def solve_reactions(self, UG):
        return np.matmul(self.Kp, UG)

    def solve_member_forces(self, UG):
        mbrForces = np.zeros(len(self.members))
        
        for n, mbr in enumerate(self.members):
            Cx, Cy, Cz = self.direction_cosines[n]
            L = self.lengths[n]
            node_i = mbr[0]
            node_j = mbr[1]
            
            # Get indices for Node i and Node j
            idx_i = [3*node_i-3, 3*node_i-2, 3*node_i-1]
            idx_j = [3*node_j-3, 3*node_j-2, 3*node_j-1]
            
            u_i = UG[idx_i].flatten()
            u_j = UG[idx_j].flatten()
            
            # Delta = (uj - ui) dot DirectionVector
            delta = (u_j[0] - u_i[0])*Cx + (u_j[1] - u_i[1])*Cy + (u_j[2] - u_i[2])*Cz
            
            mbrForces[n] = (self.A[n] * self.E / L) * delta
            
        return mbrForces

    def calculate_weight(self):
        return np.sum(self.A * self.lengths * self.ρ)

    def get_stress_displacement_forces(self):
        if self._U_all is None:
            self._U_all = []
            self._UG_all = []
            self._FG_all = []
            self._mbrForces_all = []
            self._stresses_all = []

            #print("Solving 3D static load cases...")
            for forceVector in self.forceVectors:
                U = self.solve_displacements(forceVector)
                UG = self.construct_global_displacement_vector(U)
                FG = self.solve_reactions(UG)
                mbrForces = self.solve_member_forces(UG)
            
                self._U_all.append(U)
                self._UG_all.append(UG)
                self._FG_all.append(FG)
                self._mbrForces_all.append(mbrForces)
                
                stresses = mbrForces / self.A
                self._stresses_all.append(stresses)

        return (self._U_all, self._UG_all, self._FG_all, 
                self._mbrForces_all, self._stresses_all)
    
    def get_max_displacement(self):
        _, UG_all, _, _, _ = self.get_stress_displacement_forces()
        return np.max(np.abs([np.max(np.abs(UG)) for UG in UG_all]))

    def get_max_stress(self):
        _, _, _, _, stresses_all = self.get_stress_displacement_forces()
        return np.max(np.abs([np.max(np.abs(stresses)) for stresses in stresses_all]))    

    def calculate_buckling_constraints(self, k_factors=1):
        if k_factors is None:
            k_factors = np.full(len(self.members), np.pi**2)
        
        σ = self.mbrForces / self.A
        σ_cr = (k_factors * self.A * self.E) / (self.lengths ** 2)
        compressive_members = (0.6 * σ) < 0
        buckling_constraints = np.zeros(len(σ))
        buckling_constraints[compressive_members] = np.abs(σ[compressive_members]) - σ_cr[compressive_members]
        return buckling_constraints

    def get_buckling_constraints(self):
        if self._buckling_constraints is None:
            _, _, _, mbrForces_all, _ = self.get_stress_displacement_forces()
            self._buckling_constraints_all = []
            for mbrForces in mbrForces_all:
                self.mbrForces = mbrForces
                buckling_constraints = self.calculate_buckling_constraints()
                self._buckling_constraints_all.append(buckling_constraints)
            self._buckling_constraints = np.max(self._buckling_constraints_all, axis=0)
        return self._buckling_constraints
    
    def calculate_aisc_constraints(self, Fy):
        # Calculate the maximum stress violation based on AISC ASd 1989.
        # Returns the maximum stress excess (Stress - Allowable). Returns 0 if safe.

        #1. Get stresses for all load cases
        _, _, _, _, stresses_all = self.get_stress_displacement_forces()

        #2. Calculate Geometrx properties 

        A_cm2 = self.A * 10000.0

        #Calculate Radius of Gyration (rm) in cm using power law
        # r = a * A^b
        a = 0.4993
        b = 0.6777
        r_cm = a * np.power(A_cm2, b)
        r_m = r_cm / 100 # Convert from cm to m

        #Slenderness Ratio lambda = k * L/r here k = 1 beacuse the problem is pin jointed.
        lam = self.lengths / r_m

        #3 Calculate Critical Slenderness Cc
        #Cc = sqrt(2* pi^2 * E / Fy)
        Cc = np.sqrt(2 * (np.pi**2) * (2.1e11) / Fy)

        #4 Allowable Buckling stress (Sigma_mb)

        # A. Elastic Buckling (lambda >= Cc)
        sigma_mb_elastic = (12 * (np.pi**2) * (2.1e11)) / (23 * lam**2)

        # B. Inelastic Buckling (lambda < Cc)
        # Factor of Safety denominator

        fs_inelastic = 5/3 + (3 * lam) / (8.0 * Cc) - (lam **3) / (8.0 * Cc**3)

        #Numerator 
        numerator = (1.0 - (lam**2) / (2.0 * (Cc**2))) * Fy

        sigma_mb_inelastic = numerator / fs_inelastic

        #Combine based on lambda threshold
        sigma_mb = np.where(lam >= Cc, sigma_mb_elastic, sigma_mb_inelastic)

        #Check Constraints for each load case

        max_voilation_val = 0.0 

        for stresses in stresses_all:

            #Create Allowable Stress vector
            sigma_all = np.zeros_like(stresses)

            #Tension Members (Stress >=0): Allowable = 0.6 * Fy 

            mask_ten = stresses >=0
            sigma_all[mask_ten] = 0.6 * Fy

            #Compression Members (Stress >=0): Allowable = sigma_mb

            mask_comp = ~mask_ten
            sigma_all[mask_comp] = sigma_mb[mask_comp]

            #Calculate Violation = |Actual Stress| - Allowable Stress
            #If Result > 0, it's a violation
            violations = np.abs(stresses) - sigma_all

            current_max = np.max(violations)
            if current_max > max_voilation_val:
                max_voilation_val = current_max

        #Return the maximum violation found (0 if all safe)

        return max(0.0, max_voilation_val)





    def plot_structure(self):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for mbr in self.members:
            node_i = mbr[0]
            node_j = mbr[1]
            ix, iy, iz = self.nodes[node_i - 1]
            jx, jy, jz = self.nodes[node_j - 1]
            ax.plot([ix, jx], [iy, jy], [iz, jz], 'b-o', markersize=4)
            
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D Structure Geometry')
        
        self._set_axes_equal(ax)
        plt.show()

    def plot_deflected_shape(self, load_case=0):
        _, UG_all, _, _, _ = self.get_stress_displacement_forces()
        
        if load_case >= len(UG_all):
            print(f"Error: Load case {load_case} does not exist.")
            return

        current_UG = UG_all[load_case]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for mbr in self.members:
            node_i = mbr[0]
            node_j = mbr[1]
            ix, iy, iz = self.nodes[node_i - 1]
            jx, jy, jz = self.nodes[node_j - 1]
            
            # Indices
            idx_i = [3*node_i-3, 3*node_i-2, 3*node_i-1]
            idx_j = [3*node_j-3, 3*node_j-2, 3*node_j-1]
            
            # Original
            ax.plot([ix, jx], [iy, jy], [iz, jz], 'grey', lw=0.5, linestyle='--')
            
            # Deflected
            new_ix = ix + current_UG[idx_i[0], 0] * self.xFac
            new_iy = iy + current_UG[idx_i[1], 0] * self.xFac
            new_iz = iz + current_UG[idx_i[2], 0] * self.xFac
            
            new_jx = jx + current_UG[idx_j[0], 0] * self.xFac
            new_jy = jy + current_UG[idx_j[1], 0] * self.xFac
            new_jz = jz + current_UG[idx_j[2], 0] * self.xFac

            ax.plot([new_ix, new_jx], [new_iy, new_jy], [new_iz, new_jz], 'r')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'3D Deflected Shape (Load Case {load_case + 1})')
        self._set_axes_equal(ax)
        plt.show()

    def _set_axes_equal(self, ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def summary_output(self):
        _, UG_all, FG_all, mbrForces_all, stresses_all = self.get_stress_displacement_forces()
    
        print("-" * 50)
        print("3D STATIC STRUCTURAL ANALYSIS SUMMARY")
        print("-" * 50)

        for i, (mbrForces, UG, FG) in enumerate(zip(mbrForces_all, UG_all, FG_all)):
            print(f"\n>>> LOAD CASE {i + 1} <<<")
            
            print("\n--- REACTIONS ---")
            for j in range(len(self.restrainedDoF)):
                index = self.restrainedDoF[j]
                val = FG[index].item()
                if abs(val) > 1e-5:
                    dof_mod = (index + 1) % 3
                    direction = "Z" if dof_mod == 0 else ("X" if dof_mod == 1 else "Y")
                    node_num = (index // 3) + 1
                    print(f"Node {node_num} ({direction}): {val / 1000:.2f} kN")
            
            print("\n--- MEMBER FORCES ---")
            for n in range(len(self.members)):
                force_val = mbrForces[n] / 1000
                status = "(T)" if force_val > 0 else "(C)"
                print(f"Member {n + 1}: {abs(force_val):.2f} kN {status}")