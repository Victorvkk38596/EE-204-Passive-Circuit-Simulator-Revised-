import numpy as np
import matplotlib.pyplot as plt

class Component:
    def __init__(self, name, value, nodes, ac_dc="DC", frequency=0, phase=0):
        self.name = name
        self.value = value
        self.nodes = nodes
        self.ac_dc = ac_dc
        self.frequency = frequency
        self.phase = phase
        self.prev_voltage = 0  # For capacitors (previous voltage across capacitor)
        self.prev_current = 0  # For inductors (previous current through inductor)

class Circuit:
    def __init__(self, num_nodes, reference_node):
        self.num_nodes = num_nodes
        self.reference_node = reference_node
        self.components = []
        self.branch_info = []  # Store branch information
        self.branch_voltages_over_time = None  # Store branch voltages over time
        self.branch_currents_over_time = None  # Store branch currents over time
        self.nodal_voltages_over_time = None  # Store nodal voltages over time

    def add_component(self, component):
        self.components.append(component)

    def generate_netlist(self):
        netlist = []
        for comp in self.components:
            netlist.append(f"{comp.name} {comp.nodes[0]} {comp.nodes[1]} {comp.value} {comp.ac_dc} {comp.frequency}Hz {comp.phase}°")
        return netlist

    def transform_components(self, timestep):
        transformed_components = []
        for comp in self.components:
            if 'C' in comp.name:  # Capacitor transformation
                if comp.value == 0:
                    print(f"Error: Zero capacitance for {comp.name}. Skipping component.")
                    continue  # Skip capacitors with zero capacitance
                R_eq = timestep / (2 * comp.value)  # Equivalent resistance for capacitor
                I_eq = -(2 * comp.value * comp.prev_voltage / timestep) - comp.prev_current  # Correct capacitor current source
                transformed_components.append(Component(comp.name + "_R_eq", R_eq, comp.nodes))
                transformed_components.append(Component(comp.name + "_I_eq", I_eq, comp.nodes))
            elif 'L' in comp.name:  # Inductor transformation
                if comp.value == 0:
                    print(f"Error: Zero inductance for {comp.name}. Skipping component.")
                    continue  # Skip inductors with zero inductance
                R_eq = 2 * comp.value / timestep  # Equivalent resistance for inductor
                I_eq = (timestep * comp.prev_voltage / (2 * comp.value)) + comp.prev_current  # Correct inductor current source
                transformed_components.append(Component(comp.name + "_R_eq", R_eq, comp.nodes))
                transformed_components.append(Component(comp.name + "_I_eq", I_eq, comp.nodes))
            else:
                transformed_components.append(comp)
        self.components = transformed_components

    def mna_analysis(self, t=0, timestep=0.0001):
        voltage_sources = [c for c in self.components if 'V' in c.name]
        num_voltage_sources = len(voltage_sources)
        num_vars = self.num_nodes + num_voltage_sources - 1

        G_matrix = np.zeros((num_vars, num_vars))
        I_vector = np.zeros((num_vars, 1))

        voltage_index = self.num_nodes - 1

        for comp in self.components:
            n1, n2 = comp.nodes
            if n1 == self.reference_node:
                n1 = None
            elif n1 > self.reference_node:
                n1 -= 1
            if n2 == self.reference_node:
                n2 = None
            elif n2 > self.reference_node:
                n2 -= 1

            if 'R' in comp.name:  # Resistor
                resistance = comp.value
                if n1 is not None:
                    G_matrix[n1, n1] += 1 / resistance
                    if n2 is not None:
                        G_matrix[n1, n2] -= 1 / resistance
                if n2 is not None:
                    G_matrix[n2, n2] += 1 / resistance
                    if n1 is not None:
                        G_matrix[n2, n1] -= 1 / resistance
            elif 'I' in comp.name:  # Current source
                if comp.ac_dc == "AC":
                    current = comp.value * np.sin(2 * np.pi * comp.frequency * t + np.deg2rad(comp.phase))
                else:
                    current = comp.value
                if n1 is not None:
                    I_vector[n1] -= current
                if n2 is not None:
                    I_vector[n2] += current
            elif 'V' in comp.name:  # Voltage source
                if comp.ac_dc == "AC":
                    voltage = comp.value * np.sin(2 * np.pi * comp.frequency * t + np.deg2rad(comp.phase))
                else:
                    voltage = comp.value
                if n1 is not None:
                    G_matrix[n1, voltage_index] = 1
                    G_matrix[voltage_index, n1] = 1
                if n2 is not None:
                    G_matrix[n2, voltage_index] = -1
                    G_matrix[voltage_index, n2] = -1
                I_vector[voltage_index] = voltage
                voltage_index += 1

        if np.linalg.matrix_rank(G_matrix) < G_matrix.shape[0]:
            print("Warning: G_matrix is singular, indicating isolated nodes or missing connections.")
            return None

        voltages = np.linalg.solve(G_matrix, I_vector)
        self.calculate_branch_info(voltages, voltage_sources)
        return voltages[:self.num_nodes - 1]

    def calculate_branch_info(self, voltages, voltage_sources):
        self.branch_info = []  # Reset branch information for each call
        for comp in self.components:
            n1, n2 = comp.nodes
            v1 = voltages[n1 - 1] if n1 > 0 else 0  # Voltage at node n1
            v2 = voltages[n2 - 1] if n2 > 0 else 0  # Voltage at node n2
            branch_voltage = v1 - v2  # Voltage across the component
            branch_current = 0

            if 'R' in comp.name:  # Resistor
                branch_current = branch_voltage / comp.value
            elif 'C' in comp.name:  # Capacitor
                if comp.value == 0:
                    print(f"Error: Zero capacitance for {comp.name}. Skipping capacitor.")
                    continue
                # Correct capacitor's current source
                I_eq = -(2 * comp.value * comp.prev_voltage / timestep) - comp.prev_current
                branch_current = I_eq + (2 * comp.value * branch_voltage / timestep)
                comp.prev_voltage += branch_current / comp.value * timestep  # Update capacitor's voltage dynamically
            elif 'L' in comp.name:  # Inductor
                if comp.value == 0:
                    print(f"Error: Zero inductance for {comp.name}. Skipping inductor.")
                    continue
                # Correct inductor's voltage source
                I_eq = (timestep * comp.prev_voltage / (2 * comp.value)) + comp.prev_current
                branch_current = I_eq + (branch_voltage * timestep / (2 * comp.value))
                comp.prev_current += branch_current * timestep  # Update inductor's current dynamically
            elif 'I' in comp.name:  # Current source
                branch_current = comp.value
            elif 'V' in comp.name:  # Voltage source
                source_index = voltage_sources.index(comp)
                branch_current = voltages[self.num_nodes - 1 + source_index][0]

            self.branch_info.append((comp.name, n1, n2, branch_voltage, branch_current))

    def time_domain_simulation(self, total_time, timestep):
        self.transform_components(timestep)  # Transform reactive components
        times = np.arange(0, total_time, timestep)

        # Preallocate arrays for branch voltages, currents, and nodal voltages
        num_components = len(self.components)
        self.branch_voltages_over_time = np.zeros((len(times), num_components))
        self.branch_currents_over_time = np.zeros((len(times), num_components))
        self.nodal_voltages_over_time = np.zeros((len(times), self.num_nodes))

        # Iterate through each timestep
        for i, t in enumerate(times):
            voltages = self.mna_analysis(t, timestep)
            if voltages is None:
                print("Error: Singular matrix at time t =", t)
                return None

            # Update branch data
            self.branch_voltages_over_time[i, :] = [float(info[3]) for info in self.branch_info]
            self.branch_currents_over_time[i, :] = [float(info[4]) for info in self.branch_info]

            # Update nodal voltages (insert reference node voltage as 0)
            self.nodal_voltages_over_time[i, :] = np.insert(voltages.flatten(), self.reference_node, 0)

        self.plot_branch_voltages_currents(times)
        self.plot_nodal_voltages(times)

    def plot_branch_voltages_currents(self, times):
        # Plot branch voltages
        plt.figure()
        for i, comp in enumerate(self.components):
            plt.plot(times, self.branch_voltages_over_time[:, i], label=f"{comp.name} Voltage")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Branch Voltages Over Time")
        plt.legend()
        plt.show()

        # Plot branch currents
        plt.figure()
        for i, comp in enumerate(self.components):
            plt.plot(times, self.branch_currents_over_time[:, i], label=f"{comp.name} Current")
        plt.xlabel("Time (s)")
        plt.ylabel("Current (A)")
        plt.title("Branch Currents Over Time")
        plt.legend()
        plt.show()

    def plot_nodal_voltages(self, times):
        # Plot nodal voltages
        plt.figure()
        for node in range(self.nodal_voltages_over_time.shape[1]):
            plt.plot(times, self.nodal_voltages_over_time[:, node], label=f"Node {node}")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Nodal Voltages Over Time")
        plt.legend()
        plt.show()

# Define the circuit
num_nodes = 3
reference_node = 0
circuit = Circuit(num_nodes, reference_node)

components = [
    Component("V1", 1, [1, 0], "AC", 50),  # AC Voltage Source
    Component("R2", 1, [1, 0]),  # Resistor
    Component("R3", 1, [2, 0]),  # Resistor
    Component("V2", 1, [2, 0], "DC"),  # DC Voltage Source
    Component("L1", 0.1, [2, 1]),  # Inductor
]

for comp in components:
    circuit.add_component(comp)

# Display netlist
print("Netlist:")
for line in circuit.generate_netlist():
    print(line)

# Perform time-domain simulation
timestep = 0.0001  # Smaller timestep to capture the transient effects
total_time = 0.1
circuit.time_domain_simulation(total_time, timestep)
# Perform time-domain simulation
timestep = 0.001
total_time = 0.1
circuit.time_domain_simulation(total_time, timestep)
