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


class Circuit:
    def __init__(self, num_nodes, reference_node):
        self.num_nodes = num_nodes
        self.reference_node = reference_node
        self.components = []
        self.branch_info = []  # Store branch information

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
            if 'C' in comp.name:
                eq_resistance = timestep / (2 * comp.value)
                transformed_components.append(Component(comp.name+"_R_eq", eq_resistance, comp.nodes))
            elif 'L' in comp.name:
                eq_resistance = 2 * comp.value / timestep
                transformed_components.append(Component(comp.name+"_R_eq", eq_resistance, comp.nodes))
            else:
                transformed_components.append(comp)
        self.components = transformed_components

    def mna_analysis(self, t=0):
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

            if 'R' in comp.name:
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
        self.branch_info = []
        for comp in self.components:
            n1, n2 = comp.nodes
            v1 = voltages[n1 - 1] if n1 > 0 else 0  # Voltage at node n1
            v2 = voltages[n2 - 1] if n2 > 0 else 0  # Voltage at node n2
            branch_voltage = v1 - v2  # Voltage across the component
            branch_current = 0

            if 'R' in comp.name:  # Resistor
                branch_current = branch_voltage / comp.value
            elif 'C' in comp.name or 'L' in comp.name:  # Capacitor or Inductor
                branch_current = branch_voltage / comp.value
            elif 'I' in comp.name:  # Current source
                branch_current = comp.value
            elif 'V' in comp.name:  # Voltage source
                source_index = voltage_sources.index(comp)
                branch_current = voltages[self.num_nodes - 1 + source_index][0]

            self.branch_info.append((comp.name, n1, n2, branch_voltage, branch_current))

    def display_branch_info(self):
        print("\nBranch Information:")
        for branch in self.branch_info:
            voltage = branch[3]  # Voltage across the branch
            current = branch[4]  # Current through the branch
            if isinstance(voltage, np.ndarray):  # Check if voltage is a NumPy array
                voltage = voltage.item()  # Extract scalar value
            if isinstance(current, np.ndarray):  # Check if current is a NumPy array
                current = current.item()  # Extract scalar value
            print(f"Component {branch[0]}: Nodes {branch[1]}-{branch[2]}, Voltage = {voltage:.4f} V, Current = {current:.4f} A")

    def time_domain_simulation(self, total_time, timestep):
        self.transform_components(timestep)  # Transform reactive components
        times = np.arange(0, total_time, timestep)
        voltages_over_time = []

        for t in times:
            voltages = self.mna_analysis(t)
            if voltages is not None:
                voltages_with_reference = np.insert(voltages.flatten(), self.reference_node, 0)
                voltages_over_time.append(voltages_with_reference)
            else:
                print("Error: Singular matrix at time t =", t)
                return None

        voltages_over_time = np.array(voltages_over_time)
        self.plot_node_voltages_over_time(times, voltages_over_time)

    def plot_node_voltages_over_time(self, times, voltages_over_time):
        plt.figure()
        for node in range(voltages_over_time.shape[1]):
            plt.plot(times, voltages_over_time[:, node], label=f"Node {node}")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Node Voltages Over Time")
        plt.legend()
        plt.show()


# Define the circuit
num_nodes = 3
reference_node = 0
circuit = Circuit(num_nodes, reference_node)

components = [
    Component("I1", 1, [1, 0], "AC", 50),  # AC Current source
    Component("R1", 1, [1, 0]), 
    Component("R1", 1, [2, 0]),  
    Component("V1", 1, [2, 0], "DC"), 
    Component("R1", 1, [2, 1]),  
]

for comp in components:
    circuit.add_component(comp)

# Display netlist
print("Netlist:")
for line in circuit.generate_netlist():
    print(line)

# Perform time-domain simulation
timestep = 0.001
total_time = 0.1
circuit.time_domain_simulation(total_time, timestep)

# Display branch information
circuit.display_branch_info()

