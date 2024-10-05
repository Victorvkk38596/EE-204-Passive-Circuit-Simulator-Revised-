import numpy as np
import matplotlib.pyplot as plt

class ComplexResistorCircuitSolver:
    def __init__(self):
        self.components = {}
        self.nodes = set()
        self.reference_node = None
        self.voltage_sources = []

    def add_resistor(self, name, resistance, node1, node2):
        """Add a resistor to the circuit."""
        if resistance <= 0:
            raise ValueError(f"Resistance must be positive. Got {resistance} for {name}.")
        self.components[name] = {'type': 'resistor', 'value': resistance, 'nodes': (node1, node2)}
        self.nodes.update([node1, node2])

    def add_capacitor(self, name, capacitance, node1, node2):
        """Add a capacitor to the circuit."""
        if capacitance <= 0:
            raise ValueError(f"Capacitance must be positive. Got {capacitance} for {name}.")
        self.components[name] = {'type': 'capacitor', 'value': capacitance, 'nodes': (node1, node2), 'voltage': 0}
        self.nodes.update([node1, node2])

    def add_inductor(self, name, inductance, node1, node2):
        """Add an inductor to the circuit."""
        if inductance <= 0:
            raise ValueError(f"Inductance must be positive. Got {inductance} for {name}.")
        self.components[name] = {'type': 'inductor', 'value': inductance, 'nodes': (node1, node2), 'current': 0}
        self.nodes.update([node1, node2])

    def add_voltage_source(self, name, voltage_function, node1, node2):
        """Add a time-dependent voltage source to the circuit."""
        self.components[name] = {'type': 'voltage_source', 'value_function': voltage_function, 'nodes': (node2, node1)}
        self.nodes.update([node1, node2])
        self.voltage_sources.append(name)

    def set_reference_node(self, node):
        """Set the reference node (ground) for the circuit."""
        if node not in self.nodes:
            raise ValueError(f"Reference node {node} is not in the circuit.")
        self.reference_node = node

    def build_matrices_dynamic(self, prev_voltages, prev_currents, delta_t, t):
        """Build matrices for solving with capacitors and inductors in the dynamic domain."""
        if self.reference_node is None:
            self.reference_node = min(self.nodes)

        nodes = sorted(self.nodes - {self.reference_node})
        node_indices = {node: i for i, node in enumerate(nodes)}

        n = len(nodes)
        m = len(self.voltage_sources)
        size = n + m

        A = np.zeros((size, size))
        z = np.zeros(size)

        for name, component in self.components.items():
            n1, n2 = component['nodes']
            n1_is_ref = n1 == self.reference_node
            n2_is_ref = n2 == self.reference_node

            index_n1 = node_indices[n1] if not n1_is_ref else None
            index_n2 = node_indices[n2] if not n2_is_ref else None

            if component['type'] == 'resistor':
                g = 1 / component['value']
                if index_n1 is not None:
                    A[index_n1, index_n1] += g
                if index_n2 is not None:
                    A[index_n2, index_n2] += g
                if index_n1 is not None and index_n2 is not None:
                    A[index_n1, index_n2] -= g
                    A[index_n2, index_n1] -= g

            elif component['type'] == 'capacitor':
                g = component['value'] / delta_t
                i_prev = component['value'] * (prev_voltages.get(n1, 0) - prev_voltages.get(n2, 0)) / delta_t
                if index_n1 is not None:
                    A[index_n1, index_n1] += g
                    z[index_n1] += i_prev
                if index_n2 is not None:
                    A[index_n2, index_n2] += g
                    z[index_n2] -= i_prev
                if index_n1 is not None and index_n2 is not None:
                    A[index_n1, index_n2] -= g
                    A[index_n2, index_n1] -= g

            elif component['type'] == 'inductor':
                r_eq = delta_t / component['value']
                v_prev = prev_currents[name] * r_eq
                if index_n1 is not None:
                    A[index_n1, index_n1] += 1 / r_eq
                    z[index_n1] += v_prev / r_eq
                if index_n2 is not None:
                    A[index_n2, index_n2] += 1 / r_eq
                    z[index_n2] -= v_prev / r_eq
                if index_n1 is not None and index_n2 is not None:
                    A[index_n1, index_n2] -= 1 / r_eq
                    A[index_n2, index_n1] -= 1 / r_eq

            elif component['type'] == 'voltage_source':
                index = n + self.voltage_sources.index(name)
                if index_n1 is not None:
                    A[index, index_n1] = 1
                    A[index_n1, index] = 1
                if index_n2 is not None:
                    A[index, index_n2] = -1
                    A[index_n2, index] = -1
                z[index] = component['value_function'](t)

        return A, z, node_indices

    def solve_dynamic(self, time_end=1.0, time_step=0.01):
        """Solve the circuit dynamically over time for capacitors and inductors."""
        times = np.arange(0, time_end, time_step)
        node_voltages = {node: [] for node in self.nodes}

        prev_voltages = {node: 0 for node in self.nodes}
        prev_currents = {name: 0 for name in self.components if self.components[name]['type'] == 'inductor'}

        for t in times:
            A, z, node_indices = self.build_matrices_dynamic(prev_voltages, prev_currents, time_step, t)
            solution = np.linalg.solve(A, z)

            voltages = self.get_node_voltages(solution, node_indices)

            # Update previous voltages and currents
            for node in prev_voltages:
                prev_voltages[node] = voltages.get(node, 0)

            for name, component in self.components.items():
                if component['type'] == 'inductor':
                    n1, n2 = component['nodes']
                    v_n1 = voltages.get(n1, 0)
                    v_n2 = voltages.get(n2, 0)
                    voltage_across = v_n1 - v_n2
                    prev_currents[name] += (voltage_across / component['value']) * time_step

            for node in node_voltages:
                node_voltages[node].append(voltages.get(node, 0))

        return times, node_voltages

    def get_node_voltages(self, solution, node_indices):
        """Return the voltages at each node."""
        nodes = sorted(self.nodes - {self.reference_node})
        voltages = {self.reference_node: 0}
        for i, node in enumerate(nodes):
            voltages[node] = solution[i]
        return voltages


# Example Circuit Setup
circuit = ComplexResistorCircuitSolver()

# Add components: resistors, capacitors, inductors
circuit.add_resistor('R1', 100, 0, 1)
circuit.add_resistor('R2', 200, 1, 2)
circuit.add_capacitor('C1', 1e-6, 1, 2)  # 1 microfarad capacitor
circuit.add_inductor('L1', 1e-3, 2, 0)   # 1 millihenry inductor

# Define a sinusoidal voltage source function (e.g., 50 Hz AC source)
def voltage_function(t):
    return 5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave

# Add voltage source
circuit.add_voltage_source('V1', voltage_function, 0, 1)

# Set reference node (ground)
circuit.set_reference_node(0)

# Solve circuit dynamically over time
times, node_voltages = circuit.solve_dynamic(time_end=0.1, time_step=1e-4)

# Plot voltage vs. time for each node
plt.figure(figsize=(10, 6))
for node, voltages in node_voltages.items():
    plt.plot(times, voltages, label=f'Node {node}')

plt.title('Voltage vs Time at Each Node')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.grid(True)
plt.show()

