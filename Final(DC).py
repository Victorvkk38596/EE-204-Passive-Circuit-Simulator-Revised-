import numpy as np
import matplotlib.pyplot as plt


class CircuitElement:
    def __init__(self, node1, node2, value, name):
        self.node1 = node1
        self.node2 = node2
        self.value = value
        self.name = name


class Circuit:
    def __init__(self):
        self.resistors = []
        self.capacitors = []
        self.inductors = []
        self.voltage_sources = []
        self.num_nodes = 0
        self.dt = 1e-6
        self.max_time = 0.01

    def add_resistor(self, node1, node2, resistance, name=None):
        if name is None:
            name = f"R{len(self.resistors) + 1}"
        self.resistors.append(CircuitElement(node1, node2, resistance, name))
        self.num_nodes = max(self.num_nodes, node1, node2)

    def add_capacitor(self, node1, node2, capacitance, name=None):
        if name is None:
            name = f"C{len(self.capacitors) + 1}"
        self.capacitors.append(CircuitElement(node1, node2, capacitance, name))
        self.num_nodes = max(self.num_nodes, node1, node2)

    def add_inductor(self, node1, node2, inductance, name=None):
        if name is None:
            name = f"L{len(self.inductors) + 1}"
        self.inductors.append(CircuitElement(node1, node2, inductance, name))
        self.num_nodes = max(self.num_nodes, node1, node2)

    def add_voltage_source(self, node1, node2, voltage, name=None):
        if name is None:
            name = f"V{len(self.voltage_sources) + 1}"
        self.voltage_sources.append(CircuitElement(node2, node1, voltage, name))
        self.num_nodes = max(self.num_nodes, node1, node2)

    def build_system_matrices(self, t, prev_voltages=None, prev_currents=None):
        n = self.num_nodes
        m = len(self.voltage_sources) + len(self.inductors)
        Y = np.zeros((n + m, n + m))
        I = np.zeros(n + m)

        # Add resistor contributions
        for r in self.resistors:
            if r.node1 > 0:
                Y[r.node1 - 1, r.node1 - 1] += 1 / r.value
                if r.node2 > 0:
                    Y[r.node1 - 1, r.node2 - 1] -= 1 / r.value
            if r.node2 > 0:
                Y[r.node2 - 1, r.node2 - 1] += 1 / r.value
                if r.node1 > 0:
                    Y[r.node2 - 1, r.node1 - 1] -= 1 / r.value

        # Add capacitor contributions using trapezoidal integration
        for i, c in enumerate(self.capacitors):
            conductance = 2 * c.value / self.dt
            if c.node1 > 0:
                Y[c.node1 - 1, c.node1 - 1] += conductance
                if c.node2 > 0:
                    Y[c.node1 - 1, c.node2 - 1] -= conductance
            if c.node2 > 0:
                Y[c.node2 - 1, c.node2 - 1] += conductance
                if c.node1 > 0:
                    Y[c.node2 - 1, c.node1 - 1] -= conductance

            if prev_voltages is not None:
                i_hist = 2 * c.value / self.dt * (
                        (prev_voltages[c.node1 - 1] if c.node1 > 0 else 0) -
                        (prev_voltages[c.node2 - 1] if c.node2 > 0 else 0)
                )
                if c.node1 > 0:
                    I[c.node1 - 1] += i_hist
                if c.node2 > 0:
                    I[c.node2 - 1] -= i_hist

        # Add inductor contributions
        for i, l in enumerate(self.inductors):
            curr_idx = n + i

            if l.node1 > 0:
                Y[curr_idx, l.node1 - 1] = 1
                Y[l.node1 - 1, curr_idx] = 1
            if l.node2 > 0:
                Y[curr_idx, l.node2 - 1] = -1
                Y[l.node2 - 1, curr_idx] = -1

            Y[curr_idx, curr_idx] = -l.value / self.dt

            if prev_currents is not None:
                I[curr_idx] = -l.value / self.dt * prev_currents[i]

        # Add voltage source contributions
        offset = n + len(self.inductors)
        for i, v in enumerate(self.voltage_sources):
            curr_idx = offset + i

            if v.node1 > 0:
                Y[curr_idx, v.node1 - 1] = 1
                Y[v.node1 - 1, curr_idx] = 1
            if v.node2 > 0:
                Y[curr_idx, v.node2 - 1] = -1
                Y[v.node2 - 1, curr_idx] = -1

            I[curr_idx] = v.value

        return Y, I

    def calculate_component_currents(self, voltages, currents, t_idx):
        component_currents = {}

        # Calculate currents through resistors
        for r in self.resistors:
            v1 = voltages[t_idx, r.node1 - 1] if r.node1 > 0 else 0
            v2 = voltages[t_idx, r.node2 - 1] if r.node2 > 0 else 0
            current = (v1 - v2) / r.value
            component_currents[r.name] = current

        # Calculate currents through capacitors
        for c in self.capacitors:
            v1 = voltages[t_idx, c.node1 - 1] if c.node1 > 0 else 0
            v2 = voltages[t_idx, c.node2 - 1] if c.node2 > 0 else 0
            if t_idx > 0:
                v1_prev = voltages[t_idx - 1, c.node1 - 1] if c.node1 > 0 else 0
                v2_prev = voltages[t_idx - 1, c.node2 - 1] if c.node2 > 0 else 0
                current = c.value * ((v1 - v2) - (v1_prev - v2_prev)) / self.dt
            else:
                current = c.value * (v1 - v2) / self.dt
            component_currents[c.name] = current

        # Get currents through inductors (already calculated in solve)
        for i, l in enumerate(self.inductors):
            component_currents[l.name] = currents[t_idx, i]

        # Get currents through voltage sources
        offset = len(self.inductors)
        for i, v in enumerate(self.voltage_sources):
            component_currents[v.name] = -currents[t_idx, offset + i]

        return component_currents

    def solve(self):
        t = np.arange(0, self.max_time, self.dt)
        n = self.num_nodes
        m = len(self.voltage_sources) + len(self.inductors)

        voltages = np.zeros((len(t), n))
        currents = np.zeros((len(t), m))
        component_currents = {elem.name: np.zeros(len(t)) for elem in
                              self.resistors + self.capacitors + self.inductors + self.voltage_sources}

        # Initial condition (DC solution)
        Y, I = self.build_system_matrices(0)
        solution = np.linalg.solve(Y, I)
        voltages[0, :] = solution[:n]
        currents[0, :] = solution[n:]

        # Calculate initial component currents
        initial_currents = self.calculate_component_currents(voltages, currents, 0)
        for name, current in initial_currents.items():
            component_currents[name][0] = current

        for i in range(1, len(t)):
            Y, I = self.build_system_matrices(t[i], voltages[i - 1, :], currents[i - 1, :])
            solution = np.linalg.solve(Y, I)
            voltages[i, :] = solution[:n]
            currents[i, :] = solution[n:]

            # Calculate and store component currents
            step_currents = self.calculate_component_currents(voltages, currents, i)
            for name, current in step_currents.items():
                component_currents[name][i] = current

        return t, voltages, component_currents

    def plot_results(self, t, voltages, component_currents):
        plt.figure(figsize=(12, 8))

        # Plot node voltages
        plt.subplot(2, 1, 1)
        for i in range(voltages.shape[1]):
            plt.plot(t, voltages[:, i], label=f'Node {i + 1}')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Node Voltages')
        plt.legend()

        # Plot component currents
        plt.subplot(2, 1, 2)
        for name, currents in component_currents.items():
            plt.plot(t, currents, label=name)
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Component Currents')
        plt.legend()

        plt.tight_layout()
        plt.show()


def simulate_rlc_circuit():
    # Create a series RLC circuit with DC source
    circuit = Circuit()

    # Circuit parameters
    V = 12.0  # Voltage source (V)
    R1 = 1  # Resistance (Ω)
    R2 = 5
    R3 = 4
    L = 2  # Inductance (H)
    C = 1  # Capacitance (F)

    # Add circuit elements
    # Node 0 is ground
    '''
    circuit.add_voltage_source(1, 0, V)
    circuit.add_resistor(1, 2, R)
    circuit.add_capacitor(2, 0, C)
    '''
    '''
    circuit.add_voltage_source(1, 0, V)
    circuit.add_resistor(1, 2, R1)
    circuit.add_resistor(2, 3, R2)
    circuit.add_resistor(2, 4, R3)
    circuit.add_capacitor(4, 0, C)
    circuit.add_inductor(3, 0, L)
    '''

    '''
    circuit.add_voltage_source(1, 0, 40)
    circuit.add_resistor(1, 2, 30)
    circuit.add_inductor(2, 3, 4)
    circuit.add_resistor(3, 0, 50)
    circuit.add_capacitor(3, 0, 2)
    '''


    circuit.add_voltage_source(0, 1, 20)
    circuit.add_resistor(1, 2, 1)
    circuit.add_resistor(2, 3, 1)
    circuit.add_capacitor(2, 0, 0.5)
    circuit.add_capacitor(3, 0, 0.3333)

    '''
    circuit.add_voltage_source(0, 1, 20)
    circuit.add_inductor(1, 2, 1)
    circuit.add_resistor(2, 0, 1)
    '''

    circuit.dt = 1e-3
    circuit.max_time = 10

    t, voltages, component_currents = circuit.solve()
    circuit.plot_results(t, voltages, component_currents)


if __name__ == "__main__":
    simulate_rlc_circuit()