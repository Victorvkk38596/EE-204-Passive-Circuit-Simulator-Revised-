import numpy as np
import matplotlib.pyplot as plt


class CircuitElement:
    def __init__(self, node1, node2, value):
        self.node1 = node1
        self.node2 = node2
        self.value = value


class Circuit:
    def __init__(self):
        self.resistors = []
        self.capacitors = []
        self.inductors = []
        self.voltage_sources = []
        self.num_nodes = 0
        self.dt = 1e-6  # Default timestep
        self.max_time = 0.01  # Default simulation time

    def add_resistor(self, node1, node2, resistance):
        self.resistors.append(CircuitElement(node1, node2, resistance))
        self.num_nodes = max(self.num_nodes, node1, node2)

    def add_capacitor(self, node1, node2, capacitance):
        self.capacitors.append(CircuitElement(node1, node2, capacitance))
        self.num_nodes = max(self.num_nodes, node1, node2)

    def add_inductor(self, node1, node2, inductance):
        self.inductors.append(CircuitElement(node1, node2, inductance))
        self.num_nodes = max(self.num_nodes, node1, node2)

    def add_voltage_source(self, node1, node2, voltage):
        self.voltage_sources.append(CircuitElement(node1, node2, voltage))
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
                # Add historical current contribution
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
            # Add extra row/column for inductor current
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

    def solve(self):
        t = np.arange(0, self.max_time, self.dt)
        n = self.num_nodes
        m = len(self.voltage_sources) + len(self.inductors)

        # Initialize solution arrays
        voltages = np.zeros((len(t), n))
        currents = np.zeros((len(t), m))

        # Initial condition (DC solution)
        Y, I = self.build_system_matrices(0)
        solution = np.linalg.solve(Y, I)
        voltages[0, :] = solution[:n]
        currents[0, :] = solution[n:]

        # Time stepping
        for i in range(1, len(t)):
            Y, I = self.build_system_matrices(
                t[i],
                voltages[i - 1, :],
                currents[i - 1, len(self.inductors):]
            )
            solution = np.linalg.solve(Y, I)
            voltages[i, :] = solution[:n]
            currents[i, :] = solution[n:]

        for i in range(1, len(t)):
            Y, I = self.build_system_matrices(
                t[i],
                voltages[i - 1, :],
                currents[i - 1, :]
            )
            solution = np.linalg.solve(Y, I)
            voltages[i, :] = solution[:n]
            currents[i, :] = solution[n:]

            # Debugging outputs
            print(f"Step {i}, Time {t[i]}:")
            print(f"Y matrix:\n{Y}")
            print(f"I vector:\n{I}")
            print(f"Voltages:\n{voltages[i, :]}")
            print(f"Currents:\n{currents[i, :]}")

        return t, voltages, currents

    def plot_results(self, t, voltages, currents):
        plt.figure(figsize=(12, 6))

        # Plot node voltages
        plt.subplot(2, 1, 1)
        for i in range(voltages.shape[1]):
            plt.plot(t, voltages[:, i], label=f'Node {i + 1}')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title('Node Voltages')
        plt.legend()

        # Plot currents
        plt.subplot(2, 1, 2)
        for i in range(currents.shape[1]):
            plt.plot(t, currents[:, i], label=f'Branch {i + 1}')
        plt.grid(True)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title('Branch Currents')
        plt.legend()

        plt.tight_layout()
        plt.show()


# Example usage
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
    circuit.add_voltage_source(1, 0, 40)
    circuit.add_resistor(1, 2, 30)
    circuit.add_inductor(2, 3, 4)
    circuit.add_resistor(3, 0, 50)
    circuit.add_capacitor(3, 0, 2)
    # Set simulation parameters
    circuit.dt = 1e-2  # Timestep: 1 µs
    circuit.max_time = 30  # Simulate for 10 ms

    # Solve and plot
    t, voltages, currents = circuit.solve()
    circuit.plot_results(t, voltages, currents)


if __name__ == "__main__":
    simulate_rlc_circuit()
