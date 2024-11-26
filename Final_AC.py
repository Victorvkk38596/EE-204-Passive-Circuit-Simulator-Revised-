##AC Solver
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import networkx as nx


@dataclass
class Component:
    type: str  # 'R', 'L', 'C', 'V', 'I'
    value: complex
    node1: int
    node2: int
    name: str
    phase: float = 0  # Phase in radians for sources


class CircuitSolver:
    def __init__(self, frequency: float):
        """
        Initialize the circuit solver

        Args:
            frequency (float): Operating frequency in Hz
        """
        self.f = frequency
        self.omega = 2 * np.pi * frequency
        self.components: List[Component] = []
        self.nodes = set()
        self.ground_node = 0
        self.graph = nx.Graph()

    def add_component(self,
                      component_type: str,
                      value: float,
                      node1: int,
                      node2: int,
                      name: str = None,
                      phase: float = 0):
        """
        Add a component to the circuit

        Args:
            component_type (str): 'R', 'L', 'C', 'V', or 'I'
            value (float): Component value in SI units
            node1 (int): First node number
            node2 (int): Second node number
            name (str): Component identifier
            phase (float): Phase angle in radians for sources
        """
        if name is None:
            name = f"{component_type}{len(self.components)}"

        component = Component(component_type, value, node1, node2, name, phase)
        self.components.append(component)
        self.nodes.add(node1)
        self.nodes.add(node2)

        # Add to graph for topology analysis
        self.graph.add_edge(node1, node2, component=component)

    def get_impedance(self, component: Component) -> complex:
        """Calculate complex impedance for a component"""
        if component.type == 'R':
            return component.value
        elif component.type == 'L':
            return 1j * self.omega * component.value
        elif component.type == 'C':
            return -1j / (self.omega * component.value)
        elif component.type in ['V', 'I']:
            return 0
        else:
            raise ValueError(f"Unknown component type: {component.type}")

    def check_topology(self) -> bool:
        """
        Verify circuit topology is valid
        Returns:
            bool: True if topology is valid
        """
        if not nx.is_connected(self.graph):
            raise ValueError("Circuit is not fully connected")
        return True

    def solve(self) -> Tuple[Dict[int, complex], Dict[str, complex]]:
        """
        Solve the circuit using modified nodal analysis

        Returns:
            Tuple[Dict[int, complex], Dict[str, complex]]:
                Node voltages and branch currents in phasor form
        """
        self.check_topology()

        # Get non-ground nodes and voltage sources
        non_ground_nodes = sorted(self.nodes - {self.ground_node})
        voltage_sources = [c for c in self.components if c.type == 'V']

        # Matrix size is number of non-ground nodes plus number of voltage sources
        n = len(non_ground_nodes) + len(voltage_sources)

        # Create node index mapping
        node_indices = {node: idx for idx, node in enumerate(non_ground_nodes)}

        # Initialize MNA matrices
        A = np.zeros((n, n), dtype=complex)
        b = np.zeros(n, dtype=complex)

        # Fill in admittance matrix and current vector
        for comp in self.components:
            n1, n2 = comp.node1, comp.node2

            if comp.type == 'V':
                # Handle voltage sources separately
                continue

            # Get admittance
            Y = 1 / self.get_impedance(comp) if comp.type not in ['V', 'I'] else 0

            # Add admittance matrix entries
            if n1 != self.ground_node and n2 != self.ground_node:
                i, j = node_indices[n1], node_indices[n2]
                A[i, i] += Y
                A[j, j] += Y
                A[i, j] -= Y
                A[j, i] -= Y
            elif n1 != self.ground_node:
                i = node_indices[n1]
                A[i, i] += Y
            elif n2 != self.ground_node:
                j = node_indices[n2]
                A[j, j] += Y

            # Add current sources to b vector
            if comp.type == 'I':
                if n1 != self.ground_node:
                    b[node_indices[n1]] -= comp.value * np.exp(1j * comp.phase)
                if n2 != self.ground_node:
                    b[node_indices[n2]] += comp.value * np.exp(1j * comp.phase)

        # Add voltage source equations
        for idx, v_source in enumerate(voltage_sources):
            v_idx = len(non_ground_nodes) + idx

            if v_source.node1 != self.ground_node:
                i = node_indices[v_source.node1]
                A[v_idx, i] = 1
                A[i, v_idx] = 1

            if v_source.node2 != self.ground_node:
                i = node_indices[v_source.node2]
                A[v_idx, i] = -1
                A[i, v_idx] = -1

            b[v_idx] = v_source.value * np.exp(1j * v_source.phase)

        # Solve the system
        x = solve(A, b)

        # Extract results
        voltages = {self.ground_node: 0.0}  # Ground node voltage
        currents = {}

        # Extract node voltages
        for node, idx in node_indices.items():
            voltages[node] = x[idx]

        # Calculate branch currents
        for comp in self.components:
            if comp.type != 'V':
                v1 = voltages[comp.node1]
                v2 = voltages[comp.node2]
                Z = self.get_impedance(comp)
                currents[comp.name] = (v1 - v2) / Z if Z != 0 else 0
            else:
                # For voltage sources, get current from MNA solution
                v_idx = len(non_ground_nodes) + voltage_sources.index(comp)
                currents[comp.name] = x[v_idx]

        return voltages, currents

    def plot_results(self, t_span: Optional[Tuple[float, float]] = None, n_points: int = 1000):
        """
        Plot node voltages and branch currents over time

        Args:
            t_span (tuple): Time span (start, end) in seconds
            n_points (int): Number of points to plot
        """
        if t_span is None:
            t_span = (0, 4 / self.f)  # Plot 4 periods by default

        t = np.linspace(t_span[0], t_span[1], n_points)
        voltages, currents = self.solve()

        # Plot voltages
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        for node, v_phasor in voltages.items():
            v_t = np.abs(v_phasor) * np.cos(self.omega * t + np.angle(v_phasor))
            plt.plot(t * 1000, -v_t, label=f'Node {node}')
        print()
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (V)')
        plt.title('Node Voltages vs Time')
        plt.grid(True)
        plt.legend()

        # Plot currents
        plt.subplot(2, 1, 2)
        for comp_name, i_phasor in currents.items():
            i_t = np.abs(i_phasor) * np.cos(self.omega * t + np.angle(i_phasor))
            plt.plot(t * 1000, -i_t, label=f'Component {comp_name}')
        plt.xlabel('Time (ms)')
        plt.ylabel('Current (A)')
        plt.title('Branch Currents vs Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


def example_complex_rlc_circuit():
    """Example of a complex RLC circuit with multiple branches"""
    solver = CircuitSolver(frequency=10/(2*np.pi))  # 1 kHz
    '''
    # Voltage source
    solver.add_component('V', 10, 0, 1, 'Vs', 0)

    # Left branch
    solver.add_component('R', 100, 1, 2, 'R1')
    solver.add_component('L', 0.1, 2, 3, 'L1')

    # Right branch
    solver.add_component('R', 200, 1, 4, 'R2')
    solver.add_component('C', 1e-6, 4, 3, 'C1')

    # Bottom branch
    solver.add_component('R', 150, 3, 0, 'R3')
    '''

    '''
    solver.add_component('V', 20, 0, 1, 'Vs', 165*np.pi/180) ##(-15 degree phase)
    solver.add_component('R', 60, 1, 2, 'R1')
    solver.add_component('C', 1e-2, 2, 0, 'C1')
    solver.add_component('L', 5, 2, 0, 'L1')
    '''


    solver.add_component('V', 50, 0, 1, 'Vs', 30*np.pi/180) ##(30 degree phase)
    solver.add_component('R', 10, 1, 2, 'R1')
    solver.add_component('L', 0.5, 1, 2, 'L1')
    solver.add_component('C', 0.05, 2, 0, 'C1')


    '''
    solver.add_component('V', 45, 1, 0, 'Vs', 30*np.pi/180)  ##(0 degree phase)
    solver.add_component('L', 4, 1, 2, 'L1')
    solver.add_component('C', 0.3333, 1, 3, 'C1')
    solver.add_component('R', 8, 2, 4, 'R1')
    solver.add_component('L', 5, 4, 3, 'L2')
    solver.add_component('R', 5, 2, 5, 'R2')
    solver.add_component('C', 0.5, 5, 0, 'C2')
    solver.add_component('R', 10, 3, 0, 'R3')
    '''

    '''
    solver.add_component('V', 50, 1, 0, 'Vs', 0)  ##(0 degree phase)
    solver.add_component('R', 12, 1, 2, 'R1')
    solver.add_component('L', 4, 2, 3, 'L1')
    solver.add_component('C', 0.3333, 3, 0, 'C1')
    solver.add_component('R', 8, 3, 4, 'R2')
    solver.add_component('R', 2, 2, 5, 'R3')
    solver.add_component('C', 0.25, 5, 4, 'C2')
    solver.add_component('L', 6, 4, 6, 'L2')
    solver.add_component('R', 8, 6, 0, 'R4')
    '''
    # Solve and plot
    solver.plot_results()

    return solver


if __name__ == "__main__":
    solver = example_complex_rlc_circuit()
