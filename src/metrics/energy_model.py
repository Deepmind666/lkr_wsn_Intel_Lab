"""
Energy model for WSN simulation.

Provides simple radio and MCU power model with configurable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnergyModelConfig:
    radio_electronics_energy_per_bit: float = 50e-9
    radio_amplifier_energy_per_bit_per_m2: float = 100e-12
    radio_bitrate_bps: int = 250_000
    cpu_active_power_mw: float = 5.0
    lpm_power_mw: float = 0.05

    def radio_tx_energy(self, num_bits: int, distance_m: float) -> float:
        e_elec = self.radio_electronics_energy_per_bit * num_bits
        e_amp = self.radio_amplifier_energy_per_bit_per_m2 * num_bits * (max(0.0, distance_m) ** 2)
        return e_elec + e_amp

    def radio_rx_energy(self, num_bits: int) -> float:
        return self.radio_electronics_energy_per_bit * num_bits

    def cpu_energy(self, active_seconds: float) -> float:
        return (self.cpu_active_power_mw / 1000.0) * max(0.0, active_seconds)

    def lpm_energy(self, lpm_seconds: float) -> float:
        return (self.lpm_power_mw / 1000.0) * max(0.0, lpm_seconds)


