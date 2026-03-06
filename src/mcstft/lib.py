from __future__ import annotations

import logging

import numpy as np
from swiift.model.frac_handlers import BinaryFracture, BinaryStrainFracture
from swiift.model.model import (DiscreteSpectrum, FloatingIce, Ice, Ocean,
                                WavesUnderElasticPlate, WavesUnderFloe,
                                WavesUnderIce)

logger = logging.getLogger(__name__)
gravity = 9.8
ocean = Ocean(depth=np.inf)


def prep_wui_and_amp(
    amplitude: float,
    period: float,
    poissons_ratio: float,
    thickness: float,
    youngs_modulus: float,
    phase: float,
    frac_toughness: float | None = None,
    strain_threshold: float | None = None,
):
    # Passing None to Ice shouldn't be a problem, but a type checker
    # might complain, so somewhat gracefully fall back on the defaults.
    if frac_toughness is None:
        frac_toughness = Ice().frac_toughness
    if strain_threshold is None:
        strain_threshold = Ice().strain_threshold
    ice = Ice(
        frac_toughness=frac_toughness,
        strain_threshold=strain_threshold,
        thickness=thickness,
        youngs_modulus=youngs_modulus,
    )
    # NOTE: because WUFs are initialised manually, `spectrum.phase` is
    # not passed down to them.
    # The phase must therefore be set manually by passing a complex
    # `edge_amplitudes` when instantiating the WUF object.
    spectrum = DiscreteSpectrum(amplitude, 1 / period)
    wui = WavesUnderIce.without_attenuation(
        WavesUnderElasticPlate.from_ocean(
            ice,
            ocean,
            spectrum,
            gravity,
        )
    )
    c_amplitude = np.atleast_1d(amplitude * np.exp(1j * phase))
    return wui, c_amplitude


def init_length_max(
    wui: WavesUnderIce,
    c_amplitude: np.ndarray,
    fracture_handler: BinaryFracture | BinaryStrainFracture,
    max_its: int | None = None,
):
    wavelength = 2 * np.pi / wui.wavenumbers[0]
    if max_its is None:
        max_its = 6
    for i in range(max_its + 1):
        length = 2**i * wavelength
        wuf = WavesUnderFloe(
            left_edge=0,
            length=length,
            wui=wui,
            edge_amplitudes=c_amplitude,
        )
        xf = fracture_handler.search(wuf, None, True, None)
        if xf is not None:
            return length
    return -1


def min_length_search(
    wui: WavesUnderIce,
    c_amplitude: np.ndarray,
    fracture_handler: BinaryFracture | BinaryStrainFracture,
    length_min: float | None = None,
    length_max: float | None = None,
    length_atol: float | None = None,
    verbose: bool = False,
):
    if length_min is None:
        length_min = wui.ice.thickness
    if length_max is None:
        length_max = init_length_max(wui, c_amplitude, fracture_handler, max_its=2)
        if length_max == -1:
            return np.inf  # Assume it never breaks
    if length_atol is None:
        length_atol = 1e-3  # metre

    if verbose:
        logger.info("Length search serching.")
        logger.info(f"Min length set to {length_min} m")
        logger.info(f"Max length set to {length_max} m")
        logger.info(f"Tolerance set to {length_max} m")
        logger.info("-------------------------------")
        it_count = 0

    xf = None
    # Second condition to ensure that we return a length that does break
    while length_max - length_min > length_atol or xf is None:
        length = (length_min + length_max) / 2
        wuf = WavesUnderFloe(
            left_edge=0,
            length=length,
            wui=wui,
            edge_amplitudes=c_amplitude,
        )
        # Consider the sea fully developed for simplicity
        xf = fracture_handler.search(wuf, None, True, None)
        if xf is None:
            length_min = length
        else:
            length_max = length
        if verbose:
            it_count += 1
            logger.info("Iteration: {it_count: 4d}")
            logger.info(f"Min length is now {length_min} m")
            logger.info(f"Max length is now {length_max} m")
            logger.info(f"Difference is {length_max - length_min}")

    return length


def init_max_strain(wui: WavesUnderIce, c_amplitude: np.ndarray):
    wavenumber = wui.wavenumbers.squeeze()
    scaled_amplitude = np.abs(c_amplitude.squeeze()) / (
        1 + wavenumber**4 * wui.ice.elastic_length_pow4
    )
    return scaled_amplitude * wavenumber**2 * wui.ice.thickness / 2


def strain_threshold_search(
    wui: WavesUnderIce,
    c_amplitude: np.ndarray,
    fracture_handler: BinaryStrainFracture,
    length: float,
    eps_min: float | None = None,
    eps_max: float | None = None,
    atol: float | None = None,
):
    if eps_min is None:
        wuf = WavesUnderFloe(
            left_edge=0,
            length=length,
            wui=wui,
            edge_amplitudes=c_amplitude,
        )
        diag = fracture_handler.diagnose(wuf)
        # This should give a lower bound to the max strain admissible by
        # the plate.
        eps_min = np.abs(diag.strain).max()
    if eps_max is None:
        eps_max = init_max_strain(wui, c_amplitude)
    if atol is None:
        atol = 1e-8  # arbitrary
    if atol >= eps_min:
        atol = eps_min / 1000

    xf = None
    while eps_max - eps_min > atol or xf is None:
        eps_th = (eps_min + eps_max) / 2
        wui = WavesUnderIce(
            FloatingIce(
                wui.ice.density,
                poissons_ratio=wui.ice.poissons_ratio,
                strain_threshold=eps_th,
                thickness=wui.ice.thickness,
                youngs_modulus=wui.ice.youngs_modulus,
                draft=wui.ice.draft,
                dud=wui.ice.dud,
                elastic_length_pow4=wui.ice.elastic_length_pow4,
            ),
            wui.wavenumbers,
            wui.attenuations,
        )
        wuf = WavesUnderFloe(
            left_edge=0,
            length=length,
            wui=wui,
            edge_amplitudes=c_amplitude,
        )
        # Consider the sea fully developed for simplicity
        xf = fracture_handler.search(wuf, None, True, None)
        if xf is None:
            eps_max = eps_th
        else:
            eps_min = eps_th
    return eps_th


def _length_optimiser(
    phase: float,
    fracture_toughness: float,
    parameters: tuple[float, float, float, float, float],
    fracture_handler: BinaryFracture,
    verbose: bool = False,
) -> float:
    wui, c_amp = prep_wui_and_amp(
        *parameters, phase=phase, frac_toughness=fracture_toughness
    )
    return min_length_search(wui, c_amp, fracture_handler, verbose)


def length_optimiser(
    phase: float,
    frac_toughnesses: np.ndarray,
    parameters: tuple[float, float, float, float, float],
    fracture_handler: BinaryFracture,
) -> np.ndarray:
    out = np.full(frac_toughnesses.size, np.nan)
    for j, _ft in enumerate(frac_toughnesses):
        out[j] = _length_optimiser(phase, _ft, parameters, fracture_handler)
    return out


def _strain_optimiser(
    phase,
    length,
    parameters,
    fracture_handler,
):
    # No need to provide a strain threshold; this WUI will only
    # serve as a base to instantiate others in the optimising
    # function.
    wui, c_amp = prep_wui_and_amp(*parameters, phase=phase)
    return strain_threshold_search(wui, c_amp, fracture_handler, length)


def strain_optimiser(
    phase: float,
    lengths: np.ndarray,
    parameters: tuple[float, float, float, float, float],
    fracture_handler: BinaryStrainFracture,
):
    out = np.full(lengths.size, np.nan)
    for j, _lgth in enumerate(lengths):
        if not np.isfinite(_lgth):
            # No energy fracture, give up on looking for a strain fracture.
            out[j] = np.nan
            continue
        out[j] = _strain_optimiser(phase, _lgth, parameters, fracture_handler)
        # No need to provide a strain threshold; this WUI will only
        # serve as a base to instantiate others in the optimising
        # function.
        # wui, c_amp = prep_wui_and_amp(*parameters, phase=phase)
        # out[j] = strain_threshold_search(wui, c_amp, fracture_handler, _lgth)
    return out
