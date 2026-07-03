import builtins
import importlib
import sys

import numpy as np
import pytest


INSTALL_HINT = 'pip install "hornlab-sim[metal]"'
MODULES_TO_REIMPORT = (
    "hornlab_sim.methods.bandpass",
    "hornlab_sim.methods.helmholtz",
    "hornlab_sim.methods.transfer_matrix",
    "hornlab_sim.methods.port_acoustics",
    "hornlab_sim.methods.driver_coupling",
    "hornlab_sim.methods.lem_to_bem",
    "hornlab_sim.methods.radiation_impedance",
)


@pytest.fixture
def block_metal_import(monkeypatch):
    for name in list(sys.modules):
        if (
            name == "hornlab_metal_bem"
            or name.startswith("hornlab_metal_bem.")
            or name in MODULES_TO_REIMPORT
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hornlab_metal_bem" or name.startswith("hornlab_metal_bem."):
            raise ModuleNotFoundError(
                "No module named 'hornlab_metal_bem'",
                name="hornlab_metal_bem",
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_pure_python_modules_import_without_metal(block_metal_import):
    for module_name in MODULES_TO_REIMPORT[:5]:
        importlib.import_module(module_name)


def test_bem_adapters_raise_guided_import_error_without_metal(block_metal_import):
    lem_to_bem = importlib.import_module("hornlab_sim.methods.lem_to_bem")
    radiation_impedance = importlib.import_module(
        "hornlab_sim.methods.radiation_impedance"
    )

    with pytest.raises(ImportError) as lem_exc:
        lem_to_bem.solve(
            mesh=object(),
            lem_velocities={"slot": np.array([1.0 + 0.0j])},
            aperture_tags={"slot": [1]},
            frequencies_hz=np.array([100.0]),
        )
    assert INSTALL_HINT in str(lem_exc.value)

    with pytest.raises(ImportError) as matrix_exc:
        radiation_impedance.solve_aperture_matrix(
            mesh=object(),
            aperture_tags={"slot": [1]},
            frequencies_hz=np.array([100.0]),
        )
    assert INSTALL_HINT in str(matrix_exc.value)
