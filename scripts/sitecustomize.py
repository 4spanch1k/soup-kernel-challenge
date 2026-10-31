"""Enable the opt-in T4 memory recorder only in harness solution training."""

from t4_mempeak import install_if_training_process

install_if_training_process()
