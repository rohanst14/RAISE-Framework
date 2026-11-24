# Reproducibility guide

Required external artifacts:

- `CSTR_sindy_data.npz`: training archive.
- `global_norm_params.npy`, `coeff_norm_params.npy`: normalization parameters. Training writes these under `CSTR data profiles/`; inference and risk analysis expect them in the working directory.
- `gru_sindy_model.pth`: training writes this in the working directory; inference and validation expect it under `save model/`.
- `save model/gru_good_sindy1.pth`: separate checkpoint referenced by risk analysis. Do not assume checkpoints are interchangeable.
- `dnn_CA_model.h5`, `dnn_T_model.h5`, `scaler_X1.joblib`, `scaler_Y_CA1.joblib`, `scaler_Y_T1.joblib`: adaptive validation artifacts.
- `distribution_parameters.json`, `copula_fitted_parameters.json`: fitted parameters used in risk analysis; the latter is opened in binary mode.

Create required output directories before execution, including `CSTR data profiles/` and `save model/`. Review the original path conventions, record package versions, random seeds, data provenance and checkpoint identities. Only load trusted serialized objects. Original script contents are preserved; execution has not been independently validated.
