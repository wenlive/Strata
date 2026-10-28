# Clover adapter references

The Clover operations and state semantics follow DiffuSpace's official HF
implementation, revision `4b0402b65e523d0183a14c431955438714cb16ff`, and SGLang
PR 46927, revision `058939af24d5cdc44bb4b8852d0ddfab555ed71e`. Those references
are Apache-2.0 licensed. Clover's HF implementation also credits Qwen and
Hugging Face Transformers. The model input directory retains the official
LICENSE and ATTRIBUTIONS files. The Apache-2.0 license is included here.

The new CUDA formulas, storage adapter and CLI are integrated into Strata.
Existing `DirectFile` and `prefill::Gemm` components retain Strata's licensing.
The JSON dependency comes from Strata's pinned llama.cpp vendor tree:
`nlohmann/json` 3.12.0, MIT, copyright Niels Lohmann and contributors.

The standalone Python tests load reference functions from explicit source
paths for numerical comparison. They do not bundle or run a separate HF model
as the native inference path.
