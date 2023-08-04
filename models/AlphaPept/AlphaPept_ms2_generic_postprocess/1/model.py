import re
import json
import numpy as np
import triton_python_backend_utils as pb_utils
from pyteomics import proforma


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = None
        self.logger = pb_utils.Logger

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "intensities")
        anno_config = pb_utils.get_output_config_by_name(model_config, "annotation")

        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])
        self.anno_dtype = pb_utils.triton_string_to_numpy(anno_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            peptide_in = (
                pb_utils.get_input_tensor_by_name(request, "peptides_in:0")
                .as_numpy()
                .tolist()
            )

            peptide_in = [x[0].decode("utf-8") for x in peptide_in]
            peaks_in = pb_utils.get_input_tensor_by_name(
                request, "peaks_in:0"
            ).as_numpy()

            fragmentmz = self.get_fragments(peptide_in, peaks_in.shape[1])
            peaks_norm = self.normalize_intensity(peaks_in)
            anno_arr = self.gen_annotation(peaks_in.shape[0], peaks_in.shape[1])

            output_tensors = [
                pb_utils.Tensor("intensities", peaks_norm.astype(self.output_dtype)),
                pb_utils.Tensor("mz", fragmentmz.astype(self.output_dtype)),
                pb_utils.Tensor("annotation", anno_arr.astype(self.anno_dtype)),
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=output_tensors))

        return responses

    def gen_annotation(self, nseq, max_fragment_number):
        anno_arr = np.full([nseq, 2, 2, max_fragment_number], "", dtype="U5")
        ionnumber = np.array(range(1, max_fragment_number + 1), dtype="U2")

        anno_arr[:, 0, :, :] = np.char.add(anno_arr[:, 0, :, :], "y")
        anno_arr[:, 1, :, :] = np.char.add(anno_arr[:, 1, :, :], "b")

        anno_arr = np.char.add(anno_arr, ionnumber)

        anno_arr[:, :, 0, :] = np.char.add(anno_arr[:, :, 0, :], "+1")
        anno_arr[:, :, 1, :] = np.char.add(anno_arr[:, :, 1, :], "+2")
        return anno_arr.reshape(nseq, -1)

    def get_fragments(self, sequences, max_fragment_number):
        tensor_inputs = [
            pb_utils.Tensor("ProForma", np.array(sequences, dtype=np.object_)),
            pb_utils.Tensor("charges", np.array([1, 2], dtype=np.int32)),
            pb_utils.Tensor("ion_series", np.array(["y", "b"], dtype=np.object_)),
        ]

        infer_request = pb_utils.InferenceRequest(
            model_name="fragment_mz",
            requested_output_names=["output_fragmentmz"],
            inputs=tensor_inputs,
        )

        resp = infer_request.exec()

        if resp.has_error():
            raise pb_utils.TritonModelException(resp.error().message())
        else:
            tmp = pb_utils.get_output_tensor_by_name(
                resp, "output_fragmentmz"
            ).as_numpy()

            return tmp[:, :, :, :max_fragment_number].reshape(len(sequences), -1)

    def normalize_intensity(self, peaks):
        apex_intens = peaks.reshape((peaks.shape[0], -1)).max(axis=1)
        apex_intens[apex_intens <= 0] = 1
        peaks /= apex_intens.reshape((-1, 1, 1))
        peaks[peaks < 0] = 0.0
        peaks = peaks[:, :, :4]  # remove neutral losses
        return peaks.reshape((peaks.shape[0], -1))