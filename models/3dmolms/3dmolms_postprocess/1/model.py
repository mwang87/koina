import json
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = []

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        
        qtofoutput_config = pb_utils.get_output_config_by_name(model_config, "qtof_out")
        self.qtofoutput_dtype = pb_utils.triton_string_to_numpy(qtofoutput_config["data_type"])
        
        orbioutput_config = pb_utils.get_output_config_by_name(model_config, "orbi_out")
        self.orbioutput_dtype = pb_utils.triton_string_to_numpy(orbioutput_config["data_type"])

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "3dmolms_out")
            norm = raw.as_numpy()
            norm = norm[0]

            # comvert it to mz and intensity pairs
            result = []
            bucket_size = 0.2
            threshold = 0.1
            for i in range(len(norm)):
                if norm[i] > threshold:
                    result.append([i * bucket_size, norm[i]])
            # convert to a -1 x 2 numpy array
            norm = np.array(result, dtype=np.float32)
            qtof_ce_tensor = pb_utils.Tensor("qtof_out", norm.astype(self.qtofoutput_dtype))
            orbi_ce_tensor = pb_utils.Tensor("orbi_out", norm.astype(self.orbioutput_dtype))
            responses.append(pb_utils.InferenceResponse(output_tensors=[qtof_ce_tensor, orbi_ce_tensor]))

        return responses

    def finalize(self):
        pass