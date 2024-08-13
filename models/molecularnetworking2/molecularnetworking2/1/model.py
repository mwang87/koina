import json
import triton_python_backend_utils as pb_utils
import os
import subprocess
import numpy as np

class TritonPythonModel:
    def __init__(self):
        super().__init__()
        self.output_dtype = []

    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        output0_config = pb_utils.get_output_config_by_name(model_config, "prediction")
        self.output_dtype = pb_utils.triton_string_to_numpy(output0_config["data_type"])

        # check if the python environment exists
        env_name = "predictModifications"
        env_path = get_mamba_env_path(env_name)
        if env_path is None:
            #  create the environment if it does not exist
            result = subprocess.run(["conda", "env", "create", "--file", "environment.yml", "-n", env_name],
                                    capture_output=True, text=True, check=True)
            env_path = get_mamba_env_path(env_name)
            if env_path is None:
                raise ValueError(f"Environment '{env_name}' not found.")
        
        self.env_path = env_path


    def execute(self, requests):
        responses = []

        for request in requests:
            # Extract input tensors by their names
            mz_array1 = pb_utils.get_input_tensor_by_name(request, "mz_array1").as_numpy()
            intensity_array1 = pb_utils.get_input_tensor_by_name(request, "intensity_array1").as_numpy()
            mz_array2 = pb_utils.get_input_tensor_by_name(request, "mz_array2").as_numpy()
            intensity_array2 = pb_utils.get_input_tensor_by_name(request, "intensity_array2").as_numpy()
            precursor_mass_diff = pb_utils.get_input_tensor_by_name(request, "precursor_mass_diff").as_numpy()
            precursor_mz1 = pb_utils.get_input_tensor_by_name(request, "precursor_mz1").as_numpy()
            precursor_mz2 = pb_utils.get_input_tensor_by_name(request, "precursor_mz2").as_numpy()

            # Perform the computation
            mz_array1 = pad_sequence(mz_array1)
            intensity_array1 = pad_sequence(intensity_array1)
            mz_array2 = pad_sequence(mz_array2)
            intensity_array2 = pad_sequence(intensity_array2)
            
            result = subprocess.run([self.env_path + "/bin/python", "handler.py", 
                                     str(mz_array1), str(intensity_array1), str(mz_array2),
                                     str(intensity_array2), str(precursor_mass_diff),
                                     str(precursor_mz1), str(precursor_mz2)], capture_output=True,
                                     text=True, check=True)
            
            # Convert the output to the appropriate data type
            output0 = np.array(result.stdout.split(), dtype=self.output_dtype)
            output0_pb = pb_utils.Tensor(output0)

            # Create the response
            response = pb_utils.InferenceResponse(output0_pb)
            responses.append(response)

        return responses

    def finalize(self):
        """Clean up resources."""
        pass

def pad_sequence(sequence, max_length=100):
            if len(sequence) < max_length:
                padding_length = max_length - len(sequence)
                sequence = np.pad(sequence, (0, padding_length), 'constant')
            return sequence[:max_length]


def get_mamba_env_path(env_name):
    try:
        # Use 'mamba' or 'conda' depending on your installation
        result = subprocess.run(["conda", "info", "--envs"],
                                capture_output=True, text=True, check=True)

        # Parse the output to find the environment path
        for line in result.stdout.splitlines():
            if line.startswith(env_name):
                # The environment name is followed by its path
                return line.split()[-1]

        return None  # Return None if the environment is not found
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return None