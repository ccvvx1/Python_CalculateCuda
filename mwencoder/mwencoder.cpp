#include <torch/extension.h>
#include <vector>

// Function to perform frequency encoding
std::vector<int64_t> frequency_encode(const std::vector<int64_t>& input) {
    std::unordered_map<int64_t, int64_t> freq_map;
    std::vector<int64_t> output(input.size());

    // Count frequencies
    for (const auto& item : input) {
        freq_map[item]++;
    }

    // Assign frequencies to output
    for (size_t i = 0; i < input.size(); i++) {
        output[i] = freq_map[input[i]];
    }

    return output;
}

// PyTorch binding for the frequency encoding function
torch::Tensor frequency_encode_py(torch::Tensor input) {
    // Ensure the input is a 1D tensor of integers
    TORCH_CHECK(input.dim() == 1, "Input must be a 1D tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "Input must be of type int64");

    std::vector<int64_t> input_vec(input.data_ptr<int64_t>(), input.data_ptr<int64_t>() + input.size(0));
    std::vector<int64_t> output_vec = frequency_encode(input_vec);

    return torch::tensor(output_vec, torch::dtype(torch::kInt64).requires_grad(false));
}

// Bindings to expose the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("frequency_encode", &frequency_encode_py, "Frequency encoding");
}
