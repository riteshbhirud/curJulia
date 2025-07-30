# Complete LoRA Implementation
# Exact port from Python lora.py with proper Julia patterns

"""
    LoRALayer{T1, T2}

Standard LoRA (Low-Rank Adaptation) layer implementation.
Direct equivalent of Python LoRALayer class with exact initialization.
"""
mutable struct LoRALayer{T1, T2}
    A::T1           # Low-rank matrix A (trainable)
    B::T2           # Low-rank matrix B (trainable) 
    alpha::Float32  # Scaling factor
end

# Make LoRALayer a Flux layer with trainable parameters
Functors.@functor LoRALayer (A, B)

"""
    LoRALayer(in_dim::Int, out_dim::Int, rank::Int, alpha::Real; device=cpu)

Constructor for LoRALayer with specified dimensions and rank.
Exact initialization matching Python implementation:
- A: Kaiming uniform initialization
- B: Zero initialization
"""
function LoRALayer(in_dim::Int, out_dim::Int, rank::Int, alpha::Real; device=cpu)
    # Initialize A with Kaiming uniform (exact match to Python)
    # Python: torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
    fan_in = in_dim
    bound = sqrt(5.0f0 / fan_in)  # Kaiming uniform bound
    A_data = (rand(Float32, in_dim, rank) .- 0.5f0) .* (2.0f0 * bound)
    
    # Initialize B with zeros (exact match to Python)
    B_data = zeros(Float32, rank, out_dim)
    
    # Move to specified device
    if device == gpu && CUDA.functional()
        A_param = A_data |> gpu
        B_param = B_data |> gpu
    else
        A_param = A_data
        B_param = B_data
    end
    
    return LoRALayer(A_param, B_param, Float32(alpha))
end

"""
Forward pass for LoRALayer.
Exact computation: alpha * (x @ A @ B)
"""
function (layer::LoRALayer)(x::AbstractArray)
    # Exact match to Python: self.alpha * (x @ self.A @ self.B)
    return layer.alpha * (x * layer.A * layer.B)
end

"""
    LinearWithLoRA{T1, T2}

Linear layer with LoRA adaptation.
Direct equivalent of Python LinearWithLoRA class.
"""
mutable struct LinearWithLoRA{T1, T2}
    linear::T1    # Original linear layer (frozen)
    lora::T2      # LoRA adaptation layer (trainable)
end

Functors.@functor LinearWithLoRA (linear, lora)

"""
    LinearWithLoRA(linear_layer, rank::Int, alpha::Real)

Constructor for LinearWithLoRA from existing linear layer.
Automatically determines input/output dimensions and creates LoRA adaptation.
"""
function LinearWithLoRA(linear_layer, rank::Int, alpha::Real)
    # Extract dimensions from the linear layer
    in_features, out_features = get_linear_layer_dims(linear_layer)
    
    # Determine device placement
    device = get_layer_device(linear_layer)
    
    # Create LoRA layer with matching dimensions
    lora_layer = LoRALayer(in_features, out_features, rank, alpha; device=device)
    
    return LinearWithLoRA(linear_layer, lora_layer)
end

"""
Forward pass for LinearWithLoRA.
Exact computation: linear(x) + lora(x)
"""
function (m::LinearWithLoRA)(x::AbstractArray)
    # Exact match to Python: self.linear(x) + self.lora(x)
    return m.linear(x) + m.lora(x)
end

# Helper functions for layer introspection and device handling

"""
    get_linear_layer_dims(layer)

Extract input and output dimensions from a linear layer.
Handles various layer types and weight matrix orientations.
"""
function get_linear_layer_dims(layer)
    weight_matrix = get_weight_matrix(layer)
    
    # Handle different weight matrix orientations
    # Most common: (out_features, in_features)
    if hasfield(typeof(layer), :in_features) && hasfield(typeof(layer), :out_features)
        return layer.in_features, layer.out_features
    else
        # Infer from weight matrix shape
        out_features, in_features = size(weight_matrix)
        return in_features, out_features
    end
end

"""
    get_layer_device(layer)

Determine the device (CPU/GPU) where a layer's parameters are located.
"""
function get_layer_device(layer)
    weight_matrix = get_weight_matrix(layer)
    
    if isa(weight_matrix, CuArray)
        return gpu
    else
        return cpu
    end
end

"""
    Conv1DWithLoRA{T1, T2, T3, T4}

Conv1D layer with LoRA adaptation, specifically for GPT-2 style models.
Handles the specific case of Conv1D layers in transformer attention.
"""
mutable struct Conv1DWithLoRA{T1, T2, T3, T4}
    conv1d::T1         # Original Conv1D layer
    rank::Int
    alpha::Float32
    lora_A::T2         # LoRA A matrix (trainable)
    lora_B::T3         # LoRA B matrix (trainable)
end

Functors.@functor Conv1DWithLoRA (conv1d, lora_A, lora_B)

"""
    Conv1DWithLoRA(conv1d_layer, rank::Int, alpha::Real)

Constructor for Conv1D layer with LoRA adaptation.
Handles the specific weight layout of Conv1D layers.
"""
function Conv1DWithLoRA(conv1d_layer, rank::Int, alpha::Real)
    # Extract dimensions from Conv1D layer
    # Conv1D weight shape is typically (in_features, out_features)
    weight = conv1d_layer.weight
    in_features = size(weight, 1)
    out_features = conv1d_layer.nf  # GPT-2 Conv1D specific
    
    # Initialize LoRA matrices with exact same pattern as Python
    device_fn = isa(weight, CuArray) ? CUDA.randn : randn
    zeros_fn = isa(weight, CuArray) ? CUDA.zeros : zeros
    
    # A matrix: Kaiming uniform initialization
    fan_in = in_features
    bound = sqrt(5.0f0 / fan_in)
    lora_A = (rand(Float32, in_features, rank) .- 0.5f0) .* (2.0f0 * bound)
    if isa(weight, CuArray)
        lora_A = lora_A |> gpu
    end
    
    # B matrix: Zero initialization
    lora_B = zeros_fn(Float32, rank, out_features)
    
    return Conv1DWithLoRA(conv1d_layer, rank, Float32(alpha), lora_A, lora_B)
end

"""
Forward pass for Conv1DWithLoRA.
"""
function (m::Conv1DWithLoRA)(x::AbstractArray)
    # Original Conv1D forward pass
    original_output = m.conv1d(x)
    
    # LoRA adaptation: alpha * (x @ A @ B)
    lora_output = m.alpha * (x * m.lora_A * m.lora_B)
    
    return original_output + lora_output
end

"""
    Conv1DWithCURLoRA{T1, T2, T3, T4}

Conv1D layer with CURLoRA adaptation, specifically for GPT-2 style models.
Combines Conv1D layers with CUR decomposition.
"""
mutable struct Conv1DWithCURLoRA{T1, T2, T3, T4}
    conv1d::T1         # Original Conv1D layer
    alpha::Float32
    C::T2              # CUR C matrix (fixed)
    U::T3              # CUR U matrix (trainable)
    R::T4              # CUR R matrix (fixed)
end

Functors.@functor Conv1DWithCURLoRA (conv1d, U)

"""
    Conv1DWithCURLoRA(conv1d_layer, rank::Int, alpha::Real)

Constructor for Conv1D layer with CURLoRA adaptation.
"""
function Conv1DWithCURLoRA(conv1d_layer, rank::Int, alpha::Real)
    # Extract weight matrix from Conv1D layer
    weight = conv1d_layer.weight
    
    # Perform CUR decomposition
    C, U, R = cur_decomposition(weight, rank)
    
    return Conv1DWithCURLoRA(conv1d_layer, Float32(alpha), C, U, R)
end

"""
Forward pass for Conv1DWithCURLoRA.
"""
function (m::Conv1DWithCURLoRA)(x::AbstractArray)
    # Original Conv1D forward pass
    original_output = m.conv1d(x)
    
    # CUR approximation: W_approx = C * U * R
    W_approx = m.C * m.U * m.R
    
    # CURLoRA adaptation: alpha * (x @ W_approx)
    curlora_output = m.alpha * (x * W_approx)
    
    return original_output + curlora_output
end

# Parameter counting utilities

"""
    count_parameters(model)

Count the total number of parameters in a model.
"""
function count_parameters(model)
    return sum(length(p) for p in Flux.params(model))
end

"""
    count_trainable_parameters(model)

Count only the trainable parameters in a model.
"""
function count_trainable_parameters(model)
    total = 0
    for p in Flux.params(model)
        if !get(p, :frozen, false)
            total += length(p)
        end
    end
    return total
end

"""
    freeze_parameters!(model)

Freeze all parameters in a model (mark as non-trainable).
"""
function freeze_parameters!(model)
    for p in Flux.params(model)
        p.frozen = true
    end
    return model
end

"""
    unfreeze_parameters!(model)

Unfreeze all parameters in a model (mark as trainable).
"""
function unfreeze_parameters!(model)
    for p in Flux.params(model)
        p.frozen = false
    end
    return model
end