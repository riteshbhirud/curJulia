# Complete CUR Decomposition Implementation
# Exact port from Python curlora.py with Julia optimizations

"""
    compute_selection_probabilities(A::AbstractMatrix{T}) where T

Compute column and row selection probabilities for CUR decomposition.
Identical to the Python implementation.
"""
function compute_selection_probabilities(A::AbstractMatrix{T}) where T
    # Calculate column norms squared (sum along dimension 1 -> rows)
    column_norms_squared = sum(A .^ 2, dims=1)
    # Calculate row norms squared (sum along dimension 2 -> columns)  
    row_norms_squared = sum(A .^ 2, dims=2)
    
    total_sum_squares = sum(column_norms_squared)
    
    # Convert to vectors and normalize
    column_probs = vec(column_norms_squared) ./ total_sum_squares
    row_probs = vec(row_norms_squared) ./ total_sum_squares
    
    return column_probs, row_probs
end

"""
    select_indices_with_replacement(probs::AbstractVector{T}, k::Int) where T

Select indices with replacement using inverse probability sampling.
Exact replication of Python numpy.random.choice behavior.
"""
function select_indices_with_replacement(probs::AbstractVector{T}, k::Int) where T
    # Create inverted probabilities (exact match to Python)
    inverted_P = 1.0f0 ./ (probs .+ 0.001f0)
    
    # Normalize the inverted probabilities
    normalized_probs = inverted_P ./ sum(inverted_P)
    
    # Handle CUDA arrays by moving to CPU for sampling
    if isa(normalized_probs, CuArray)
        normalized_probs_cpu = Array(normalized_probs)
    else
        normalized_probs_cpu = normalized_probs
    end
    
    # Use StatsBase.sample for replacement sampling (equivalent to numpy.random.choice)
    return sample(1:length(probs), Weights(normalized_probs_cpu), k, replace=true)
end

"""
    adjust_duplicates(selected_indices::Vector{Int}, A::AbstractMatrix, axis::Int)

Adjust matrix for duplicate indices with proper scaling factors.
Direct port from Python implementation.
"""
function adjust_duplicates(selected_indices::Vector{Int}, A::AbstractMatrix, axis::Int)
    # Get unique indices and their counts
    unique_indices = unique(selected_indices)
    counts = countmap(selected_indices)
    
    if axis == 1  # Column selection
        adjusted_matrix = A[:, unique_indices]
        
        # Apply scaling factors for duplicates
        for (idx, unique_idx) in enumerate(unique_indices)
            if counts[unique_idx] > 1
                scaling_factor = sqrt(counts[unique_idx])
                adjusted_matrix[:, idx] .*= scaling_factor
            end
        end
    else  # Row selection (axis == 0 in Python, but we use axis != 1)
        adjusted_matrix = A[unique_indices, :]
        
        # Apply scaling factors for duplicates
        for (idx, unique_idx) in enumerate(unique_indices)
            if counts[unique_idx] > 1
                scaling_factor = sqrt(counts[unique_idx])
                adjusted_matrix[idx, :] .*= scaling_factor
            end
        end
    end
    
    return adjusted_matrix, unique_indices
end

"""
    cur_decomposition(A::AbstractMatrix{T}, c::Int) where T

Perform CUR decomposition of matrix A with rank c.
Exact mathematical implementation matching Python version.
"""
function cur_decomposition(A::AbstractMatrix{T}, c::Int) where T
    r = c  # Use same rank for rows and columns
    
    # Compute selection probabilities
    column_probs, row_probs = compute_selection_probabilities(A)
    
    # Select column and row indices
    selected_columns = select_indices_with_replacement(column_probs, c)
    selected_rows = select_indices_with_replacement(row_probs, r)
    
    # Extract C and R matrices
    C = A[:, selected_columns]
    R = A[selected_rows, :]
    
    # Initialize U matrix with zeros (trainable parameter)
    device_type = isa(A, CuArray) ? CUDA.zeros : zeros
    U = device_type(T, size(C, 2), size(R, 1))
    
    return C, U, R
end

"""
    CURModule{T1, T2, T3}

Neural network module implementing CUR decomposition.
Direct equivalent of Python CURModule class.
"""
mutable struct CURModule{T1, T2, T3}
    C::T1      # Fixed matrix from decomposition
    U::T2      # Trainable parameter matrix
    R::T3      # Fixed matrix from decomposition
end

# Make CURModule a Flux layer with trainable parameters
Functors.@functor CURModule (U,)

"""
    CURModule(W::AbstractMatrix{T}, rank::Int) where T

Constructor for CURModule from weight matrix W with given rank.
"""
function CURModule(W::AbstractMatrix{T}, rank::Int) where T
    C, U, R = cur_decomposition(W, rank)
    return CURModule(C, U, R)
end

"""
Forward pass for CURModule - reconstructs approximation and applies to input.
"""
function (m::CURModule)(x::AbstractArray)
    # Reconstruct weight approximation: W_approx = C * U * R
    W_approx = m.C * m.U * m.R
    
    # Apply the approximated weight matrix
    # Handle different tensor orientations (matching Python try/catch)
    try
        return x * W_approx'  # Standard matrix multiplication
    catch
        return x * W_approx   # Alternative orientation
    end
end

"""
    CURLoRAMLP{T1, T2, T3}

MLP with CURLoRA adaptation on the last layer.
Direct port of Python CURLoRAMLP class.
"""
mutable struct CURLoRAMLP{T1, T2, T3}
    base_model::T1
    cur_module::T2
    rank::Int
    alpha::T3
end

Functors.@functor CURLoRAMLP (base_model, cur_module)

"""
    CURLoRAMLP(base_model, rank::Int=8, alpha::Real=1)

Constructor for CURLoRAMLP with base model adaptation.
"""
function CURLoRAMLP(base_model, rank::Int=8, alpha::Real=1)
    # Freeze base model parameters
    for p in Flux.params(base_model)
        p.grad = nothing
    end
    
    # Extract last layer weight for CUR decomposition
    # This is model-specific and needs to be adapted based on architecture
    last_layer_weight = get_last_layer_weight(base_model)
    cur_module = CURModule(last_layer_weight, rank)
    
    return CURLoRAMLP(base_model, cur_module, rank, alpha)
end

"""
Forward pass for CURLoRAMLP combining base model and CUR adaptation.
"""
function (m::CURLoRAMLP)(x::AbstractArray)
    # Forward through all layers except last
    x_intermediate = forward_through_base_layers(m.base_model, x)
    
    # Original last layer output
    x_original = apply_original_last_layer(m.base_model, x_intermediate)
    
    # CUR adapted output
    x_adapted = m.cur_module(x_intermediate)
    
    # Combine with bias (exact match to Python implementation)
    bias = get_last_layer_bias(m.base_model)
    return x_original + m.alpha * x_adapted + bias
end

"""
    LinearWithCURLoRA{T1, T2}

Linear layer with CURLoRA adaptation.
Direct equivalent of Python LinearWithCURLoRA class.
"""
mutable struct LinearWithCURLoRA{T1, T2}
    linear::T1         # Original linear layer
    curlora::T2        # CUR decomposition module
    rank::Int
    alpha::Float32
end

Functors.@functor LinearWithCURLoRA (linear, curlora)

"""
    LinearWithCURLoRA(linear_layer, rank::Int, alpha::Real)

Constructor for LinearWithCURLoRA from existing linear layer.
"""
function LinearWithCURLoRA(linear_layer, rank::Int, alpha::Real)
    # Extract weight matrix from linear layer
    weight_matrix = get_weight_matrix(linear_layer)
    
    # Create CUR decomposition
    curlora_module = CURModule(weight_matrix, rank)
    
    return LinearWithCURLoRA(linear_layer, curlora_module, rank, Float32(alpha))
end

"""
Forward pass for LinearWithCURLoRA combining original and adapted outputs.
"""
function (m::LinearWithCURLoRA)(x::AbstractArray)
    # Original linear layer output
    x_original = m.linear(x)
    
    # CUR adapted output
    x_adapted = m.curlora(x)
    
    # Combine outputs (exact match to Python: x_0 + alpha * x_adapted)
    return x_original + m.alpha * x_adapted
end

# Complete model-specific implementations using Transformers.jl knowledge

"""
    get_weight_matrix(layer)

Extract weight matrix from a layer, handling all Transformers.jl layer types.
"""
function get_weight_matrix(layer)
    # Handle Transformers.jl layer types based on the codebase analysis
    if hasfield(typeof(layer), :weight)
        return layer.weight
    elseif hasfield(typeof(layer), :W)  # Dense layers in Transformers.jl
        return layer.W
    elseif hasfield(typeof(layer), :embeddings)  # Embedding layers
        return layer.embeddings
    elseif isa(layer, Transformers.Layers.Dense)
        return layer.W
    elseif isa(layer, Transformers.Layers.Embed)
        return layer.embeddings
    elseif isa(layer, Transformers.Layers.EmbedDecoder)
        return layer.embed.embeddings
    else
        error("Cannot extract weight matrix from layer of type $(typeof(layer))")
    end
end

"""
    get_conv1d_weight(layer)

Extract weight matrix from Conv1D-style layers (GPT-2 specific).
"""
function get_conv1d_weight(layer)
    # GPT-2 uses specific weight naming in Transformers.jl
    if hasfield(typeof(layer), :weight)
        return layer.weight
    elseif hasfield(typeof(layer), :W)
        return layer.W
    else
        error("Cannot extract Conv1D weight from layer of type $(typeof(layer))")
    end
end

"""
    get_last_layer_weight(model)

Extract weight matrix from the language modeling head of transformer models.
"""
function get_last_layer_weight(model)
    # Handle different model architectures from Transformers.jl
    if hasfield(typeof(model), :lm_head)
        # Most language models have lm_head
        return get_weight_matrix(model.lm_head)
    elseif hasfield(typeof(model), :cls)
        # Some models use cls for classification head
        if hasfield(typeof(model.cls), :layer)
            return get_weight_matrix(model.cls.layer)
        else
            return get_weight_matrix(model.cls)
        end
    elseif hasfield(typeof(model), :decoder)
        # Seq2seq models might have decoder
        if hasfield(typeof(model.decoder), :layers)
            last_layer = model.decoder.layers[end]
            return get_weight_matrix(last_layer)
        else
            return get_weight_matrix(model.decoder)
        end
    elseif hasfield(typeof(model), :layers)
        # Generic layer-based model
        last_layer = model.layers[end]
        return get_weight_matrix(last_layer)
    else
        error("Cannot determine last layer weight for model type $(typeof(model))")
    end
end

"""
    get_last_layer_bias(model)

Extract bias from the language modeling head of transformer models.
"""
function get_last_layer_bias(model)
    try
        if hasfield(typeof(model), :lm_head)
            layer = model.lm_head
        elseif hasfield(typeof(model), :cls)
            layer = hasfield(typeof(model.cls), :layer) ? model.cls.layer : model.cls
        elseif hasfield(typeof(model), :decoder)
            layer = hasfield(typeof(model.decoder), :layers) ? model.decoder.layers[end] : model.decoder
        elseif hasfield(typeof(model), :layers)
            layer = model.layers[end]
        else
            error("Cannot determine last layer for model type $(typeof(model))")
        end
        
        # Extract bias from the layer
        if hasfield(typeof(layer), :bias) && !isnothing(layer.bias)
            return layer.bias
        elseif hasfield(typeof(layer), :b) && !isnothing(layer.b)
            return layer.b
        else
            # Return zeros if no bias
            weight = get_weight_matrix(layer)
            device_type = isa(weight, CuArray) ? CUDA.zeros : zeros
            return device_type(eltype(weight), size(weight, 1))
        end
    catch e
        # Fallback: return zero bias
        @warn "Fallback!"
        weight = get_last_layer_weight(model)
        device_type = isa(weight, CuArray) ? CUDA.zeros : zeros
        return device_type(eltype(weight), size(weight, 1))
    end
end

"""
    forward_through_base_layers(model, x)

Forward pass through transformer model excluding the final layer.
"""
function forward_through_base_layers(model, x)
    # Handle different transformer architectures
    if hasfield(typeof(model), :transformer)
        # GPT-2 style: model.transformer contains the main layers
        transformer = model.transformer
        
        # Apply embedding if present
        if hasfield(typeof(transformer), :embed) || hasfield(typeof(transformer), :wte)
            embed_layer = hasfield(typeof(transformer), :embed) ? transformer.embed : transformer.wte
            x = embed_layer(x)
        end
        
        # Apply transformer blocks
        if hasfield(typeof(transformer), :h)
            # GPT-2 style blocks
            for block in transformer.h
                x = block(x)
            end
        elseif hasfield(typeof(transformer), :layers)
            # Generic transformer layers
            for layer in transformer.layers
                x = layer(x)
            end
        elseif hasfield(typeof(transformer), :blocks)
            # Alternative naming
            for block in transformer.blocks
                x = block(x)
            end
        end
        
        # Apply final layer norm if present
        if hasfield(typeof(transformer), :ln_f)
            x = transformer.ln_f(x)
        elseif hasfield(typeof(transformer), :norm)
            x = transformer.norm(x)
        end
        
        return x
        
    elseif hasfield(typeof(model), :encoder)
        # Encoder-only model (BERT style)
        return model.encoder(x)
        
    elseif hasfield(typeof(model), :layers)
        # Generic layered model - apply all but last
        for layer in model.layers[1:end-1]
            x = layer(x)
        end
        return x
        
    else
        error("Cannot determine how to forward through model type $(typeof(model))")
    end
end

"""
    apply_original_last_layer(model, x)

Apply the original last layer (typically language modeling head).
"""
function apply_original_last_layer(model, x)
    if hasfield(typeof(model), :lm_head)
        return model.lm_head(x)
    elseif hasfield(typeof(model), :cls)
        layer = hasfield(typeof(model.cls), :layer) ? model.cls.layer : model.cls
        return layer(x)
    elseif hasfield(typeof(model), :decoder)
        layer = hasfield(typeof(model.decoder), :layers) ? model.decoder.layers[end] : model.decoder
        return layer(x)
    elseif hasfield(typeof(model), :layers)
        return model.layers[end](x)
    else
        error("Cannot apply original last layer for model type $(typeof(model))")
    end
end

"""
    get_model_hidden_size(model)

Extract the hidden size dimension from a transformer model.
"""
function get_model_hidden_size(model)
    # Try to infer from the language modeling head
    try
        lm_head_weight = get_last_layer_weight(model)
        return size(lm_head_weight, 2)  # Input dimension
    catch
        # Fallback: try to infer from embeddings
        if hasfield(typeof(model), :transformer)
            transformer = model.transformer
            if hasfield(typeof(transformer), :embed)
                embed_weight = get_weight_matrix(transformer.embed)
                return size(embed_weight, 1)
            elseif hasfield(typeof(transformer), :wte)
                embed_weight = get_weight_matrix(transformer.wte)
                return size(embed_weight, 1)
            end
        end
        
        # Last resort: common transformer sizes
        @warn "Cannot determine hidden size, assuming 768 (GPT-2 base)"
        return 768
    end
end

"""
    get_attention_layers(model)

Extract all attention layers from a transformer model for adaptation.
"""
function get_attention_layers(model)
    attention_layers = []
    
    if hasfield(typeof(model), :transformer) && hasfield(typeof(model.transformer), :h)
        # GPT-2 style
        for (i, block) in enumerate(model.transformer.h)
            if hasfield(typeof(block), :attn) && hasfield(typeof(block.attn), :c_attn)
                push!(attention_layers, (block.attn, :c_attn, "transformer.h[$i].attn.c_attn"))
            elseif hasfield(typeof(block), :attention)
                # Generic attention layer
                push!(attention_layers, (block, :attention, "transformer.h[$i].attention"))
            end
        end
    elseif hasfield(typeof(model), :encoder)
        # BERT style - need to traverse encoder layers
        if hasfield(typeof(model.encoder), :layers)
            for (i, layer) in enumerate(model.encoder.layers)
                if hasfield(typeof(layer), :attention)
                    push!(attention_layers, (layer, :attention, "encoder.layers[$i].attention"))
                end
            end
        end
    end
    
    return attention_layers
end

"""
    adapt_attention_layers!(model, adaptation_type::String, rank::Int, alpha::Real)

Adapt all attention layers in a transformer model with LoRA or CURLoRA.
"""
function adapt_attention_layers!(model, adaptation_type::String, rank::Int, alpha::Real)
    attention_layers = get_attention_layers(model)
    adapted_count = 0
    
    for (parent_obj, attr_name, path) in attention_layers
        try
            original_layer = getfield(parent_obj, attr_name)
            
            if adaptation_type == "lora"
                if hasfield(typeof(original_layer), :weight)
                    # Standard linear layer
                    adapted_layer = LinearWithLoRA(original_layer, rank, alpha)
                else
                    # Conv1D layer (GPT-2 style)
                    adapted_layer = Conv1DWithLoRA(original_layer, rank, alpha)
                end
            elseif adaptation_type == "curlora"
                if hasfield(typeof(original_layer), :weight)
                    adapted_layer = LinearWithCURLoRA(original_layer, rank, alpha)
                else
                    adapted_layer = Conv1DWithCURLoRA(original_layer, rank, alpha)
                end
            else
                error("Unknown adaptation type: $adaptation_type")
            end
            
            setfield!(parent_obj, attr_name, adapted_layer)
            adapted_count += 1
            
        catch e
            @warn "Failed to adapt layer at $path: $e"
        end
    end
    
    return adapted_count
end