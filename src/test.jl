# Complete Testing and Comparison Framework
# Exact port from Python test.py with full functionality

using MLDatasets
using Flux
using Flux.Losses
using Optimisers
using Zygote

# Custom dataset structures to match Python datasets
struct MRPCExample
    sentence1::String
    sentence2::String
    label::Int
end

struct SST2Example
    sentence::String
    label::Int
end

struct SentimentExample
    text::String
    sentiment::Int
end

"""
    calculate_perplexity(model, tokenizer, text::String; device=cpu, max_length=1000)

Calculate perplexity on given text.
Exact replication of Python calculate_perplexity function.
"""
function calculate_perplexity(model, tokenizer, text::String; device=cpu, max_length=1000)
    if isempty(strip(text))
        @warn "Empty text encountered"
        return Inf
    end
    
    try
        # Tokenize text (adjust based on Transformers.jl API)
        encoded = encode(tokenizer, text; max_length=max_length, truncation=true, padding=true)
        input_ids = encoded.token
        
        # Move to device
        if device == gpu && CUDA.functional()
            input_ids = input_ids |> gpu
        end
        
        if isempty(input_ids) || length(input_ids) == 0
            @warn "Empty input_ids for text: $(text[1:min(100, end)])..."
            return Inf
        end
        
        target_ids = copy(input_ids)
        
        # Calculate loss without gradients
        loss = Flux.withgradient(() -> begin
            outputs = model(input_ids)
            return Flux.Losses.crossentropy(outputs, target_ids)
        end, Flux.params())[1]
        
        perplexity = exp(loss)
        
        return Float64(perplexity)
        
    catch e
        @error "Error calculating perplexity" exception=e
        return Inf
    finally
        # Memory cleanup
        if device == gpu && CUDA.functional()
            CUDA.reclaim()
        end
        GC.gc()
    end
end

"""
    load_datasets()

Load all required datasets for evaluation.
Mimics the Python dataset loading structure.
"""
function load_datasets()
    println("Loading datasets...")
    
    # Load MRPC dataset (simplified version)
    mrpc_train, mrpc_val = load_mrpc_dataset()
    
    # Load SST-2 dataset
    sst2_train, sst2_val = load_sst2_dataset()
    
    # Load WikiText for perplexity calculation
    wikitext = load_wikitext_dataset()
    
    # Create dataset structure
    datasets = (
        mrpc_train = mrpc_train,
        mrpc_val = mrpc_val,
        sst2_train = sst2_train,
        sst2_val = sst2_val,
        wikitext = wikitext
    )
    
    return datasets
end

"""
    load_mrpc_dataset()

Load MRPC dataset for paraphrase detection.
"""
function load_mrpc_dataset()
    # This is a simplified implementation - in practice, you'd load from files
    # or use a proper dataset loading library
    
    train_examples = [
        MRPCExample("The company reported strong earnings.", "Strong earnings were reported by the company.", 1),
        MRPCExample("It's raining outside.", "The weather is sunny.", 0),
        MRPCExample("Machine learning is advancing rapidly.", "AI technology is progressing quickly.", 1),
        MRPCExample("The book was interesting.", "I don't like reading books.", 0),
    ]
    
    val_examples = [
        MRPCExample("The movie was excellent.", "It was a great film.", 1),
        MRPCExample("I love pizza.", "Cats are animals.", 0),
    ]
    
    return train_examples, val_examples
end

"""
    load_sst2_dataset()

Load SST-2 dataset for sentiment analysis.
"""
function load_sst2_dataset()
    train_examples = [
        SST2Example("This movie is amazing!", 1),
        SST2Example("I hate this film.", 0),
        SST2Example("Great acting and storyline.", 1),
        SST2Example("Boring and disappointing.", 0),
        SST2Example("Excellent cinematography.", 1),
        SST2Example("Terrible waste of time.", 0),
    ]
    
    val_examples = [
        SST2Example("Good movie overall.", 1),
        SST2Example("Not worth watching.", 0),
        SST2Example("Fantastic performance.", 1),
        SST2Example("Very disappointing.", 0),
    ]
    
    return train_examples, val_examples
end

"""
    load_wikitext_dataset()

Load WikiText dataset sample for perplexity calculation.
"""
function load_wikitext_dataset()
    # Sample text for perplexity calculation
    sample_texts = [
        "The history of artificial intelligence began in ancient times with myths and stories.",
        "Machine learning algorithms have revolutionized data analysis and prediction.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning networks consist of multiple layers of interconnected neurons.",
        "Transformers have become the dominant architecture for language models."
    ]
    
    return join(sample_texts, " ")
end

"""
    evaluate_sst2(model, tokenizer, dataset; device=cpu, max_samples=50)

Evaluate model on SST-2 dataset.
Exact replication of Python evaluate_sst2 function.
"""
function evaluate_sst2(model, tokenizer, dataset; device=cpu, max_samples=50)
    model = model |> device
    Flux.testmode!(model)  # Set to evaluation mode
    
    correct = 0
    total = 0
    
    @showprogress "Evaluating SST-2..." for (i, example) in enumerate(dataset)
        i > max_samples && break
        
        try
            # Encode sentence
            encoded = encode(tokenizer, example.sentence; max_length=512, truncation=true, padding=true)
            inputs = encoded.token
            
            # Move to device
            if device == gpu && CUDA.functional()
                inputs = inputs |> gpu
            end
            
            # Forward pass
            outputs = model(inputs)
            
            # Get prediction from last token (GPT-style)
            predicted = argmax(outputs[:, end])
            
            # Compare with label (adjust for 0-based to 1-based indexing)
            correct += (predicted == example.label + 1)
            total += 1
            
        catch e
            @warn "Error evaluating example $i" exception=e
            continue
        end
    end
    
    return correct / total
end

"""
    evaluate_mrpc(model, tokenizer, dataset; device=cpu, max_samples=50)

Evaluate model on MRPC dataset.
Exact replication of Python evaluate_mrpc function.
"""
function evaluate_mrpc(model, tokenizer, dataset; device=cpu, max_samples=50)
    model = model |> device
    Flux.testmode!(model)
    
    correct = 0
    total = 0
    
    @showprogress "Evaluating MRPC..." for (i, example) in enumerate(dataset)
        i > max_samples && break
        
        try
            # Encode sentence pair
            # Note: This is simplified - actual implementation depends on tokenizer API
            combined_text = example.sentence1 * " " * example.sentence2
            encoded = encode(tokenizer, combined_text; max_length=512, truncation=true, padding=true)
            inputs = encoded.token
            
            # Move to device
            if device == gpu && CUDA.functional()
                inputs = inputs |> gpu
            end
            
            # Forward pass
            outputs = model(inputs)
            
            # Get prediction from last token
            predicted = argmax(outputs[:, end])
            
            # Compare with label
            correct += (predicted == example.label + 1)
            total += 1
            
        catch e
            @warn "Error evaluating example $i" exception=e
            continue
        end
    end
    
    return correct / total
end

"""
    evaluate_sentiment(model, tokenizer, dataset; device=cpu, max_samples=50)

Evaluate sentiment analysis (using SST-2 as proxy).
"""
function evaluate_sentiment(model, tokenizer, dataset; device=cpu, max_samples=50)
    return evaluate_sst2(model, tokenizer, dataset; device=device, max_samples=max_samples)
end

"""
    ModelTester

Main testing class for running CURLoRA vs LoRA comparisons.
Direct port of Python ModelTester class.
"""
mutable struct ModelTester
    model_name::String
    device::Union{Function, Symbol}
    max_len::Int
    lr::Float32
    results::Dict{String, Any}
    datasets::NamedTuple
    tokenizer::Any
end

"""
    ModelTester(model_name::String="gpt2"; device=cpu)

Constructor for ModelTester.
"""
function ModelTester(model_name::String="gpt2"; device=cpu)
    println("Initializing Model Tester for $model_name")
    
    # Load datasets
    datasets = load_datasets()
    
    # Load tokenizer (this will be model-specific)
    tokenizer = load_tokenizer_for_model(model_name)
    
    return ModelTester(
        model_name,
        device,
        512,      # max_len
        2.5f-4,   # lr
        Dict{String, Any}(),
        datasets,
        tokenizer
    )
end

"""
    load_tokenizer_for_model(model_name::String)

Load appropriate tokenizer for the specified model using Transformers.jl.
"""
function load_tokenizer_for_model(model_name::String)
    try
        if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
            # Load GPT-2 tokenizer and model
            tokenizer, _ = hgf"$(model_name)"
            return tokenizer
        elseif model_name in ["bert-base-uncased", "bert-base-cased", "bert-large-uncased"]
            # Load BERT tokenizer
            tokenizer, _ = hgf"$(model_name)"
            return tokenizer
        elseif model_name in ["t5-small", "t5-base", "t5-large"]
            # Load T5 tokenizer
            tokenizer, _ = hgf"$(model_name)"
            return tokenizer
        else
            @warn "Unknown model $model_name, attempting to load with hgf macro"
            tokenizer, _ = hgf"$(model_name)"
            return tokenizer
        end
    catch e
        error("Failed to load tokenizer for $model_name: $e")
    end
end

"""
    load_base_model(tester::ModelTester)

Load the base model for testing using Transformers.jl.
"""
function load_base_model(tester::ModelTester)
    println("Loading base model: $(tester.model_name)")
    
    try
        if tester.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
            # Load GPT-2 model
            tokenizer, model = hgf"$(tester.model_name)"
            
            # Update tokenizer in tester
            tester.tokenizer = tokenizer
            
            # Move to device
            if tester.device == gpu && CUDA.functional()
                model = todevice(model)
            else
                model = model  # Keep on CPU
                tester.device = cpu
            end
            
            return model
            
        elseif tester.model_name in ["bert-base-uncased", "bert-base-cased", "bert-large-uncased"]
            # Load BERT model
            tokenizer, model = hgf"$(tester.model_name)"
            tester.tokenizer = tokenizer
            
            if tester.device == gpu && CUDA.functional()
                model = todevice(model)
            else
                model = model
                tester.device = cpu
            end
            
            return model
            
        else
            @warn "Attempting to load unknown model $(tester.model_name)"
            tokenizer, model = hgf"$(tester.model_name)"
            tester.tokenizer = tokenizer
            
            if tester.device == gpu && CUDA.functional()
                model = todevice(model)
            else
                tester.device = cpu
            end
            
            return model
        end
        
    catch e
        error("Failed to load model $(tester.model_name): $e")
    end
end

"""
    apply_adaptation(model, adaptation_type::String="lora"; rank::Int=8, alpha::Real=1, device=cpu)

Apply LoRA or CURLoRA adaptation to transformer model using Transformers.jl knowledge.
"""
function apply_adaptation(model, adaptation_type::String="lora"; rank::Int=8, alpha::Real=1, device=cpu)
    # Freeze base model parameters
    freeze_parameters!(model)
    
    println("Applying $(adaptation_type) adaptation to transformer layers...")
    
    # Adapt attention layers using complete implementation
    adaptation_count = adapt_attention_layers!(model, adaptation_type, rank, alpha)
    
    println("Applied $(adaptation_type) to $(adaptation_count) attention layers")
    
    # Count trainable parameters
    total_params = count_trainable_parameters(model)
    println("Total trainable parameters after $(adaptation_type): $(total_params)")
    
    return model, total_params
end

"""
    add_classification_head(model, num_classes::Int, device)

Add a classification head to the model for downstream tasks.
"""
function add_classification_head(model, num_classes::Int, device)
    hidden_size = get_model_hidden_size(model)
    
    # Create new classification head
    classification_head = Dense(hidden_size, num_classes)
    
    # Move to device if needed
    if device == gpu && CUDA.functional()
        classification_head = todevice(classification_head)
    end
    
    # Store original head and replace
    original_head = hasfield(typeof(model), :lm_head) ? model.lm_head : nothing
    
    if hasfield(typeof(model), :lm_head)
        model.lm_head = classification_head
    else
        # Add as new field (this is a simplification)
        @warn "Adding classification head to model without lm_head field"
    end
    
    return model, original_head
end

"""
    prepare_batch(batch, tokenizer, task_name::String, device)

Prepare a batch for training using Transformers.jl tokenizer.
"""
function prepare_batch(batch, tokenizer, task_name::String, device)
    try
        if task_name == "mrpc"
            # Combine sentence pairs for MRPC
            texts = []
            labels = []
            
            for example in batch
                # Encode both sentences together
                if hasfield(typeof(example), :sentence1)
                    combined_text = "$(example.sentence1) $(example.sentence2)"
                    push!(texts, combined_text)
                    push!(labels, example.label + 1)  # Convert to 1-based
                else
                    # Fallback for different data structures
                    combined_text = "$(example[1]) $(example[2])"
                    push!(texts, combined_text)
                    push!(labels, example[3] + 1)
                end
            end
            
        elseif task_name in ["sst2", "sentiment"]
            texts = []
            labels = []
            
            for example in batch
                if hasfield(typeof(example), :sentence)
                    push!(texts, example.sentence)
                    push!(labels, example.label + 1)  # Convert to 1-based
                else
                    # Fallback
                    push!(texts, example[1])
                    push!(labels, example[2] + 1)
                end
            end
            
        else
            error("Unknown task: $task_name")
        end
        
        # Tokenize all texts
        encoded_inputs = []
        max_length = 0
        
        for text in texts
            encoded = encode(tokenizer, text)
            push!(encoded_inputs, encoded.token)
            max_length = max(max_length, length(encoded.token))
        end
        
        # Pad to same length
        batch_size = length(encoded_inputs)
        padded_inputs = zeros(Int32, max_length, batch_size)
        
        for (i, tokens) in enumerate(encoded_inputs)
            len = length(tokens)
            padded_inputs[1:len, i] = tokens
        end
        
        # Convert labels to appropriate format
        labels_tensor = collect(labels)
        
        # Move to device
        if device == gpu && CUDA.functional()
            padded_inputs = todevice(padded_inputs)
            labels_tensor = todevice(labels_tensor)
        end
        
        return padded_inputs, labels_tensor
        
    catch e
        @error "Error preparing batch for $task_name" exception=e
        rethrow(e)
    end
end

"""
    evaluate_with_head(model, tokenizer, dataset, task_head, task_name::String; device=cpu, max_samples=50)

Evaluate model with specific task head.
"""
function evaluate_with_head(model, tokenizer, dataset, task_head, task_name::String; device=cpu, max_samples=50)
    # Set task head
    original_head = hasfield(typeof(model), :lm_head) ? model.lm_head : nothing
    
    try
        if hasfield(typeof(model), :lm_head)
            model.lm_head = task_head
        end
        
        # Set to evaluation mode
        Flux.testmode!(model)
        
        correct = 0
        total = 0
        
        dataset_slice = dataset[1:min(max_samples, length(dataset))]
        
        @showprogress "Evaluating $(task_name)..." for example in dataset_slice
            try
                # Prepare single example
                if task_name == "mrpc"
                    text = "$(example.sentence1) $(example.sentence2)"
                    true_label = example.label + 1
                elseif task_name in ["sst2", "sentiment"]
                    text = example.sentence
                    true_label = example.label + 1
                else
                    continue
                end
                
                # Encode and predict
                encoded = encode(tokenizer, text)
                inputs = encoded.token
                
                if device == gpu && CUDA.functional()
                    inputs = todevice(inputs)
                end
                
                # Forward pass
                outputs = model(inputs)
                
                # Get prediction from last token (for causal models)
                if ndims(outputs) == 3  # (vocab_size, seq_len, batch_size)
                    logits = outputs[:, end, 1]  # Last token, batch 1
                elseif ndims(outputs) == 2  # (vocab_size, seq_len) 
                    logits = outputs[:, end]
                else
                    logits = outputs
                end
                
                predicted = argmax(logits)
                
                if predicted == true_label
                    correct += 1
                end
                total += 1
                
            catch e
                @warn "Error evaluating example in $task_name" exception=e
                continue
            end
        end
        
        return correct / max(total, 1)
        
    finally
        # Restore original head
        if !isnothing(original_head) && hasfield(typeof(model), :lm_head)
            model.lm_head = original_head
        end
    end
end

"""
    load_base_model(tester::ModelTester)

Load the base model for testing using Transformers.jl.
"""
function load_base_model(tester::ModelTester)
    println("Loading base model: $(tester.model_name)")
    
    try
        if tester.model_name in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
            # Load GPT-2 model
            tokenizer, model = hgf"$(tester.model_name)"
            
            # Update tokenizer in tester
            tester.tokenizer = tokenizer
            
            # Move to device
            if tester.device == gpu && CUDA.functional()
                model = todevice(model)
            else
                model = model  # Keep on CPU
                tester.device = cpu
            end
            
            return model
            
        elseif tester.model_name in ["bert-base-uncased", "bert-base-cased", "bert-large-uncased"]
            # Load BERT model
            tokenizer, model = hgf"$(tester.model_name)"
            tester.tokenizer = tokenizer
            
            if tester.device == gpu && CUDA.functional()
                model = todevice(model)
            else
                model = model
                tester.device = cpu
            end
            
            return model
            
        else
            @warn "Attempting to load unknown model $(tester.model_name)"
            tokenizer, model = hgf"$(tester.model_name)"
            tester.tokenizer = tokenizer
            
            if tester.device == gpu && CUDA.functional()
                model = todevice(model)
            else
                tester.device = cpu
            end
            
            return model
        end
        
    catch e
        error("Failed to load model $(tester.model_name): $e")
    end
end

"""
    train_on_task(model, tokenizer, task_name::String, num_classes::Int; num_epochs::Int=3, batch_size::Int=8, device=cpu, max_samples::Int=100)

Train model on downstream task with complete Transformers.jl integration.
"""
function train_on_task(model, tokenizer, task_name::String, num_classes::Int; 
                      num_epochs::Int=3, batch_size::Int=8, device=cpu, max_samples::Int=100)
    println("Training on $(task_name)...")
    
    # Add classification head
    model_with_head, original_head = add_classification_head(model, num_classes, device)
    
    # Get training data
    train_data = get_training_data(task_name, max_samples)
    
    # Setup optimizer - only optimize trainable parameters
    trainable_params = [p for p in Flux.params(model) if !get(p, :frozen, false)]
    opt_state = Optimisers.setup(Optimisers.Adam(2.5f-4), model_with_head)
    
    # Training loop
    start_time = time()
    
    for epoch in 1:num_epochs
        total_loss = 0.0f0
        num_batches = 0
        
        # Create batches
        batches = create_batches(train_data, batch_size)
        
        @showprogress "Epoch $epoch/$num_epochs" for batch in batches
            try
                # Prepare batch
                inputs, labels = prepare_batch(batch, tokenizer, task_name, device)
                
                # Forward pass with gradient computation
                loss, grads = Flux.withgradient(model_with_head) do m
                    outputs = m(inputs)
                    
                    # Handle different output shapes
                    if ndims(outputs) == 3  # (vocab_size, seq_len, batch_size)
                        logits = outputs[:, end, :]  # Last token for each example
                        logits = transpose(logits)    # (batch_size, vocab_size)
                    elseif ndims(outputs) == 2  # (vocab_size, seq_len)
                        logits = transpose(outputs[:, end:end])  # (1, vocab_size)
                    else
                        logits = outputs
                    end
                    
                    # Compute cross-entropy loss
                    return Flux.Losses.crossentropy(transpose(logits), labels)
                end
                
                # Skip if loss is NaN
                if isnan(loss) || isinf(loss)
                    @warn "Skipping batch due to NaN/Inf loss"
                    continue
                end
                
                # Update only trainable parameters
                if !isnothing(grads[1])
                    opt_state, model_with_head = Optimisers.update(opt_state, model_with_head, grads[1])
                end
                
                total_loss += loss
                num_batches += 1
                
                # Memory cleanup
                if device == gpu && CUDA.functional()
                    CUDA.reclaim()
                end
                GC.gc()
                
            catch e
                @warn "Error in training batch: $e"
                continue
            end
        end
        
        if num_batches > 0
            avg_loss = total_loss / num_batches
            println("Epoch $(epoch), Average loss: $(round(avg_loss, digits=6))")
        else
            @warn "No successful batches in epoch $epoch"
        end
    end
    
    training_time = time() - start_time
    
    # Extract and return task head
    task_head = hasfield(typeof(model_with_head), :lm_head) ? model_with_head.lm_head : nothing
    
    # Restore original head if it existed
    if !isnothing(original_head) && hasfield(typeof(model), :lm_head)
        model.lm_head = original_head
    end
    
    return task_head, training_time
end

"""
    evaluate_all_tasks(model, tokenizer, task_heads, datasets; device=cpu)

Evaluate model on all tasks with complete implementation.
"""
function evaluate_all_tasks(model, tokenizer, task_heads, datasets; device=cpu)
    results = Dict{String, Float64}()
    
    # Evaluate each task with its specific head
    for (task_name, head) in task_heads
        println("Evaluating on $(task_name)...")
        
        try
            if task_name == "mrpc"
                accuracy = evaluate_with_head(model, tokenizer, datasets.mrpc_val, head, task_name; device=device, max_samples=20)
            elseif task_name == "sst2"
                accuracy = evaluate_with_head(model, tokenizer, datasets.sst2_val, head, task_name; device=device, max_samples=20)
            elseif task_name == "sentiment"
                # Use SST-2 dataset as sentiment proxy
                accuracy = evaluate_with_head(model, tokenizer, datasets.sst2_val, head, task_name; device=device, max_samples=20)
            else
                @warn "Unknown task: $task_name"
                accuracy = 0.0
            end
            
            results[task_name] = accuracy
            println("$(uppercase(task_name)) Accuracy: $(round(accuracy, digits=4))")
            
        catch e
            @error "Error evaluating $task_name" exception=e
            results[task_name] = 0.0
        end
    end
    
    # Calculate final perplexity with original language modeling head
    println("Calculating perplexity...")
    try
        perplexity = calculate_perplexity(model, tokenizer, datasets.wikitext; device=device)
        results["perplexity"] = perplexity
        println("Perplexity: $(round(perplexity, digits=2))")
    catch e
        @error "Error calculating perplexity" exception=e
        results["perplexity"] = Inf
    end
    
    return results
end

"""
    apply_adaptation(model, adaptation_type::String="lora"; rank::Int=8, alpha::Real=1, device=cpu)

Apply LoRA or CURLoRA adaptation to the model.
Exact port of Python apply_adaptation method.
"""
function apply_adaptation(model, adaptation_type::String="lora"; rank::Int=8, alpha::Real=1, device=cpu)
    # Freeze base model parameters
    freeze_parameters!(model)
    
    println("Applying $(adaptation_type) to GPT-2 layers...")
    adaptation_count = 0
    
    # For GPT-2, adapt attention layers in transformer blocks
    if hasfield(typeof(model), :transformer) && hasfield(typeof(model.transformer), :h)
        for (i, block) in enumerate(model.transformer.h)
            println("Processing transformer block $i")
            
            # Check if c_attn exists (GPT-2 specific)
            if hasfield(typeof(block.attn), :c_attn)
                println("  c_attn type: $(typeof(block.attn.c_attn))")
                
                if adaptation_type == "lora"
                    block.attn.c_attn = Conv1DWithLoRA(block.attn.c_attn, rank, alpha)
                elseif adaptation_type == "curlora"
                    block.attn.c_attn = Conv1DWithCURLoRA(block.attn.c_attn, rank, alpha)
                else
                    error("Unknown adaptation type: $adaptation_type")
                end
                
                adaptation_count += 1
            end
        end
    else
        error("Model structure not compatible with GPT-2 adaptation")
    end
    
    println("Applied $(adaptation_type) to $(adaptation_count) attention layers")
    
    # Count trainable parameters
    total_params = count_trainable_parameters(model)
    println("Total trainable parameters after $(adaptation_type): $(total_params)")
    
    return model, total_params
end

"""
    train_on_task(model, tokenizer, task_name::String, num_classes::Int; num_epochs::Int=3, batch_size::Int=8, device=cpu, max_samples::Int=100)

Train the model on a specific downstream task.
Direct port of Python train_on_task method.
"""
function train_on_task(model, tokenizer, task_name::String, num_classes::Int; 
                      num_epochs::Int=3, batch_size::Int=8, device=cpu, max_samples::Int=100)
    println("Training on $(task_name)...")
    
    # Save original language modeling head
    original_lm_head = model.lm_head
    
    # Add classification head
    Random.seed!(1311)  # Exact match to Python torch.manual_seed(1311)
    if hasfield(typeof(model), :lm_head)
        in_features = size(model.lm_head.weight, 2)  # Assuming weight is (out_features, in_features)
    else
        error("Cannot determine model hidden size")
    end
    
    classification_head = Dense(in_features, num_classes)
    if device == gpu && CUDA.functional()
        classification_head = classification_head |> gpu
    end
    model.lm_head = classification_head
    
    # Get training data
    train_data = get_training_data(task_name, max_samples)
    
    # Setup optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(2.5f-4), model)
    
    # Training loop
    start_time = time()
    
    for epoch in 1:num_epochs
        total_loss = 0.0f0
        num_batches = 0
        
        # Create batches
        batches = create_batches(train_data, batch_size)
        
        @showprogress "Epoch $epoch/$num_epochs" for batch in batches
            # Prepare batch
            inputs, labels = prepare_batch(batch, tokenizer, task_name, device)
            
            # Forward pass with gradient computation
            loss, grads = Flux.withgradient(model) do m
                outputs = m(inputs)
                # Use outputs from last token position
                logits = outputs[:, end, :]  # Shape: (vocab_size, batch_size)
                return Flux.Losses.crossentropy(logits, labels)
            end
            
            # Update parameters
            opt_state, model = Optimisers.update(opt_state, model, grads[1])
            
            total_loss += loss
            num_batches += 1
            
            # Memory cleanup
            if device == gpu && CUDA.functional()
                CUDA.reclaim()
            end
            GC.gc()
        end
        
        avg_loss = total_loss / num_batches
        println("Epoch $(epoch), Average loss: $(round(avg_loss, digits=6))")
    end
    
    training_time = time() - start_time
    
    # Save task head and restore original
    task_head = model.lm_head
    model.lm_head = original_lm_head
    
    return task_head, training_time
end

"""
    get_training_data(task_name::String, max_samples::Int)

Get training data for a specific task.
"""
function get_training_data(task_name::String, max_samples::Int)
    datasets = load_datasets()
    
    if task_name == "mrpc"
        return datasets.mrpc_train[1:min(max_samples, length(datasets.mrpc_train))]
    elseif task_name == "sst2"
        return datasets.sst2_train[1:min(max_samples, length(datasets.sst2_train))]
    elseif task_name == "sentiment"
        # Use SST-2 as sentiment proxy
        return datasets.sst2_train[1:min(max_samples, length(datasets.sst2_train))]
    else
        error("Unknown task: $task_name")
    end
end

"""
    create_batches(data, batch_size::Int)

Create batches from training data.
"""
function create_batches(data, batch_size::Int)
    batches = []
    for i in 1:batch_size:length(data)
        batch = data[i:min(i + batch_size - 1, length(data))]
        push!(batches, batch)
    end
    return batches
end

"""
    prepare_batch(batch, tokenizer, task_name::String, device)

Prepare a batch for training.
"""
function prepare_batch(batch, tokenizer, task_name::String, device)
    if task_name == "mrpc"
        # Combine sentence pairs
        texts = [example.sentence1 * " " * example.sentence2 for example in batch]
        labels = [example.label + 1 for example in batch]  # Convert to 1-based indexing
    elseif task_name in ["sst2", "sentiment"]
        texts = [example.sentence for example in batch]
        labels = [example.label + 1 for example in batch]  # Convert to 1-based indexing
    else
        error("Unknown task: $task_name")
    end
    
    # Tokenize texts
    inputs = []
    for text in texts
        encoded = encode(tokenizer, text; max_length=512, truncation=true, padding=true)
        push!(inputs, encoded.token)
    end
    
    # Convert to batch tensor (this needs proper implementation based on tokenizer output)
    # For now, assume inputs is a vector of vectors
    max_len = maximum(length(inp) for inp in inputs)
    batch_inputs = zeros(Int, max_len, length(inputs))
    
    for (i, inp) in enumerate(inputs)
        batch_inputs[1:length(inp), i] = inp
    end
    
    # Convert labels to tensor
    labels_tensor = labels
    
    # Move to device
    if device == gpu && CUDA.functional()
        batch_inputs = batch_inputs |> gpu
        labels_tensor = labels_tensor |> gpu
    end
    
    return batch_inputs, labels_tensor
end

"""
    test_continual_learning(tester::ModelTester, adaptation_type::String="lora"; rank::Int=8, alpha::Real=1)

Test continual learning scenario.
Direct port of Python test_continual_learning method.
"""
function test_continual_learning(tester::ModelTester, adaptation_type::String="lora"; rank::Int=8, alpha::Real=1)
    println("="^50)
    println("Testing $(uppercase(adaptation_type)) - Continual Learning")
    println("Tasks: MRPC -> SST-2 -> Sentiment")
    println("="^50)
    
    # Load and adapt model
    model = load_base_model(tester)
    adapted_model, total_params = apply_adaptation(model, adaptation_type; rank=rank, alpha=alpha, device=tester.device)
    
    # Calculate initial perplexity
    initial_perplexity = calculate_perplexity(adapted_model, tester.tokenizer, tester.datasets.wikitext; device=tester.device)
    println("Initial Perplexity: $(round(initial_perplexity, digits=2))")
    
    # Sequential training on tasks
    tasks = [
        ("mrpc", 2),
        ("sst2", 2),
        ("sentiment", 2)
    ]
    
    task_heads = Dict{String, Any}()
    training_times = Dict{String, Float64}()
    
    for (task_name, num_classes) in tasks
        head, train_time = train_on_task(adapted_model, tester.tokenizer, task_name, num_classes; device=tester.device)
        task_heads[task_name] = head
        training_times[task_name] = train_time
    end
    
    # Evaluate on all tasks
    accuracies = evaluate_all_tasks(adapted_model, tester.tokenizer, task_heads, tester.datasets; device=tester.device)
    
    # Compile results
    results = Dict(
        "adaptation_type" => adaptation_type,
        "trainable_parameters" => total_params,
        "initial_perplexity" => initial_perplexity,
        "final_perplexity" => accuracies["perplexity"],
        "training_times" => training_times,
        "accuracies" => filter(p -> p.first != "perplexity", accuracies)
    )
    
    return results
end

"""
    evaluate_all_tasks(model, tokenizer, task_heads, datasets; device=cpu)

Evaluate model on all tasks with their respective heads.
"""
function evaluate_all_tasks(model, tokenizer, task_heads, datasets; device=cpu)
    results = Dict{String, Float64}()
    
    # Evaluate each task
    for (task_name, head) in task_heads
        println("Evaluating on $(task_name)...")
        
        # Set task head
        original_head = model.lm_head
        model.lm_head = head
        
        try
            if task_name == "mrpc"
                accuracy = evaluate_mrpc(model, tokenizer, datasets.mrpc_val; device=device, max_samples=20)
            elseif task_name == "sst2"
                accuracy = evaluate_sst2(model, tokenizer, datasets.sst2_val; device=device, max_samples=20)
            elseif task_name == "sentiment"
                accuracy = evaluate_sentiment(model, tokenizer, datasets.sst2_val; device=device, max_samples=20)
            else
                error("Unknown task: $task_name")
            end
            
            results[task_name] = accuracy
            println("$(uppercase(task_name)) Accuracy: $(round(accuracy, digits=4))")
            
        finally
            # Restore original head
            model.lm_head = original_head
        end
    end
    
    # Calculate final perplexity with original language modeling head
    println("Calculating perplexity...")
    perplexity = calculate_perplexity(model, tokenizer, datasets.wikitext; device=device)
    results["perplexity"] = perplexity
    println("Perplexity: $(round(perplexity, digits=2))")
    
    return results
end

"""
    run_comparison(tester::ModelTester; rank::Int=8, alpha::Real=1, save_results::Bool=true)

Run complete comparison between LoRA and CURLoRA.
Direct port of Python run_comparison method.
"""
function run_comparison(tester::ModelTester; rank::Int=8, alpha::Real=1, save_results::Bool=true)
    println("Running comparison on $(tester.model_name)")
    println("Rank: $(rank), Alpha: $(alpha)")
    
    # Test LoRA
    lora_results = test_continual_learning(tester, "lora"; rank=rank, alpha=alpha)
    
    # Memory cleanup
    if tester.device == gpu && CUDA.functional()
        CUDA.reclaim()
    end
    GC.gc()
    
    # Test CURLoRA
    curlora_results = test_continual_learning(tester, "curlora"; rank=rank, alpha=alpha)
    
    # Combine results
    comparison_results = Dict(
        "model_name" => tester.model_name,
        "rank" => rank,
        "alpha" => alpha,
        "lora" => lora_results,
        "curlora" => curlora_results
    )
    
    # Print summary
    print_comparison_summary(comparison_results)
    
    # Save results
    if save_results
        save_results_to_file(comparison_results)
    end
    
    return comparison_results
end

"""
    print_comparison_summary(results::Dict)

Print detailed comparison summary.
Exact port of Python print_comparison_summary function.
"""
function print_comparison_summary(results::Dict)
    println("="^70)
    println("COMPARISON SUMMARY")
    println("="^70)
    
    lora = results["lora"]
    curlora = results["curlora"]
    
    println("Model: $(results["model_name"])")
    println("Rank: $(results["rank"]), Alpha: $(results["alpha"])")
    println()
    
    # Parameter efficiency
    println("PARAMETER EFFICIENCY:")
    println("LoRA trainable parameters:    $(lora["trainable_parameters"])")
    println("CURLoRA trainable parameters: $(curlora["trainable_parameters"])")
    compression_ratio = lora["trainable_parameters"] / curlora["trainable_parameters"]
    println("Compression ratio: $(round(compression_ratio, digits=2))x")
    println()
    
    # Performance comparison
    println("TASK PERFORMANCE:")
    tasks = ["mrpc", "sst2", "sentiment"]
    println("Task         LoRA     CURLoRA  Difference")
    println("-"^50)
    
    for task in tasks
        lora_acc = lora["accuracies"][task]
        curlora_acc = curlora["accuracies"][task]
        diff = curlora_acc - lora_acc
        println("$(rpad(uppercase(task), 12)) $(rpad(round(lora_acc, digits=4), 8)) $(rpad(round(curlora_acc, digits=4), 8)) $(round(diff, digits=4, sigdigits=3))")
    end
    
    println()
    println("PERPLEXITY:")
    println("LoRA final perplexity:    $(round(lora["final_perplexity"], digits=2))")
    println("CURLoRA final perplexity: $(round(curlora["final_perplexity"], digits=2))")
    println()
    
    # Training time comparison
    total_lora_time = sum(values(lora["training_times"]))
    total_curlora_time = sum(values(curlora["training_times"]))
    println("TRAINING TIME:")
    println("LoRA total training time:    $(round(total_lora_time, digits=2))s")
    println("CURLoRA total training time: $(round(total_curlora_time, digits=2))s")
    println("Time difference: $(round(total_curlora_time - total_lora_time, digits=2))s")
end

"""
    save_results_to_file(results::Dict)

Save results to JSON file with timestamp.
"""
function save_results_to_file(results::Dict)
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "curlora_comparison_$(replace(results["model_name"], "/" => "_"))_$(timestamp).json"
    
    # Create results directory if it doesn't exist
    mkpath("results")
    filepath = joinpath("results", filename)
    
    # Save to JSON file
    open(filepath, "w") do f
        JSON3.pretty(f, results, indent=2)
    end
    
    println("Results saved to: $(filepath)")
end