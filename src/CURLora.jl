module CURLoRA

using Transformers
using Flux
using LinearAlgebra
using Statistics
using Random
using StatsBase
using JSON3
using ProgressMeter
using CUDA
using Functors
using Dates
using DataStructures
using ChainRulesCore
using Zygote
using Optimisers

# Export CUR-related functionality
export CURModule, CURLoRAMLP, LinearWithCURLoRA
export compute_selection_probabilities, select_indices_with_replacement
export adjust_duplicates, cur_decomposition

# Export LoRA functionality
export LoRALayer, LinearWithLoRA

# Export testing and evaluation
export ModelTester, run_comparison, test_continual_learning
export calculate_perplexity, evaluate_sst2, evaluate_mrpc, evaluate_sentiment
export load_datasets, prepare_batch, create_batches

# Export utility functions
export print_comparison_summary, save_results_to_file
export apply_adaptation, train_on_task, evaluate_all_tasks

# Include implementation files
include("cur.jl")
include("lora.jl") 
include("test.jl")

end # module