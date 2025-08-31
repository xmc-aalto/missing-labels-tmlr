using NPZ
using Statistics

include("psrec.jl")

#=
This script calculates the PSRecall values for a dataset. We assume a
structured way of naming data files, so that the specification of which
setting to run can be configured by setting the `dataset` and `mode` variables
below, and letting `data_dir` point to the root directory of the datasets.

We then assume that all data pertaining to the dataset can be found in a
subfolder named after the dataset. In particular, we expect the files
`${dataset}-${mode}.predict` (i.e. for amazoncat and mode ps a file
`amazoncat/amazoncat-ps.predict` should exist) that contains the predictions
as a space-separated list of index:value pairs ordered by value, so that the
top-3 predictions e.g. would look like `5:2.3 3:1.2 9:0.5` where `(5, 3, 9)`
are the top-3 labels. For the ground-truth, a file named
`amazoncat/tfidf-amazoncat-test.txt` would be expected that is in the format
as in the extreme classification repository.

ATTENTION: Note that our training scripts assume label indices starting with
0, but julia uses arrays starting with 1. Therefore, the `parse_gt` and
`parse_pred` functions below increment the label values by one. If you use
data that is already index 1-based, you need to adapt these functions.

Finally, a file `amazoncat/inv_prop.npy` that contains a numpy array with the
inverse propensities is expected.

If you want to use another naming scheme, you can adapt the bottom part of
this script, and just call `process_files` with the file names you want to
use.
=#

# set file locations
dataset = "amazon14"
mode = "ps"
data_dir = "data"

# reads the labels from a line in the test data file
function parse_gt(line::AbstractString)
    splitted = split(line, " ")
    labels = map(x->parse(Int, x), split(splitted[1], ","))
    # TODO check that this works for empty lines
    return labels  .+ 1
end

# reads the labels from a line in the prediction file
function parse_pred(line::AbstractString)
    splitted = split(line, " ")
    labels = map(x->split(x, ":")[1], splitted)
    # TODO check that this works for empty lines
    return map(x->parse(Int, x), labels)  .+ 1
end


function process_files(predictions_file_name::AbstractString, gt_file_name::AbstractString, 
                       prop_file_name::AbstractString, base_path::AbstractString, mode::AbstractString, k::Int) 
    println("Predictions: $predictions_file_name, ground truth: $gt_file_name, Recall@$k")

    # open the files and skip the headers
    prediction_file = open("$base_path/$predictions_file_name")
    readline(prediction_file)
    gt_file = open("$base_path/$gt_file_name")
    readline(gt_file)

    propensity::Array{Float64} = 1 ./ npzread("$base_path/$prop_file_name")

    total::Array{Float64} = []
    vanilla::Array{Float64} = []

    for data in zip(eachline(prediction_file), eachline(gt_file))
        pred, gt = data
        labels = parse_gt(gt)
        predicted = parse_pred(pred)
        predicted = predicted[1:k]
        ps_r_at_k = ps_recall(predicted, labels, propensity)
        r_at_k = recall(predicted, labels)
        append!(total, ps_r_at_k)
        append!(vanilla, r_at_k)

        if length(total) % 10000 == 0 
            println("processed $(length(total)) items")
        end
    end

    println("PsRecall@$k $(mean(total)) ± $(std(total)/sqrt(length(total)))")
    println("median: $(median(total))")
    println("Recall@$k ", mean(vanilla))

    npzwrite("$base_path/ps-recall-$mode-$k.npy", total)
    npzwrite("$base_path/recall-$mode-$k.npy", vanilla)
end

base_path = "$data_dir/$dataset"
predictions_file_name = "$dataset-$mode.predict"
gt_file_name = "tfidf-$dataset-test.txt"
prop_file_name = "inv_prop.npy"

process_files(predictions_file_name, gt_file_name, prop_file_name, base_path, mode, 1)
process_files(predictions_file_name, gt_file_name, prop_file_name, base_path, mode, 3)
process_files(predictions_file_name, gt_file_name, prop_file_name, base_path, mode, 5)
