
library(diffusr)
library(matrixcalc)
library(MASS)
library(Matrix)


dir_data <- '../data'

dir_sparse_index <- paste(dir_data, 'rwr_mat.txt', sep = '/')

adjacency_mat_sparse_index <- read.table(dir_sparse_index, header = TRUE, sep ='\t')

row_index <- as.matrix(adjacency_mat_sparse_index$row_idx)
col_index <- as.matrix(adjacency_mat_sparse_index$col_idx)

input_value <- 1 * (1:length(row_index))

graph_adjacency_mat <- sparseMatrix(i = row_index, j = col_index, x = input_value)

graph_adjacency_mat_full <- as.matrix(graph_adjacency_mat)

total_nodes_num <- dim(graph_adjacency_mat)[1]

dir_histone <- paste(dir_data, 'rwr_histone_input.txt', sep = '/')
p0_histone <- read.table(dir_histone, header = TRUE, sep ='\t')

p0_histone_mat <- as.matrix(p0_histone$val)
p0_histone_mat <- p0_histone_mat / sum(p0_histone_mat)

print(p0_histone_mat[which(p0_histone_mat != 0)])


# computation of stationary distribution
pt <- random.walk(p0_histone_mat, graph_adjacency_mat_full, r = 0.6, correct.for.hubs = TRUE)

p0_histone_imputation <- pt$p.inf

print(max(p0_histone_imputation))

print(p0_histone_imputation[1:10])

p0_histone_imputation <- p0_histone_imputation / max(p0_histone_imputation)

dir_histone_imputation <- paste(dir_data, 'histone_imputation_0.6.txt', sep = '/')
write.table(p0_histone_imputation, file = dir_histone_imputation, row.names=FALSE, col.names=FALSE, sep = '\t')
