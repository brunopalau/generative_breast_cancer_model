compareClusters <- function(cluster_list, second){
  n = length(cluster_list)
  final_mat_median <- matrix(1, ncol = n, nrow = n)
  colnames(final_mat_median) <- names(cluster_list)
  rownames(final_mat_median) <- names(cluster_list)
  
  final_mat_allCells <- matrix(1, ncol = n, nrow = n)
  colnames(final_mat_allCells) <- names(cluster_list)
  rownames(final_mat_allCells) <- names(cluster_list)
  
  for (i in seq_along(cluster_list)){
    for (j in seq_along(cluster_list)){
      if(i == j){
        cluster_1 <- cluster_list[[i]]
        cluster_2 <- second[[j]]
      }
      else{
        cluster_1 <- cluster_list[[i]]
        cluster_2 <- cluster_list[[j]]
      }
      
      final_mat_median[i,j] <- median(rowMax(unclass(proportions(table(cluster_1, cluster_2), margin = 1))))
      final_mat_allCells[i,j] <- sum(rowMax(unclass(table(cluster_1, cluster_2)))) / length(cluster_1)
    }     
  }
  return(list(final_mat_median = final_mat_median,
              final_mat_allCells = final_mat_allCells))
}
