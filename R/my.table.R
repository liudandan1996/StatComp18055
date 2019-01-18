#' @title A faster version of table()
#' @description A faster version of table() for the case of an input of two integer vectors with no missing values
#' @param x an integer vector with no missing values
#' @param y an integer vector with no missing values
#' @return a contingency table
#' @examples
#' \dontrun{
#' x <- c(1, 2, 2, 3, 1, 3, 3, 2)
#' y <- c(1, 1, 2, 1, 1, 1, 1, 2)
#' my.table(x, y)
#' }
#' @export
my.table <- function(x, y) {
  x_unique <- unique(x) # extract unique elements of x
  y_unique <- unique(y) # extract unique elements of y
  mat <- matrix(0L, length(x_unique), length(y_unique)) # storage for replicates
  for (i in seq_along(x)) {
    mat[which(x_unique == x[[i]]), which(y_unique == y[[i]])] <-
      mat[which(x_unique == x[[i]]),  which(y_unique == y[[i]])] + 1L
  } # build a contingency table of the counts at each combination of factor levels
  # optimize the form of output
  dimnames <- list(x_unique, y_unique)
  names(dimnames) <- as.character(as.list(match.call())[-1])
  tab <- array(mat, dim = dim(mat), dimnames = dimnames)
  class(tab) <- "table"
  tab
}
