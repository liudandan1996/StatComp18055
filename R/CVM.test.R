#' @title Two-sample Cramer-von Mises test
#' @description Choose Cramer-von Mises statistic to measure the difference between two distributions and test for equal distributions in the univariate case
#' @param x independent random samples from distribution F
#' @param y independent random samples from distribution G
#' @param R number of repeated sampling
#' @return the p-value of hypothesis test
#' @examples
#' \dontrun{
#' attach(chickwts)
#' x <- sort(as.vector(weight[feed == "soybean"]))
#' y <- sort(as.vector(weight[feed == "linseed"]))
#' detach(chickwts)
#' CVM.test(x,y,R=999)
#' }
#' @export
CVM.test <- function(x,y,R) {
  CVM.stat <- function(x,y) {
    n <- length(x)
    m <- length(y)
    F_n <- ecdf(x)
    G_m <- ecdf(y)
    T <- ((m*n)/(m+n)^2)*(sum((F_n(x)-G_m(x))^2) + sum((F_n(y)-G_m(y))^2)) # the Cramer-Von Mises (CVM) statistic
    return(T)
  } # use Cramer-von Mises statistic to measure the difference between two distributions
  n1 <- length(x)
  n2 <- length(y)
  n <- n1 + n2
  z <- c(x, y) # pooled sample
  K <- 1:n
  reps <- numeric(R) # storage for replicates
  t0 <- CVM.stat(x,y) # the observed statistic t0
  for (i in 1:R) { # permutation samples
    k <- sample(K, size = n1, replace = FALSE)
    x1 <- z[k]
    y1 <- z[-k] # complement of x1
    reps[i] <- CVM.stat(x1, y1) # the Cramer-Von Mises (CVM) statistic T
  }
  p <- mean(c(t0, reps) >= t0)
  print(p)
}
