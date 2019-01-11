## ----eval=TRUE-----------------------------------------------------------
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

## ----eval=TRUE-----------------------------------------------------------
# obtain data
attach(chickwts)
x <- sort(as.vector(weight[feed == "soybean"]))
y <- sort(as.vector(weight[feed == "linseed"]))
detach(chickwts)
set.seed(1)
CVM.test(x,y,R=999)

## ----eval=TRUE-----------------------------------------------------------
my.pcauchy <- function (x, eta, theta) {
  stopifnot(theta > 0)
  my.dcauchy <- function (x, eta, theta) {
    return(1/(theta*pi*(1 + ((x - eta)/theta)^2)))
  } # the density function of the Cauchy distribution
  integral <- function (x) {
    my.dcauchy(x, eta, theta)
  }
  return(integrate(integral, lower = -Inf, upper = x, rel.tol=.Machine$double.eps^0.25)$value)
} # the value of the integrand is the cdf of the Cauchy distribution

## ----eval=TRUE-----------------------------------------------------------
# Cauchy(0,1)
xs <- seq(-4, 5)
names(xs) <-c(seq(-4, 5))
my.pcauchy.val <- sapply(xs, function(x) my.pcauchy(x, 0, 1))
pcauchy.val <- sapply(xs, function(x) pcauchy(x, 0, 1))
round(rbind(my.pcauchy=my.pcauchy.val, pcauchy=pcauchy.val), 4)
# Cauchy(1,2)
my.pcauchy.val <- sapply(xs, function(x) my.pcauchy(x, 1, 2))
pcauchy.val <- sapply(xs, function(x) pcauchy(x, 1, 2))
round(rbind(my.pcauchy=my.pcauchy.val, pcauchy=pcauchy.val), 4)

## ----eval=TRUE-----------------------------------------------------------
my.chisq.test <- function(x, y) {
  total <- sum(x) + sum(y)
  rowsum_x <- sum(x)
  rowsum_y <- sum(y)
  chistat <- 0
  expected <- function(colsum, rowsum, total) {
    (colsum / total) * (rowsum / total) * total
  } # the expected (theoretical) count of type i
  chi_stat <- function(observed, expected) {
    ((observed - expected) ^ 2) / expected
  } # a measure of the approximation between observed value and expected value
  for (i in seq_along(x)) {
    colsum <- x[i] + y[i]
    expected_x <- expected(colsum, rowsum_x, total)
    expected_y <- expected(colsum, rowsum_y, total)
    chistat <- chistat + chi_stat(x[i], expected_x)
    chistat <- chistat + chi_stat(y[i], expected_y)
  }
  chistat
} # code from the mathematical definition

## ----eval=TRUE-----------------------------------------------------------
my.chisq.test(c(367,342,266,329),c(56,40,20,16))

## ----eval=TRUE-----------------------------------------------------------
# validate the function my.chisq.test() is reasonable
chisq.test(as.table(rbind(c(367,342,266,329),c(56,40,20,16))))
# Validate the function my.chisq.test() is faster than chisq.test()
print(microbenchmark::microbenchmark(
  my.chisq.test = my.chisq.test(c(367,342,266,329),c(56,40,20,16)),
  chisq.test = chisq.test(as.table(rbind(c(367,342,266,329),c(56,40,20,16))))
))

## ----eval=TRUE-----------------------------------------------------------
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

## ----eval=TRUE-----------------------------------------------------------
x <- c(1, 2, 2, 3, 1, 3, 3, 2)
y <- c(1, 1, 2, 1, 1, 1, 1, 2)
# validate the function my.table() is reasonable
my.table(x, y)
table(x, y)
# validate the function my.table() is faster than table()
print(microbenchmark::microbenchmark(
  my.table = my.table(x, y),
  table = table(x, y)
))

