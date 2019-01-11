#' @title A faster version of chisq.test()
#' @description A faster version of chisq.test() that only computes the chi-square test statistic when the input is two numeric vectors with no missing values.
#' @param x a numeric vector with no missing values
#' @param y a numeric vector with no missing values
#' @return the chi-square test statistic
#' @examples
#' \dontrun{
#' my.chisq.test(c(367,342,266,329),c(56,40,20,16))
#' }
#' @export
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
