#' @title Compute the cdf of the Cauchy distribution
#' @description Compute the cdf of the Cauchy distribution
#' @param x the upper bound
#' @param eta the location parameter of the Cauchy distribution
#' @param theta the scale parameter of the Cauchy distribution
#' @return the value of the cdf of the Cauchy distribution at x
#' @examples
#' \dontrun{
#' my.pcauchy(x=1, eta=0, theta=1)
#' }
#' @export
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
