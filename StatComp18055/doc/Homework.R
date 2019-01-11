## ----eval=FALSE----------------------------------------------------------
#  m1<-matrix(1,nr=2,nc=2)
#  m2<-matrix(2,nr=2,nc=2)
#  rbind(m1,m2)
#  cbind(m1,m2)
#  rbind(m1,m2) %*% cbind(m1,m2)
#  cbind(m1,m2) %*% rbind(m1,m2)
#  diag(m1)
#  diag(rbind(m1,m2) %*% cbind(m1,m2))
#  diag(m1)<-10
#  m1
#  diag(3)
#  v<-c(10,20,30)
#  diag(v)
#  diag(2.1,nr=3,nc=5)

## ----eval=FALSE----------------------------------------------------------
#  mat<-matrix(1:4,2,2)
#  mat
#  layout(mat)
#  layout.show(4)
#  layout(matrix(1:6,3,2))
#  layout.show(6)
#  m<-matrix(c(1:3,3),2,2)
#  layout(m)
#  layout.show(3)
#  m<-matrix(1:4,2,2)
#  layout(m,widths=c(1,3),heights=c(3,1))
#  layout.show(4)

## ----eval=FALSE----------------------------------------------------------
#  x<-rnorm(10)
#  y<-rnorm(10)
#  plot(x,y)
#  plot(x,y,xlab="Ten random values",ylab="Ten other values",xlim=c(-2,2),ylim=c(-2,2),pch=22,col="red",bg="yellow",bty="l",tcl=0.4,
#  main="How to customize a plot with R ",las=1,cex=1.5)
#  opar<-par()
#  par(bg="lightyellow",col.axis="blue",mar=c(4,4,2.5,0.25))
#  plot(x,y,xlab="Ten random values",ylab="Ten other values",xlim=c(-2,2),ylim=c(-2,2),pch=22,col="red",bg="yellow",bty="l",tcl=-.25,las=1,cex=1.5)
#  title("How to customize a plot with R(bis)",font.main=3,adj=1)
#  opar<-par()
#  par(bg="lightgray",mar=c(2.5,1.5,2.5,0.25))
#  plot(x,y,type="n",xlab="",ylab="",xlim=c(-2,2),ylim=c(-2,2),xaxt="n",yaxt="n")
#  rect(-3,-3,3,3,col="cornsilk")
#  points(x,y,pch=10,col="red",cex=2)
#  axis(side=1,c(-2,0,2),tcl=-0.2,labels=FALSE)
#  axis(side=2,-1:1,tcl=-0.2,labels=FALSE)
#  title("How to customize a plot with R(ter)",font.main=4,adj=1,cex.main=1)
#  mtext("Ten random values",side=1,line=1,at=1,cex=0.9,font=3)
#  mtext("Ten other values",line=0.5,at=-1.8,cex=0.9,font=3)
#  mtext(c(-2,0,2),side=1,las=1,at=c(-2,0,2),line=0.3,col="blue",cex=0.9)
#  mtext(-1:1,side=2,las=1,at=-1:1,line=0.2,col="blue",cex=0.9)

## ----eval=FALSE----------------------------------------------------------
#   A<-c(321,266,256,388,330,329,303,334,299,221,365,250,258,342,343,298,238,317,354)
#   B<-c(488,598,507,428,807,342,512,350,672,589,665,549,451,481,514,391,366,468)
#   diff<-median(B)-median(A)
#   A<-A+diff
#   mood.test(A,B)

## ----eval=FALSE----------------------------------------------------------
#  x<-c(274,180,375,205,86,265,98,330,195,53,430,372,236,157,370)
#  y<-c(162,120,223,131,67,169,81,192,116,55,252,234,144,103,212)
#  A<-data.frame(x,y)
#  plot(x,y)
#  lm.reg<-lm(y~x)
#  summary(lm.reg)
#  abline(lm.reg)

## ----eval=FALSE----------------------------------------------------------
#  x <- c(0,1,2,3,4)
#  p <- c(0.1,0.2,0.2,0.2,0.3)
#  cp <- cumsum(p)  #cumulative probability
#  m <- 1e3
#  r <- numeric(m)
#  r <- x[findInterval(runif(m),cp)+1] #a random sample of size 1000 from the distribution of X
#  r
#  table(r)  #a relative frequency table
#  ct <- as.vector(table(r))
#  ct/sum(ct)/p  #compare the empirical with the theoretical probabilities

## ----eval=FALSE----------------------------------------------------------
#  n <- 1e3
#  k <- 0  #counter for accepted
#  j <- 0  #iterations
#  y <- numeric(n)
#  while (k < n) {
#    u <- runif(1)
#    j <- j + 1
#    x <- runif(1)  #random variate from g
#    if (27/4 * x^2 * (1-x) > u) {
#      #we accept x
#      k <- k + 1
#      y[k] <- x
#    }
#  }
#  y  #a random sample of size 1000 from the Beta(3,2) distribution
#  j  #experiments for 1000 random numbers
#  hist(y, prob = TRUE, main = expression(f(x)==12*x^2*(1-x)))  #the histogram of the sample
#  z <- seq(0, 1, .01)
#  lines(z, 12*z^2*(1-z))  #the theoretical Beta(3,2) density

## ----eval=FALSE----------------------------------------------------------
#  #generate a Exponential-Gamma mixture
#  n <- 1e3
#  r <- 4
#  beta <- 2
#  lambda <- rgamma(n, r, beta)  #lambda is random
#  y <- rexp(n,lambda)
#  y  #1000 random observations from this mixture

## ----eval=FALSE----------------------------------------------------------
#  #  write a function to compute a Monte Carlo estimate of the Beta(3, 3) cdf
#  MC.Phi <- function(x, m = 10000) {
#    u <- runif(m)
#    cdf <- numeric(length(x))
#    for (i in 1:length(x)) {
#      g <- x[i] * 30 * (u * x[i])^2 * (1-(u * x[i]))^2  #the expectation of g is the target parameter
#      cdf[i] <- mean(g)
#    }
#    cdf  #a Monte Carlo estimate of the Beta(3, 3) cdf
#  }

## ----eval=FALSE----------------------------------------------------------
#  #  use the function to estimate F(x) for x = 0.1, 0.2, . . . , 0.9
#  x <- seq(.1, .9, length=9)  #x = 0.1, 0.2, . . . , 0.9
#  set.seed(123)
#  MC <- MC.Phi(x)  #a Monte Carlo estimate of  F(x) for x = 0.1, 0.2, . . . , 0.9.

## ----eval=FALSE----------------------------------------------------------
#  #compare the estimates with the values returned by the pbeta function in R
#  Phi <- pbeta(x,3,3)  #values returned by the pbeta function
#  print(round(rbind(x, MC, Phi), 9))  #comparison between MC and Phi

## ----eval=FALSE----------------------------------------------------------
#  rayleigh_red <- function(sigma, n) {
#    rayleigh <- antithetic <- numeric(n)
#    for (i in 1:n) {
#      U <- runif(n)
#      V <- 1 - U  #use antithetic variables
#      rayleigh = sigma * sqrt(-2 * log(1-U))
#      antithetic = sigma * sqrt(-2 * log(1-V))  #'rayleigh' and 'antithetic' are the  samples we generate from a Rayleigh distribution, and they are negatively correlated.
#      var1 <- var(rayleigh)  #the variance of (X1+X2)/2
#      var2 <- (var(rayleigh) + var(antithetic) + 2 * cov(rayleigh, antithetic)) / 4 #the variance of (X+X')/2
#      reduction <- ((var1 - var2) / var1)
#      percent <- paste0(formatC(100 * reduction, format = "f", digits = 4), "%")
#    }  #the percent reduction of variance
#    return(noquote(percent))
#  }
#  set.seed(123)
#  sigma = 1  #set the value of unknown parameter sigma be 1
#  n <- 1e3
#  rayleigh_red(sigma, n)

## ----eval=FALSE----------------------------------------------------------
#  x <- seq(1, 5, .01)  #the value of g(x) tends to 0 when x>5, so limit x in [1, 5] is more helpful to observe the gragh
#  w <- 2
#  g <- x^2*exp(-x^2/2)/ sqrt(2*pi)
#  f1<- x*exp(-x)  #the importance function f1 is the density function of Gamma(2,1)
#  f2<- exp(-(x-sqrt(2))^2/2)/sqrt(2*pi)  #the importance function f2 is the density function of Normal(sqrt(2),1)
#  gs <- c(expression(g(x)==x^2*e^{-x^2/2}/ sqrt(2*pi)),expression(f[1](x)==x*e^{-x}),expression(f[2](x)==e^{-(x-sqrt(2))^2/2}/sqrt(2*pi)))
#  #for color change lty to col
#  par(mfrow=c(1,2))
#  #figure (a)
#  plot(x, g, type = "l", ylab = "",
#       ylim = c(0,4), lwd = w,col=1,main='(a)')
#  lines(x, f1, lty = 2, lwd = w,col=2)
#  lines(x, f2, lty = 3, lwd = w,col=3)
#  legend("topright", legend = gs,
#         lty = 1:3, lwd = w, inset = 0.02,col=1:3)
#  #figure (b)
#  plot(x, g/f1, type = "l", ylab = "",
#       ylim = c(0,4), lwd = w, lty = 2,col=2,main='(b)')
#  lines(x, g/f2, lty = 3, lwd = w,col=3)
#  legend("topright", legend = gs[-1],
#         lty = 2:3, lwd = w, inset = 0.02,col=2:3)

## ----eval=FALSE----------------------------------------------------------
#  set.seed(123)
#  m <- 1e5
#  theta.hat <- se <- numeric(2)
#  g <- function(x) {
#    x^2*exp(-x^2/2)/ sqrt(2*pi) * (x > 1)
#  }
#  x <- rgamma(m, 2,1)  #using f1, f1 is the density function of Gamma(2,1)
#  fg <- g(x) / (x*exp(-x))
#  theta.hat[1] <- mean(fg)  #a Monte Carlo estimate by importance sampling, the importance function is f1
#  se[1] <- sd(fg)  #standard error of estimation by using f1
#  x <- rnorm(m, mean = sqrt(2), sd = 1)  #using f2, f2 is the density function of Normal(sqrt(2),1)
#  fg <- g(x) / (exp(-(x-sqrt(2))^2/2)/sqrt(2*pi))
#  theta.hat[2] <- mean(fg)  ##a Monte Carlo estimate by importance sampling, the importance function is f2
#  se[2] <- sd(fg)  #standard error of estimation by using f2
#  res <- rbind(theta=round(theta.hat,7), se=round(se,7))
#  res

## ----eval=FALSE----------------------------------------------------------
#  integrate(g, 1, Inf)

## ----eval=FALSE----------------------------------------------------------
#  m <- 1e3 #number of samples of per simulation
#  n <- 20 #number of simulations
#  Gx <- Gy <- Gz <- numeric(m)
#  set.seed(123)
#  for (j in 1:m){ #simulation procedure
#    x <- sort(rlnorm(n)) #sort x, x is generated from standard lognormal distribution
#    y <- sort(runif(n)) #sort y, y is generated from uniform(0,1) distribution
#    z <- sort(rbinom(n,size=1,prob=0.1)) #sort z, z is generated from Bernoulli(0.1) distribution
#    mu1 <- mean(x);mu2 <- mean(y);mu3 <- mean(z) #use mean(x) replace ¦Ì
#    for (i in 1:n) {
#        x1 <- (2*i-n-1)*x[i]
#        y1 <- (2*i-n-1)*y[i]
#        z1 <- (2*i-n-1)*z[i]
#      }
#    #generate the estimated value of Gini ratio(use mean(x) replace ¦Ì)
#    Gx[j] <- sum(x1)/(n^2*mu1)
#    Gy[j] <- sum(y1)/(n^2*mu2)
#    Gz[j] <- sum(z1)/(n^2*mu3)
#    }
#  Gx_mean <- mean(Gx);Gx_median <- median(Gx);Gx_deciles <- quantile(Gx, probs = seq(0, 1, 0.1)) #estimate by simulation the mean, median and deciles of Ghat
#  print(c(Gx_mean,Gx_median))
#  print(Gx_deciles)
#  hist(Gx,prob=TRUE,main="density histogram of X(X is standard lognormal)") #construct density histogram
#  
#  Gy_mean <- mean(Gy);Gy_median <- median(Gy);Gy_deciles <- quantile(Gy, probs = seq(0, 1, 0.1)) #estimate by simulation the mean, median and deciles of Ghat
#  print(c(Gy_mean,Gy_median))
#  print(Gy_deciles)
#  hist(Gy,prob=TRUE,main="density histogram of Y(Y is uniform(0,1))") #construct density histogram
#  
#  Gz_mean <- mean(Gz,na.rm = TRUE);Gz_median <- median(Gz,na.rm = TRUE);Gz_deciles <- quantile(Gz,probs=seq(0, 1, 0.1),na.rm = TRUE) #estimate by simulation the mean, median and deciles of Ghat
#  print(c(Gz_mean,Gz_median))
#  print(Gz_deciles)
#  hist(Gz,prob=TRUE,main="density histogram of Z(Z is Bernoulli(0.1))") #construct density histogram

## ----eval=FALSE----------------------------------------------------------
#  m <- 1e3 #number of samples of per simulation
#  n <- 20 #number of simulations
#  #set the parameter of lognormal distribution
#  a <- 0;b <- 1
#  G <- numeric(m)
#  set.seed(123)
#  for (j in 1:m){ #simulation procedure
#    x <- sort(rlnorm(n,a,b)) #sort x, x is generated from standard lognormal distribution
#    mu <- mean(x) #use mean(x) replace ¦Ì
#    for (i in 1:n) {
#      x1 <- (2*i-n-1)*x[i]
#    }
#    #generate the estimated value of Gini ratio(use mean(x) replace ¦Ì)
#    G[j] <- sum(x1)/(n^2*mu)
#  }
#  G_mean <- mean(G) #estimate by simulation the mean of Ghat
#  G_se <- sd(G) #estimate by simulation the variance of Ghat
#  alpha <- 0.05 #the significant level
#  UCL <- G_mean+qt(1-(alpha/2), df=n-1)*G_se/sqrt(n) #obtain the confidence interval upper limit
#  LCL <- G_mean-qt(1-(alpha/2), df=n-1)*G_se/sqrt(n) #obtain the lower confidence interval
#  CI <- c(LCL,UCL) #an approximate 95% confidence interval for the Gini ratio
#  print(CI)
#  
#  #assess the coverage rate of the estimation procedure with a Monte Carlo experiment
#  m <- 1e3 #number of samples of per simulation
#  n <- 20 #number of simulations
#  #set the parameter of lognormal distribution
#  a <- 0;b <- 1
#  G <- numeric(m)
#  set.seed(123)
#  for (j in 1:m){ #simulation procedure
#    x <- sort(rlnorm(n,a,b)) #sort x, x is generated from standard lognormal distribution
#    mu <- mean(x) #use mean(x) replace ¦Ì
#    for (i in 1:n) {
#      x1 <- (2*i-n-1)*x[i]
#    }
#    #generate the estimated value of Gini ratio(use mean(x) replace ¦Ì)
#    G[j] <- sum(x1)/(n^2*mu)
#  }
#  CI <- c(LCL,UCL)
#  coverage <- 1-mean(G< UCL & G >LCL)
#  coverage_rate <- paste0(format(100*coverage, digits = 3), "%") #the coverage rate

## ----eval=FALSE----------------------------------------------------------
#  library(MASS)
#  n <- 1e3 #number of random samples
#  alpha <- 0.05 #the significant level
#  #set the parameters of bivariate normal distribution
#  mu <- c(0,0) #mean
#  sigma <- matrix(c(1,0,0,1),2,2) #covariance matrix
#  p.value_pearson <- p.value_spearman <- p.value_kendall <- numeric(n)
#  set.seed(123)
#  for(i in 1:n){
#    samples <- mvrnorm(n,Sigma=sigma,mu=mu) #obtain samples from bivariate normal distribution
#    x <- samples[,1] #the samples of x
#    y <- samples[,2] #the samples of y
#    p.value_pearson[i] <- cor.test(x,y,method="pearson")$p.value #the p_value of the correlation test
#    p.value_spearman[i] <- cor.test(x,y,method="spearman")$p.value #the p_value of the nonparametric tests based on ¦Ñs
#    p.value_kendall[i] <- cor.test(x,y,method="kendall")$p.value #the p_value of the nonparametric tests based on ¦Ó
#  }
#  power_pearson <- mean(p.value_pearson <= alpha) #the power of the correlation test
#  power_spearman <- mean(p.value_spearman <= alpha) #the power of the nonparametric tests based on ¦Ñs
#  power_kendall <- mean(p.value_kendall <= alpha) #the power of the nonparametric tests based on ¦Ó
#  print(c(power_pearson,power_spearman,power_kendall))

## ----eval=FALSE----------------------------------------------------------
#  library(MASS)
#  n <- 1e3 #number of random samples
#  alpha <- 0.05 #the significant level
#  #set the parameters of bivariate normal distribution
#  mu <- c(0,0) #mean
#  sigma <- matrix(c(1,0.15,0.15,1),2,2) #covariance matrix
#  p.value_pearson <- p.value_spearman <- p.value_kendall <- numeric(n)
#  set.seed(123)
#  for(i in 1:n){
#    samples <- mvrnorm(n,Sigma=sigma,mu=mu) #obtain samples from bivariate normal distribution
#    x <- samples[,1] #the samples of x
#    y <- samples[,2] #the samples of y
#    p.value_pearson[i] <- cor.test(x,y,method="pearson")$p.value #the p_value of the correlation test
#    p.value_spearman[i] <- cor.test(x,y,method="spearman")$p.value #the p_value of the nonparametric tests based on ¦Ñs
#    p.value_kendall[i] <- cor.test(x,y,method="kendall")$p.value #the p_value of the nonparametric tests based on ¦Ó
#  }
#  power_pearson <- mean(p.value_pearson <= alpha) #the power of the correlation test
#  power_spearman <- mean(p.value_spearman <= alpha) #the power of the nonparametric tests based on ¦Ñs
#  power_kendall <- mean(p.value_kendall <= alpha) #the power of the nonparametric tests based on ¦Ó
#  print(c(power_pearson,power_spearman,power_kendall))

## ----eval=FALSE----------------------------------------------------------
#  data(law, package = "bootstrap") #load data set "law"
#  theta.hat <- cor(law$LSAT, law$GPA) #compute the original value of the correlation statistic
#  
#  #compute the jackknife replicates, leave-one-out estimates
#  n <- nrow(law) #number of replicates
#  theta.jack <- numeric(n) #storage for replicates
#  for (i in 1:n){
#    LSAT <- law$LSAT[-i]
#    GPA <- law$GPA[-i]
#    theta.jack[i] <- cor(LSAT, GPA)
#  }
#  bias <- (n - 1) * (mean(theta.jack) - theta.hat) #jackknife estimate of the bias of the correlation statistic
#  se <- sqrt((n-1)/n *sum((theta.jack - mean(theta.jack))^2)) #jackknife estimate of the standard error of the correlation statistic
#  print(c(original=theta.hat, bias.jack=bias, se.jack=se))

## ----eval=FALSE----------------------------------------------------------
#  library(boot) #for boot and boot.ci
#  data(aircondit, package = "boot") #load data set "aircondit"
#  set.seed(1)
#  boot.obj <- boot(data=aircondit, statistic = function(x, i) mean(x[i,]), R = 2000) #use boot function
#  print(boot.ci(boot.obj, type=c("basic","norm","perc","bca"))) #Compute 95% bootstrap confidence intervals for the mean time between failures 1/¦Ë by the standard normal, basic, percentile, and BCa methods.

## ----eval=FALSE----------------------------------------------------------
#  hist(boot.obj$t, main='histogram of 1/¦Ë', xlab=expression(1/lambda), prob=TRUE) #the bootstrap distribution of 1/¦Ë
#  points(boot.obj$t0, 0, pch = 19) #the MLE of 1/¦Ë

## ----eval=FALSE----------------------------------------------------------
#  data(scor, package = "bootstrap") #load data set "scor"
#  cov.matrix <- cov(scor) #compute the MLE of covariance matrix
#  lambda.hat <- eigen(cov(scor))$values #compute the estimated value of ¦Ë
#  theta.hat <- lambda.hat[1]/sum(lambda.hat) #compute the original value
#  
#  #compute the jackknife replicates, leave-one-out estimates
#  n <- nrow(scor) #number of replicates
#  theta.jack <- numeric(n) #storage for replicates
#  theta <- function(x){
#    eigen(cov(x))$values[1]/sum(eigen(cov(x))$values)
#  } #Write a function
#  x <- as.matrix(scor) #convert "scor" to matrix
#  for (i in 1:n){
#    theta.jack[i] <- theta(x[-i,])
#  }
#  bias <- (n - 1) * (mean(theta.jack) - theta.hat) #jackknife estimate of the bias
#  se <- sqrt((n-1)/n *sum((theta.jack - mean(theta.jack))^2)) #jackknife estimate of the standard error
#  print(c(original=theta.hat, bias.jack=bias, se.jack=se))

## ----eval=FALSE----------------------------------------------------------
#  data(ironslag, package = "DAAG") #load data set "ironslag"
#  magnetic <- ironslag$magnetic
#  chemical <- ironslag$chemical
#  n <- length(magnetic) #in DAAG ironslag
#  a <- n
#  b <- n - 1
#  e1 <- e2 <- e3 <- e4 <- array(dim = c(n, n-1)) #storage for replicates
#  
#  # fit models on leave-two-out samples
#  for( i in 1:a ) { #outer 1 to n
#    u <- magnetic[-i]
#    v <- chemical[-i]
#    for( j in 1:b ) { #inner 1 to n-1
#      y <- u[-j]
#      x <- v[-j]
#  
#      J1 <- lm(y ~ x) #Linear model
#      yhat1 <- J1$coef[1] + J1$coef[2] * v[j]
#      e1[i,j] <- u[j] - yhat1
#  
#      J2 <- lm(y ~ x + I(x^2)) #Quadratic model
#      yhat2 <- J2$coef[1] + J2$coef[2] * v[j] +
#        J2$coef[3] * v[j]^2
#      e2[i,j] <- u[j] - yhat2
#  
#      J3 <- lm(log(y) ~ x) #Exponential model
#      logyhat3 <- J3$coef[1] + J3$coef[2] * v[j]
#      yhat3 <- exp(logyhat3)
#      e3[i,j] <- u[j] - yhat3
#  
#      J4 <- lm(log(y) ~ log(x)) #Log-Log model
#      logyhat4 <- J4$coef[1] + J4$coef[2] * log(v[j])
#      yhat4 <- exp(logyhat4)
#      e4[i,j] <- u[j] - yhat4
#      }
#  }
#  
#  c(mean(e1^2), mean(e2^2), mean(e3^2), mean(e4^2)) #estimates for prediction error

## ----eval=FALSE----------------------------------------------------------
#  ## Write a function to implement the two-sample Cram¨¦r-von Mises test for equal distributions
#  CVM <- function(x,y){
#    n <- length(x)
#    m <- length(y)
#    F_n <- ecdf(x)
#    G_m <- ecdf(y)
#    T <- ((m*n)/(m+n)^2)*(sum((F_n(x)-G_m(x))^2) + sum((F_n(y)-G_m(y))^2)) #the Cram¨¦r-Von Mises (CVM) statistic
#    return(T)
#  }
#  
#  ##Obtain data
#  attach(chickwts)
#  x <- sort(as.vector(weight[feed == "soybean"]))
#  y <- sort(as.vector(weight[feed == "linseed"]))
#  detach(chickwts)
#  
#  set.seed(1)
#  R <- 999 #number of replicates
#  z <- c(x, y) #pooled sample
#  K <- 1:26
#  reps <- numeric(R) #storage for replicates
#  t0 <- CVM(x,y) #the observed statistic t0
#  
#  for (i in 1:R) { #permutation samples
#  #generate indices k for the first sample
#  k <- sample(K, size = 14, replace = FALSE)
#  x1 <- z[k]
#  y1 <- z[-k] #complement of x1
#  reps[i] <- CVM(x1, y1) #the Cram¨¦r-Von Mises (CVM) statistic T
#  }
#  
#  p <- mean(c(t0, reps) >= t0)
#  print(p)
#  
#  hist(reps, main = "Permutation distribution of Cram¨¦r-Von Mises (CVM) statistic", freq = FALSE, xlab = "T (p = 0.421)", breaks = "scott")
#  points(t0, 0, cex = 1, pch = 16)

## ----eval=FALSE----------------------------------------------------------
#  library(RANN) #for locating nearest neighbors
#  library(boot)
#  library(energy)
#  library(Ball)
#  
#  Tn <- function(z, ix, sizes,k) {
#  n1 <- sizes[1]
#  n2 <- sizes[2]
#  n <- n1 + n2
#  if(is.vector(z)) z <- data.frame(z,0)
#  z <- z[ix, ]
#  NN <- nn2(data=z, k=k+1)
#  block1 <- NN$nn.idx[1:n1,-1]
#  block2 <- NN$nn.idx[(n1+1):n,-1]
#  i1 <- sum(block1 < n1 + .5)
#  i2 <- sum(block2 > n1+.5)
#  return((i1 + i2) / (k * n))
#  }
#  
#  eqdist.nn <- function(z,sizes,k){
#    boot.obj <- boot(data=z,statistic=Tn,R=R,sim = "permutation", sizes = sizes,k=k)
#    ts <- c(boot.obj$t0,boot.obj$t)
#    p.value <- mean(ts>=ts[1])
#    list(statistic=ts[1],p.value=p.value)
#  }
#  
#  m <- 50
#  k <- 3
#  p <- 2
#  n1 <- n2 <- 50
#  n <- n1 + n2
#  N <- c(n1,n2)
#  R <- 999
#  p.values <- matrix(NA,m,3) #storage for p.values
#  set.seed(1)
#  
#  
#  ##Situation 1: Unequal variances and equal expectation
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*p,mean = 1,sd = 1),ncol=p)
#    y <- matrix(rnorm(n2*p,mean = 1,sd = 1.5),ncol=p) #x and y have unequal variances and equal expectations
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value  #Nearest neighbor (NN) test
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value  #Energy test
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed = i*1)$p.value  #Ball test
#  }
#  alpha <- 0.1  #the confidence level
#  pow <- colMeans(p.values < alpha)  #compute the mean of p.values which is less than 0.1
#  print(pow)
#  barplot(pow, main = "unequal variances and equal expectations", xlab = "power comparison", names.arg=c("NN","energy","ball"), col="lightblue")
#  
#  
#  ##Situation 2: Unequal variances and unequal expectations
#  for(i in 1:m){
#    x <- matrix(rnorm(n1*p,mean = 0.5,sd = 1),ncol=p)
#    y <- matrix(rnorm(n2*p,mean = 0.1,sd = 1.4),ncol=p) #x and y have unequal variances and unequal expectations
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed = i*1)$p.value
#  }
#  alpha <- 0.1
#  pow <- colMeans(p.values < alpha)
#  print(pow)
#  barplot(pow, main = "unequal variances and unequal expectations", xlab = "power comparison", names.arg=c("NN","energy","ball"), col="gray")
#  
#  
#  ##Situation 3: Non-normal distributions
#  for(i in 1:m){
#    x <- matrix(rt(n1*p,df = 1),ncol=p) #t distribution with 1 df (heavy-tailed distribution)
#    y <- cbind(rnorm(n2,mean = 0,sd = 1),rnorm(n2,mean = 0.2,sd = 2)) #bimodel distribution (mixture of two normal distributions)
#    z <- rbind(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed = i*1)$p.value
#  }
#  alpha <- 0.1;
#  pow <- colMeans(p.values<alpha)
#  print(pow)
#  barplot(pow, main = "non-normal distributions", xlab = "power comparison", names.arg=c("NN","energy","ball"), col="orange")
#  
#  
#  ##Situation 4: Unbalanced samples
#  n1 <- 10
#  n2 <- 100
#  n <- n1+n2
#  N <- c(n1,n2)
#  for(i in 1:m){
#    x <- c(rnorm(n1,mean = 1,sd = 1)) #the number of samples for x is 10
#    y <- c(rnorm(n2,mean = 2,sd = 2)) #the number of samples for y is 100
#    z <- c(x,y)
#    p.values[i,1] <- eqdist.nn(z,N,k)$p.value
#    p.values[i,2] <- eqdist.etest(z,sizes=N,R=R)$p.value
#    p.values[i,3] <- bd.test(x=x,y=y,R=999,seed = i*12)$p.value
#  }
#  alpha <- 0.1;
#  pow <- colMeans(p.values<alpha)
#  print(pow)
#  barplot(pow, main = "unbalanced samples", xlab = "power comparison", names.arg=c("NN","energy","ball"), col="lightgreen")

## ----eval=FALSE----------------------------------------------------------
#  ##Generate random variables from a standard Cauchy distribution
#  f <- function(x, theta=1, eta=0) {
#    return(1/(pi*theta*(1+((x-eta)/theta)^2)))
#  }  #the standard Cauchy density function
#  
#  set.seed(1)
#  m <- 10000
#  x <- numeric(m)
#  x[1] <- rnorm(1,mean=0,sd=10) #generate X0 from distribution N(0,10) and store in x[1]
#  k <- 0
#  u <- runif(m) #generate U from Uniform(0,1)
#  
#  for (i in 2:m) {
#    xt <- x[i-1]
#    y <- rnorm(1, mean = xt, sd=10)
#    num <- f(y, theta=1, eta=0)*dnorm(xt, mean = y, sd=10)
#    den <- f(xt, theta=1, eta=0)*dnorm(y, mean = xt, sd=10)
#    if(u[i] <= num/den){
#      x[i] <- y
#    }
#    else {
#      x[i] <- xt
#      k <- k+1  #y is rejected
#    }
#  }
#  
#  plot(x, type="l", main="", ylab="x")

## ----eval=FALSE----------------------------------------------------------
#  ##Discard the first 1000 of the chain
#  x1 <- x[1001:m]
#  
#  ##Compare the deciles of the generated observations with the deciles of the standard Cauchy distribution
#  generated_observations <- quantile(x1, seq(0, 1, 0.1))
#  standard_Cauchy <- qcauchy(seq(0, 1, 0.1))
#  decile <- data.frame(generated_observations, standard_Cauchy)
#  decile
#  
#  ##Compare the quantiles of the generated observations with the quantiles of the standard Cauchy distribution
#  b <- 1001  #discard the burnin sample
#  y <- x[b:m]
#  a <- ppoints(100)
#  QR <- qcauchy(a)  #quantiles of the standard Cauchy distribution
#  Q <- quantile(x, a)
#  qqplot(QR, Q, xlim=c(-2,2), ylim=c(-2,2), main="",xlab="Standard Cauchy Distribution Quantiles", ylab="Generated Observations Quantiles")
#  
#  hist(y, breaks="scott", main="",  xlim=c(-10,10), xlab="", freq=FALSE)
#  lines(QR, f(QR, theta=1, eta=0))

## ----eval=FALSE----------------------------------------------------------
#  w <- 0.25 #width of the uniform support set
#  m <- 5000  #length of the chain
#  burn <- 1000  #burn-in time
#  group_size <- c(125,18,20,34)  #group size
#  x <- numeric(m) #the chain
#  
#  prob <- function(theta,group_size){
#    if(theta<0 || theta>1)
#      return(0)
#    else
#    return((1/2+theta/4)^group_size[1]*((1-theta)/4)^group_size[2]*((1-theta)/4)^group_size[3]*(theta/4)^group_size[4])
#  }
#  set.seed(12345)
#  u <- runif(m)  #for accept/reject step
#  v <- runif(m, -w, w)  #proposal distribution
#  x[1] <- 0.25
#  for (i in 2:m) {
#    theta <- x[i-1]+v[i]
#    if(u[i]<= prob(theta,group_size)/prob(x[i-1],group_size))
#      x[i] <- theta
#    else
#      x[i] <- x[i-1]
#  }
#  
#  xtheta <- x[(burn+1):m]
#  theta.hat <- mean(xtheta) #the estimator of posterior distribution of ¦È
#  print(theta.hat)
#  
#  ##Obtain the estimated values of group sizes
#  group_size.hat <- sum(group_size) * c((2+theta.hat)/4, (1-theta.hat)/4, (1-theta.hat)/4, theta.hat/4)
#  round(group_size.hat)

## ----eval=FALSE----------------------------------------------------------
#  ##EM algorithm
#  em_function <- function(theta0) {
#    maxit = 1000
#    releps = 1e-09
#    i <- 0
#    theta1 <- theta0
#    theta0 <- theta1 + 1
#    while((i != maxit) && (abs(theta1 - theta0) > releps * abs(theta0))) {
#     i <- i + 1
#     theta0 <- theta1
#     theta1 <- ((125 * theta1)/(2 + theta1) + 34)/((125 * theta1)/(2 + theta1) + 34 + 18 + 20)
#     print(c(theta0,theta1,i))
#    }
#    return(theta1)
#  }
#  
#  em_theta.hat <- em_function(0.5) #the estimator of posterior distribution of ¦È
#  print(em_theta.hat)
#  
#  ##Obtain the estimated values of group sizes
#  em_group_size.hat <- sum(group_size) * c((2+em_theta.hat)/4, (1-em_theta.hat)/4, (1-em_theta.hat)/4, em_theta.hat/4)
#  round(em_group_size.hat)

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1234)
#  group<-c(125,18,20,34) #group sizes
#  k<-4 #number of chains to generate
#  N<-15000 #length of chains
#  b<-1000 #burn-in length
#  
#  Gelman.Rubin <-function(psi){
#    #psi[i,j] is the statistic psi(X[i,1:j])
#    #for chain in i-th row of X
#    psi<-as.matrix(psi)
#    n<-ncol(psi)
#    k<-nrow(psi)
#  
#    psi.means<-rowMeans(psi) #row means
#    B<-n*var(psi.means) #between variance est.
#    psi.w<-apply(psi, 1, "var") #within variances
#    W<-mean(psi.w) #within est.
#    v.hat<-W*(n-1)/n+(B/n) #upper variance est.
#    r.hat<-v.hat/W #G-R statistic
#    return(r.hat)
#  }
#  
#  prob<-function(theta,group){
#    if(theta<0||theta>=1)
#      return(0)
#    else
#      return((1/2+theta/4)^group[1]*((1-theta)/4)^group[2]*((1-theta)/4)^group[3]*(theta/4)^group[4])
#  }
#  
#  chain<-function(group,N,X1){
#    #generates a Metropolis chain for Normal(0,1)
#    #with Normal(X[t], sigma) proposal distribution
#    #and starting value X1
#    x<-numeric(N)
#    x[1]<-X1
#    w<-0.25
#    u<-runif(N) #for accept/reject step
#    v<-runif(N,-w,w) #proposal distribution
#    for (i in 2:N){
#      theta<-x[i-1]+v[i]
#      if(u[i]<=prob(theta,group)/prob(x[i-1],group))
#        x[i]<-theta
#      else
#        x[i]<-x[i-1]
#    }
#    return(x)
#  }
#  
#  #choose overdispersed initial values
#  x0<-c(0.2,0.4,0.6,0.8)
#  
#  #generate the chains
#  X<-matrix(0,nrow = k,ncol = N)
#  for (i in 1:k){
#     X[i, ]<-chain(group,N,x0[i])
#  }
#  
#  #compute diagnostic statistics
#  psi<-t(apply(X,1,cumsum))
#  for (i in 1:nrow(psi)){
#     psi[i, ] <- psi[i, ]/(1:ncol(psi))
#  }
#  print(Gelman.Rubin(psi))
#  
#  #plot psi for the four chains
#  par(mfrow=c(2,2))
#  for (i in 1:k){
#     plot(psi[i,(b+1):N],type = "l",xlab = i,ylab = bquote(psi))
#  }
#  par(mfrow=c(1,1))
#  
#  #plot the sequence of R-hat statistics
#  rhat<-rep(0,N)
#  for (j in (b+1):N){
#     rhat[j]<-Gelman.Rubin(psi[,1:j])
#  }
#  plot(rhat[(b+1):N],type = "l",xlab = " ",ylab = "R")
#  abline(h=1.2,lty=2)
#  abline(h=1.1,lty=2)

## ----eval=FALSE----------------------------------------------------------
#  intersection <- function (k) {
#    s.k.minus.one <- function (a) {
#      1-pt(sqrt(a^2 * (k - 1) / (k - a^2)), df = k-1)
#    } #the function of S_{k-1}(a)
#    s.k <- function (a) {
#      1-pt(sqrt(a^2 * k / (k + 1 - a^2)), df = k)
#    } #the function of S_{k}(a)
#    f <- function (a) {
#      s.k(a) - s.k.minus.one(a)
#    } #the root of f is the intersection points
#  
#    eps <- .Machine$double.eps^0.5
#    return(uniroot(f, interval = c(eps, sqrt(k)-eps))$root) #find the intersection points A(k) in (0,sqrt(k))
#  }
#  
#  k <- c(4:25, 100, 500, 1000)
#  rs <- sapply(k, function (k) {
#    intersection(k)
#    })
#  points <- cbind(k,rs)
#  print(points)

## ----eval=FALSE----------------------------------------------------------
#  # Write a function to compute the cdf of the Cauchy distribution
#  my.dcauchy <- function (x, eta, theta) {
#    stopifnot(theta > 0)
#    return(1/(theta*pi*(1 + ((x - eta)/theta)^2)))
#  } #the density function of the Cauchy distribution
#  
#  my.pcauchy <- function (x, eta, theta) {
#    stopifnot(theta > 0)
#    integral <- function (x) {
#      my.dcauchy(x, eta, theta)
#    }
#    return(integrate(integral, lower = -Inf, upper = x, rel.tol=.Machine$double.eps^0.25)$value)
#  } #the value of the integrand is the cdf of the Cauchy distribution
#  
#  
#  # Compare my results to the results from the R function pcauchy
#  # Cauchy(0,1)
#  eta <- 0 #the parameter value of eta is 0
#  theta <- 1 #the parameter value of theta is 1
#  xs <- seq(-10, 10)
#  names(xs) <-c(seq(-10, 10))
#  estimate <- sapply(xs, function(x) my.pcauchy(x, eta, theta))
#  truth <- sapply(xs, function(x) pcauchy(x, eta, theta))
#  round(rbind(estimate, truth), 4)
#  
#  # Cauchy(1,2)
#  eta <- 1 #the parameter value of eta is 1
#  theta <- 2 #the parameter value of eta is 2
#  xs <- seq(-10, 10)
#  names(xs) <-c(seq(-10, 10))
#  estimate <- sapply(xs, function(x) my.pcauchy(x, eta, theta))
#  truth <- sapply(xs, function(x) pcauchy(x, eta, theta))
#  round(rbind(estimate, truth), 4)

## ----eval=FALSE----------------------------------------------------------
#  # Write the log-likelihood function
#  lnL <- function(p, q, nA = 28, nB = 24, nOO = 41, nAB = 70) {
#    r = 1 - p - q
#    nA * log(p^2 + 2*p*r) + nB * log(q^2 + 2 * q * r) + 2 * nOO * log(r) + nAB * log(2 * p * q)
#  }
#  
#  # Write the E-M function
#  EM <- function (p, q, nA = 28, nB = 24, nOO = 41, nAB = 70, debug = FALSE) {
#  
#    # Evaluate the likelihood using initial estimates
#    llk <- lnL(p, q, nA, nB, nOO, nAB)
#  
#    # Count the number of iterations so far
#    iter <- 1
#  
#    # Loop until convergence
#    while (TRUE)
#    {
#      # Estimate the frequency for allele O
#      r = 1 - p - q
#  
#      # First we carry out the E-step
#  
#      # The counts for genotypes O/O and A/B are effectively observed
#      # Estimate the counts for the other genotypes
#      nAA <- nA * p / (p + 2*r)
#      nAO <- nA - nAA
#      nBB <- nB * q / (q + 2*r)
#      nBO <- nB - nBB
#  
#      # Print debugging information
#      if (debug)
#      {
#        cat("Round #", iter, "lnLikelihood = ", llk, "\n")
#        cat("    Allele frequencies: p = ", p, ", q = ", q, ", r = ", r, "\n")
#        cat("    Genotype counts:    nAA = ", nAA, ", nAO = ", nAO, ", nBB = ", nBB,
#            ", nBO = ", nBO, "\n")
#      }
#  
#      # Then the M-step
#      p <- (2 * nAA + nAO + nAB) / (2 * (nA + nB + nOO + nAB))
#      q <- (2 * nBB + nBO + nAB) / (2 * (nA + nB + nOO + nAB))
#  
#      # Then check for convergence
#      llk1 <- lnL(p, q, nA, nB, nOO, nAB)
#  
#      if (abs(llk1 - llk) < (abs(llk) + abs(llk1)) * 1e-6) break
#  
#      # Otherwise keep going
#      llk <- llk1
#      iter <- iter + 1
#    }
#    list(p = p, q = q, r=1-p-q)
#  }
#  
#  # Set the initial estimate of the target parameters, then run the E-M function.
#  EM(0.3,0.25,nA = 28, nB = 24, nOO = 41, nAB = 70, debug = TRUE)

## ----eval=FALSE----------------------------------------------------------
#  out <- vector("list", length(x))
#  for (i in seq_along(x)) {
#    out[[i]] <- f(x[[i]], ...)
#  }

## ----eval=FALSE----------------------------------------------------------
#  formulas <- list(
#    mpg ~ disp,
#    mpg ~ I(1 / disp),
#    mpg ~ disp + wt,
#    mpg ~ I(1 / disp) + wt
#  )
#  
#  # Use for loops to fit linear models to the mtcars
#  mtcars_models_loop <- vector("list", length(formulas)) # storage for replicates
#  for(i in seq_along(formulas))
#  {
#    mtcars_models_loop[[i]] <- lm(formulas[[i]], data = mtcars) # fit linear models
#    print(mtcars_models_loop[[i]])
#  }
#  
#  # Use lapply() to fit linear models to the mtcars
#  mtcars_models_lapply <- lapply(formulas, function(x) lm(x, data = mtcars))
#  print(mtcars_models_lapply)

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  bootstraps <- lapply(1:10, function(i) {
#    rows <- sample(1:nrow(mtcars), rep = TRUE)
#    mtcars[rows, ]
#  })
#  
#  # Use a for loop
#  bootstrap_models_loop <- vector("list",length(bootstraps)) # storage for replicates
#  for(i in seq_along(bootstraps)){
#    bootstrap_models_loop[[i]] <- lm(mpg ~ disp, data = bootstraps[[i]]) # fit the model
#    print(bootstrap_models_loop[[i]])
#  }
#  
#  # Use lapply() without an anonymous function
#  bootstrap_models_lapply <- lapply(bootstraps,lm,formula=mpg ~ disp)
#  print(bootstrap_models_lapply)

## ----eval=FALSE----------------------------------------------------------
#  # Model in question 1
#  formulas <- list(
#    mpg ~ disp,
#    mpg ~ I(1 / disp),
#    mpg ~ disp + wt,
#    mpg ~ I(1 / disp) + wt
#  )
#  
#  # Use for loops to fit linear models to the mtcars
#  mtcars_models_loop <- vector("list", length(formulas)) # storage for replicates
#  for(i in seq_along(formulas))
#  {
#    mtcars_models_loop[[i]] <- lm(formulas[[i]], data = mtcars) # fit linear models
#  }
#  
#  # Use lapply() to fit linear models to the mtcars
#  mtcars_models_lapply <- lapply(formulas, function(x) lm(x, data = mtcars))
#  
#  
#  # Model in question 2
#  set.seed(1)
#  bootstraps <- lapply(1:10, function(i) {
#    rows <- sample(1:nrow(mtcars), rep = TRUE)
#    mtcars[rows, ]
#  })
#  
#  # Use a for loop
#  bootstrap_models_loop <- vector("list",length(bootstraps)) # storage for replicates
#  for(i in seq_along(bootstraps)){
#    bootstrap_models_loop[[i]] <- lm(mpg ~ disp, data = bootstraps[[i]]) # fit the model
#  }
#  
#  # Use lapply() without an anonymous function
#  bootstrap_models_lapply <- lapply(bootstraps,lm,formula=mpg ~ disp)
#  
#  
#  rsq <- function(mod) summary(mod)$r.squared # the function to extract R2
#  
#  # Extract R2 for the model in question 1
#  mtcars_R2_loop <- sapply(mtcars_models_loop, rsq)
#  mtcars_R2_lapply <- sapply(mtcars_models_lapply, rsq)
#  rbind(mtcars_R2_loop,mtcars_R2_lapply)
#  
#  # Extract R2 for the model in question 2
#  bootstrap_R2_loop <- sapply(bootstrap_models_loop, rsq)
#  bootstrap_R2_lapply <- sapply(bootstrap_models_lapply, rsq)
#  rbind(bootstrap_R2_loop,bootstrap_R2_lapply)

## ----eval=FALSE----------------------------------------------------------
#  set.seed(1)
#  
#  # Simulate the performance of a t-test for non-normal data
#  trials <- replicate(
#    100,
#    t.test(rpois(10, 10), rpois(7, 10)),
#    simplify = FALSE
#  )
#  
#  # Use sapply() and an anonymous function to extract the p-value from every trial
#  p1 <- sapply(trials, function(x) x$p.value)
#  
#  # Get rid of the anonymous function by using [[ directly
#  p2 <- sapply(trials, '[[', i = "p.value")
#  cbind(p1,p2)

## ----eval=FALSE----------------------------------------------------------
#  library(parallel)
#  vapply_Map <- function(x, f, FUN.VALUE, ...){
#    vapply(x, Map(f, ...), FUN.VALUE)
#  }

## ----eval=FALSE----------------------------------------------------------
#  expected <- function(colsum, rowsum, total) {
#    (colsum / total) * (rowsum / total) * total
#  } # the expected (theoretical) count of type i
#  
#  chi_stat <- function(observed, expected) {
#    ((observed - expected) ^ 2) / expected
#  } # a measure of the approximation between observed value and expected value
#  
#  # Make a faster version of chisq.test() that only computes the chi-square test statistic when the input is two numeric vectors with no missing values
#  chisq_test_faster <- function(x, y) {
#    total <- sum(x) + sum(y)
#    rowsum_x <- sum(x)
#    rowsum_y <- sum(y)
#    chistat <- 0
#    for (i in seq_along(x)) {
#      colsum <- x[i] + y[i]
#      expected_x <- expected(colsum, rowsum_x, total)
#      expected_y <- expected(colsum, rowsum_y, total)
#      chistat <- chistat + chi_stat(x[i], expected_x)
#      chistat <- chistat + chi_stat(y[i], expected_y)
#    }
#   chistat
#  } # code from the mathematical definition
#  
#  # Validate that the function chisq_test_faster() is reasonable
#  chisq_test_faster(c(367,342,266,329),c(56,40,20,16))
#  chisq.test(as.table(rbind(c(367,342,266,329),c(56,40,20,16))))
#  
#  # Validate that the function chisq_test_faster() is faster than chisq.test()
#  print(microbenchmark::microbenchmark(
#    chisq_test_faster = chisq_test_faster(c(367,342,266,329),c(56,40,20,16)),
#    chisq.test = chisq.test(as.table(rbind(c(367,342,266,329),c(56,40,20,16))))
#  ))
#  

## ----eval=FALSE----------------------------------------------------------
#  # Make a faster version of table() for the case of an input of two integer vectors with no missing values
#  table_faster <- function(x, y) {
#    x_unique <- unique(x) # extract unique elements of x
#    y_unique <- unique(y) # extract unique elements of y
#    mat <- matrix(0L, length(x_unique), length(y_unique)) # storage for replicates
#    for (i in seq_along(x)) {
#      mat[which(x_unique == x[[i]]), which(y_unique == y[[i]])] <-
#        mat[which(x_unique == x[[i]]),  which(y_unique == y[[i]])] + 1L
#    } #build a contingency table of the counts at each combination of factor levels
#    # Optimize the form of output
#    dimnames <- list(x_unique, y_unique)
#    names(dimnames) <- as.character(as.list(match.call())[-1])
#    tab <- array(mat, dim = dim(mat), dimnames = dimnames)
#    class(tab) <- "table"
#    tab
#  }
#  
#  # Validate that the function table_faster() is reasonable
#  x <- c(1, 2, 2, 3, 1, 3, 3, 2)
#  y <- c(1, 1, 2, 1, 1, 1, 1, 2)
#  table_faster(x, y)
#  table(x, y)
#  
#  # Validate that the function table_faster() is faster than table()
#  print(microbenchmark::microbenchmark(
#    table_faster = table_faster(x, y),
#    table = table(x, y)
#  ))
#  

