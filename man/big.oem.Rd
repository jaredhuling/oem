% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/big_oem.R
\name{big.oem}
\alias{big.oem}
\title{Orthogonalizing EM for big.matrix objects}
\usage{
big.oem(
  x,
  y,
  family = c("gaussian", "binomial"),
  penalty = c("elastic.net", "lasso", "ols", "mcp", "scad", "mcp.net", "scad.net",
    "grp.lasso", "grp.lasso.net", "grp.mcp", "grp.scad", "grp.mcp.net", "grp.scad.net",
    "sparse.grp.lasso"),
  weights = numeric(0),
  lambda = numeric(0),
  nlambda = 100L,
  lambda.min.ratio = NULL,
  alpha = 1,
  gamma = 3,
  tau = 0.5,
  groups = numeric(0),
  penalty.factor = NULL,
  group.weights = NULL,
  standardize = TRUE,
  intercept = TRUE,
  maxit = 500L,
  tol = 1e-07,
  irls.maxit = 100L,
  irls.tol = 0.001,
  compute.loss = FALSE,
  gigs = 4,
  hessian.type = c("full", "upper.bound")
)
}
\arguments{
\item{x}{input big.matrix object pointing to design matrix 
Each row is an observation, each column corresponds to a covariate}

\item{y}{numeric response vector of length nobs.}

\item{family}{\code{"gaussian"} for least squares problems, \code{"binomial"} for binary response. 
\code{"binomial"} currently not available.}

\item{penalty}{Specification of penalty type. Choices include:
\itemize{
   \item \code{"elastic.net"} - elastic net penalty, extra parameters: \code{"alpha"}
   \item \code{"lasso"} - lasso penalty
   \item \code{"ols"} - ordinary least squares
   \item \code{"mcp"} - minimax concave penalty, extra parameters: \code{"gamma"}
   \item \code{"scad"} - smoothly clipped absolute deviation, extra parameters: \code{"gamma"}
   \item \code{"mcp.net"} - minimax concave penalty + l2 penalty, extra parameters: 
   \code{"gamma"}, \code{"alpha"}
   \item \code{"scad.net"} - smoothly clipped absolute deviation + l2 penalty, extra parameters: 
   \code{"gamma"}, \code{"alpha"}
   \item \code{"grp.lasso"} - group lasso penalty
   \item \code{"grp.lasso.net"} - group lasso penalty + l2 penalty, extra parameters: \code{"alpha"}
   \item \code{"grp.mcp"} - group minimax concave penalty, extra parameters: \code{"gamma"}
   \item \code{"grp.scad"} - group smoothly clipped absolute deviation, extra parameters: \code{"gamma"}
   \item \code{"grp.mcp.net"} - group minimax concave penalty + l2 penalty, extra parameters: \code{"gamma"}, \code{"alpha"}
   \item \code{"grp.scad.net"} - group smoothly clipped absolute deviation + l2 penalty, extra parameters: \code{"gamma"}, \code{"alpha"}
   \item \code{"sparse.grp.lasso"} - sparse group lasso penalty (group lasso + lasso), extra parameters: \code{"tau"}
}
Careful consideration is required for the group lasso, group MCP, and group SCAD penalties. Groups as specified by the \code{groups} argument 
should be chosen in a sensible manner.}

\item{weights}{observation weights. Not implemented yet. Defaults to 1 for each observation (setting weight vector to 
length 0 will default all weights to 1)}

\item{lambda}{A user supplied lambda sequence. By default, the program computes
its own lambda sequence based on \code{nlambda} and \code{lambda.min.ratio}. Supplying
a value of lambda overrides this.}

\item{nlambda}{The number of lambda values - default is 100.}

\item{lambda.min.ratio}{Smallest value for lambda, as a fraction of \code{lambda.max}, the (data derived) entry
value (i.e. the smallest value for which all coefficients are zero). The default
depends on the sample size nobs relative to the number of variables nvars. If
\code{nobs > nvars}, the default is 0.0001, close to zero. If \code{nobs < nvars}, the default
is 0.01. A very small value of \code{lambda.min.ratio} will lead to a saturated fit
when \code{nobs < nvars}.}

\item{alpha}{mixing value for \code{elastic.net}, \code{mcp.net}, \code{scad.net}, \code{grp.mcp.net}, \code{grp.scad.net}. 
penalty applied is (1 - alpha) * (ridge penalty) + alpha * (lasso/mcp/mcp/grp.lasso penalty)}

\item{gamma}{tuning parameter for SCAD and MCP penalties. must be >= 1}

\item{tau}{mixing value for \code{sparse.grp.lasso}. penalty applied is (1 - tau) * (group lasso penalty) + tau * (lasso penalty)}

\item{groups}{A vector of describing the grouping of the coefficients. See the example below. All unpenalized variables
should be put in group 0}

\item{penalty.factor}{Separate penalty factors can be applied to each coefficient. 
This is a number that multiplies lambda to allow differential shrinkage. Can be 0 for some variables, 
which implies no shrinkage, and that variable is always included in the model. Default is 1 for all 
variables.}

\item{group.weights}{penalty factors applied to each group for the group lasso. Similar to \code{penalty.factor}, 
this is a number that multiplies lambda to allow differential shrinkage. Can be 0 for some groups, 
which implies no shrinkage, and that group is always included in the model. Default is sqrt(group size) for all
groups.}

\item{standardize}{Logical flag for x variable standardization, prior to fitting the models. 
The coefficients are always returned on the original scale. Default is \code{standardize = TRUE}. If 
variables are in the same units already, you might not wish to standardize. Keep in mind that 
standardization is done differently for sparse matrices, so results (when standardized) may be
slightly different for a sparse matrix object and a dense matrix object}

\item{intercept}{Should intercept(s) be fitted (\code{default = TRUE}) or set to zero (\code{FALSE})}

\item{maxit}{integer. Maximum number of OEM iterations}

\item{tol}{convergence tolerance for OEM iterations}

\item{irls.maxit}{integer. Maximum number of IRLS iterations}

\item{irls.tol}{convergence tolerance for IRLS iterations. Only used if \code{family != "gaussian"}}

\item{compute.loss}{should the loss be computed for each estimated tuning parameter? Defaults to \code{FALSE}. Setting
to \code{TRUE} will dramatically increase computational time}

\item{gigs}{maximum number of gigs of memory available. Used to figure out how to break up calculations
involving the design matrix x}

\item{hessian.type}{only for logistic regression. if \code{hessian.type = "full"}, then the full hessian is used. If
\code{hessian.type = "upper.bound"}, then an upper bound of the hessian is used. The upper bound can be dramatically
faster in certain situations, ie when n >> p}
}
\value{
An object with S3 class "oem"
}
\description{
Orthogonalizing EM for big.matrix objects
}
\examples{
\dontrun{
set.seed(123)
nrows <- 50000
ncols <- 100
bkFile <- "bigmat.bk"
descFile <- "bigmatk.desc"
bigmat <- filebacked.big.matrix(nrow=nrows, ncol=ncols, type="double",
                                backingfile=bkFile, backingpath=".",
                                descriptorfile=descFile,
                                dimnames=c(NULL,NULL))

# Each column value with be the column number multiplied by
# samples from a standard normal distribution.
set.seed(123)
for (i in 1:ncols) bigmat[,i] = rnorm(nrows)*i

y <- rnorm(nrows) + bigmat[,1] - bigmat[,2]

fit <- big.oem(x = bigmat, y = y, 
               penalty = c("lasso", "elastic.net", 
                           "ols", 
                           "mcp",       "scad", 
                           "mcp.net",   "scad.net",
                           "grp.lasso", "grp.lasso.net",
                           "grp.mcp",   "grp.scad",
                           "sparse.grp.lasso"), 
               groups = rep(1:20, each = 5))
               
fit2 <- oem(x = bigmat[,], y = y, 
            penalty = c("lasso", "grp.lasso"), 
            groups = rep(1:20, each = 5))   
           
max(abs(fit$beta[[1]] - fit2$beta[[1]]))            

layout(matrix(1:2, ncol = 2))
plot(fit)
plot(fit, which.model = 2)
}

}
\references{
Huling. J.D. and Chien, P. (2022), Fast Penalized Regression and Cross Validation for Tall Data with the oem Package.
Journal of Statistical Software 104(6), 1-24. doi:10.18637/jss.v104.i06
}
