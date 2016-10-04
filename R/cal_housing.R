

#' California Median House Prices
#'
#' Data of California median housing prices
#'
#' @docType data
#'
#' @usage data(calHousing)
#'
#' @format An object of class \code{"data.frame"}.
#'
#' @keywords datasets
#'
#' @examples
#' data(calHousing)
#' medianValue <- log(calHousing$medianValue)
#' x.matrix    <- data.matrix(calHousing[,-match("medianValue", colnames(calHousing))])
#' 
#' # Fit with oem
#' 
#' calfit    <- oem(x = x.matrix, y = medianValue, 
#'                  penalty = "lasso", gamma = 3, maxit = 2500L)
#' 
#' plot(calfit)
#' 
#' calfit.cv <- xval.oem(x = x.matrix, y = medianValue,
#'                       penalty = "lasso", maxit = 2500L)
#'                       
#' plot(calfit.cv)
#' 
"calHousing"