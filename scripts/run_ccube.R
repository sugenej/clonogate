library(ccube)
library(dplyr)
args <- commandArgs(TRUE)
set.seed(17)
input <- read.delim(args[1], stringsAsFactors=FALSE)
result <- RunCcubePipeline(ssm=input,numOfClusterPool=1:6,numOfRepeat=3,
                          runAnalysis=TRUE,runQC=TRUE,multiCore=FALSE)
stopifnot(nrow(result$ssm)==length(result$res$label))
position <- match(result$ssm$mutation_id,input$mutation_id)
stopifnot(!anyNA(position), !anyDuplicated(position))
labels <- rep(NA_integer_,nrow(input))
labels[position] <- match(result$res$label,sort(unique(result$res$label)))
cp <- rep(0,nrow(input))
cp[position] <- result$ssm$ccube_ccf_mean*input$purity[position]
if(any(!is.finite(cp)) || any(cp<0) || any(cp>1)) stop('Invalid CCube CP output')
k <- max(labels,na.rm=TRUE)
# Any rows omitted by the public pipeline are retained as an explicit zero-CP
# group for scoring. This output adapter is not a historical submission.
labels[is.na(labels)] <- k+1L
tab <- do.call(rbind,lapply(sort(unique(labels)),function(x) c(x,sum(labels==x),mean(cp[labels==x]))))
write.table(input$purity[1], '1A.txt',row.names=FALSE,col.names=FALSE,quote=FALSE)
write.table(k,'1B.txt',row.names=FALSE,col.names=FALSE,quote=FALSE)
write.table(tab,'1C.txt',sep='\t',row.names=FALSE,col.names=FALSE,quote=FALSE)
write.table(labels,'2A.txt',row.names=FALSE,col.names=FALSE,quote=FALSE)
saveRDS(result,'result.rds')
