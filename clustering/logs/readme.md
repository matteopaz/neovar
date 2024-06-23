## masterlog.txt
For errors and successes on all partitions
## errors.txt
Just errors
## progress.txt
Internal file used to track progress and generate plots. Format is

Partition id _ Status value _ Num rows processed _ num rows included _ time completed
Status value is 0 for error, 1 for in progress, 2 for complete
## individual log files
Used to track individual jobs in more detail