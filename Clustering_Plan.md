# Approaches

Two potential approaches that might work:

1. Cluster AND Run on Cloud

Pro:
- Lots of CPU and RAM => fast clustering
- Fast transfer with database

Con:
- No GPU!!!
- Still have to transfer back to IRSA for new columns

So, would have to explore getting GPUs on the cloud.

2. Cluster on Cloud, run on IRSAGPU
Pro: 
- Fast transfer on cloud
- Fast clustering
- Fast inference
- Data already transfered for new columns

Con:
- Need to write on cloud
- Need to transfer from cloud to IRSA after clustering

So, would need to make sure transfer of this data is tractable. *ONLY NEED TO TRANSFER 2 cols: sourceid and clusternum

1. Cluster AND Run on IRSA
(Possible ONLY if parquet methods possible on inhouse data)

Pro:
- No transfer involved
- Fast inference w GPU

Con:
- Low CPU and RAM

So, need to see what speed my current RAM and CPU on irsagpu2 will yield for clustering the entire dataset.
