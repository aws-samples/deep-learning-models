# Fields to be tracked in logs

`time`: training time per iteration (including data access time)

`data_time`: time to access data per iteration

`bbox_mAP_copypaste`: bounding box scores

`segm_mAP_copypaste`: segmentation scores (if available,)

`eval_time`: time taken to run evaluation

Throughput (imgs/sec) can be calculated as: `(imgs_per_gpu * total gpus) / time` averaged over multiple entries
